#!/usr/bin/env python3
"""Nyra - lanceur 24/7 : serveur webchat + sessions successives surveillees.

Une session, c'est ``session_min`` minutes d'entrainement texte, puis un
checkpoint, sa promotion en ``data/genome.pt``, une ligne ``data/metrics.json``,
et on recommence. Deux surveillances tournent a cote :

* la VRAM prise par les AUTRES processus (le jeu d'Alex) : au-dela du seuil,
  pause propre avec sauvegarde, reprise automatique quand elle est liberee ;
* la temperature du GPU : pause de refroidissement au-dessus de ``temp_pause``,
  arret complet au-dessus de ``temp_stop``, plus une pause planifiee toutes les
  ``intervalle_thermique_min`` minutes d'entrainement cumulees.

Chaque regle de securite sauvegarde AVANT de s'arreter ou de se mettre en
pause : un crash ne doit jamais couter plus que l'intervalle de sauvegarde.
La configuration est relue dans ``data/nyra_config.json`` avant chaque
session : rien de ce qui decide d'une duree, d'un seuil ou d'un batch n'est
ecrit ici en dur.

    python scripts/nyra_launcher.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from mycelia import nyra_control as nc
from mycelia.config import load_config
from mycelia.train import Trainer, acquire_lock, process_memory, release_lock

PORT = 8080
DEVICE = "cuda"
WEBCHAT = ROOT / "scripts" / "webchat.py"
WEBCHAT_LOG = ROOT / "data" / "runs" / "logs_webchat_nyra2.log"
LOG_PATH = ROOT / "data" / "runs" / "nyra_launcher.log"
LOG_LOCK = threading.Lock()
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
MAX_GUARD_HITS = 3
MAX_CRASH_STREAK = 3


class StopLauncher(Exception):
    """Arret demande par une regle de securite ou par le panel."""


def log(message: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    with LOG_LOCK:
        print(line, flush=True)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass


# ----------------------------------------------------------------------
# GPU : une seule requete nvidia-smi, compteurs par processus indisponibles
# ----------------------------------------------------------------------
def gpu_stats() -> dict:
    """Temperature, VRAM totale utilisee, occupation — en une requete.

    ``pynvml`` n'est pas installe sur cette machine, et les compteurs par
    processus de ``nvidia-smi`` repondent ``[N/A]`` sous WDDM (verifie) : la
    VRAM des AUTRES processus se deduit donc du total utilise par le GPU moins
    celle que ce processus a reservee. C'est exact sur ce portable ou rien
    d'autre que le jeu et nous ne partage la carte intelligemment.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0 or not (out.stdout or "").strip():
        return {}
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 4:
        return {}
    try:
        return {
            "temp_c": float(parts[0]),
            "gpu_used_mb": float(parts[1]),
            "gpu_total_mb": float(parts[2]),
            "gpu_pct": float(parts[3]),
        }
    except ValueError:
        return {}


# ----------------------------------------------------------------------
# Surveillance : jeu, temperature, arret demande par le panel
# ----------------------------------------------------------------------
class Surveillance(threading.Thread):
    """Les regards lateraux, hors de la boucle d'entrainement."""

    SAMPLE_SECONDS = 15.0

    def __init__(self) -> None:
        super().__init__(daemon=True, name="nyra-surveillance")
        self._stop = threading.Event()
        self.snapshot: dict = {
            "temp_c": None,
            "gpu_used_mb": None,
            "gpu_total_mb": None,
            "gpu_pct": None,
            "other_vram_mb": None,
        }
        self.stop_reason: str | None = None
        self.stop_after_session = False
        # Mis a jour par la boucle d'entrainement.
        self.own_reserved_mb = 0.0
        self.training_seconds = 0.0
        # Etat des pauses.
        self.game_pause = False
        self.game_reason: str | None = None
        self.cooling_until = 0.0
        self.cool_reason: str | None = None
        self.last_break_at = 0.0

    # -- boucle -------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - thread
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception as error:  # un regard ne tue jamais la boucle
                log(f"surveillance: {type(error).__name__}: {error}")
            if self._stop.wait(self.SAMPLE_SECONDS):
                return

    def halt(self) -> None:
        self._stop.set()

    def _sample(self) -> None:
        cfg = nc.read_config()
        self.stop_after_session = bool(cfg["stop_after_session"])
        stats = gpu_stats()
        if stats:
            other = max(0.0, stats["gpu_used_mb"] - float(self.own_reserved_mb or 0.0))
            self.snapshot.update(stats)
            self.snapshot["other_vram_mb"] = round(other, 1)
        temp = self.snapshot.get("temp_c")
        now = time.time()

        # 1. Arret thermique : la seule regle qui coupe tout.
        if temp is not None and temp >= float(cfg["temp_stop"]):
            self.stop_reason = (
                f"temperature GPU {temp:.0f} C >= temp_stop {cfg['temp_stop']} C"
            )
            log(f"STOP thermique: {self.stop_reason}")
            self._stop.set()
            return

        # 2. Refroidissement en cours : on le termine sous le seuil.
        if self.cooling_until and now >= self.cooling_until:
            if temp is None or temp < float(cfg["temp_pause"]):
                log(f"refroidissement termine (GPU {temp if temp is not None else '?'} C)")
                self.cooling_until = 0.0
                self.cool_reason = None
            else:
                self.cooling_until = now + float(cfg["pause_thermique_min"]) * 60.0
                log(f"toujours {temp:.0f} C : refroidissement prolonge")

        # 3. Temperature au-dessus du seuil : pause de refroidissement.
        if temp is not None and temp >= float(cfg["temp_pause"]) and not self.cooling_until:
            self.cooling_until = now + float(cfg["pause_thermique_min"]) * 60.0
            self.cool_reason = (
                f"temperature {temp:.0f} C > temp_pause {cfg['temp_pause']} C"
            )
            log(f"pause thermique: {self.cool_reason}")

        # 4. Pause planifiee toutes les N minutes cumulees.
        if (
            cfg["pauses_thermiques"]
            and not self.cooling_until
            and not self.game_pause
            and self.training_seconds - self.last_break_at
            >= float(cfg["intervalle_thermique_min"]) * 60.0
        ):
            self.last_break_at = self.training_seconds
            self.cooling_until = now + float(cfg["pause_thermique_min"]) * 60.0
            self.cool_reason = (
                f"pause thermique planifiee ({cfg['pause_thermique_min']} min apres "
                f"{cfg['intervalle_thermique_min']} min d'entrainement)"
            )
            log(self.cool_reason)

        # 5. Coexistence : la VRAM des autres processus.
        other = self.snapshot.get("other_vram_mb")
        threshold = float(cfg["game_vram_mb"])
        if other is not None:
            if not self.game_pause and other > threshold:
                self.game_pause = True
                self.game_reason = (
                    f"jeu: {other:.0f} Mo de VRAM pris par un autre processus "
                    f"(> {threshold:.0f} Mo)"
                )
                log(f"pause: {self.game_reason}")
            elif self.game_pause and other <= threshold - 256:
                self.game_pause = False
                self.game_reason = None
                log(f"reprise: la VRAM des autres est redescendue a {other:.0f} Mo")

    # -- lecture ------------------------------------------------------
    def status(self) -> tuple[str, str | None]:
        if self.stop_reason:
            return "stopped", self.stop_reason
        if self.game_pause:
            return "paused", self.game_reason
        if self.cooling_until and time.time() < self.cooling_until:
            return "cooling", self.cool_reason
        return "training", None


# ----------------------------------------------------------------------
# Serveur webchat
# ----------------------------------------------------------------------
def server_alive(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/api/status", timeout=timeout
        ) as handle:
            return int(getattr(handle, "status", 200)) == 200
    except Exception:
        return False


def start_server(wait_seconds: float = 150.0):
    """Demarre le panel s'il ne repond pas deja, sans jamais bloquer le train."""
    if server_alive():
        log(f"serveur webchat deja actif sur {PORT} : reutilise")
        return None
    WEBCHAT_LOG.parent.mkdir(parents=True, exist_ok=True)
    handle = WEBCHAT_LOG.open("a", encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(WEBCHAT)],
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    log(f"serveur webchat lance (PID {proc.pid}) -> {WEBCHAT_LOG.name}")
    started = time.time()
    while time.time() - started < wait_seconds:
        if server_alive():
            log(f"serveur webchat pret : http://127.0.0.1:{PORT}/dashboard")
            return proc
        if proc.poll() is not None:
            log("serveur webchat mort au demarrage — le lanceur continue sans panel")
            return None
        time.sleep(2.0)
    log(f"serveur muet apres {wait_seconds:.0f} s — le lanceur continue sans panel")
    return proc


# ----------------------------------------------------------------------
# Sauvegardes, etat publie, rapport
# ----------------------------------------------------------------------
def save_checkpoint(trainer: Trainer, session_dir: Path, reason: str,
                    promote: bool = False) -> Path | None:
    """Ecrit le genome complet, et le promeut en ``data/genome.pt`` si demande."""
    path = session_dir / f"genome_step{trainer.step}.pt"
    try:
        trainer.save(path, latest=False)
    except Exception as error:
        log(f"save impossible ({reason}) a step {trainer.step}: "
            f"{type(error).__name__}: {error}")
        return None
    if promote:
        try:
            shutil.copyfile(path, nc.GENOME_PATH)
        except OSError as error:
            log(f"promotion impossible ({reason}): {error}")
            return path
    log(f"checkpoint ({reason}) step {trainer.step} -> "
        f"{path.relative_to(nc.ROOT)}" + (" + data/genome.pt" if promote else ""))
    return path


def publish_state(**fields) -> dict:
    state = dict(nc.read_state() or {})
    state.update(fields)
    nc.write_state(state)
    return state


def session_mots_fr(started: float) -> int:
    """Somme des mots francais comptes par /chat pendant la session."""
    start_iso = datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds")
    total = 0
    for entry in nc.read_eloquence():
        if str(entry.get("ts", "")) >= start_iso:
            total += int(entry.get("mots_fr") or 0)
    return total


def write_report(reason: str, state: dict | None = None) -> Path:
    nc.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = nc.SESSIONS_DIR / f"RAPPORT-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    state = state or (nc.read_state() or {})
    lines = [
        "# Nyra — rapport d'arret du lanceur",
        "",
        f"- Date : {nc.now_iso()}",
        f"- Raison : {reason}",
        f"- PID : {os.getpid()}",
        "",
    ]
    for key in (
        "status", "reason", "session", "step", "loss", "tok_s", "vram_mb",
        "vram_reserved_mb", "ram_mb", "gpu_pct", "temp_c", "other_vram_mb",
        "cumulative_training_min", "sessions_done", "vram_guard_hits", "device",
    ):
        if key in state:
            lines.append(f"- {key} : {state[key]}")
    metrics = nc.read_metrics()
    if metrics:
        lines += [
            "",
            "## Derniere ligne metrics.json",
            "",
            "```json",
            json.dumps(metrics[-1], indent=2, ensure_ascii=False),
            "```",
        ]
    lines += [
        "",
        "Le genome est conserve dans `data/genome.pt` ; les checkpoints de session",
        "sont dans `data/runs/nyra_sessions/`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"rapport ecrit : {path.relative_to(nc.ROOT)}")
    return path


def next_session_index() -> int:
    highest = 0
    if nc.SESSIONS_DIR.exists():
        for child in nc.SESSIONS_DIR.iterdir():
            if child.is_dir() and child.name.startswith("session-"):
                try:
                    highest = max(highest, int(child.name.split("-", 1)[1]))
                except ValueError:
                    continue
    return highest + 1


# ----------------------------------------------------------------------
# Une session
# ----------------------------------------------------------------------
def run_session(index: int, config: dict, cfg: dict, surv: Surveillance) -> dict:
    """Entraine pendant ``session_min`` minutes puis sauvegarde et publie."""
    session_dir = nc.SESSIONS_DIR / f"session-{index:03d}"
    session_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer.resume(
        nc.GENOME_PATH,
        config=config,
        device=DEVICE,
        data="text",
        batch_size=int(cfg["batch_size"]),
        seq_len=int(cfg["seq_len"]),
    )
    # La config du panel s'applique a CETTE session : warmup et lr max lus ici.
    config["training"]["warmup"] = int(cfg["warmup"])
    config["training"]["text_steps"] = max(
        int(config["training"]["text_steps"]), trainer.step + 1
    )
    trainer.base_lr = float(cfg["lr_max"])
    trainer.report_every = 0

    started = time.time()
    deadline = started + float(cfg["session_min"]) * 60.0
    paused_seconds = 0.0
    pause_started: float | None = None
    last_save = started
    losses: list[float] = []
    metrics_last: dict | None = None
    tok_s = 0.0
    tokens_total = 0.0
    train_seconds = 0.0
    guard_hits = 0
    steps = 0
    first_loss: float | None = None

    log(f"=== session {index} : {cfg['session_min']} min, "
        f"batch {trainer.batch_size}, seq {trainer.seq_len}, lr {cfg['lr_max']}, "
        f"warmup {cfg['warmup']}, device {trainer.device} ===")
    log(f"reprise depuis data/genome.pt : step {trainer.step}, pool "
        f"{trainer.brain.n_nodes}, arêtes {trainer.brain.ne}, "
        f"vocab {trainer.brain.vocab}")

    def publish(status: str, reason: str | None) -> None:
        memory = process_memory()
        snap = surv.snapshot
        nc.write_state({
            "status": status,
            "reason": reason,
            "session": index,
            "session_minutes_left": round(
                max(0.0, (deadline + paused_seconds - time.time())) / 60.0, 1
            ),
            "step": trainer.step,
            "loss": None if metrics_last is None else round(metrics_last["loss"], 4),
            "tok_s": round(tok_s, 1),
            "vram_mb": None if metrics_last is None else metrics_last.get("gpu_alloc_mb"),
            "vram_reserved_mb": (None if metrics_last is None
                                 else metrics_last.get("gpu_reserved_mb")),
            "ram_mb": round(memory["working_set"], 0),
            "commit_mb": round(memory["commit"], 0),
            "gpu_pct": snap.get("gpu_pct"),
            "temp_c": snap.get("temp_c"),
            "other_vram_mb": snap.get("other_vram_mb"),
            "gpu_used_mb": snap.get("gpu_used_mb"),
            "gpu_total_mb": snap.get("gpu_total_mb"),
            "cumulative_training_min": round(surv.training_seconds / 60.0, 1),
            "sessions_done": index - 1,
            "vram_guard_hits": guard_hits,
            "device": str(trainer.device),
            "batch_size": trainer.batch_size,
            "seq_len": trainer.seq_len,
            "pid": os.getpid(),
            "session_steps": steps,
            "session_first_loss": (None if first_loss is None
                                   else round(first_loss, 4)),
            "session_tok_s": round(tokens_total / train_seconds, 1)
            if train_seconds > 0 else None,
        })

    # Publie des l'ouverture : sans cela le panel affiche l'etat de la session
    # precedente pendant tout le chargement du genome.
    publish("training", None)

    try:
        while True:
            status, reason = surv.status()
            if status == "stopped":
                raise StopLauncher(reason or "arret demande")
            if status != "training":
                if pause_started is None:
                    pause_started = time.time()
                    save_checkpoint(trainer, session_dir, reason="pause", promote=True)
                    log(f"pause propre ({status}: {reason}) a step {trainer.step}")
                publish(status, reason)
                time.sleep(5.0)
                continue
            now = time.time()
            if pause_started is not None:
                paused_seconds += now - pause_started
                pause_started = None
                log(f"reprise apres {paused_seconds / 60.0:.1f} min de pause "
                    f"(step {trainer.step})")
            if now >= deadline + paused_seconds:
                break

            step_started = time.perf_counter()
            metrics = trainer.train_step()
            seconds = max(1e-6, time.perf_counter() - step_started)
            steps += 1
            losses.append(metrics["loss"])
            metrics_last = metrics
            if first_loss is None:
                first_loss = metrics["loss"]
            tok_s = trainer.batch_size * trainer.seq_len / seconds
            tokens_total += trainer.batch_size * trainer.seq_len
            train_seconds += seconds
            surv.training_seconds += seconds
            surv.own_reserved_mb = float(metrics.get("gpu_reserved_mb") or 0.0)

            if steps % 25 == 0:
                log(f"step {trainer.step:6d} | perte {metrics['loss']:.4f} | "
                    f"{tok_s:.0f} tok/s | VRAM {metrics.get('gpu_alloc_mb', 0):.0f} Mo "
                    f"allouee ({metrics.get('gpu_reserved_mb', 0):.0f} reservee) | "
                    f"device {trainer.device} | pool {trainer.brain.n_nodes}")

            alloc = float(metrics.get("gpu_alloc_mb") or 0.0)
            if alloc > float(cfg["seuil_vram"]):
                guard_hits += 1
                save_checkpoint(trainer, session_dir, reason="garde VRAM",
                                promote=True)
                log(f"garde VRAM: {alloc:.0f} Mo > {cfg['seuil_vram']} Mo "
                    f"— save + pause ({guard_hits}/{MAX_GUARD_HITS})")
                if guard_hits >= MAX_GUARD_HITS:
                    raise StopLauncher(
                        f"garde VRAM declenchee {guard_hits} fois dans la session {index}"
                    )
                surv.cooling_until = time.time() + 300.0
                surv.cool_reason = f"garde VRAM ({alloc:.0f} Mo allouee)"
                continue

            if (time.time() - last_save) >= float(cfg["save_min"]) * 60.0:
                save_checkpoint(trainer, session_dir, reason="horloge", promote=True)
                last_save = time.time()

            publish("training", None)
    except StopLauncher:
        save_checkpoint(trainer, session_dir, reason="arret", promote=True)
        publish("stopped", surv.stop_reason or "arret demande")
        raise
    except KeyboardInterrupt:
        save_checkpoint(trainer, session_dir, reason="interruption", promote=True)
        raise
    except Exception as error:
        log(f"session {index} interrompue a step {trainer.step}: "
            f"{type(error).__name__}: {error}")
        log(traceback.format_exc(limit=8))
        save_checkpoint(trainer, session_dir, reason="crash", promote=True)
        publish("paused", f"session {index} interrompue: {type(error).__name__}")
        result = {"crash": f"{type(error).__name__}: {error}", "steps": steps}
        return result

    # Fin de session : ce qui compte est sauvegarde, promu, mesure, publie.
    duration = (time.time() - started) / 60.0
    save_checkpoint(trainer, session_dir, reason="fin de session", promote=True)
    brain = trainer.brain
    row = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "nodes": int(brain.n_nodes),
        "nodes_pool": int(brain.n_nodes),
        "nodes_alive": int(brain.node_alive.sum()),
        "nodes_born": int(getattr(brain, "node_births", 0)),
        "aretes": int(brain.ne),
        "step": int(trainer.step),
        "perte": round(losses[-1], 4) if losses else None,
        "session": f"{index}-nyra",
        "duree_min": round(duration, 1),
        "corpus": "corpus_fr.txt",
        "genome": "v2-texte",
        "mots_fr": session_mots_fr(started),
        "tok_s": round(tokens_total / train_seconds, 1) if train_seconds > 0 else None,
        "pas": steps,
        "garde_vram": guard_hits,
        "device": str(trainer.device),
    }
    nc.append_metrics(row)
    log(f"session {index} terminee : {steps} pas en {duration:.1f} min, "
        f"perte {row['perte']}, tok/s {row['tok_s']}, mots_fr {row['mots_fr']} "
        f"-> data/metrics.json")
    publish("idle", "session suivante dans un instant")
    result = {"row": row, "steps": steps}
    return result


# ----------------------------------------------------------------------
# Boucle principale
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="lanceur 24/7 de Nyra : serveur webchat + sessions successives"
    )
    parser.add_argument("--no-server", action="store_true",
                        help="ne pas demarrer le serveur webchat")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda (defaut) ou cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global DEVICE
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"centre de commandement Nyra — device {DEVICE}, PID {os.getpid()}")
    log(f"config : {nc.CONFIG_PATH.relative_to(nc.ROOT)}")
    log(f"etat   : {nc.STATE_PATH.relative_to(nc.ROOT)}")

    if not nc.GENOME_PATH.exists():
        log("aucun genome dans data/genome.pt : rien a reprendre, arret")
        return 2

    if not args.no_server:
        start_server()

    lock = acquire_lock()
    surv = Surveillance()
    surv.start()

    index = next_session_index()
    crash_streak = 0
    exit_reason = "arret"
    code = 0
    try:
        while True:
            cfg = nc.read_config()
            log(f"config lue : session {cfg['session_min']} min, batch {cfg['batch_size']}, "
                f"warmup {cfg['warmup']}, lr {cfg['lr_max']}, seuil VRAM "
                f"{cfg['seuil_vram']} Mo, pause {cfg['temp_pause']} C, stop "
                f"{cfg['temp_stop']} C, 24/7 {'on' if cfg['mode_247'] else 'off'}, "
                f"pauses thermiques {'on' if cfg['pauses_thermiques'] else 'off'}")
            try:
                result = run_session(index, load_config(), cfg, surv)
            except StopLauncher as stop:
                exit_reason = str(stop)
                code = 1
                break
            except KeyboardInterrupt:
                exit_reason = "interruption (Ctrl+C)"
                break
            except SystemExit as stop:
                exit_reason = f"verrou : {stop}"
                code = 2
                break
            except Exception as error:
                # Filet de securite : une session ne doit jamais tuer le lanceur.
                log(f"session {index} : erreur hors boucle {type(error).__name__}: {error}")
                log(traceback.format_exc(limit=8))
                result = {"crash": f"{type(error).__name__}: {error}", "steps": 0}

            if result.get("crash"):
                crash_streak = crash_streak if not result.get("steps") else 0
                crash_streak += 1
                log(f"crash sans progression ({crash_streak}/{MAX_CRASH_STREAK}) : "
                    f"{result['crash']}")
                if crash_streak >= MAX_CRASH_STREAK:
                    exit_reason = (f"{crash_streak} sessions consecutives sans un pas : "
                                   f"{result['crash']}")
                    code = 1
                    break
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                time.sleep(30.0)
            else:
                crash_streak = 0

            index += 1
            cfg = nc.read_config()
            if cfg["stop_after_session"]:
                exit_reason = "arret propre demande par le panel (fin de session)"
                break
            if not cfg["mode_247"]:
                exit_reason = "mode 24/7 desactive : une seule session"
                break
            log("session suivante")
    finally:
        surv.halt()
        state = publish_state(status="stopped", reason=exit_reason)
        write_report(exit_reason, state)
        release_lock(lock)
        log(f"lanceur arrete : {exit_reason}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
