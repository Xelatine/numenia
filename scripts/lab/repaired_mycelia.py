"""S4 R1-R2 — la lecture reparee du MYCELIA jouet.

Le rapport S3-BIS a mesure le defaut : l'etat de la colonie contient le contexte
(sonde "toutes membranes" a 4,914 nats) mais la tete ne lit qu'**une** membrane
(sonde "noeud 1" a 5,365), donc le modele plafonne au plancher unigramme (5,08).
Ce module corrige l'interface de lecture, **sans toucher ``mycelia/brain.py``** :
il en derive une sous-classe, ce qui laisse le genome de Nyra et tous les runs
existants au bit pres.

Trois decisions, chacune payee par une mesure (voir ``RAPPORT-S4.md``)
----------------------------------------------------------------------

* **agregat de toutes les membranes** (voie lente) : les 8 noeuds du jouet
  donnent 8 x 16 = 128 dimensions, normalisees puis projetees vers la dimension
  de lecture (16). Le noeud 1 reste present a cote, normalise lui aussi.
* **voie rapide = table d'embedding DEDIEE** : la premiere version reutilisait
  l'embedding d'injection, et la mesure l'a condamnee -- ses gradients servent
  le graphe, pas la lecture, et la voie rapide n'apprenait rien (5,16 apres
  150 pas, contre 4,24 pour le meme bigramme isole). Table dediee : 312 x 16.
* **mixage initialise en bigramme** : le bloc de la voie rapide est l'identite,
  les deux autres sont a zero. Sans cela, l'agregat (echelle ~111) ecrase tout
  et la tete reste au plancher ; avec cela, la tete part d'un modele qu'elle
  sait deja lire et n'ouvre la memoire du graphe que si elle aide.

Les trois vecteurs sont concatenes puis passes par une lineaire (concat +
lineaire, la combinaison la plus simple a documenter et a compter) :

    lecture = LayerNorm( mix[ slow(128->16) , noeud1(16) , rapide(16) ] )

``output_head`` et ``readout_norm`` du cerveau d'origine sont reutilises : la
seule chose qui change est ce qu'ils recoivent.

Cout exact (compte par ``readout_accounting``, jamais suppose) :

    voie lente    : 2 064 (128->16) + 256 (LN 128) = 2 320
    temoin noeud1 :    32 (LN 16)
    voie rapide   :    32 (LN 16) + 4 992 (table 312x16 dediee)
    mixage        :   784 (48->16)
    ------------------------------------------------
    ajoutes       : 8 160   (modele : 23 748 -> 31 908)

Le jumeau equitable de S3-BIS comptait 16 711 parametres actifs par token ; le
MYCELIA repare en compte ~24 871. C'est **plus** que le jumeau (1,49x) : la
comparaison est donc defavorable a MYCELIA, et c'est le sens honnete.

LE DECALAGE D'UN CRAN (mesure pendant cette mission)
----------------------------------------------------

``lab_common.mycelia_loss`` compare les logits de MYCELIA au couple ``y`` des
lots, c'est-a-dire au token **suivant** ; or MYCELIA predit, a la position
``t``, le token ``t`` (il lit avant d'injecter). La validation de S2/S3-BIS
demandait donc au cerveau un pas de plus que ce qu'il apprend. Sur un modele au
plancher (5,08) les deux cibles se valent ; sur un modele qui predit, l'ecart
devient enorme (mesure : 4,06 honnete contre 6,64 decale, meme instantane).
``aligned_loss`` est la mesure honnete, celle qui compare a ``x`` -- exactement
la cible d'entrainement, et exactement la tache du dense (dont la metrique
d'origine etait deja alignee). Les deux chiffres sont donnes partout.

Entrainement
------------

``train_repaired`` reprend **le meme metabolisme** que ``micro_mycelia`` :
naissances d'aretes payees par le trafic, adresses honorees entre deux pas,
coupe des synapses faminees et enterrement des noeuds isoles. Il ajoute la
distillation (R5) : ``L = KL(logits_MYC || logits_prof) + lambda * d(P h_MYC, h_prof)``
ou ``P`` est un projecteur lineaire **jete apres usage** (il ne fait pas partie du
cerveau). L'alignement compare le pas ``t`` de MYCELIA (qui predit le token ``t``)
au pas ``t-1`` du dense (qui predit le meme token a partir du meme prefixe) ; le
pas 0 de MYCELIA n'a aucun contexte et est donc exclu des deux termes.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_common import LAB, SEED, ToyCorpus, set_seed
from micro_mycelia import KW, _swap_population, n_parameters, snapshot as base_snapshot
from mycelia.brain import OUTPUT_NODE, Brain
from mycelia.growth import apply_pending_node_births
from mycelia.lifecycle import apply_lifecycle

SEQ = 8


class RepairedBrain(Brain):
    """Le cerveau jouet + la tete de lecture pleine, et la voie rapide."""

    def __init__(self, *args, nodes_read: int = 8, two_speeds: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodes_read = max(1, min(int(nodes_read), self.n_nodes))
        self.two_speeds = bool(two_speeds)
        width = self.nodes_read * self.dim
        # L'entree de l'agregat est une somme de messages : son echelle depend de
        # la duree de vie du cerveau. Meme raison que readout_norm pour le noeud 1.
        self.read_in_norm = nn.LayerNorm(width)
        self.read_slow = nn.Linear(width, self.dim)
        # Le noeud 1 est garde comme temoin de comparaison avec S3-BIS, mais
        # normalise : sa membrane non normalisee ecrasait les autres voies.
        self.read_node1_norm = nn.LayerNorm(self.dim)
        self.read_fast_norm = nn.LayerNorm(self.dim) if self.two_speeds else None
        if self.two_speeds:
            # Table dediee : voir l'en-tete. La partager avec l'injection du
            # graphe a ete mesure et refuse (deux objectifs inconciliables).
            self.fast_embed = nn.Embedding(self.vocab, self.dim)
        self.read_mix = nn.Linear(self.dim * (3 if self.two_speeds else 2), self.dim)
        d = self.dim
        with torch.no_grad():
            # Initialisation bigramme : la voie rapide passe telle quelle, les
            # autres sont coupees. Une tete qui part a zero apprend a lire au
            # lieu d'apprendre a predire (mesure : 5,08 -> 5,08 pendant 300 pas).
            self.read_mix.weight.zero_()
            self.read_mix.bias.zero_()
            block = 2 * d if self.two_speeds else 0
            self.read_mix.weight[:, block : block + d] = torch.eye(d)
        self._last_token: torch.Tensor | None = None
        self._h_rows: list[torch.Tensor] = []
        self.embedding.register_forward_pre_hook(self._remember_token)
        # Pour que ``restore`` reconstruise la meme tete : config() doit porter
        # les deux arguments propres a la sous-classe.
        self._init_kwargs["nodes_read"] = int(self.nodes_read)
        self._init_kwargs["two_speeds"] = bool(self.two_speeds)

    # -- capture du dernier token, sans toucher au forward -----------------
    def _remember_token(self, _module, inputs) -> None:
        self._last_token = inputs[0].detach()

    def _fresh_state(self, batch: int) -> torch.Tensor:
        # Un etat neuf = un nouveau document : le dernier token de l'appel
        # precedent ne doit pas fuir dans la premiere prediction.
        self._last_token = None
        return super()._fresh_state(batch)

    # -- la lecture --------------------------------------------------------
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        batch = int(state.shape[1])
        window = state[: self.nodes_read].transpose(0, 1).reshape(batch, -1)
        parts = [
            self.read_slow(self.read_in_norm(window)),
            self.read_node1_norm(state[OUTPUT_NODE]),
        ]
        if self.two_speeds:
            ids = self._last_token
            if ids is None or ids.shape[0] != batch:
                ids = torch.zeros(batch, dtype=torch.long, device=state.device)
            parts.append(self.read_fast_norm(self.fast_embed(ids)))
        h = self.read_mix(torch.cat(parts, dim=-1))
        # Uniquement pendant un forward : une prediction appelee a la main
        # (top-3, generation) ne fait pas partie d'une sequence de lecture.
        if isinstance(self._h_rows, list):
            self._h_rows.append(h)
        return self.output_head(self.readout_norm(h))

    def forward(self, *args, **kwargs):
        # Une ligne de lecture par position, dans l'ordre : c'est ce que la
        # distillation aligne sur le dense. Reinitialise a chaque appel.
        self._h_rows = []
        out = super().forward(*args, **kwargs)
        self._h_rows = (
            torch.stack(self._h_rows, dim=1) if self._h_rows else torch.zeros(0)
        )
        return out

    def readout_rows(self) -> torch.Tensor:
        """[batch, seq, dim] : les vecteurs de lecture du dernier forward."""
        return self._h_rows

    def readout_modules(self) -> list[nn.Module]:
        modules = [self.read_in_norm, self.read_slow, self.read_node1_norm, self.read_mix]
        if self.two_speeds:
            modules += [self.fast_embed, self.read_fast_norm]
        return modules


def build_repaired(
    vocab: int,
    seed: int = SEED,
    two_speeds: bool = True,
    overrides: dict | None = None,
) -> RepairedBrain:
    """Meme construction que ``micro_mycelia.build_mycelia``, tete reparee."""
    set_seed(seed)
    kwargs = dict(KW)
    kwargs.update(overrides or {})
    n_nodes = int(kwargs.pop("n_nodes", 24))
    return RepairedBrain(
        n_nodes=n_nodes, vocab=int(vocab), two_speeds=bool(two_speeds), **kwargs
    )


def readout_accounting(brain: RepairedBrain) -> dict:
    """Parametres ajoutes par la reparation, module par module (R1)."""
    def count(*modules) -> int:
        return int(sum(p.numel() for m in modules if m is not None for p in m.parameters()))

    slow = count(brain.read_in_norm, brain.read_slow)
    node1 = count(brain.read_node1_norm)
    fast = count(brain.read_fast_norm, getattr(brain, "fast_embed", None))
    mix = count(brain.read_mix)
    return {
        "noeuds_lus": int(brain.nodes_read),
        "dimensions_lues": int(brain.nodes_read * brain.dim),
        "voie_lente": slow,
        "temoin_noeud1": node1,
        "voie_rapide": fast,
        "mixage": mix,
        "ajoutes": slow + node1 + mix + fast,
        "total_modele": n_parameters(brain),
        "deux_vitesses": bool(brain.two_speeds),
    }


def snapshot(brain: Brain) -> dict:
    return base_snapshot(brain)


def save_brain(path, brain: Brain, step: int = 0, extra: dict | None = None) -> None:
    """Sauvegarde complete d'un cerveau repare (config comprise).

    ``mycelia.checkpoint.save_genome`` n'est pas utilise : il reconstruit un
    ``Brain`` nu et ignore la tete ajoutee par la reparation.
    """
    torch.save(
        {
            "state_dict": brain.state_dict(),
            "config": brain.config(),
            "clock": int(brain.clock),
            "ne": int(brain.ne),
            "step": int(step),
            "extra": extra or {},
        },
        str(path),
    )


def load_brain(path) -> Brain:
    blob = torch.load(str(path), weights_only=False)
    brain = RepairedBrain(**blob["config"])
    brain.load_state_dict(blob["state_dict"])
    brain.ne = int(blob.get("ne", brain.ne))
    brain.clock = int(blob.get("clock", 0))
    brain._rebuild_edge_keys()
    brain.eval()
    return brain


def restore(brain: Brain, snap: dict) -> Brain:
    """Reconstruit un cerveau *de la meme classe* au point sauvegarde."""
    config = dict(brain.config())
    config["n_nodes"] = int(snap["n_nodes"])
    fresh = type(brain)(**config)
    fresh.load_state_dict(snap["state_dict"])
    fresh.ne = int(snap["ne"])
    fresh.clock = int(snap["clock"])
    fresh.births = int(snap["births"])
    fresh.node_births = int(snap["node_births"])
    fresh.edge_deaths = int(snap["edge_deaths"])
    fresh.node_deaths = int(snap["node_deaths"])
    fresh.cemetery = {
        k: {n: t.clone() for n, t in v.items()} for k, v in snap["cemetery"].items()
    }
    fresh._rebuild_edge_keys()
    fresh.eval()
    return fresh


# ---------------------------------------------------------------------------
# Mesure : la cible d'entrainement, pas le token suivant
# ---------------------------------------------------------------------------
@torch.no_grad()
def aligned_loss(brain: Brain, batches) -> float:
    """Perte honnete : MYCELIA predit le token ``t``, on compare a ``x[:, t]``.

    ``lab_common.mycelia_loss`` compare au couple ``y`` (token suivant) : c'est
    un pas de plus que ce que le cerveau apprend, et sur un modele qui predit
    l'ecart n'est plus une constante (mesure : 4,06 contre 6,64 au meme
    instantane). ``Brain.forward`` lit avant d'injecter, donc la cible de la
    position ``t`` est ``x[:, t]`` -- exactement ``corpus.targets``.
    """
    total = 0.0
    count = 0
    for x, _y in batches:
        logits = brain(x, grow=False, trace=False)[0]
        total += float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1))
        )
        count += 1
    return total / max(1, count)


@torch.no_grad()
def shifted_loss(brain: Brain, batches) -> float:
    """La metrique publiee de S2/S3-BIS (cible ``y``), pour la continuite."""
    total = 0.0
    count = 0
    for x, y in batches:
        logits = brain(x, grow=False, trace=False)[0]
        total += float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        )
        count += 1
    return total / max(1, count)


@torch.no_grad()
def per_position_loss(brain: Brain, batches, seq: int = SEQ) -> list[float]:
    """Perte honnete par position : la position 0 est sans contexte."""
    total = torch.zeros(int(seq))
    count = 0
    for x, _y in batches:
        logits = brain(x, grow=False, trace=False)[0]
        for t in range(int(seq)):
            total[t] += float(F.cross_entropy(logits[:, t], x[:, t]))
        count += 1
    return [round(float(v) / max(1, count), 4) for v in total]


# ---------------------------------------------------------------------------
# Distillation : le projecteur jetable et la lecture des deux cerveaux
# ---------------------------------------------------------------------------
def make_projector(dim: int) -> nn.Linear:
    """P = identite au depart : d(P h, h_prof) mesure alors l'ecart reel.

    Il est optimise avec le cerveau puis **jete** : ce n'est pas une partie du
    cerveau, seulement l'echelle qui rend les deux espaces comparables.
    """
    proj = nn.Linear(dim, dim)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(dim))
        proj.bias.zero_()
    return proj


@torch.no_grad()
def teacher_parts(model, tokens: torch.Tensor, perm: torch.Tensor | None = None):
    """Logits et etat cache du professeur dense, alignes sur les pas de MYCELIA.

    MYCELIA predit le token ``t`` a partir des tokens ``0..t-1`` ; le dense predit
    ``x[t+1]`` a partir de ``0..t``. Le pas ``t-1`` du dense est donc exactement
    la prediction du token ``t`` : on rend ``hidden[:, :-1]`` et ``logits[:, :-1]``,
    de forme [B, seq-1, ...], utilisables contre les pas 1..seq-1 de MYCELIA.
    """
    hidden, logits = model.parts(tokens)
    if perm is not None:
        # Controle anti-bruit : le meme professeur, etiquettes melangees. La
        # forme et l'entropie sont identiques, le lien mot -> distribution ne
        # l'est plus.
        logits = logits[..., perm]
    return logits[:, :-1], hidden[:, :-1]


def train_repaired(
    brain: RepairedBrain,
    corpus: ToyCorpus,
    steps: int,
    lr: float = 2e-3,
    batch: int = 16,
    seed: int = SEED,
    log_every: int = 10,
    lifecycle_every: int = 100,
    val_batches=None,
    weight_decay: float = 0.01,
    restore_best: bool = True,
    teacher=None,
    lam: float = 0.5,
    teacher_perm: torch.Tensor | None = None,
    clip: float = 1.0,
) -> dict:
    """Entraine (ou distille) le cerveau repare, metabolisme actif.

    ``teacher`` est soit None (controle M0 : cross-entropy seule), soit un dense
    jouet. Dans ce cas la perte est exactement celle de la mission :

        KL(logits_MYC || logits_prof) + lambda * MSE(P h_MYC, h_prof)

    sans terme de cross-entropy : le savoir doit venir du professeur, pas des
    etiquettes (sinon le gain de distillation serait indiscernable d'un simple
    entrainement supervise).

    La validation est ``aligned_loss`` (cible du cerveau) ; ``best_step`` et la
    courbe sont choisis dessus. La metrique decalee de S3-BIS est ajoutee au
    retour, pour la table de continuite.
    """
    extra = []
    projector = None
    if teacher is not None:
        projector = make_projector(int(brain.dim))
        extra = list(projector.parameters())
    optimizer = torch.optim.AdamW(
        list(brain.parameters()) + extra, lr=lr, weight_decay=weight_decay
    )
    best = {"val": float("inf"), "step": 0, "snap": snapshot(brain)}
    generator = torch.Generator().manual_seed(int(seed))
    val = val_batches if val_batches is not None else corpus.fixed(16, "val")
    curve: list[dict] = []
    events: list[dict] = []
    started = time.time()
    last_n = int(brain.n_nodes)
    brain.train()
    for step in range(1, int(steps) + 1):
        x, _y = corpus.batch(batch, generator=generator)
        target = corpus.targets(x)
        brain.clock = step
        logits = brain(x, grow=True, trace=False)[0]
        if teacher is None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            parts = {"ce": float(loss.detach()), "kl": None, "align": None}
        else:
            t_logits, t_hidden = teacher_parts(teacher, x, teacher_perm)
            logp = F.log_softmax(logits[:, 1:], dim=-1)
            logq = F.log_softmax(t_logits, dim=-1)
            kl = (logp.exp() * (logp - logq)).sum(-1).mean()
            align = F.mse_loss(projector(brain.readout_rows()[:, 1:]), t_hidden)
            loss = kl + float(lam) * align
            parts = {"ce": None, "kl": float(kl.detach()), "align": float(align.detach())}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        brain.freeze_newborn_gradients()
        torch.nn.utils.clip_grad_norm_(list(brain.parameters()) + extra, clip)
        optimizer.step()

        if step % 25 == 0:
            born = apply_pending_node_births(brain)
            if born:
                events.append({"step": step, "naissances": len(born), "pool": int(brain.n_nodes)})
        if brain.n_nodes != last_n:
            optimizer = _swap_population(brain, optimizer, lr)
            last_n = int(brain.n_nodes)
        if lifecycle_every and step % int(lifecycle_every) == 0:
            report = apply_lifecycle(brain, step=step)
            if report["edge_deaths"] or report["node_deaths"]:
                events.append({"step": step, **report, "pool": int(brain.n_nodes)})

        if step % int(log_every) == 0 or step == 1:
            brain.eval()
            value = aligned_loss(brain, val)
            brain.train()
            if value < best["val"]:
                best = {"val": value, "step": step, "snap": snapshot(brain)}
            curve.append(
                {
                    "step": step,
                    "loss": round(float(loss.detach()), 4),
                    "val": round(float(value), 4),
                    "pool": int(brain.n_nodes),
                    "edges": int(brain.ne),
                    **{k: (None if v is None else round(v, 4)) for k, v in parts.items()},
                }
            )
    if restore_best and best["step"]:
        events.append({"step": step, "arret": "meilleur", "best_step": best["step"]})
        brain = restore(brain, best["snap"])
    brain.eval()
    return {
        "brain": brain,
        "projector": projector,
        "best_val": round(float(best["val"]), 4),
        "best_step": int(best["step"]),
        "best_val_s3bis": round(shifted_loss(brain, val), 4),
        "final_val": round(aligned_loss(brain, val), 4),
        "per_position": per_position_loss(brain, val),
        "curve": curve,
        "events": events,
        "seconds": round(time.time() - started, 1),
        "edges": int(brain.ne),
        "pool": int(brain.n_nodes),
        "edge_deaths": int(brain.edge_deaths),
        "births": int(brain.births),
    }


@torch.no_grad()
def free_life(brain: RepairedBrain, corpus: ToyCorpus, steps: int = 1000, batch: int = 16,
              seed: int = SEED, lifecycle_every: int = 50) -> dict:
    """R8 : le metabolisme tourne LIBRE, sans entrainement ni gradient.

    Naissances d'aretes (payees par le trafic), adresses honorees, coupe des
    synapses faminees : exactement la vie du cerveau, mais aucun pas
    d'optimiseur. C'est la seule facon de demander si le savoir greffe survit
    a l'organisme qui le porte.
    """
    generator = torch.Generator().manual_seed(int(seed) + 991)
    brain.eval()
    events: list[dict] = []
    for step in range(1, int(steps) + 1):
        x, _y = corpus.batch(batch, generator=generator)
        brain.clock = step
        brain(x, grow=True, trace=False)
        if step % 25 == 0:
            born = apply_pending_node_births(brain)
            if born:
                events.append({"step": step, "naissances": len(born)})
        if lifecycle_every and step % int(lifecycle_every) == 0:
            report = apply_lifecycle(brain, step=step)
            if report["edge_deaths"] or report["node_deaths"]:
                events.append({"step": step, **report})
    return {
        "pas": int(steps),
        "aretes_fin": int(brain.ne),
        "aretes_mortes": int(brain.edge_deaths),
        "naissances": int(brain.births),
        "noeuds_morts": int(brain.node_deaths),
        "evenements": events[:40],
    }


__all__ = [
    "LAB",
    "SEQ",
    "RepairedBrain",
    "aligned_loss",
    "build_repaired",
    "free_life",
    "make_projector",
    "per_position_loss",
    "readout_accounting",
    "restore",
    "shifted_loss",
    "snapshot",
    "teacher_parts",
    "train_repaired",
]
