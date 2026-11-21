"""Panel Nyra : les onglets ajoutes au dashboard existant.

Le dashboard de ``scripts/webchat.py`` garde ses cartes d'origine (etat,
cerveau, dernieres paroles, conversations, croissance). Ce module y ajoute
quatre onglets et le style qui va avec :

* 📊 Performance — etat du lanceur, perte du pas en cours, tok/s, VRAM, RAM,
  GPU, temperature, rafraichi toutes les 5 s (fetch, pas de framework) ;
* 📈 Evolution — toutes les lignes de metrics.json, perte vs step, population
  vs date, heures cumulees, jalon suivant avec son ETA ;
* 🗣️ Eloquence — mots francais par session, courbe, derniere reponse et
  verdict, champ de chat branche sur le serveur existant ;
* ⚙️ Configuration — les neuf champs de data/nyra_config.json, lus a chaque
  session par le lanceur, et le bouton d'arret propre.

Tout est du HTML/CSS/JS vanilla servi par Flask : aucun CDN, aucune
dependance. Les chaines sont brutes (r\"\"\"...\"\"\") pour que le JS garde ses
echappements intacts.
"""

from __future__ import annotations

PANEL_STYLE = r"""
  nav.tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
  nav.tabs button { background: #101317; color: #8b939e; border: 1px solid #1e242c;
                    border-radius: 8px; padding: 6px 11px; font: inherit; font-size: 12.5px;
                    cursor: pointer; }
  nav.tabs button:hover { color: #dfe3e8; border-color: #2a323c; }
  nav.tabs button.on { background: #0c1611; color: #6ee7a8; border-color: #2c4a3a; }
  .badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11.5px;
           border: 1px solid #1e242c; background: #101317; color: #8b939e;
           font-variant-numeric: tabular-nums; }
  .badge.ok { color: #6ee7a8; border-color: #2c4a3a; background: #0c1611; }
  .badge.warn { color: #f5a86b; border-color: #4a3a2c; background: #16110c; }
  .badge.muted { color: #7c848f; }
  main > section[data-panel] { display: none; }
  main > section[data-panel].vis { display: block; }
  .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
  .grid.two { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
  .card .card { background: #0c0f13; }
  dl.kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; }
  dl.kv dt { color: #7c848f; }
  dl.kv dd { margin: 0; color: #e9edf2; font-variant-numeric: tabular-nums; }
  canvas.chart { display: block; width: 100%; background: #0a0c0f;
                 border: 1px solid #1a1f26; border-radius: 8px; }
  form.cfg { display: grid; gap: 12px; margin: 0;
             grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); }
  form.cfg label { display: grid; gap: 4px; font-size: 12px; color: #8b939e; }
  form.cfg input[type=number] { background: #0a0c0f; border: 1px solid #2a323c;
                                color: #e9edf2; border-radius: 7px; padding: 7px 9px;
                                font: inherit; }
  form.cfg label.check { display: flex; align-items: center; gap: 8px; color: #dfe3e8;
                         font-size: 13px; }
  button.act { background: #0c1611; color: #6ee7a8; border: 1px solid #2c4a3a;
               border-radius: 8px; padding: 8px 14px; font: inherit; cursor: pointer; }
  button.act.danger { background: #16110c; color: #f5a86b; border-color: #4a3a2c; }
  button.act:hover { filter: brightness(1.3); }
  .note { color: #7c848f; font-size: 12px; margin: 8px 0 0; }
  .chatline { display: flex; gap: 8px; }
  .chatline input { flex: 1; background: #0a0c0f; border: 1px solid #2a323c; color: #e9edf2;
                    border-radius: 8px; padding: 9px 11px; font: inherit; }
  .verdict { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
"""

PANEL_TABS = r"""
<nav class="tabs">
  <button type="button" data-tab="home">🫀 Nyra</button>
  <button type="button" data-tab="live">📊 Performance</button>
  <button type="button" data-tab="history">📈 Évolution</button>
  <button type="button" data-tab="speech">🗣️ Éloquence</button>
  <button type="button" data-tab="labo">🧪 Labo</button>
  <button type="button" data-tab="config">⚙️ Configuration</button>
</nav>
<span id="hdr-badge" class="badge" title="état du lanceur">—</span>
"""

PANEL_SECTIONS = r"""
  <section class="card wide" data-panel="live">
    <h2>📊 Performance temps réel</h2>
    <div class="grid">
      <div class="card">
        <h2>État du lanceur</h2>
        <dl class="kv">
          <dt>Status</dt><dd><span id="lv-badge" class="badge">—</span></dd>
          <dt>Raison</dt><dd id="lv-reason">—</dd>
          <dt>Session</dt><dd id="lv-session">—</dd>
          <dt>Device</dt><dd id="lv-device">—</dd>
          <dt>Source</dt><dd id="lv-source">—</dd>
          <dt>Vu à</dt><dd id="lv-updated">—</dd>
        </dl>
      </div>
      <div class="card">
        <h2>Entraînement</h2>
        <dl class="kv">
          <dt>Step</dt><dd id="lv-step">—</dd>
          <dt>Perte du step</dt><dd id="lv-loss">—</dd>
          <dt>Débit</dt><dd id="lv-toks">—</dd>
          <dt>Heures cumulées</dt><dd id="lv-hours">—</dd>
          <dt>Garde VRAM</dt><dd id="lv-guard">—</dd>
        </dl>
      </div>
      <div class="card">
        <h2>Mémoire et GPU</h2>
        <dl class="kv">
          <dt>VRAM Nyra</dt><dd id="lv-vram">—</dd>
          <dt>VRAM autres</dt><dd id="lv-other">—</dd>
          <dt>RAM</dt><dd id="lv-ram">—</dd>
          <dt>GPU</dt><dd id="lv-gpu">—</dd>
          <dt>Température</dt><dd id="lv-temp">—</dd>
        </dl>
      </div>
    </div>
    <p class="note">Rafraîchi toutes les 5 s. Les mesures GPU sont échantillonnées
    toutes les 15 s par le lanceur (nvidia-smi, une seule requête).</p>
  </section>

  <section class="card wide" data-panel="history">
    <h2>📈 Évolution</h2>
    <div class="grid two">
      <div class="card">
        <h2>Perte vs step (toutes sessions)</h2>
        <canvas id="ch-loss" class="chart" data-h="190"></canvas>
      </div>
      <div class="card">
        <h2>Population vs date</h2>
        <canvas id="ch-pop" class="chart" data-h="190"></canvas>
      </div>
      <div class="card">
        <h2>Heures d'entraînement cumulées</h2>
        <canvas id="ch-hours" class="chart" data-h="190"></canvas>
      </div>
      <div class="card">
        <h2>Prochain jalon</h2>
        <dl class="kv">
          <dt>Cible</dt><dd id="hist-target">—</dd>
          <dt>Perte actuelle</dt><dd id="hist-loss">—</dd>
          <dt>Pas restants</dt><dd id="hist-steps">—</dd>
          <dt>ETA</dt><dd id="hist-eta">—</dd>
          <dt>Total cumulé</dt><dd id="hist-hours">—</dd>
        </dl>
        <p class="note" id="hist-note"></p>
      </div>
    </div>
    <table style="margin-top:14px">
      <thead><tr><th>Date</th><th>Session</th><th>Step</th><th>Perte</th>
      <th>mots_fr</th><th>Durée (min)</th></tr></thead>
      <tbody id="hist-body"><tr><td colspan="6" class="muted">chargement…</td></tr></tbody>
    </table>
  </section>

  <section class="card wide" data-panel="speech">
    <h2>🗣️ Éloquence</h2>
    <div class="grid two">
      <div class="card">
        <h2>Mots français par session</h2>
        <canvas id="ch-elo" class="chart" data-h="190"></canvas>
        <p class="note">Compteur appliqué à chaque réponse de /chat, somme par session
        dans metrics.json (« mots_fr »).</p>
      </div>
      <div class="card">
        <h2>Dernière réponse</h2>
        <dl class="kv">
          <dt>Verdict</dt><dd><span id="elo-verdict" class="badge">—</span></dd>
          <dt>Mots français</dt><dd id="elo-count">—</dd>
          <dt>Question</dt><dd id="elo-prompt">—</dd>
        </dl>
        <div id="elo-last" class="log" style="margin-top:10px">—</div>
      </div>
    </div>
    <div class="card" style="margin-top:14px">
      <h2>Parler à Nyra</h2>
      <form id="elo-form" class="chatline">
        <input id="elo-input" type="text" placeholder="votre question…" autocomplete="off">
        <button class="act" type="submit">Envoyer</button>
      </form>
      <div id="elo-now" class="log" style="margin-top:10px">en attente</div>
      <h2 style="margin-top:16px">Échanges récents</h2>
      <div id="elo-log" class="log scroll">—</div>
    </div>
  </section>

  <section class="card wide" data-panel="config">
    <h2>⚙️ Configuration — data/nyra_config.json</h2>
    <form id="cfg-form" class="cfg">
      <label>session_min (minutes par session)
        <input id="cfg-session_min" type="number" min="1" max="10080" step="1"></label>
      <label>batch_size
        <input id="cfg-batch_size" type="number" min="1" max="64" step="1"></label>
      <label>warmup (pas)
        <input id="cfg-warmup" type="number" min="0" max="200000" step="10"></label>
      <label>lr_max
        <input id="cfg-lr_max" type="number" min="0" max="1" step="0.00001"></label>
      <label>seuil_vram (Mo alloués)
        <input id="cfg-seuil_vram" type="number" min="100" max="8000" step="50"></label>
      <label>temp_pause (°C)
        <input id="cfg-temp_pause" type="number" min="40" max="100" step="1"></label>
      <label>temp_stop (°C)
        <input id="cfg-temp_stop" type="number" min="40" max="110" step="1"></label>
      <label class="check"><input id="cfg-mode_247" type="checkbox"> mode_247</label>
      <label class="check"><input id="cfg-pauses_thermiques" type="checkbox">
        pauses_thermiques</label>
    </form>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px;align-items:center">
      <button id="cfg-save" class="act" type="button">Enregistrer</button>
      <button id="cfg-stop" class="act danger" type="button">Arrêter après cette session</button>
      <span id="cfg-state" class="note">—</span>
    </div>
    <p class="note">Le lanceur relit ce fichier avant chaque session : un changement
    s'applique à la session suivante, sans redémarrer. Aucun seuil n'est codé en dur.</p>
  </section>

  <section class="card wide" data-panel="labo">
    <h2>🧪 Labo — MYCELI-SCAN et traducteur</h2>
    <div class="grid">
      <div class="card">
        <h2>Atlas de Nyra</h2>
        <dl class="kv">
          <dt>Vivants</dt><dd id="lab-alive">—</dd>
          <dt>Actifs au scan</dt><dd id="lab-active">—</dd>
          <dt>Familles</dt><dd id="lab-fam">—</dd>
          <dt>Arêtes jamais co-actives</dt><dd id="lab-dead">—</dd>
          <dt>Paires co-actives sans arête</dt><dd id="lab-free">—</dd>
          <dt>Engagement (scan)</dt><dd id="lab-engagement">—</dd>
          <dt>Reproductible S1 → S2</dt><dd id="lab-repro">—</dd>
          <dt>Scans enregistrés</dt><dd id="lab-scans">—</dd>
        </dl>
      </div>
      <div class="card">
        <h2>dense → MYCELIA</h2>
        <dl class="kv">
          <dt>Avant</dt><dd id="lab-d-avant">—</dd>
          <dt>Après greffe</dt><dd id="lab-d-apres">—</dd>
          <dt>Contrôle</dt><dd id="lab-d-controle">—</dd>
          <dt>Gain</dt><dd id="lab-d-gain">—</dd>
        </dl>
      </div>
      <div class="card">
        <h2>MYCELIA → dense</h2>
        <dl class="kv">
          <dt>Avant</dt><dd id="lab-m-avant">—</dd>
          <dt>Après greffe</dt><dd id="lab-m-apres">—</dd>
          <dt>Contrôle</dt><dd id="lab-m-controle">—</dd>
          <dt>Gain</dt><dd id="lab-m-gain">—</dd>
        </dl>
      </div>
    </div>
    <h2 style="margin-top:16px">Familles fonctionnelles</h2>
    <table>
      <thead><tr><th>#</th><th>Nœuds</th><th>Taille</th><th>Activation</th><th>Top stimulus</th></tr></thead>
      <tbody id="lab-fam-body"></tbody>
    </table>
    <h2 style="margin-top:16px">Film d'activation</h2>
    <div class="chatline">
      <select id="lab-film" style="flex:1;background:#0a0c0f;border:1px solid #2a323c;
        color:#e9edf2;border-radius:8px;padding:8px 10px;font:inherit"></select>
      <button class="act" id="lab-play" type="button">▶︎ Lire</button>
      <span id="lab-frame" class="badge">token —</span>
    </div>
    <canvas id="lab-canvas" class="chart" width="760" height="240"></canvas>
    <p class="note">Une colonne par token, un point par nœud actif (plus il est haut,
    plus il est actif). Les 5 stimuli filmés sont lus depuis data/lab/film_nyra.json.</p>
    <h2 style="margin-top:16px">Plafond S3-bis — courbe longue et verdict A/B</h2>
    <div class="grid two">
      <div class="card">
        <h2>Verdict</h2>
        <dl class="kv">
          <dt>Verdict</dt><dd id="lab-s3-verdict">—</dd>
          <dt>L∞ MYCELIA</dt><dd id="lab-s3-linf">—</dd>
          <dt>Dense (contrôle)</dt><dd id="lab-s3-dense">—</dd>
          <dt>Suspect D2</dt><dd id="lab-s3-suspect">—</dd>
          <dt>L_ref (unigramme)</dt><dd id="lab-s3-ref">—</dd>
        </dl>
      </div>
      <div class="card">
        <h2>L(t) sur 500 pas</h2>
        <canvas id="lab-curve" class="chart" width="520" height="200"></canvas>
        <p class="note">Orange : MYCELIA · vert : dense jumeau. Ordonnée : perte de
        validation (borne basse = meilleure validation atteinte), abscisse : pas.
        Lecture seule, aucun CDN.</p>
      </div>
    </div>
    <h2 style="margin-top:16px">Lecture réparée S4 — avant/après et survie au métabolisme</h2>
    <div class="grid two">
      <div class="card">
        <h2>Verdict S4</h2>
        <dl class="kv">
          <dt>Verdict</dt><dd id="lab-s4-verdict">—</dd>
          <dt>Ancien MYCELIA</dt><dd id="lab-s4-avant">—</dd>
          <dt>Réparé (deux vitesses)</dt><dd id="lab-s4-repare">—</dd>
          <dt>Dense / bigramme</dt><dd id="lab-s4-refs">—</dd>
          <dt>Distillation (R)</dt><dd id="lab-s4-dist">—</dd>
          <dt>Survie 1000 pas</dt><dd id="lab-s4-survie">—</dd>
        </dl>
      </div>
      <div class="card">
        <h2>Perte de validation, 600 pas</h2>
        <canvas id="lab-s4-curve" class="chart" width="520" height="200"></canvas>
        <p class="note">Orange : MYCELIA réparé (deux vitesses, semence 1). Gris :
        l'ancien plafond (5,15) ; vert : le dense jumeau (3,79). Lecture seule,
        aucun CDN.</p>
      </div>
    </div>
    <h2 style="margin-top:16px">Plafond physique S5 — F avant/après la vie</h2>
    <div class="grid two">
      <div class="card">
        <h2>Score F (plafond inter-donneurs)</h2>
        <dl class="kv">
          <dt>F avant la vie</dt><dd id="lab-s5-f-avant">—</dd>
          <dt>F après la vie</dt><dd id="lab-s5-f-apres">—</dd>
          <dt>Plafond KL(A‖B)</dt><dd id="lab-s5-plafond">—</dd>
          <dt>Survie du receveur</dt><dd id="lab-s5-verdict">—</dd>
          <dt>Vie libre (faim)</dt><dd id="lab-s5-vie">—</dd>
          <dt>Mesure corrigée</dt><dd id="lab-s5-mesure">—</dd>
        </dl>
      </div>
      <div class="card">
        <h2>Survie du savoir, 1000 pas libres</h2>
        <canvas id="lab-s5-curve" class="chart" width="520" height="200"></canvas>
        <p class="note">Orange : R (gain relatif à l'unigramme) pendant la vie libre ;
        rouge : arêtes coupées cumulées. Lecture seule, aucun CDN.</p>
      </div>
    </div>
  </section>
"""

PANEL_SCRIPT = r"""
// --- Onglets -----------------------------------------------------------
const NYRA_TABS = ['home', 'live', 'history', 'speech', 'labo', 'config'];
let CFG_CACHE = {};

function showTab(name) {
  if (NYRA_TABS.indexOf(name) < 0) name = 'home';
  document.querySelectorAll('nav.tabs button').forEach(b => {
    b.classList.toggle('on', b.dataset.tab === name);
  });
  document.querySelectorAll('main > section[data-panel]').forEach(s => {
    s.classList.toggle('vis', s.dataset.panel === name);
  });
  try { localStorage.setItem('nyra-tab', name); } catch (e) {}
  if (name === 'live') loadLive();
  if (name === 'history') loadHistory();
  if (name === 'speech') loadSpeech();
  if (name === 'labo') loadLab();
  if (name === 'config') loadConfig();
}

document.querySelectorAll('nav.tabs button').forEach(b => {
  b.addEventListener('click', () => showTab(b.dataset.tab));
});

// --- Petits utilitaires ------------------------------------------------
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = (value === null || value === undefined || value === '')
    ? '—' : String(value);
}
function setHTML(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = value;
}
function mb(v) {
  if (v === null || v === undefined || v === '') return '—';
  const n = Number(v);
  return isFinite(n) ? Math.round(n) + ' Mo' : '—';
}
function hours(minutes) {
  if (minutes === null || minutes === undefined) return '—';
  const n = Number(minutes);
  if (!isFinite(n)) return '—';
  return (n / 60).toFixed(2) + ' h';
}
function fmtNum(v) {
  const n = Math.abs(Number(v));
  if (!isFinite(n)) return '—';
  if (n >= 1000) return Math.round(n).toString();
  if (n >= 10) return n.toFixed(1);
  if (n >= 1) return n.toFixed(2);
  return n.toFixed(3);
}

const STATUS_LABEL = {
  training: 'training', paused: 'paused: jeu', cooling: 'cooling',
  idle: 'idle', stopped: 'stopped'
};
const STATUS_CLASS = {
  training: 'badge ok', paused: 'badge warn', cooling: 'badge warn',
  idle: 'badge muted', stopped: 'badge muted'
};
function paintStatus(status) {
  const label = STATUS_LABEL[status] || status || '—';
  const cls = STATUS_CLASS[status] || 'badge';
  ['lv-badge', 'hdr-badge'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.textContent = label; el.className = cls; }
  });
}

// --- Mini-moteur de courbe (canvas, sans CDN) --------------------------
const LAST_CHARTS = {};
function drawChart(id, series, opts) {
  opts = opts || {};
  const canvas = document.getElementById(id);
  if (!canvas) return;
  LAST_CHARTS[id] = { series: series, opts: opts };
  const w = canvas.clientWidth || 520;
  const h = parseInt(canvas.getAttribute('data-h') || '190', 10);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.font = '11px ui-monospace, Consolas, monospace';
  const pad = { l: 56, r: 14, t: 14, b: 26 };
  const pw = Math.max(10, w - pad.l - pad.r);
  const ph = Math.max(10, h - pad.t - pad.b);
  const pts = [];
  series.forEach(s => s.points.forEach(p => {
    if (isFinite(p[0]) && isFinite(p[1])) pts.push(p);
  }));
  if (!pts.length) {
    ctx.fillStyle = '#5d656f';
    ctx.fillText("pas encore de données", pad.l, h / 2);
    return;
  }
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  pts.forEach(p => {
    x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]);
    y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]);
  });
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) y1 = y0 + (Math.abs(y0) * 0.1 || 1);
  const span = y1 - y0;
  y0 -= span * 0.08;
  y1 += span * 0.08;
  const X = v => pad.l + (v - x0) / (x1 - x0) * pw;
  const Y = v => pad.t + ph - (v - y0) / (y1 - y0) * ph;
  ctx.strokeStyle = '#1a1f26';
  ctx.fillStyle = '#5d656f';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ph * i / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + pw, y);
    ctx.stroke();
    ctx.fillText(fmtNum(y1 - (y1 - y0) * i / 4), 6, y + 4);
  }
  series.forEach(s => {
    const clean = s.points.filter(p => isFinite(p[0]) && isFinite(p[1]));
    if (!clean.length) return;
    ctx.strokeStyle = s.color || '#6ee7a8';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    clean.forEach((p, i) => {
      const x = X(p[0]), y = Y(p[1]);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
    const last = clean[clean.length - 1];
    ctx.fillStyle = s.color || '#6ee7a8';
    ctx.beginPath();
    ctx.arc(X(last[0]), Y(last[1]), 2.6, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.fillStyle = '#5d656f';
  const left = opts.xLabel ? opts.xLabel(x0) : fmtNum(x0);
  const right = opts.xLabel ? opts.xLabel(x1) : fmtNum(x1);
  ctx.fillText(left, pad.l, h - 8);
  ctx.fillText(right, pad.l + pw - ctx.measureText(right).width, h - 8);
  if (series.length > 1) {
    let lx = pad.l;
    series.forEach(s => {
      ctx.fillStyle = s.color || '#6ee7a8';
      ctx.fillRect(lx, 4, 8, 3);
      ctx.fillStyle = '#8b939e';
      ctx.fillText(s.label, lx + 12, 8);
      lx += ctx.measureText(s.label).width + 32;
    });
  }
}
window.addEventListener('resize', () => {
  Object.keys(LAST_CHARTS).forEach(id => {
    const canvas = document.getElementById(id);
    if (canvas && canvas.clientWidth) {
      drawChart(id, LAST_CHARTS[id].series, LAST_CHARTS[id].opts);
    }
  });
});

// --- 📊 Performance ----------------------------------------------------
async function loadLive() {
  try {
    const d = await (await fetch('/api/live')).json();
    paintStatus(d.status);
    setText('lv-reason', d.reason || '—');
    setText('lv-session', '#' + d.session + ' · ' + (d.session_minutes_left === undefined
      ? '—' : d.session_minutes_left + ' min restantes'));
    setText('lv-device', d.device);
    setText('lv-source', d.source === 'lanceur' ? 'lanceur' : 'serveur seul');
    setText('lv-updated', d.updated);
    setText('lv-step', d.step);
    setText('lv-loss', d.loss === null || d.loss === undefined
      ? '—' : Number(d.loss).toFixed(4));
    setText('lv-toks', d.tok_s ? Math.round(d.tok_s) + ' tok/s' : '—');
    setText('lv-hours', hours(d.cumulative_training_min));
    setText('lv-guard', (d.vram_guard_hits || 0) + ' / 3');
    setText('lv-vram', mb(d.vram_mb) + ' / ' + mb(d.vram_reserved_mb) + ' réservée');
    setText('lv-other', mb(d.other_vram_mb));
    setText('lv-ram', mb(d.ram_mb));
    setText('lv-gpu', d.gpu_pct === null || d.gpu_pct === undefined
      ? '—' : Math.round(d.gpu_pct) + ' %');
    setText('lv-temp', d.temp_c === null || d.temp_c === undefined
      ? '—' : Math.round(d.temp_c) + ' °C');
  } catch (e) {
    setText('lv-reason', 'API injoignable');
  }
}

// --- 📈 Évolution ------------------------------------------------------
const PALETTE = ['#6ee7a8', '#7aa2f7', '#f5a86b', '#c792ea', '#e06c75', '#56b6c2'];
async function loadHistory() {
  let d;
  try {
    d = await (await fetch('/api/history')).json();
  } catch (e) {
    setText('hist-note', 'API injoignable');
    return;
  }
  const rows = d.rows || [];
  setHTML('hist-body', rows.length ? rows.map(r => '<tr><td>' + esc(r.date)
    + '</td><td>' + esc(r.session === undefined ? '' : r.session)
    + '</td><td>' + esc(r.step) + '</td><td>' + esc(r.perte)
    + '</td><td>' + esc(r.mots_fr === undefined ? '—' : r.mots_fr)
    + '</td><td>' + esc(r.duree_min === undefined ? '—' : r.duree_min)
    + '</td></tr>').join('')
    : '<tr><td colspan="6" class="muted">aucune session</td></tr>');

  const groups = {};
  rows.forEach(r => {
    const key = String(r.session === undefined ? '?' : r.session);
    const step = Number(r.step), loss = Number(r.perte);
    if (!isFinite(step) || !isFinite(loss)) return;
    if (!groups[key]) groups[key] = [];
    groups[key].push([step, loss]);
  });
  const lossSeries = Object.keys(groups).slice(-6).map((k, i) => ({
    label: 'session ' + k, color: PALETTE[i % PALETTE.length], points: groups[k]
  }));
  drawChart('ch-loss', lossSeries, { xLabel: v => 'step ' + Math.round(v) });

  const popSeries = [
    { label: 'pool', color: '#7aa2f7', points: [] },
    { label: 'vivants', color: '#6ee7a8', points: [] },
    { label: 'arêtes', color: '#f5a86b', points: [] }
  ];
  rows.forEach((r, i) => {
    if (r.nodes_pool !== undefined) popSeries[0].points.push([i, Number(r.nodes_pool)]);
    if (r.nodes_alive !== undefined) popSeries[1].points.push([i, Number(r.nodes_alive)]);
    if (r.aretes !== undefined) popSeries[2].points.push([i, Number(r.aretes)]);
  });
  const dateAt = i => {
    const row = rows[Math.round(i)];
    return row ? String(row.date).slice(5, 10) : '';
  };
  drawChart('ch-pop', popSeries, { xLabel: dateAt });

  let running = 0;
  const hourPoints = [];
  rows.forEach((r, i) => {
    running += Number(r.duree_min || 0);
    hourPoints.push([i, running / 60]);
  });
  drawChart('ch-hours', [{ label: 'heures', color: '#6ee7a8', points: hourPoints }],
    { xLabel: dateAt });
  setText('hist-hours', hours(running));

  const m = d.next_milestone || {};
  setText('hist-target', m.cible === null || m.cible === undefined
    ? 'aucun' : m.cible);
  setText('hist-loss', m.perte_actuelle === null || m.perte_actuelle === undefined
    ? '—' : Number(m.perte_actuelle).toFixed(4));
  setText('hist-steps', m.pas_restants === null || m.pas_restants === undefined
    ? '—' : Math.round(m.pas_restants));
  setText('hist-eta', m.heures === null || m.heures === undefined
    ? '—' : Number(m.heures).toFixed(1) + ' h');
  setText('hist-note', m.note || '');
}

// --- 🗣️ Éloquence ------------------------------------------------------
async function loadSpeech() {
  let d;
  try {
    d = await (await fetch('/api/eloquence')).json();
  } catch (e) {
    return;
  }
  const series = d.series || [];
  drawChart('ch-elo', [{ label: 'mots_fr', color: '#6ee7a8',
    points: series.map((r, i) => [i, Number(r.mots_fr) || 0]) }],
    { xLabel: i => {
        const row = series[Math.round(i)];
        return row ? String(row.date).slice(5, 10) : '';
      } });
  const verdict = d.verdict || 'premier point';
  const el = document.getElementById('elo-verdict');
  if (el) {
    el.textContent = verdict;
    el.className = 'badge ' + (verdict === 'mieux' ? 'ok'
      : verdict === 'pire' ? 'warn' : 'muted');
  }
  const last = d.last;
  setText('elo-count', last ? (last.mots_fr + ' / ' + last.mots_total + ' mots') : '—');
  setText('elo-prompt', last ? last.prompt : '—');
  setText('elo-last', last ? last.reponse : 'aucune réponse enregistrée');
  const entries = (d.entries || []).slice(-8).reverse();
  setHTML('elo-log', entries.length ? entries.map(e =>
    '<div class="row"><span class="ts">' + esc(String(e.ts).slice(11, 16)) + '</span>'
    + '<span><span class="q">' + esc(e.prompt) + '</span><br>'
    + '<span class="a">' + esc(e.reponse) + '</span> '
    + '<span class="muted">· ' + e.mots_fr + ' mots fr</span></span></div>').join('')
    : '<span class="muted">aucune réponse</span>');
}

document.addEventListener('submit', async ev => {
  if (ev.target.id !== 'elo-form') return;
  ev.preventDefault();
  const input = document.getElementById('elo-input');
  const msg = (input.value || '').trim();
  if (!msg) return;
  input.value = '';
  setText('elo-now', '…');
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    });
    const data = await res.json();
    setText('elo-now', (data['réponse'] || data.erreur || '(vide)')
      + (data.mots_fr === undefined ? '' : '  [' + data.mots_fr + ' mots fr]'));
  } catch (e) {
    setText('elo-now', 'échec : ' + e);
  }
  loadSpeech();
});

// --- ⚙️ Configuration --------------------------------------------------
const CFG_FIELDS = ['session_min', 'batch_size', 'warmup', 'lr_max',
  'seuil_vram', 'temp_pause', 'temp_stop'];

async function loadConfig() {
  try {
    CFG_CACHE = await (await fetch('/api/config')).json();
  } catch (e) {
    setText('cfg-state', 'API injoignable');
    return;
  }
  CFG_FIELDS.forEach(k => {
    const el = document.getElementById('cfg-' + k);
    if (el) el.value = CFG_CACHE[k];
  });
  const m7 = document.getElementById('cfg-mode_247');
  const pt = document.getElementById('cfg-pauses_thermiques');
  if (m7) m7.checked = !!CFG_CACHE.mode_247;
  if (pt) pt.checked = !!CFG_CACHE.pauses_thermiques;
  const stop = document.getElementById('cfg-stop');
  if (stop) {
    stop.textContent = CFG_CACHE.stop_after_session
      ? "Annuler l'arrêt programmé" : 'Arrêter après cette session';
  }
  setText('cfg-state', CFG_CACHE.stop_after_session
    ? 'arrêt programmé en fin de session' : 'aucun arrêt programmé');
}

async function saveConfig(patch) {
  try {
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch)
    });
    const data = await res.json();
    if (data.ok) {
      CFG_CACHE = data.config;
      setText('cfg-state', 'enregistré — appliqué à la session suivante');
      setTimeout(loadConfig, 1200);
    } else {
      setText('cfg-state', 'refusé : ' + (data.error || 'inconnu'));
    }
  } catch (e) {
    setText('cfg-state', 'échec : ' + e);
  }
}

document.addEventListener('click', ev => {
  if (ev.target.id === 'cfg-save') {
    const patch = {};
    CFG_FIELDS.forEach(k => {
      const el = document.getElementById('cfg-' + k);
      if (el) patch[k] = Number(el.value);
    });
    const m7 = document.getElementById('cfg-mode_247');
    const pt = document.getElementById('cfg-pauses_thermiques');
    if (m7) patch.mode_247 = m7.checked;
    if (pt) patch.pauses_thermiques = pt.checked;
    saveConfig(patch);
  }
  if (ev.target.id === 'cfg-stop') {
    saveConfig({ stop_after_session: !CFG_CACHE.stop_after_session });
  }
});

// --- 🧪 Labo -----------------------------------------------------------
const LAB = { atlas: null, film: null, scan: null, s3bis: null, s4: null, s5: null, timer: null };

function paintConversion(prefix, data) {
  const v = data && data.verdict;
  if (!v) {
    ['avant', 'apres', 'controle', 'gain'].forEach(k => setText('lab-' + prefix + '-' + k, '—'));
    return;
  }
  setText('lab-' + prefix + '-avant', Number(v.avant).toFixed(4));
  setText('lab-' + prefix + '-apres', Number(v.apres_greffe).toFixed(4));
  setText('lab-' + prefix + '-controle', Number(v.controle).toFixed(4));
  setText('lab-' + prefix + '-gain',
    (100 * Number(v.gain_vs_avant)).toFixed(1) + ' % vs avant · '
    + (100 * Number(v.gain_vs_controle)).toFixed(2) + ' % vs contrôle');
}

function paintScan() {
  // C4 : le taux d'engagement est la lecture la plus courte du scan recurrent.
  // Il doit rester identique a S1 tant qu'aucun entrainement n'a tourne ; une
  // variation signale soit un cerveau qui a bouge, soit un instrument instable.
  const scans = (LAB.scan && LAB.scan.scans) || [];
  if (!scans.length) {
    setText('lab-engagement', 'aucun scan');
    setText('lab-repro', '—');
    setText('lab-scans', '0');
    return;
  }
  const last = scans[scans.length - 1];
  setText('lab-engagement', (100 * Number(last.engagement)).toFixed(1) + ' % · '
    + last.actifs + '/' + last.vivants + ' nœuds · step ' + last.step);
  setText('lab-repro', (last.reproductible ? 'oui' : 'non')
    + ' — écart ' + last.ecart_max_matrice);
  setText('lab-scans', scans.length + ' point(s) dans scan_history.json');
}

function drawS3bis() {
  // Deux courbes, une seule echelle : c'est la comparaison A/B qu'on doit voir.
  const cv = document.getElementById('lab-curve');
  const data = LAB.s3bis && LAB.s3bis.long;
  if (!cv || !data) return;
  const my = (data.mycelia && data.mycelia.envelope) || [];
  const de = (data.dense && data.dense.envelope) || [];
  if (!my.length) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, padL = 34, padB = 18, padT = 8, padR = 6;
  const end = Number(my[my.length - 1].step) || 1;
  const vals = my.concat(de).map(r => Number(r.val));
  const lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  const span = (hi - lo) || 1;
  const x = s => padL + (W - padL - padR) * Number(s) / end;
  const y = v => padT + (H - padT - padB) * (1 - (Number(v) - lo) / span);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#1e242c';
  ctx.fillStyle = '#8b939e';
  ctx.font = '10px ui-monospace, monospace';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const v = lo + span * i / 3;
    const py = y(v);
    ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(W - padR, py); ctx.stroke();
    ctx.fillText(v.toFixed(2), 2, py + 3);
  }
  const line = (rows, color) => {
    if (!rows.length) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    rows.forEach((r, i) => {
      const px = x(r.step), py = y(r.val);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  };
  line(de, '#6ee7a8');
  line(my, '#f5a86b');
  ctx.fillStyle = '#8b939e';
  ctx.fillText('pas 0', padL, H - 4);
  ctx.fillText('pas ' + end, W - 52, H - 4);
}

const SUSPECT_LABEL = {
  agregation: 'agrégation', lecture: 'lecture', routage: 'routage',
  'agregation+lecture': 'agrégation + lecture', aucun: 'aucun'
};

function paintS3bis() {
  const d = LAB.s3bis || {};
  const verdict = d.summary && d.summary.verdict;
  setText('lab-s3-verdict', verdict || 'S3-bis non exécuté');
  const long = d.long || {};
  const my = long.mycelia || {}, de = long.dense || {};
  const myL = (my.decision && my.decision.L_inf_effectif) || my.best_val;
  const deL = (de.decision && de.decision.L_inf_effectif) || de.best_val;
  setText('lab-s3-linf', myL === undefined ? '—'
    : Number(myL).toFixed(4) + ' (meilleur pas ' + my.best_step + ', A/B : '
      + ((long.verdict) || '—') + ')');
  setText('lab-s3-dense', deL === undefined ? '—' : Number(deL).toFixed(4));
  const suspect = d.context && d.context.suspect;
  setText('lab-s3-suspect', suspect ? (SUSPECT_LABEL[suspect] || suspect) : '—');
  setText('lab-s3-ref', (d.floors && d.floors.L_ref !== undefined)
    ? Number(d.floors.L_ref).toFixed(4) : '—');
  drawS3bis();
}

function drawS4() {
  // Une courbe, deux reperes horizontaux : l'ancien plafond et le dense.
  const cv = document.getElementById('lab-s4-curve');
  const s = LAB.s4 && LAB.s4.summary;
  if (!cv || !s) return;
  const rows = (s.courbes && s.courbes.apres) || [];
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, padL = 34, padB = 18, padT = 8, padR = 6;
  ctx.clearRect(0, 0, W, H);
  if (!rows.length) {
    ctx.fillStyle = '#8b939e';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('S4 non exécuté', padL, H / 2);
    return;
  }
  const refs = [Number(s.ancien || 5.15), Number(s.dense || 3.79)];
  const vals = rows.map(r => Number(r.val)).concat(refs);
  const lo = Math.min.apply(null, vals) - 0.1, hi = Math.max.apply(null, vals) + 0.1;
  const span = (hi - lo) || 1;
  const end = Number(rows[rows.length - 1].step) || 1;
  const x = v => padL + (W - padL - padR) * Number(v) / end;
  const y = v => padT + (H - padT - padB) * (1 - (Number(v) - lo) / span);
  ctx.font = '10px ui-monospace, monospace';
  ctx.lineWidth = 1;
  [s.ancien, s.dense].forEach((v, i) => {
    if (v === undefined || v === null) return;
    ctx.strokeStyle = i === 0 ? '#5d656f' : '#6ee7a8';
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(padL, y(v)); ctx.lineTo(W - padR, y(v)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = i === 0 ? '#8b939e' : '#6ee7a8';
    ctx.fillText(Number(v).toFixed(2), W - padR - 26, y(v) - 3);
  });
  ctx.strokeStyle = '#f5a86b'; ctx.lineWidth = 1.6; ctx.beginPath();
  rows.forEach((r, i) => {
    const px = x(r.step), py = y(r.val);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.fillStyle = '#8b939e';
  ctx.fillText('pas 0', padL, H - 4);
  ctx.fillText('pas ' + end, W - 52, H - 4);
}

function paintS4() {
  const s = (LAB.s4 && LAB.s4.summary) || {};
  const verdict = s.verdict;
  setText('lab-s4-verdict', verdict ? (verdict + ' — ' + (s.verdict_phrase || ''))
    : 'S4 non exécuté');
  setText('lab-s4-avant', s.ancien === undefined ? '—' : Number(s.ancien).toFixed(4));
  setText('lab-s4-repare', s.mycelia_repere === undefined || s.mycelia_repere === null
    ? '—' : Number(s.mycelia_repere).toFixed(4));
  setText('lab-s4-refs',
    (s.dense === undefined ? 'dense —' : 'dense ' + Number(s.dense).toFixed(4))
    + ' · ' + (s.bigramme === undefined ? 'bigramme —'
      : 'bigramme ' + Number(s.bigramme).toFixed(4))
    + (s.dense_meme_validation === undefined ? ''
      : ' · même validation : dense ' + Number(s.dense_meme_validation).toFixed(4)));
  const d = s.distillation || {};
  const fmtR = v => (v === undefined || v === null) ? '—' : Number(v).toFixed(3);
  setText('lab-s4-dist', (d.R_distille === undefined || d.R_distille === null)
    ? '—'
    : 'M0 ' + fmtR(d.R_m0) + ' → distillé ' + fmtR(d.R_distille)
      + ' (gain ' + fmtR(d.gain_isole) + ', bruit ' + fmtR(d.R_melange) + ')');
  const v = s.survie || {};
  setText('lab-s4-survie', (v.survit === undefined || v.survit === null) ? '—'
    : v.survit + '/' + v.sur + ' semences · R ' + fmtR(v.R_avant)
      + ' → ' + fmtR(v.R_apres));
  drawS4();
}

function drawS5() {
  // Deux axes : R (orange, gauche) et arêtes coupées cumulées (rouge, droite).
  const cv = document.getElementById('lab-s5-curve');
  const s = (LAB.s5 && LAB.s5.summary) || {};
  const rows = s.courbe_survie || [];
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, padL = 34, padB = 18, padT = 8, padR = 34;
  ctx.clearRect(0, 0, W, H);
  if (!rows.length) {
    ctx.fillStyle = '#8b939e';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('S5 non exécuté', padL, H / 2);
    return;
  }
  const rs = rows.map(r => Number(r.R));
  const cs = rows.map(r => Number(r.coupees || 0));
  const rlo = Math.min.apply(null, rs) - 0.1, rhi = Math.max.apply(null, rs) + 0.1;
  const cmax = Math.max(1, Math.max.apply(null, cs));
  const end = Number(rows[rows.length - 1].pas) || 1;
  const x = v => padL + (W - padL - padR) * Number(v) / end;
  const yR = v => padT + (H - padT - padB) * (1 - (Number(v) - rlo) / ((rhi - rlo) || 1));
  const yC = v => padT + (H - padT - padB) * (1 - Number(v) / cmax);
  ctx.font = '10px ui-monospace, monospace';
  ctx.strokeStyle = '#e06c75'; ctx.lineWidth = 1.2; ctx.beginPath();
  rows.forEach((r, i) => {
    const px = x(r.pas), py = yC(r.coupees || 0);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.strokeStyle = '#f5a86b'; ctx.lineWidth = 1.8; ctx.beginPath();
  rows.forEach((r, i) => {
    const px = x(r.pas), py = yR(r.R);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.fillStyle = '#8b939e';
  ctx.fillText('pas 0', padL, H - 4);
  ctx.fillText('pas ' + end, W - 52, H - 4);
  ctx.fillText('R', padL - 10, padT + 8);
  ctx.fillStyle = '#e06c75';
  ctx.fillText('coupes ' + cmax, W - padR + 2, padT + 8);
}

function paintS5() {
  const s = (LAB.s5 && LAB.s5.summary) || {};
  const fmt = v => (v === undefined || v === null) ? '—' : Number(v).toFixed(3);
  setText('lab-s5-f-avant', fmt(s.F_avant));
  setText('lab-s5-f-apres', fmt(s.F_apres));
  setText('lab-s5-plafond', fmt(s.plafond_kl))
  setText('lab-s5-verdict', (s.survie_code ? (s.survie_code + ' — ' + (s.survie_phrase || ''))
    : 'S5 non exécuté'));
  setText('lab-s5-vie', (s.vie_creees === undefined || s.vie_creees === null) ? '—'
    : 'créées ' + s.vie_creees + ' · coupées ' + s.vie_coupees
      + ' · densité ' + (s.densite === null || s.densite === undefined
        ? '—' : Math.round(Number(s.densite) * 100) + ' %'));
  setText('lab-s5-mesure', (s.mesure_mycelia === undefined || s.mesure_mycelia === null)
    ? '—' : 'MYCELIA ' + Number(s.mesure_mycelia).toFixed(3)
      + ' · dense ' + fmt(s.mesure_dense) + ' · bigramme ' + fmt(s.mesure_bigramme));
  drawS5();
}

function paintAtlas() {
  const a = LAB.atlas;
  if (!a || !a.meta) { setText('lab-alive', 'labo non exécuté'); return; }
  setText('lab-alive', a.meta.alive + ' / ' + a.meta.n_nodes);
  setText('lab-active', a.noeuds.length);
  setText('lab-fam', a.familles.length + ' (k=' + a.meta.k + ')');
  const dead = a.divergences.cablees_jamais_coactives;
  const free = a.divergences.coactifs_sans_arete;
  setText('lab-dead', dead.nombre + ' (' + Math.round(dead.part * 100) + ' %)');
  setText('lab-free', free.nombre + ' paires');
  setHTML('lab-fam-body', a.familles.map(f => '<tr><td>' + f.famille
    + '</td><td>' + f.noeuds.slice(0, 5).join(', ') + (f.taille > 5 ? '…' : '')
    + '</td><td>' + f.taille + '</td><td>' + Number(f.activation_moyenne).toFixed(3)
    + '</td><td>' + esc((f.top_stimuli[0] || {}).texte || '—') + '</td></tr>').join(''));
}

function paintFilmOptions() {
  const sel = document.getElementById('lab-film');
  if (!sel) return;
  const items = (LAB.film && LAB.film.stimuli) || [];
  sel.innerHTML = items.map((s, i) => '<option value="' + i + '">'
    + esc(s.texte) + '</option>').join('');
  if (items.length) drawFilm(0);
}

function drawFilm(index, frame) {
  const cv = document.getElementById('lab-canvas');
  const items = (LAB.film && LAB.film.stimuli) || [];
  if (!cv || !items[index]) return;
  const s = items[index];
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, padL = 14, padB = 26;
  ctx.clearRect(0, 0, W, H);
  const frames = s.frames;
  const n = frames.length ? frames[0].length : 0;
  let max = 0;
  frames.forEach(f => f.forEach(v => { if (v > max) max = v; }));
  if (!max) max = 1;
  const colW = (W - padL - 8) / Math.max(1, frames.length);
  for (let t = 0; t < frames.length; t++) {
    for (let i = 0; i < n; i++) {
      const v = frames[t][i];
      if (v <= 0.2 * max) continue;
      const a = Math.min(1, v / max);
      const x = padL + t * colW + colW * 0.5;
      const y = 10 + (1 - a) * (H - padB - 26);
      ctx.fillStyle = 'rgba(110,231,168,' + (0.18 + 0.82 * a).toFixed(2) + ')';
      ctx.beginPath();
      ctx.arc(x, y, 1.2 + 3.6 * a, 0, 6.283);
      ctx.fill();
    }
  }
  ctx.font = '11px ui-monospace, monospace';
  s.tokens.forEach((tok, t) => {
    ctx.fillStyle = (t === frame) ? '#6ee7a8' : '#5d656f';
    ctx.fillText(String(tok).slice(0, 9), padL + t * colW + 2, H - 8);
  });
  if (frame !== undefined) {
    ctx.strokeStyle = '#6ee7a8';
    ctx.lineWidth = 1;
    ctx.strokeRect(padL + frame * colW + 1, 6, Math.max(4, colW - 2), H - padB - 14);
    setText('lab-frame', 'token ' + (frame + 1) + '/' + s.tokens.length
      + ' « ' + s.tokens[frame] + ' »');
  }
}

async function loadLab() {
  try { LAB.atlas = await (await fetch('/api/lab/atlas')).json(); } catch (e) { LAB.atlas = null; }
  try { LAB.film = await (await fetch('/api/lab/film')).json(); } catch (e) { LAB.film = null; }
  try {
    const c = await (await fetch('/api/lab/conversions')).json();
    paintConversion('d', c.dense_to_mycelia);
    paintConversion('m', c.mycelia_to_dense);
  } catch (e) {
    paintConversion('d', null);
    paintConversion('m', null);
  }
  try { LAB.scan = await (await fetch('/api/lab/scan-history')).json(); } catch (e) { LAB.scan = null; }
  try { LAB.s3bis = await (await fetch('/api/lab/s3bis')).json(); } catch (e) { LAB.s3bis = null; }
  try { LAB.s4 = await (await fetch('/api/lab/s4')).json(); } catch (e) { LAB.s4 = null; }
  try { LAB.s5 = await (await fetch('/api/lab/s5')).json(); } catch (e) { LAB.s5 = null; }
  paintAtlas();
  paintScan();
  paintS3bis();
  paintS4();
  paintS5();
  paintFilmOptions();
}

document.addEventListener('click', ev => {
  if (ev.target.id !== 'lab-play') return;
  const sel = document.getElementById('lab-film');
  const items = (LAB.film && LAB.film.stimuli) || [];
  const s = items[Number(sel && sel.value ? sel.value : 0)];
  if (!s) return;
  if (LAB.timer) { clearInterval(LAB.timer); LAB.timer = null; }
  let t = 0;
  drawFilm(Number(sel.value || 0), 0);
  LAB.timer = setInterval(() => {
    t += 1;
    if (t >= s.tokens.length) {
      clearInterval(LAB.timer);
      LAB.timer = null;
      return;
    }
    drawFilm(Number(sel.value || 0), t);
  }, 420);
});

document.addEventListener('change', ev => {
  if (ev.target.id === 'lab-film') drawFilm(Number(ev.target.value || 0));
});

// --- Démarrage ---------------------------------------------------------
let savedTab = 'home';
try { savedTab = localStorage.getItem('nyra-tab') || 'home'; } catch (e) {}
showTab(savedTab);
setInterval(loadLive, 5000);
loadLive();
"""
