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
"""

PANEL_SCRIPT = r"""
// --- Onglets -----------------------------------------------------------
const NYRA_TABS = ['home', 'live', 'history', 'speech', 'config'];
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

// --- Démarrage ---------------------------------------------------------
let savedTab = 'home';
try { savedTab = localStorage.getItem('nyra-tab') || 'home'; } catch (e) {}
showTab(savedTab);
setInterval(loadLive, 5000);
loadLive();
"""
