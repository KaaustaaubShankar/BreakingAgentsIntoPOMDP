from flask import Flask
import pandas as pd

app = Flask(__name__)

def _num(x):
    sv = str(x).replace('%', '').strip()
    if sv == '' or sv.lower() == 'nan' or sv in ('-', '.', '--', '- '):
        return None
    try:
        f = float(sv)
    except Exception:
        return None
    if f != f:  # NaN
        return None
    return f

# Models that live ONLY in the Overview sheet (not in Detailed_Results).
# We melt these from wide->long; GPTs/haiku come from Detailed to avoid double-counting.
_OV_MODEL_MAP = {
    'deepseek v4-pro': 'deepseek-v4-pro',
    'grok 4.3': 'grok-4.3',
    'grok 4.1 fast': 'grok-4.1-fast',
}
_OV_CFG_MAP = [
    ('Baseline', 'baseline'),
    ('Win% world_hard', 'world_hard'),
    ('Win% goal_hard', 'goal_hard'),
    ('Win% mechanics_hard', 'mechanics_hard'),
    ('Win% feedback_hard', 'feedback_hard'),
]

def _melt_overview_extra():
    ov = pd.read_csv('/home/exedev/jkj results - Overview(5).csv')
    rows = []
    for _, r in ov.iterrows():
        key = str(r.get('Model', '')).strip().lower()
        if key not in _OV_MODEL_MAP:
            continue
        model = _OV_MODEL_MAP[key]
        game = str(r.get('Game', '')).strip()
        if game == '' or game.lower() == 'nan':
            continue
        n = _num(r.get('N'))
        if n is None:
            continue
        reasoning = str(r.get('Reasoning?', '')).strip()
        if reasoning.lower() == 'nan':
            reasoning = ''
        for col, cfg in _OV_CFG_MAP:
            if col not in ov.columns:
                continue
            v = _num(r.get(col))
            if v is None:
                continue
            rows.append({'Model': model, 'Game': game, 'Reasoning': reasoning,
                         'Config': cfg, 'N': n, 'Win%': v})
    return rows

def load_data():
    df = pd.read_csv('/home/exedev/jkj results - Detailed_Results.csv')
    df['Win%'] = df['Win%'].astype(str).str.replace('%', '').str.strip()
    df['Win%'] = pd.to_numeric(df['Win%'], errors='coerce')
    extra = _melt_overview_extra()
    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df

def load_overview():
    df = pd.read_csv('/home/exedev/jkj results - Overview(5).csv')
    df = df.dropna(how='all')
    for col in ['Baseline','Win% world_hard','Win% goal_hard','Win% mechanics_hard','Win% feedback_hard']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%','').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['Model'].notna() & (df['Model'].astype(str).str.strip() != '')]
    return df

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Breaking Agents — COLM 2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0a0a0a;--s1:#111113;--s2:#141416;--b1:#1e1e22;--b2:#252528;
--tx:#e2e2e8;--mu:#888890;--di:#444450;
--ac:#7c3aed;--al:#a78bfa;--re:#ef4444;--or:#f97316;--gr:#22c55e;--bl:#3b82f6;--ye:#fbbf24;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--tx);font-family:-apple-system,'Inter',sans-serif;font-size:14px;line-height:1.5;}

.hdr{padding:24px 44px 20px;border-bottom:1px solid var(--b1);display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.hdr h1{font-size:20px;font-weight:700;color:#fff;letter-spacing:-.4px;}
.hdr p{font-size:12px;color:var(--mu);flex-basis:100%;margin-top:2px;}
.pill{background:#1a1030;border:1px solid var(--ac);color:var(--al);font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;}
.pill.g{background:#0d2010;border-color:var(--gr);color:var(--gr);}
.cd{background:#0c1322;border:1px solid var(--bl);color:#93c5fd;font-size:10px;font-weight:600;padding:3px 10px;border-radius:20px;font-variant-numeric:tabular-nums;letter-spacing:.2px;}
.cd.ws{background:#1a1030;border-color:var(--ac);color:var(--al);}
.cd.done{background:#1c0b0b;border-color:var(--re);color:var(--re);}

.tabs{display:flex;border-bottom:1px solid var(--b1);padding:0 44px;overflow-x:auto;}
.tab{padding:11px 16px;font-size:12px;font-weight:500;color:var(--di);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;}
.tab:hover{color:var(--mu);}
.tab.on{color:var(--al);border-bottom-color:var(--ac);}

.pg{padding:28px 44px;display:flex;flex-direction:column;gap:28px;}
.tp{display:flex;flex-direction:column;gap:28px;height:0;overflow:hidden;opacity:0;pointer-events:none;}
.tp.on{height:auto;overflow:visible;opacity:1;pointer-events:auto;}

.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;}
.st{background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:16px 18px;}
.sv{font-size:28px;font-weight:800;color:#fff;line-height:1;letter-spacing:-1px;}
.sv.ac{color:var(--al);} .sv.re{color:var(--re);}
.sl{font-size:10px;color:var(--di);margin-top:5px;text-transform:uppercase;letter-spacing:.8px;font-weight:500;}

.card{background:var(--s1);border:1px solid var(--b1);border-radius:10px;padding:22px;}
.stl{font-size:10px;font-weight:600;color:var(--di);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:16px;}
.r2{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
.r3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;}
canvas{max-height:280px;}
.chw{position:relative;width:100%;}
.chw canvas{max-height:none!important;height:100%!important;width:100%!important;}

/* Findings */
.fg{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;align-items:start;}
.fi{background:var(--bg);border:1px solid var(--b2);border-radius:10px;padding:18px;position:relative;overflow:hidden;}
.fi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.fi.cl::before{background:var(--re);} .fi.rv::before{background:var(--gr);}
.fi.dc::before{background:var(--bl);} .fi.fl::before{background:var(--or);}
.fi.an::before{background:var(--ye);}
.fn{font-size:10px;font-weight:700;color:var(--di);text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;}
.fs{font-size:24px;font-weight:800;letter-spacing:-1px;margin-bottom:4px;}
.fs.re{color:var(--re);} .fs.gr{color:var(--gr);} .fs.bl{color:var(--bl);} .fs.ye{color:var(--ye);}
.ft{font-size:13px;font-weight:700;color:#fff;margin-bottom:7px;line-height:1.3;}
.fb{font-size:11px;color:var(--mu);line-height:1.65;}

/* Heatmap */
.hm{width:100%;border-collapse:collapse;}
.hm th{font-size:10px;color:var(--di);font-weight:600;text-transform:uppercase;letter-spacing:.5px;padding:7px 6px;text-align:center;border-bottom:1px solid var(--b1);}
.hm th:first-child{text-align:left;}
.hm td{padding:6px 6px;text-align:center;border-bottom:1px solid #13131500;}
.hm td:first-child{text-align:left;}
.hm tr:hover td{background:#ffffff04;}
.ce{display:inline-flex;align-items:center;justify-content:center;min-width:44px;height:22px;border-radius:4px;font-size:10px;font-weight:700;padding:0 4px;}
.c5{background:#14532d;color:#4ade80;} .ch{background:#15412a;color:#6ee7b7;}
.cm{background:#422006;color:#fb923c;} .cl2{background:#3b0f0f;color:#f87171;}
.c0{background:#1c0b0b;color:#5a1a1a;} .cn{background:#111;color:#2a2a2a;}
.ab{display:inline-block;background:#422007;color:#fbbf24;font-size:9px;padding:1px 5px;border-radius:3px;margin-left:4px;font-weight:600;}
.ml{font-weight:600;color:var(--tx);font-size:12px;} .rl{color:var(--di);font-size:10px;}

/* Research controls */
.rc{background:var(--s2);border:1px solid var(--b1);border-radius:10px;padding:16px 18px;display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;}
.cg{display:flex;flex-direction:column;gap:5px;}
.cg label{font-size:10px;color:var(--di);text-transform:uppercase;letter-spacing:.7px;font-weight:600;}
.cg select{background:var(--bg);border:1px solid var(--b2);color:var(--tx);padding:6px 10px;border-radius:6px;font-size:12px;font-family:inherit;cursor:pointer;}
.cg select:focus{outline:none;border-color:var(--ac);}
.mdd{position:relative;}
.mdd>summary{list-style:none;background:var(--bg);border:1px solid var(--b2);color:var(--tx);padding:6px 10px;border-radius:6px;font-size:12px;cursor:pointer;white-space:nowrap;}
.mdd>summary::-webkit-details-marker{display:none;}
.mdd[open]>summary{border-color:var(--ac);}
.mbox{position:absolute;z-index:30;margin-top:4px;background:var(--s2);border:1px solid var(--b2);border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:6px;max-height:280px;overflow:auto;min-width:180px;box-shadow:0 10px 28px rgba(0,0,0,.6);}
.mbox label{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--tx);text-transform:none;letter-spacing:0;font-weight:400;cursor:pointer;white-space:nowrap;}
.mbox input{width:14px;height:14px;accent-color:var(--ac);cursor:pointer;margin:0;}
.mbox .mall{border-bottom:1px solid var(--b1);padding-bottom:7px;margin-bottom:2px;color:var(--mu);}
.ct{display:flex;align-items:center;gap:7px;padding-bottom:2px;}
.ct input{width:14px;height:14px;cursor:pointer;accent-color:var(--ac);}
.ct label{font-size:12px;color:var(--mu);cursor:pointer;}
.btn{background:var(--ac);color:#fff;border:none;padding:7px 14px;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;font-family:inherit;}
.btn:hover{background:#6d28d9;}
.btn.sec{background:transparent;border:1px solid var(--b2);color:var(--mu);}
.btn.sec:hover{border-color:var(--ac);color:var(--al);}

/* CI table */
.cit{width:100%;border-collapse:collapse;font-size:11px;}
.cit th{font-size:10px;color:var(--di);font-weight:600;text-transform:uppercase;letter-spacing:.7px;padding:7px 9px;text-align:left;border-bottom:1px solid var(--b1);white-space:nowrap;}
.cit td{padding:7px 9px;border-bottom:1px solid #0f0f0f;color:var(--mu);}
.cit td:first-child{color:var(--tx);font-weight:500;}
.cit tr:hover td{background:#ffffff03;}
.cb{display:flex;align-items:center;gap:6px;}
.ct2{flex:1;height:5px;background:#1a1a1a;border-radius:3px;position:relative;min-width:70px;}
.cf2{position:absolute;height:100%;border-radius:3px;top:0;}
.cm2{position:absolute;width:2px;height:9px;top:-2px;background:#fff;border-radius:1px;}
.cl3{position:absolute;width:1px;height:9px;top:-2px;background:#555;}
.ch3{position:absolute;width:1px;height:9px;top:-2px;background:#555;}
.wp{display:inline-flex;align-items:center;justify-content:center;min-width:38px;padding:2px 6px;border-radius:4px;font-weight:700;font-size:11px;}
.sg{font-size:9px;padding:1px 5px;border-radius:3px;font-weight:700;}
.sy{background:#0d2010;color:var(--gr);} .sn{background:#1a1a1a;color:var(--di);}

/* Data table */
.tf{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;}
.tf select{background:var(--bg);border:1px solid var(--b2);color:var(--mu);padding:5px 9px;border-radius:6px;font-size:11px;font-family:inherit;cursor:pointer;}
.tf select:focus{outline:none;border-color:var(--ac);}
.dt{width:100%;border-collapse:collapse;font-size:11px;}
.dt th{font-size:10px;color:var(--di);font-weight:600;text-transform:uppercase;letter-spacing:.7px;padding:7px 10px;text-align:left;border-bottom:1px solid var(--b1);white-space:nowrap;}
.dt td{padding:7px 10px;border-bottom:1px solid #0f0f0f;color:var(--mu);}
.dt td:first-child{color:var(--tx);font-weight:500;}
.dt tr:hover td{background:#ffffff03;}
.scr{overflow-x:auto;-webkit-overflow-scrolling:touch;}

@media(max-width:900px){
.fg,.r2,.r3{grid-template-columns:1fr;} .fi.an{grid-column:span 1!important;}
.stats{grid-template-columns:1fr 1fr;} .sv{font-size:22px;}
.pg{padding:16px 14px;gap:20px;} .hdr{padding:18px 14px;} .tabs{padding:0 14px;}
.card{padding:16px 12px;} canvas{max-height:200px!important;}
.hm th,.hm td{padding:4px 3px;} .ce{min-width:34px;height:20px;font-size:9px;}
.dt th:nth-child(3),.dt td:nth-child(3),.dt th:nth-child(8),.dt td:nth-child(8),
.dt th:nth-child(9),.dt td:nth-child(9){display:none;}
.rc{gap:10px;}
}
</style>
</head>
<body>

<div class="hdr">
  <h1>Breaking Agents</h1>
  <span class="pill">COLM 2026 Workshop</span>
  <span class="pill g" id="dateBadge"></span>
  <span class="cd" id="cdSub"></span>
  <span class="cd ws" id="cdWs"></span>
  <p>Knockout Evaluation of Hidden-Mechanic Discovery Under Partial Observability — ka59simple · ls20 · bp35</p>
</div>

<div class="tabs">
  <div class="tab on" onclick="go('ov')">Overview</div>
  <div class="tab" onclick="go('an')">Research Analysis</div>
  <div class="tab" onclick="go('fi')">Findings</div>
  <div class="tab" onclick="go('dt')">Raw Data</div>
</div>

<div class="pg">

<!-- OVERVIEW -->
<div class="tp on" id="tp-ov">
  <div class="stats" id="kpi"></div>
  <div class="r2">
    <div class="card"><div class="stl">Win % Across Configs — All Models</div><canvas id="oc1"></canvas></div>
    <div class="card"><div class="stl">Baseline vs World Hard vs Mechanics Hard</div><canvas id="oc2"></canvas></div>
  </div>
  <div class="r3">
    <div class="card"><div class="stl">ka59simple — Axis Breakdown</div><table class="hm" id="ht1"></table></div>
    <div class="card"><div class="stl">ls20 — Axis Breakdown</div><table class="hm" id="ht2"></table></div>
    <div class="card"><div class="stl">bp35 — Axis Breakdown</div><table class="hm" id="ht3"></table></div>
  </div>
</div>

<!-- ANALYSIS -->
<div class="tp" id="tp-an">
  <div class="rc">
    <div class="cg"><label>Chart Type</label>
      <select id="ctype" onchange="rebuild()"><option value="bar">Bar</option><option value="line">Line</option></select></div>
    <div class="cg"><label>X Axis</label>
      <select id="xax" onchange="rebuild()"><option value="Model">Model</option><option value="Config">Config</option><option value="Game">Game</option><option value="Reasoning">Reasoning</option></select></div>
    <div class="cg"><label>Split By</label>
      <select id="spby" onchange="rebuild()"><option value="Config">Config</option><option value="Model">Model</option><option value="Game">Game</option><option value="Reasoning">Reasoning</option></select></div>
    <div class="cg"><label>Filter Game</label><select id="fg2" onchange="rebuild()"><option value="">All</option></select></div>
    <div class="cg"><label>Filter Model</label><details class="mdd"><summary id="fmsum">Models</summary><div id="fmbox" class="mbox"></div></details></div>
    <div class="cg"><label>Filter Config</label><select id="fc2" onchange="rebuild()"><option value="">All</option></select></div>
    <div class="cg">
      <div class="ct"><input type="checkbox" id="showCI" onchange="rebuild()" checked><label for="showCI">95% CI error bars</label></div>
      <div class="ct"><input type="checkbox" id="showSig" onchange="buildCI()"><label for="showSig">Sig. only in CI table</label></div>
    </div>
    <button class="btn sec" onclick="resetA()">Reset</button>
  </div>
  <div class="card">
    <div class="stl" id="atitle">Win Rate by Model, split by Config</div>
    <div class="chw" style="height:340px"><canvas id="ac"></canvas></div>
  </div>
  <div class="r2">
    <div class="card"><div class="stl">World Hard — 95% CI by Model</div><div class="chw" style="height:240px"><canvas id="wci"></canvas></div></div>
    <div class="card"><div class="stl">Mechanics Hard — 95% CI by Model</div><div class="chw" style="height:240px"><canvas id="mci"></canvas></div></div>
  </div>
  <div class="card">
    <div class="stl">Confidence Intervals — Wilson 95% CI per Condition</div>
    <div class="scr"><table class="cit"><thead><tr><th>Model</th><th>Game</th><th>Config</th><th>Reasoning</th><th>N</th><th>Win%</th><th>95% CI</th><th>Interval</th><th>vs Baseline</th></tr></thead><tbody id="cib"></tbody></table></div>
  </div>
</div>

<!-- FINDINGS -->
<div class="tp" id="tp-fi">
  <div class="fg">
    <div class="fi cl"><div class="fn">Finding 01</div><div class="fs re">Always 0%</div><div class="ft">World &amp; Mechanics Axes Collapse</div><div class="fb">Ablating world info or mechanics info collapses win rate to 0% across all models and all games. Upper bound ≤43% at n=5. No model circumvents this — both axes are necessary for task completion.</div></div>
    <div class="fi rv"><div class="fn">Finding 02</div><div class="fs gr">90pp swing</div><div class="ft">Goal Axis Directional Reversal</div><div class="fb">GPT-5.2 drops 60%→20% (no-R → medium) while Grok 4.1 rises 20%→70% on the same goal axis. Reasoning chain length interacts with model architecture in opposite ways — no universal benefit to reasoning.</div></div>
    <div class="fi dc"><div class="fn">Finding 03</div><div class="fs bl">56 wins</div><div class="ft">Verbal / Behavioral Decoupling</div><div class="fb">56 wins exploited the wall-transfer mechanic; verbal regex never fired once. OODA-F scaffold = 0/16 wins. Models solve behaviorally not linguistically — probing verbal output underestimates actual capability.</div></div>
    <div class="fi fl"><div class="fn">Finding 04</div><div class="fs" style="color:var(--or)">0/0/0</div><div class="ft">KA59 Canonical 0% Floor</div><div class="fb">Full KA59: GPT-4.1, GPT-5.2, and Claude Haiku all achieve 0% across every config. Used as difficulty anchor — ka59simple isolates the mechanic-discovery component specifically.</div></div>
    <div class="fi an" style="grid-column:span 2"><div class="fn">Finding 05 — Anomaly</div><div class="fs ye">60% world_hard</div><div class="ft">LS20 World_hard Anomaly — GPT-5.2 / no reasoning</div><div class="fb">The only case in the dataset where world_hard &gt; 0% without reasoning. GPT-5.2 none achieves 60% world_hard in ls20 vs 0% everywhere else. DeepSeek v4-pro also shows 5% world_hard on ls20 — the only other non-zero world_hard result. Suggests game-specific mechanic leakage or fundamentally different info structure in Josh's environment.</div></div>
  </div>
</div>

<!-- DATA -->
<div class="tp" id="tp-dt">
  <div class="card">
    <div class="stl">Full Knockout Matrix</div>
    <div class="tf">
      <select id="df1" onchange="rtbl()"><option value="">All games</option></select>
      <select id="df2" onchange="rtbl()"><option value="">All models</option></select>
      <select id="df3" onchange="rtbl()"><option value="">All reasoning</option></select>
      <select id="df4" onchange="rtbl()"><option value="">All configs</option></select>
    </div>
    <div class="scr">
      <table class="dt"><thead><tr><th>Model</th><th>Game</th><th>Reasoning</th><th>Config</th><th>N</th><th>Win%</th><th>Avg Turns</th><th>Wall Hits</th><th>Rel. diff</th></tr></thead>
      <tbody id="dtb"></tbody></table>
    </div>
  </div>
</div>

</div><!-- /pg -->

<script>
const RAW = RAWJSON;
const OV = OVJSON;

document.getElementById('dateBadge').textContent = 'Live · ' + new Date().toLocaleDateString('en-US',{month:'short',day:'numeric'});

// ── COLM 2026 countdowns (submission deadline + workshop) ──
(function(){
  const SUB = Date.UTC(2026,5,24,11,59,59); // 23 Jun 2026 23:59:59 AoE (UTC-12)
  const WS  = Date.UTC(2026,9,9,0,0,0);      // 9 Oct 2026 — Workshop on Agent Behavior @ COLM
  const fmt = ms => { const d=Math.floor(ms/864e5),h=Math.floor(ms/36e5)%24,m=Math.floor(ms/6e4)%60,s=Math.floor(ms/1e3)%60; return d+'d '+h+'h '+m+'m '+s+'s'; };
  const sub=document.getElementById('cdSub'), ws=document.getElementById('cdWs');
  function tick(){
    const now=Date.now();
    const ds=SUB-now;
    if(ds>0){ sub.textContent='Submissions close in '+fmt(ds); }
    else { sub.textContent='Submissions closed'; sub.classList.add('done'); }
    const dw=WS-now;
    ws.textContent = dw>0 ? 'Workshop in '+fmt(dw) : 'Workshop underway';
  }
  tick(); setInterval(tick,1000);
})();

// ── utils ──────────────────────────────────────────────────────────────────────
const P=['#7c3aed','#3b82f6','#f97316','#22c55e','#ec4899','#fbbf24','#14b8a6','#ef4444','#a78bfa','#60a5fa'];
function avg(a){const v=a.filter(x=>x!=null&&!isNaN(x));return v.length?v.reduce((s,x)=>s+x,0)/v.length:null;}
function wtxt(w){return w==null||isNaN(w)?'—':Math.round(w)+'%';}
function cc(w){if(w==null||isNaN(w))return'cn';if(w>=100)return'c5';if(w>=60)return'ch';if(w>=30)return'cm';if(w>0)return'cl2';return'c0';}
function wps(w){
  if(w==null||isNaN(w))return'background:#1a1a1a;color:#444';
  if(w===0)return'background:#1c0b0b;color:#6b2020';
  if(w<30)return'background:#3b0f0f;color:#f87171';
  if(w<60)return'background:#422006;color:#fb923c';
  if(w<80)return'background:#15412a;color:#6ee7b7';
  return'background:#14532d;color:#4ade80';
}
function wilson(pct,n){
  if(!n||n<=0||pct==null||isNaN(pct))return{lo:null,hi:null};
  const p=pct/100,z=1.96,z2=z*z,d=1+z2/n;
  const c=(p+z2/(2*n))/d,m=(z/d)*Math.sqrt(p*(1-p)/n+z2/(4*n*n));
  return{lo:Math.max(0,(c-m)*100),hi:Math.min(100,(c+m)*100)};
}
function noOverlap(a,b){return a.lo!=null&&b.lo!=null&&(a.hi<b.lo||b.hi<a.lo);}

// ── KPI ───────────────────────────────────────────────────────────────────────
const allM=[...new Set([...RAW.map(r=>r.Model),...OV.map(r=>r.Model)].filter(Boolean))];
const allG=[...new Set([...RAW.map(r=>r.Game),...OV.map(r=>r.Game)].filter(Boolean))];
const wCols=['Baseline','Win% world_hard','Win% goal_hard','Win% mechanics_hard','Win% feedback_hard'];
const ovVals=[];OV.forEach(r=>wCols.forEach(c=>{if(r[c]!=null&&!isNaN(r[c]))ovVals.push(r[c]);}));
const totalN=OV.filter(r=>r.N&&!isNaN(+r.N)).reduce((s,r)=>s+(+r.N),0);
document.getElementById('kpi').innerHTML=`
<div class="st"><div class="sv">${totalN}</div><div class="sl">Total Trials (N)</div></div>
<div class="st"><div class="sv ac">${allM.length}</div><div class="sl">Models Tested</div></div>
<div class="st"><div class="sv ac">${allG.length}</div><div class="sl">Environments</div></div>
<div class="st"><div class="sv">${Math.round(avg(ovVals))}%</div><div class="sl">Avg Win Rate</div></div>
<div class="st"><div class="sv re">${Math.round(ovVals.filter(x=>x===0).length/ovVals.length*100)}%</div><div class="sl">Configs at 0%</div></div>`;

// ── Tab switching ─────────────────────────────────────────────────────────────
function go(id){
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('on',['ov','an','fi','dt'][i]===id));
  document.querySelectorAll('.tp').forEach(p=>p.classList.remove('on'));
  document.getElementById('tp-'+id).classList.add('on');
  if(id==='an'){
    requestAnimationFrame(()=>{
      if(aC){aC.resize();}
      if(wC){wC.resize();}
      if(mC){mC.resize();}
    });
  }
}

// ── Overview charts ───────────────────────────────────────────────────────────
const ovMs=[...new Set(OV.map(r=>r.Model).filter(Boolean))];
const cfgC=['Baseline','Win% world_hard','Win% goal_hard','Win% mechanics_hard','Win% feedback_hard'];
const cfgL=['Baseline','World Hard','Goal Hard','Mech Hard','Feedback Hard'];

new Chart(document.getElementById('oc1'),{type:'line',
  data:{labels:cfgL,datasets:ovMs.map((m,i)=>({label:m,
    data:cfgC.map(c=>{const v=avg(OV.filter(r=>r.Model===m).map(r=>r[c]));return v!=null?Math.round(v):null;}),
    borderColor:P[i%P.length],backgroundColor:P[i%P.length]+'18',
    tension:.35,pointRadius:5,pointHoverRadius:7,borderWidth:2,spanGaps:true}))},
  options:{responsive:true,plugins:{legend:{labels:{color:'#666',font:{size:10},boxWidth:12}}},
    scales:{x:{ticks:{color:'#555',font:{size:10}},grid:{color:'#181818'}},
      y:{min:0,max:100,ticks:{color:'#555',callback:v=>v+'%',font:{size:10}},grid:{color:'#181818'}}}}});

new Chart(document.getElementById('oc2'),{type:'bar',
  data:{labels:ovMs,datasets:['Baseline','Win% world_hard','Win% mechanics_hard'].map((c,i)=>({
    label:['Baseline','World Hard','Mech Hard'][i],
    data:ovMs.map(m=>{const v=avg(OV.filter(r=>r.Model===m).map(r=>r[c]));return v!=null?Math.round(v):null;}),
    backgroundColor:[P[0],P[7],P[2]][i]+'cc',borderRadius:3}))},
  options:{responsive:true,plugins:{legend:{labels:{color:'#666',font:{size:10},boxWidth:12}}},
    scales:{x:{ticks:{color:'#555',font:{size:9},maxRotation:30},grid:{color:'#181818'}},
      y:{min:0,max:100,ticks:{color:'#555',callback:v=>v+'%',font:{size:10}},grid:{color:'#181818'}}}}});

// Heatmaps
function hm(game,id){
  const rows=OV.filter(r=>r.Game===game),el=document.getElementById(id);
  if(!rows.length){el.innerHTML='<tr><td style="color:#333;padding:10px">No data</td></tr>';return;}
  const cols=['Baseline','Win% world_hard','Win% goal_hard','Win% mechanics_hard','Win% feedback_hard'];
  const sh=['Base','World','Goal','Mech','Feed'];
  let h=`<thead><tr><th>Model/R</th>${sh.map(s=>`<th>${s}</th>`).join('')}</tr></thead><tbody>`;
  rows.forEach(r=>{
    const anom=r.Game==='ls20'&&r['Win% world_hard']>0;
    h+=`<tr><td><span class="ml">${r.Model}</span><br><span class="rl">${r['Reasoning?']||'—'}</span>${anom?'<span class="ab">⚠ anomaly</span>':''}</td>`;
    cols.forEach(c=>{const v=r[c];h+=`<td><span class="ce ${cc(v)}">${wtxt(v)}</span></td>`;});
    h+='</tr>';
  });
  el.innerHTML=h+'</tbody>';
}
hm('ka59simple','ht1');hm('ls20','ht2');hm('bp35','ht3');

// ── Error bar plugin (self-contained) ─────────────────────────────────────────
const errBarsPlugin={id:'errBars',afterDatasetsDraw(chart){
  const ctx=chart.ctx;
  chart.data.datasets.forEach((ds,di)=>{
    const meta=chart.getDatasetMeta(di);if(!meta.visible)return;
    ds.data.forEach((pt,i)=>{
      if(!pt||pt.lo==null)return;
      const el=meta.data[i];if(!el)return;
      const x=el.x,yLo=chart.scales.y.getPixelForValue(pt.lo),yHi=chart.scales.y.getPixelForValue(pt.hi);
      ctx.save();ctx.strokeStyle=ds.borderColor||'rgba(255,255,255,0.7)';ctx.lineWidth=1.5;
      ctx.beginPath();ctx.moveTo(x,yLo);ctx.lineTo(x,yHi);ctx.stroke();
      const w=7;
      [yLo,yHi].forEach(y=>{ctx.beginPath();ctx.moveTo(x-w,y);ctx.lineTo(x+w,y);ctx.stroke();});
      ctx.restore();
    });
  });
}};

// ── Analysis chart ────────────────────────────────────────────────────────────
let aC=null;
function filtered(){
  const gf=document.getElementById('fg2').value,cf=document.getElementById('fc2').value;
  const sel=new Set(selectedModels());
  return RAW.filter(r=>(!gf||r.Game===gf)&&sel.has(r.Model)&&(!cf||r.Config===cf)&&r['Win%']!=null&&!isNaN(r['Win%']));
}
function rebuild(){
  const xk=document.getElementById('xax').value,sk=document.getElementById('spby').value;
  const ct=document.getElementById('ctype').value,ci=document.getElementById('showCI').checked;
  const fd=filtered();
  const xv=[...new Set(fd.map(r=>r[xk]).filter(Boolean))].sort();
  const sv=[...new Set(fd.map(r=>r[sk]).filter(Boolean))].sort();
  document.getElementById('atitle').textContent=`Win Rate by ${xk}, split by ${sk}`;
  const ds=sv.map((s,i)=>{
    const color=P[i%P.length];
    const pts=xv.map(x=>{
      const rows=fd.filter(r=>r[xk]===x&&r[sk]===s);
      if(!rows.length)return{x:x,y:null,lo:null,hi:null};
      const w=avg(rows.map(r=>r['Win%']));
      const n=rows.reduce((sum,r)=>sum+(+r.N||5),0);
      const wl=wilson(w,n);
      return{x:x,y:w!=null?Math.round(w):null,lo:ci?Math.round(wl.lo||0):null,hi:ci?Math.round(wl.hi||0):null};
    });
    return{label:s,type:ct,data:pts,backgroundColor:color+'bb',borderColor:color,borderWidth:2,tension:.3,borderRadius:3,spanGaps:true,pointRadius:4};
  });
  if(aC)aC.destroy();
  aC=new Chart(document.getElementById('ac'),{
    data:{labels:xv,datasets:ds},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#666',font:{size:11},boxWidth:12}},
        tooltip:{callbacks:{label:c=>{const d=c.raw;if(d&&d.lo!=null)return`${c.dataset.label}: ${d.y}% (CI: ${d.lo}–${d.hi}%)`;return`${c.dataset.label}: ${d&&d.y!=null?d.y+'%':'—'}`;}}}},
      scales:{x:{ticks:{color:'#555',font:{size:10},maxRotation:30},grid:{color:'#181818'}},
        y:{min:0,max:100,ticks:{color:'#555',callback:v=>v+'%',font:{size:10}},grid:{color:'#181818'}}}},
    plugins:[errBarsPlugin]});
}
function resetA(){
  ['xax','spby','ctype','fg2','fm2','fc2'].forEach(id=>{const e=document.getElementById(id);if(e)e.selectedIndex=0;});
  document.getElementById('showCI').checked=true;document.getElementById('showSig').checked=false;
  buildModelPicker();
  rebuild();buildCICharts();buildCI();
}

// ── CI charts (world_hard & mechanics_hard) ───────────────────────────────────
let wC=null,mC=null;
function buildCICharts(){
  ['world_hard','mechanics_hard'].forEach((cfg,idx)=>{
    const sel=new Set(selectedModels());
    const rows=RAW.filter(r=>r.Config===cfg&&sel.has(r.Model)&&r['Win%']!=null&&!isNaN(r['Win%'])&&r.N);
    const ms=[...new Set(rows.map(r=>r.Model))].sort();
    const pts=ms.map(m=>{
      const mr=rows.filter(r=>r.Model===m);
      const w=avg(mr.map(r=>r['Win%']));
      const n=mr.reduce((s,r)=>s+(+r.N||5),0);
      const wl=wilson(w,n);
      return{x:m,y:w!=null?Math.round(w):0,lo:Math.round(wl.lo||0),hi:Math.round(wl.hi||0)};
    });
    const cid=idx===0?'wci':'mci';
    const existing=idx===0?wC:mC;
    if(existing)existing.destroy();
    const inst=new Chart(document.getElementById(cid),{
      data:{labels:ms,datasets:[{label:cfg.replace('_',' '),type:'bar',data:pts,
        backgroundColor:P[idx*3]+'bb',borderColor:P[idx*3],borderWidth:2,borderRadius:4}]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>{const d=c.raw;return`${d.y}% (CI: ${d.lo}–${d.hi}%)`;}}}},
        scales:{x:{ticks:{color:'#555',font:{size:9},maxRotation:30},grid:{color:'#181818'}},
          y:{min:0,max:100,ticks:{color:'#555',callback:v=>v+'%',font:{size:10}},grid:{color:'#181818'}}}},
      plugins:[errBarsPlugin]});
    if(idx===0)wC=inst;else mC=inst;
  });
}

// ── CI table ──────────────────────────────────────────────────────────────────
function buildCI(){
  const sigOnly=document.getElementById('showSig').checked;
  const sel=new Set(selectedModels());
  const rows=RAW.filter(r=>sel.has(r.Model)&&r['Win%']!=null&&!isNaN(r['Win%'])&&r.N);
  const baseCIs={};
  rows.filter(r=>r.Config==='baseline').forEach(r=>{
    const k=`${r.Model}|${r.Game}`;
    if(!baseCIs[k])baseCIs[k]=wilson(r['Win%'],+r.N);
  });
  document.getElementById('cib').innerHTML=rows
    .sort((a,b)=>(a.Model+a.Game+a.Config).localeCompare(b.Model+b.Game+b.Config))
    .map(r=>{
      const w=r['Win%'],n=+r.N,ci=wilson(w,n);
      const bci=baseCIs[`${r.Model}|${r.Game}`];
      const sig=r.Config!=='baseline'&&bci&&ci.lo!=null?noOverlap(ci,bci):null;
      if(sigOnly&&sig===false)return'';
      const lo=ci.lo!=null?ci.lo:w||0,hi=ci.hi!=null?ci.hi:w||0;
      const fillBg=w>0?'#3b82f620':'#3b0f0f40';
      return`<tr>
        <td>${r.Model}</td><td>${r.Game}</td><td>${r.Config}</td><td>${r.Reasoning||'—'}</td><td>${n}</td>
        <td><span class="wp" style="${wps(w)}">${wtxt(w)}</span></td>
        <td style="font-size:10px;color:var(--mu)">${ci.lo!=null?Math.round(ci.lo)+'–'+Math.round(ci.hi)+'%':'—'}</td>
        <td><div class="cb"><div class="ct2">
          <div class="cf2" style="left:${lo}%;width:${Math.max(0,hi-lo)}%;background:${fillBg}"></div>
          <div class="cm2" style="left:${w||0}%"></div>
          ${ci.lo!=null?`<div class="cl3" style="left:${lo}%"></div><div class="ch3" style="left:${hi}%"></div>`:''}
        </div><span style="font-size:10px;color:var(--di);min-width:26px">${wtxt(w)}</span></div></td>
        <td>${sig===null?'<span class="sg sn">baseline</span>':sig?'<span class="sg sy">✓ sig</span>':'<span class="sg sn">overlap</span>'}</td>
      </tr>`;
    }).join('');
}

// ── Populate filters ──────────────────────────────────────────────────────────
function popSel(id,vals){const e=document.getElementById(id);[...new Set(vals)].filter(Boolean).sort().forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;e.appendChild(o);});}
// ── Model multi-select picker ─────────────────────────────────────────────────
const MODELS=[...new Set(RAW.map(r=>r.Model).filter(Boolean))].sort();
const MODEL_OFF=new Set(['claude-haiku-4-5']);  // near-empty: available but off by default
function buildModelPicker(){
  const box=document.getElementById('fmbox');
  box.innerHTML='<label class="mall"><input type="checkbox" id="fmAll" onchange="toggleAllModels(this.checked)"> Select all</label>'
    +MODELS.map(m=>`<label><input type="checkbox" class="fmck" value="${m}" ${MODEL_OFF.has(m)?'':'checked'} onchange="onModels()"> ${m}</label>`).join('');
  syncModelSummary();
}
function selectedModels(){return [...document.querySelectorAll('.fmck:checked')].map(c=>c.value);}
function toggleAllModels(on){document.querySelectorAll('.fmck').forEach(c=>c.checked=on);onModels();}
function syncModelSummary(){
  const sel=selectedModels(),tot=MODELS.length,sum=document.getElementById('fmsum');
  sum.textContent=sel.length===tot?`Models · all (${tot})`:(sel.length===0?'Models · none':`Models · ${sel.length}/${tot}`);
  const all=document.getElementById('fmAll');if(all)all.checked=(sel.length===tot);
}
function onModels(){syncModelSummary();rebuild();buildCICharts();buildCI();}
popSel('fg2',RAW.map(r=>r.Game));popSel('fc2',RAW.map(r=>r.Config));buildModelPicker();
popSel('df1',RAW.map(r=>r.Game));popSel('df2',RAW.map(r=>r.Model));popSel('df3',RAW.map(r=>r.Reasoning));popSel('df4',RAW.map(r=>r.Config));

// ── Raw data table ────────────────────────────────────────────────────────────
function rtbl(){
  const g=document.getElementById('df1').value,m=document.getElementById('df2').value,
        rs=document.getElementById('df3').value,c=document.getElementById('df4').value;
  document.getElementById('dtb').innerHTML=RAW
    .filter(r=>(!g||r.Game===g)&&(!m||r.Model===m)&&(!rs||r.Reasoning===rs)&&(!c||r.Config===c))
    .map(r=>`<tr><td>${r.Model||'—'}</td><td>${r.Game||'—'}</td><td>${r.Reasoning||'—'}</td><td>${r.Config||'—'}</td><td>${r.N||'—'}</td>
      <td><span class="wp" style="${wps(r['Win%'])}">${wtxt(r['Win%'])}</span></td>
      <td>${r['Avg Turns']||'—'}</td><td>${r['Wall Hits']||'—'}</td><td>${r['Rel. diff.']||'—'}</td></tr>`).join('');
}
rtbl();
rebuild();buildCICharts();buildCI();
</script>
</body></html>"""

@app.route('/')
def index():
    df = load_data()
    ov = load_overview()
    page = PAGE.replace('RAWJSON', df.to_json(orient='records')) \
               .replace('OVJSON', ov.to_json(orient='records'))
    return page

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090, debug=False)
