/* ============================================================================
   AI Compliance Chat Widget – embeddable with runtime settings v2.10
   (v2 .10 = v2 .9 + whitespace-preserving fixes)
   All comments are in English (per user preference).
============================================================================ */
(() => {
  'use strict';

  // 1. CONFIG ──────────────────────────────────────────────────────────────
  const el         = document.currentScript;
  const BACKEND    = (el.dataset.backend || '').replace(/\/$/, '');
  const WIDGET_KEY = el.dataset.key || '';
  const USE_STREAM = el.dataset.stream !== 'false';

  if (!BACKEND || !WIDGET_KEY) {
    console.error('[aiw] backend URL and data-key are required');
    return;
  }

  // Dimensions
  const COMPACT_W = 340, EXPANDED_W = 600, DEBUG_W = 320, COMPACT_H = 420;

  // Widget state
  let isExpanded      = false;
  let chatBusy        = false;
  let debugPanelOpen  = false;

  // Settings toggles (persisted in localStorage)
  let showChunkDebug  = (localStorage.getItem('aiw-show-chunks-debug') ?? 'true') === 'true';
  let showToolResults = (localStorage.getItem('aiw-show-tool-results') ?? 'true') === 'true';

  // Request control
  let currentFetchController = null;
  let currentReader          = null;

  // History settings
  const toggleStateStr = localStorage.getItem('aiw-toggle-history');
  let useHistory = (toggleStateStr === null) ? true : (toggleStateStr === 'true');
  let prevID     = useHistory ? localStorage.getItem('aiw-prevID') || null : null;

  // Stored chat
  let chatHistory = [];
  if (useHistory) {
    try { chatHistory = JSON.parse(localStorage.getItem('aiw-chat-history') || '[]'); }
    catch { chatHistory = []; }
  }

  // Track manual open
  let userOpenedChat = false;

  // 2. STYLES ───────────────────────────────────────────────────────────────
  const css = /* css */`
:root{
  --aiw-primary:#2563eb;
  --aiw-primary-light:#3b82f6;
  --aiw-primary-dark:#1e40af;
  --aiw-text-on-primary:#ffffff;
  /* Glass variables (can be overridden dynamically) */
  --aiw-glass: rgba(255,255,255,0.55);
  --aiw-user-glass: rgba(255,255,255,0.65);
  --aiw-assistant-glass: rgba(99,179,237,0.55);
  /* Fallbacks for non-glass themes */
  --aiw-user-bg:linear-gradient(180deg, #f8fafc, #eef2ff);
  --aiw-user-text:#111827;
  --aiw-assistant-bg:linear-gradient(180deg, #dcfce7, #bee9d0);
  --aiw-assistant-text:#065f46;
  --aiw-radius:12px;
  --aiw-shadow:0 10px 28px rgba(0,0,0,.16);
  --aiw-font:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{
    --aiw-primary:#3b82f6;
    --aiw-primary-light:#60a5fa;
    --aiw-primary-dark:#1d4ed8;
  --aiw-glass: rgba(17,24,39,0.45);
  --aiw-user-glass: rgba(31,41,55,0.55);
  --aiw-assistant-glass: rgba(6,78,59,0.55);
  --aiw-user-bg:linear-gradient(180deg, #1f2937, #111827);
    --aiw-user-text:#e5e7eb;
    --aiw-assistant-bg:linear-gradient(180deg, #064e3b, #0a5e49);
    --aiw-assistant-text:#d1fae5;
  }
}
.aiw-launcher{
  position:fixed; bottom:24px; right:24px;
  display:flex; align-items:center; justify-content:center;
  width:56px; height:56px;
  background:linear-gradient(180deg, var(--aiw-primary-light), var(--aiw-primary)); color:var(--aiw-text-on-primary);
  border:1px solid rgba(255,255,255,.22); border-radius:50%;
  font-size:24px; cursor:pointer;
  box-shadow:0 12px 28px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.35);
  animation:aiwPulse 2.8s ease-in-out infinite;
  z-index:100000;
}
.aiw-launcher:focus-visible{outline:3px solid var(--aiw-primary-light);}
.aiw-box{
  position:fixed; bottom:90px; right:24px;
  display:none; flex-direction:row;
  width:${COMPACT_W}px; height:${COMPACT_H}px;
  background:var(--aiw-glass); border:1px solid rgba(255,255,255,0.45);
  border-radius:var(--aiw-radius); box-shadow:0 16px 44px rgba(0,0,0,.18);
  backdrop-filter: blur(14px) saturate(120%);
  will-change: backdrop-filter, background;
  font-family:var(--aiw-font); overflow:hidden;
  transition:width .25s, height .25s, background .18s ease, border-color .18s ease;
  scroll-behavior:smooth;
}
@media(max-width:480px){
  .aiw-box{right:12px; width:calc(100vw - 24px);}
}
.aiw-chat-main {
  display:flex; flex-direction:column; flex:1 1 0%;
  min-width:0; min-height:0;
}
.aiw-chat-main header{
  display:flex; align-items:center; gap:10px;
  background:linear-gradient(180deg, var(--aiw-primary), var(--aiw-primary-dark)); color:var(--aiw-text-on-primary);
  padding:.75rem 1rem; font-size:15px; font-weight:600;
  box-shadow:inset 0 -1px 0 rgba(0,0,0,.12);
}
.aiw-chat-main header img{
  width:28px; height:28px; border-radius:50%; background:#fff; padding:2px;
}
.aiw-chat-main header .toggle,
.aiw-chat-main header .settings{
  background:none; border:none; color:inherit;
  font-size:1.1rem; cursor:pointer; line-height:1;
  transition:transform .15s;
}
.aiw-chat-main header .toggle:hover,
.aiw-chat-main header .settings:hover{transform:scale(1.15);}
.aiw-chat-main header .reset-chat{
  background:none; border:none; color:inherit;
  font-size:1rem; cursor:pointer; line-height:1;
  transition:transform .15s; margin-left:-2px;
}
.aiw-chat-main header .reset-chat:hover{transform:scale(1.15);}
.history-wrapper{
  display:flex; align-items:center; margin-left:auto; margin-right:4px;
}
.history-label{margin-right:6px; font-size:13px; opacity:0.9;}
.toggle-history{position:relative; display:inline-block; width:36px; height:18px; margin-right:8px;}
.toggle-history input{opacity:0; width:0; height:0;}
.slider{
  position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0;
  background-color:rgba(255,255,255,0.6); border-radius:34px; transition:.2s;
}
.slider:before{
  position:absolute; content:"";
  height:14px; width:14px; left:2px; bottom:2px;
  background-color:var(--aiw-primary-dark); border-radius:50%; transition:.2s;
}
.toggle-history input:checked + .slider{background-color:rgba(255,255,255,0.85);}
.toggle-history input:checked + .slider:before{transform:translateX(18px);}
.aiw-chat-main .messages{
  flex:1; min-height:0; padding:1rem;
  display:flex; flex-direction:column; gap:.75rem; overflow-y:auto;
}
.aiw-chat-main footer{
  display:flex; gap:.5rem; padding:.75rem 1rem; border-top:1px solid #f3f4f6;
}
.aiw-chat-main textarea{
  flex:1; font:inherit; font-size:14px;
  padding:.5rem; border:1px solid #d1d5db; border-radius:8px; resize:vertical;
}
.aiw-chat-main textarea:focus-visible{outline:2px solid var(--aiw-primary-light);}
.aiw-chat-main button.send{
  background:var(--aiw-primary); color:var(--aiw-text-on-primary);
  border:none; padding:.5rem 1rem; border-radius:8px; cursor:pointer; transition:background .2s;
}
.aiw-chat-main button.send:hover{background:var(--aiw-primary-dark);}

/*────────────────────  Whitespace-preserving update  ────────────────────*/
.aiw-chat-main .message{
  max-width:90%; padding:.75rem; border-radius:10px;
  line-height:1.45; font-size:14px;
  white-space:pre-wrap;          /* MOD: keep \n, \t, double spaces            */
  tab-size:4;                    /* MOD: visual width of a TAB character        */
  word-break:break-word;         /* MOD: prevent overflow on long strings       */
  border:1px solid rgba(0,0,0,.05);
  box-shadow:0 2px 8px rgba(0,0,0,.06), inset 0 1px 0 rgba(255,255,255,.6);
}
.aiw-chat-main .message.user      {background:var(--aiw-user-glass, var(--aiw-user-bg)); color:var(--aiw-user-text); margin-left:auto;}
.aiw-chat-main .message.assistant {background:var(--aiw-assistant-glass, var(--aiw-assistant-bg)); color:var(--aiw-assistant-text); margin-right:auto;}
.aiw-chat-main .message.legal-note{background:var(--aiw-user-bg); color:#6b7280; margin-right:auto; font-size:13px;}
.aiw-chat-main .message.assistant.typing{opacity:0.8;}

/* Monospace font for explicit code/pre blocks */
.aiw-chat-main .message pre,
.aiw-chat-main .message code{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  white-space:pre-wrap;
}

.aiw-chat-main .typing::after{content:"."; animation:aiwDots 1.2s steps(3,end) infinite;}
@keyframes aiwDots{0%{content:"."}33%{content:".."}66%{content:"..."}}
.aiw-chat-main .quick-replies{
  display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.5rem;
}
.aiw-chat-main .quick-replies button{
  background:#eef2ff; color:#1e3a8a; border:none; border-radius:6px;
  padding:.4rem .75rem; font-size:0.85rem; cursor:pointer; transition:background .2s;
}
.aiw-chat-main .quick-replies button:hover{background:#e0e7ff;}
.aiw-chat-main .getInTouch{
  background:var(--aiw-primary); color:var(--aiw-text-on-primary);
  border:none; padding:.5rem 1rem; border-radius:8px;
  font-size:0.85rem; cursor:pointer; transition:background .2s;
}
.aiw-chat-main .getInTouch:hover{background:var(--aiw-primary-dark);}
.aiw-modal{
  position:fixed; inset:0; background:rgba(0,0,0,.35);
  display:flex; align-items:center; justify-content:center;
  opacity:0; visibility:hidden; transition:.2s; z-index:100001;
}
.aiw-modal.open{opacity:1; visibility:visible;}
.aiw-modal .panel{
  width:420px; max-width:90vw; background:#fff;
  border-radius:var(--aiw-radius); box-shadow:var(--aiw-shadow);
  padding:1.25rem 1.5rem; font-family:var(--aiw-font);
}
.aiw-modal .panel h2{margin:0 0 1rem; font-size:1.1rem;}
.aiw-modal label{display:block; margin-bottom:.75rem; font-size:.9rem;}
.aiw-modal select,
.aiw-modal textarea{
  width:100%; font:inherit; font-size:.92rem;
  padding:.5rem; border:1px solid #d1d5db; border-radius:8px;
}
.aiw-modal textarea{resize:vertical; min-height:110px;}
.aiw-modal .actions{
  margin-top:1.1rem; display:flex; justify-content:flex-end; gap:.5rem;
}
.aiw-modal .actions button{
  border:none; border-radius:8px; cursor:pointer; font-size:.9rem; padding:.5rem 1rem;
}
.aiw-modal .actions .save{background:var(--aiw-primary); color:var(--aiw-text-on-primary);}
.aiw-modal .actions .cancel{background:#e5e7eb;}
.aiw-modal .actions .save:hover{background:var(--aiw-primary-dark);}
.aiw-modal .actions .cancel:hover{background:#d1d5db;}
.aiw-debug-panel{
  display:none; flex-direction:column;
  width:${DEBUG_W}px; max-width:50vw; min-width:220px; height:100%;
  background:#111827; color:#e5e7eb; font-size:12px;
  border-right:1px solid #374151; overflow:hidden; z-index:2;
}
.aiw-debug-panel.open{display:flex;}
.aiw-debug-panel header{
  display:flex; align-items:center; justify-content:space-between;
  padding:.4rem .6rem; background:#1f2937; font-weight:600;
}
.aiw-debug-panel pre{
  flex:1; margin:0; padding:.5rem .75rem; overflow-y:auto; white-space:pre-wrap;
}
.aiw-debug-panel button{background:none;border:none;color:inherit;cursor:pointer;}
.aiw-box {flex-direction:row !important;}
.aiw-debug-panel {order:0;}
.aiw-chat-main   {order:1; flex:1 1 0%; display:flex; flex-direction:column; min-width:0; min-height:0;}
@media(max-width:480px){
  .aiw-box{width:100vw;min-width:0;}
  .aiw-debug-panel{max-width:45vw;}
}
@keyframes aiwPulse{
  0%  {transform:rotate(0) scale(1);   box-shadow:0 0 0 rgba(37,99,235,0.4);}
  50% {transform:rotate(-5deg) scale(1.08); box-shadow:0 0 12px rgba(37,99,235,0.6);}
  100%{transform:rotate(0) scale(1);   box-shadow:0 0 0 rgba(37,99,235,0.4);}
}
`;

  // Inject runtime CSS
  document.head.appendChild(Object.assign(document.createElement('style'), {textContent: css}));

  // 3. DOM CREATION ─────────────────────────────────────────────────────────
  const launcher = Object.assign(document.createElement('button'),{
    className:'aiw-launcher', title:'Chat', 'aria-label':'Open chat widget', textContent:'💬'
  });
  document.body.appendChild(launcher);

  const box = document.createElement('div');
  box.className = 'aiw-box';

  const chatMain = document.createElement('div');
  chatMain.className = 'aiw-chat-main';
  chatMain.innerHTML = `
<header>
  <img src="https://img.icons8.com/fluency/48/artificial-intelligence.png" alt="">
  Chat with Brita AI
  <div class="history-wrapper">
    <span class="history-label">History</span>
    <label class="toggle-history">
      <input type="checkbox" ${useHistory ? 'checked' : ''} />
      <span class="slider"></span>
    </label>
  </div>
  <button class="reset-chat" title="Refresh Chat">🔄</button>
  <button class="settings"   title="Settings">⚙</button>
  <button class="toggle" title="Expand">⛶</button>
</header>
<div class="messages"></div>
<footer>
  <textarea rows="2" placeholder="Type your message…"></textarea>
  <button class="send">➤</button>
</footer>`;

  // Debug panel (left column, hidden by default)
  const debugPanel = document.createElement('div');
  debugPanel.className = 'aiw-debug-panel';
  debugPanel.innerHTML = `
<header>
  <span>Debug Console</span>
  <div>
    <button class="clear" title="Clear log">🗑</button>
    <button class="close" title="Close">✖</button>
  </div>
</header>
<pre></pre>`;

  box.appendChild(debugPanel);
  box.appendChild(chatMain);
  document.body.appendChild(box);

  // Settings modal (+debug and chunk/tool toggles)
  const modal = document.createElement('div');
  modal.className = 'aiw-modal';
  modal.innerHTML = `
<div class="panel">
  <h2>Chat Settings</h2>
  <form>
    <label>Model <select name="model"></select></label>
    <label>Instructions <textarea name="instructions" placeholder="Optional system instructions…"></textarea></label>
    <label style="display:flex;align-items:center;gap:8px;margin-top:6px;">
      <input type="checkbox" name="autotheme" style="width:20px;height:20px;vertical-align:middle;" />
      Match site theme (colors auto‑adapt)
    </label>
    <label style="display:flex;align-items:center;gap:8px;">
      <input type="checkbox" name="showdebug" style="width:20px;height:20px;vertical-align:middle;" />
      Show debug panel
    </label>
    <label style="display:flex;align-items:center;gap:8px;margin-top:6px;">
      <input type="checkbox" name="showchunks" style="width:20px;height:20px;vertical-align:middle;" />
      Show chunk messages in debug
    </label>
    <label style="display:flex;align-items:center;gap:8px;margin-top:6px;">
      <input type="checkbox" name="showtoolresults" style="width:20px;height:20px;vertical-align:middle;" />
      Show tool/function results
    </label>
    <div class="actions">
      <button type="button" class="cancel">Cancel</button>
      <button type="submit" class="save">Save</button>
    </div>
  </form>
</div>`;
  document.body.appendChild(modal);

  // 4. REFERENCES (DOM nodes) ───────────────────────────────────────────────
  const toggleBtn        = chatMain.querySelector('.toggle');
  const settingsBtn      = chatMain.querySelector('.settings');
  const historyToggle    = chatMain.querySelector('.toggle-history input');
  const resetChatBtn     = chatMain.querySelector('.reset-chat');
  const messages         = chatMain.querySelector('.messages');
  const textarea         = chatMain.querySelector('textarea');
  const sendBtn          = chatMain.querySelector('.send');

  // Debug refs
  const dbgClose         = debugPanel.querySelector('.close');
  const dbgClear         = debugPanel.querySelector('.clear');
  const dbgOutput        = debugPanel.querySelector('pre');

  // Modal refs
  const modalForm        = modal.querySelector('form');
  const modelSelect      = modal.querySelector('select[name=model]');
  const instrTextarea    = modal.querySelector('textarea[name=instructions]');
  const showDebugInput   = modal.querySelector('input[name=showdebug]');
  const showChunksInput  = modal.querySelector('input[name=showchunks]');
  const showToolResultsInput = modal.querySelector('input[name=showtoolresults]');
  const autoThemeInput      = modal.querySelector('input[name=autotheme]');
  const modalCancelBtn   = modal.querySelector('.cancel');
  // Theme state
  let autoThemeEnabled = (localStorage.getItem('aiw-auto-theme') ?? 'false') === 'true';

  // ───────────────────────── 5a. THEME HELPERS ─────────────────────────────
  function clamp(n, min, max){ return Math.min(max, Math.max(min, n)); }
  function parseColor(str){
    if(!str) return null;
    str = String(str).trim();
    if(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(str)){
      if(str.length===4){ return '#' + str.slice(1).split('').map(c=>c+c).join(''); }
      return str.toLowerCase();
    }
    const m = str.match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)$/i);
    if(m){
      const r=+m[1], g=+m[2], b=+m[3];
      return '#'+[r,g,b].map(v=>v.toString(16).padStart(2,'0')).join('');
    }
    return null; // ignore named colors
  }
  function hexToHsl(hex){
    hex = parseColor(hex); if(!hex) return null;
    const r = parseInt(hex.slice(1,3),16)/255,
          g = parseInt(hex.slice(3,5),16)/255,
          b = parseInt(hex.slice(5,7),16)/255;
    const max=Math.max(r,g,b), min=Math.min(r,g,b);
    let h=0, s=0, l=(max+min)/2;
    const d=max-min;
    if(d!==0){
      s = l>0.5 ? d/(2-max-min) : d/(max+min);
      switch(max){
        case r: h=(g-b)/d+(g<b?6:0); break;
        case g: h=(b-r)/d+2; break;
        case b: h=(r-g)/d+4; break;
      }
      h/=6;
    }
    return {h, s, l};
  }
  function hslToHex(h,s,l){
    function f(n){
      const k=(n+h*12)%12;
      const a=s*Math.min(l,1-l);
      const c=l-a*Math.max(-1, Math.min(k-3, Math.min(9-k, 1)));
      return Math.round(255*c).toString(16).padStart(2,'0');
    }
    return `#${f(0)}${f(8)}${f(4)}`;
  }
  function lighten(hex, amt){ const h=hexToHsl(hex); if(!h) return hex; return hslToHex(h.h, h.s, clamp(h.l+amt, 0, 1)); }
  function darken (hex, amt){ const h=hexToHsl(hex); if(!h) return hex; return hslToHex(h.h, h.s, clamp(h.l-amt, 0, 1)); }
  function parseRGBA(str){
    const m = String(str||'').match(/^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)$/i);
    if(!m) return null; return { r:+m[1], g:+m[2], b:+m[3], a: m[4]===undefined ? 1 : +m[4] };
  }
  function hexFromRGB({r,g,b}){ return '#'+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,'0')).join(''); }
  function mixHex(a,b,ratio){
    const ar=parseInt(a.slice(1,3),16), ag=parseInt(a.slice(3,5),16), ab=parseInt(a.slice(5,7),16);
    const br=parseInt(b.slice(1,3),16), bg=parseInt(b.slice(3,5),16), bb=parseInt(b.slice(5,7),16);
    const t = clamp(ratio,0,1);
    return '#'+[
      Math.round(ar+(br-ar)*t),
      Math.round(ag+(bg-ag)*t),
      Math.round(ab+(bb-ab)*t)
    ].map(v=>v.toString(16).padStart(2,'0')).join('');
  }
  function luminance(hex){
    hex = parseColor(hex) || '#ffffff';
    const srgb = [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255).map(v=>v<=0.03928? v/12.92 : Math.pow((v+0.055)/1.055,2.4));
    return 0.2126*srgb[0] + 0.7152*srgb[1] + 0.0722*srgb[2];
  }

  function rgbaFromHex(hex, a){
    hex = parseColor(hex) || '#ffffff';
    const r = parseInt(hex.slice(1,3),16);
    const g = parseInt(hex.slice(3,5),16);
    const b = parseInt(hex.slice(5,7),16);
    return `rgba(${r},${g},${b},${clamp(a,0,1)})`;
  }

  // Extract a representative background color under a point (walk up if transparent)
  function colorAtPoint(x,y){
    const els = document.elementsFromPoint(Math.round(x), Math.round(y));
    for(const el of els){
      const cs = getComputedStyle(el);
      // gradient?
      const bgImg = cs.backgroundImage;
      if(bgImg && bgImg !== 'none'){
        // naive parse: take first color-like token
        const colors = bgImg.match(/#(?:[0-9a-f]{3}|[0-9a-f]{6})|rgba?\([^\)]+\)/ig) || [];
        for(const c of colors){ const hex = parseColor(c) || (parseRGBA(c)? hexFromRGB(parseRGBA(c)) : null); if(hex) return hex; }
      }
      const rgba = parseRGBA(cs.backgroundColor);
      if(rgba && rgba.a>0){ return hexFromRGB(rgba); }
    }
    const bodyRGBA = parseRGBA(getComputedStyle(document.body).backgroundColor) || {r:255,g:255,b:255};
    return hexFromRGB(bodyRGBA);
  }

  function sampleAreaColor(rect){
    // sample only center to avoid edge transitions during scroll
    const cx = rect.left + rect.width / 2;
    const cy = rect.top  + rect.height / 2;
    return colorAtPoint(cx, cy);
  }

  function applyBackplate(bgHex){
    const L = luminance(bgHex);
    const topHex = L>0.5 ? darken(bgHex, 0.03) : lighten(bgHex, 0.06);
    const botHex = L>0.5 ? darken(bgHex, 0.07) : lighten(bgHex, 0.10);
    const top = rgbaFromHex(topHex, 0.45);
    const bot = rgbaFromHex(botHex, 0.45);
    box.style.background = `linear-gradient(180deg, ${top}, ${bot})`;
    box.style.borderColor = L>0.5 ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.14)';
  }

  function updateAdaptiveTheme(){
    if(!autoThemeEnabled) return;
    const now = performance.now();
    if(now < _lockUntil && _paletteCache){ applyTheme(_paletteCache); return; }
    // Determine sampling target (box if open, else launcher)
    const target = box.style.display === 'flex' ? box : launcher;
    const rect = target.getBoundingClientRect();
    // add small hysteresis: compare to previous and skip tiny deltas
    const bg = sampleAreaColor(rect);
    if(updateAdaptiveTheme._prev && updateAdaptiveTheme._prev === bg) return;
    // Only update when change is significant (reduce color churn)
    if(updateAdaptiveTheme._prev){
      const a = hexToHsl(updateAdaptiveTheme._prev), b = hexToHsl(bg);
      if(a && b){
        const dh = Math.min(Math.abs(a.h-b.h), 1-Math.abs(a.h-b.h));
        const ds = Math.abs(a.s-b.s);
        const dl = Math.abs(a.l-b.l);
        if(dh < 0.05 && ds < 0.08 && dl < 0.08) return; // below threshold → skip
      }
    }
    updateAdaptiveTheme._prev = bg;
    // Keep primary; adapt semi‑transparent glass tints to background
    const primary = getComputedStyle(document.documentElement).getPropertyValue('--aiw-primary').trim() || '#2563eb';
    const base = parseColor(bg) || '#ffffff';
    const asstHex = mixHex(primary, base, 0.30);
    const asstGlass = rgbaFromHex(asstHex, 0.55);
    _paletteCache = {
      '--aiw-glass': 'rgba(255,255,255,0.55)',
      '--aiw-user-glass': 'rgba(255,255,255,0.65)',
      '--aiw-assistant-glass': asstGlass,
    };
    applyTheme(_paletteCache);
    _lockUntil = now + 240; // lock for ~240ms to avoid visible jumps
    applyBackplate(bg);
  }

  // trailing debounce: update only после остановки скролла
  let _idleTimer = null, _lastReq = 0, _lockUntil = 0, _paletteCache = null;
  function onAdaptiveTick(){
    clearTimeout(_idleTimer);
    _idleTimer = setTimeout(()=>{ try{ updateAdaptiveTheme(); }catch{} }, 220);
  }

  function guessSiteTheme(){
    const rootStyles = getComputedStyle(document.documentElement);
    const candVars = ['--primary','--color-primary','--brand','--brand-primary','--accent','--link-color'];
    let primary;
    for(const v of candVars){
      const val = rootStyles.getPropertyValue(v).trim();
      const hex = parseColor(val); if(hex){ primary = hex; break; }
    }
    if(!primary){
      const meta = document.querySelector('meta[name="theme-color"]');
      const hex = parseColor(meta?.getAttribute('content') || '');
      if(hex) primary = hex;
    }
    if(!primary){
      const a = document.querySelector('a');
      const col = a ? getComputedStyle(a).color : '';
      const hex = parseColor(col); if(hex) primary = hex;
    }
    if(!primary){ primary = '#2563eb'; }

    const bodyBG = parseColor(getComputedStyle(document.body).backgroundColor) || '#ffffff';
    return { primary, bodyBG };
  }

  function applyTheme(vars){
    const entries = Object.entries(vars);
    entries.forEach(([k,v])=>{ box.style.setProperty(k, v); launcher.style.setProperty(k, v); });
  }
  function resetTheme(){
    const keys = [
      '--aiw-primary','--aiw-primary-light','--aiw-primary-dark',
      '--aiw-user-bg','--aiw-assistant-bg','--aiw-glass','--aiw-user-glass','--aiw-assistant-glass'
    ];
    keys.forEach(k=>{ box.style.removeProperty(k); launcher.style.removeProperty(k); });
    // Restore original non-glass look
    box.style.background = 'linear-gradient(180deg, #ffffff, #f6f8fb)';
    box.style.borderColor = '#e5e7eb';
    box.style.backdropFilter = 'none';
  }
  function applyAutoTheme(){
    const { primary, bodyBG } = guessSiteTheme();
    const pLight = lighten(primary, 0.12);
    const pDark  = darken(primary, 0.12);
    const base = parseColor(bodyBG) || '#ffffff';
    const asstHex = mixHex(primary, base, 0.30);
    const asstGlass = rgbaFromHex(asstHex, 0.55);
    applyTheme({
      '--aiw-primary': primary,
      '--aiw-primary-light': pLight,
      '--aiw-primary-dark': pDark,
      '--aiw-glass': 'rgba(255,255,255,0.55)',
      '--aiw-user-glass': 'rgba(255,255,255,0.65)',
      '--aiw-assistant-glass': asstGlass,
    });
  }
  function applyThemeMode(){
    if(autoThemeEnabled) applyAutoTheme(); else resetTheme();
  }

  // 5. DEBUG UTILITIES ──────────────────────────────────────────────────────
  const debugLog = [];
  const MAX_LOG  = 200;

  function looksLikeToolPayload(txt){
    if(!txt) return false;
    try{
      const obj = JSON.parse(txt);
      return obj && typeof obj === 'object' && ('tool' in obj || 'output' in obj);
    }catch{ return false; }
  }

  // Add entry to debug log with filtering rules
  function addDebug(direction, payload){
    // Always log errors / meta / full responses
    if (direction.startsWith('!') || direction === '→ request' || direction === '← response' || direction === '← meta') {
      doLog(); return;
    }
    // Always log tool/function results
    if (direction === '← chunk' && looksLikeToolPayload(payload)) { doLog(); return; }
    // For plain chunks obey toggle
    if (direction === '← chunk') { if (showChunkDebug) doLog(); return; }
    doLog();

    function doLog(){
      const ts = new Date().toISOString().split('T')[1].split('Z')[0];
      debugLog.push(`[${ts}] ${direction}: ${payload}`);
      if(debugLog.length > MAX_LOG) debugLog.shift();
      if(debugPanel.classList.contains('open')){
        dbgOutput.textContent = debugLog.join('\n');
        dbgOutput.scrollTop   = dbgOutput.scrollHeight;
      }
    }
  }

  // 6. SETTINGS LOGIC (modal) ───────────────────────────────────────────────
  async function openSettings(){
    try{
      const res = await fetch(`${BACKEND}/config`, {method:'GET', mode:'cors'});
      if(!res.ok) throw new Error('Cannot fetch /config');
      const cfg = await res.json();

      const defaultModels = ['gpt-4o-mini','gpt-4o','gpt-4o-turbo','gpt-4-turbo','gpt-3.5-turbo-0125'];
      const models = (cfg.available_models || defaultModels)
        .concat(cfg.model || [])
        .filter((v,i,a)=>a.indexOf(v)===i);

      modelSelect.innerHTML = models
        .map(m=>`<option value="${m}" ${m===cfg.model?'selected':''}>${m}</option>`)
        .join('');
      instrTextarea.value           = cfg.instructions || '';
      showDebugInput.checked        = debugPanelOpen;
      showChunksInput.checked       = showChunkDebug;
      showToolResultsInput.checked  = showToolResults;
      autoThemeInput.checked        = autoThemeEnabled;
      modal.classList.add('open');
    }catch(err){
      console.error('[aiw] settings:', err);
      alert('Error loading settings');
    }
  }

  async function saveSettings(e){
    e.preventDefault();
    try{
      const res = await fetch(`${BACKEND}/config`, {
        method:'POST', mode:'cors',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          model:modelSelect.value.trim(),
          instructions:instrTextarea.value.trim(),
        }),
      });
      const j = await res.json();
      if(!res.ok || j.error) throw new Error(j.error || 'Save failed');
      setDebugPanel(showDebugInput.checked);
      showChunkDebug  = showChunksInput.checked;
      showToolResults = showToolResultsInput.checked;
      const prev = autoThemeEnabled;
      autoThemeEnabled = autoThemeInput.checked;
      localStorage.setItem('aiw-show-chunks-debug', showChunkDebug ? 'true' : 'false');
      localStorage.setItem('aiw-show-tool-results', showToolResults ? 'true' : 'false');
      localStorage.setItem('aiw-auto-theme', autoThemeEnabled ? 'true' : 'false');
      if(autoThemeEnabled){
        unbindAdaptive(); // ensure clean state before binding
        bindAdaptive();
        applyThemeMode();
        updateAdaptiveTheme();
      }else{
        unbindAdaptive();
        // clear caches used by adaptive mode to avoid stale values on next enable
        _paletteCache = null; _lockUntil = 0; updateAdaptiveTheme._prev = null;
        resetTheme();
      }
      modal.classList.remove('open');
    }catch(err){
      console.error('[aiw] save settings:', err);
      alert(err.message || 'Cannot save settings');
    }
  }

  // Show / hide debug panel and resize chat box
  function setDebugPanel(show){
    debugPanelOpen = show;
    if(show){
      debugPanel.classList.add('open');
      resizeBox();
      dbgOutput.textContent = debugLog.join('\n');
      dbgOutput.scrollTop   = dbgOutput.scrollHeight;
    }else{
      debugPanel.classList.remove('open');
      resizeBox();
    }
  }

  // 7. HELPER FUNCTIONS ────────────────────────────────────────────────────
  const sanitize = s => s.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi,'');

  function addDefaultMessages(){
    addMsg(
      `How can I assist you with telecommunications compliance?
       <div class="quick-replies">
         <button>STIR/SHAKEN</button><button>FCC 911 Rules</button><button>CPNI</button>
       </div>`,
      'assistant'
    );
    addMsg(
      `This chat provides general information only and is not legal advice.<br>
       Would you like to speak with Brita for legal advice about this?<br><br>
       <button class="getInTouch">Get in Touch</button>`,
      'legal-note'
    );
  }

  function addMsg(raw, cls){
    const div = document.createElement('div');
    div.className = `message ${cls}`;
    div.innerHTML = sanitize(raw);
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;

    chatHistory.push({role:cls, content:div.innerHTML});
    if(useHistory) localStorage.setItem('aiw-chat-history',JSON.stringify(chatHistory));
    return div;
  }

  function startDots(el){ el.classList.add('typing'); }
  function stopDots (el){ el.classList.remove('typing'); }

  function restoreHistory(){
    for(const m of chatHistory){
      const d = document.createElement('div');
      d.className = `message ${m.role}`;
      d.innerHTML = m.content;
      messages.appendChild(d);
    }
    messages.scrollTop = messages.scrollHeight;
  }

  // 8. UI EVENTS ────────────────────────────────────────────────────────────
  function resizeBox(){
    const baseWidth  = isExpanded ? EXPANDED_W : COMPACT_W;
    const totalWidth = baseWidth + (debugPanelOpen ? DEBUG_W : 0);
    Object.assign(box.style, {
      width : `${totalWidth}px`,
      height: isExpanded ? '80vh' : `${COMPACT_H}px`,
    });
  }
  function setCompact (){ isExpanded = false; resizeBox(); }
  function setExpanded(){ isExpanded = true;  resizeBox(); }

  toggleBtn        .addEventListener('click', () => { isExpanded ? setCompact() : setExpanded(); });
  settingsBtn      .addEventListener('click', openSettings);
  modalCancelBtn   .addEventListener('click', () => modal.classList.remove('open'));
  modalForm        .addEventListener('submit', saveSettings);

  launcher.addEventListener('click', () => {
    const open = box.style.display === 'flex';
    box.style.display = open ? 'none' : 'flex';
    if(!open){
      setCompact();
      userOpenedChat = true;
      textarea.focus();
      if(autoThemeEnabled){ applyThemeMode(); updateAdaptiveTheme(); }
    }
  });

  dbgClose.addEventListener('click', () => setDebugPanel(false));
  dbgClear.addEventListener('click', () => { debugLog.length = 0; dbgOutput.textContent = ''; });

  resetChatBtn.addEventListener('click', () => {
    abortActiveRequest();
    chatBusy = false;
    textarea.disabled = false;
    sendBtn .disabled = false;
    messages.innerHTML = '';
    chatHistory = [];
    if(useHistory) localStorage.removeItem('aiw-chat-history');
    prevID = null;
    if(useHistory) localStorage.removeItem('aiw-prevID');
    addDefaultMessages();
  });

  sendBtn .addEventListener('click', sendMessage);
  textarea.addEventListener('keydown', e=>{
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault();
      sendMessage();
    }
  });

  messages.addEventListener('click', e=>{
    const q = e.target.closest('.quick-replies button');
    if(q){
      textarea.value = 'I’d like to talk about: ' + q.textContent.trim();
      sendMessage();
    }
    if(e.target.classList.contains('getInTouch')){
      textarea.value = 'I’d like to get in touch.';
      sendMessage(); 
      return;
    }
  });

  historyToggle.addEventListener('change', e=>{
    useHistory = e.target.checked;
    localStorage.setItem('aiw-toggle-history', useHistory.toString());
    if(!useHistory){
      ['aiw-prevID','aiw-chat-history'].forEach(k=>localStorage.removeItem(k));
    }else{
      if(prevID) localStorage.setItem('aiw-prevID', prevID);
      if(chatHistory.length) localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
    }
  });

  // 9. CORE SEND LOGIC ──────────────────────────────────────────────────────
  async function sendMessage(){
    const msg = textarea.value.trim();
    if(!msg || chatBusy) return;

    abortActiveRequest();
    chatBusy   = true;
    textarea.value = '';

    addMsg(msg, 'user');
    const aiDiv = addMsg('', 'assistant');
    startDots(aiDiv);

    currentFetchController = new AbortController();
    const signal  = currentFetchController.signal;
    const headers = {'Content-Type':'application/json','X-Widget-Key':WIDGET_KEY};

    let assistantBuffer = '';   // text chunks
    let lastToolResult  = null; // last tool payload (JSON)

    addDebug('→ request', JSON.stringify({message:msg, prevID}));

    try{
      if(USE_STREAM){
        const res = await fetch(`${BACKEND}/chat/stream`, {
          method:'POST', mode:'cors', headers,
          body:JSON.stringify({message:msg, previous_response_id:prevID}),
          signal,
        });
        if(!res.ok || !res.body) throw new Error('Network error');
        currentReader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';

        while(true){
          const {value, done} = await currentReader.read();
          if(done) break;
          buf += dec.decode(value,{stream:true});
          const evts = buf.split('\n\n'); buf = evts.pop();

          for(const ev of evts){
            if(ev.startsWith('event: done')){ currentReader.cancel(); break; }

            if(ev.startsWith('event: meta')){
              const m = JSON.parse(ev.split('\n')[1].slice(6));
              addDebug('← meta', JSON.stringify(m));
              if(m.prev_id){
                prevID = m.prev_id;
                if(useHistory) localStorage.setItem('aiw-prevID', prevID);
              }
              continue;
            }
            const chunk = ev.split('\n')
              .filter(l=>l.startsWith('data:'))
              .map(l=>l.slice(6))
              .join('\n');
            if(!chunk) continue;
            addDebug('← chunk', chunk);

            // Tool payload?
            if(looksLikeToolPayload(chunk)){
              lastToolResult = chunk;
              continue;
            }

            // Normal text chunk → render
            if(aiDiv.classList.contains('typing')){
              stopDots(aiDiv);
              aiDiv.innerHTML = '';
            }
            assistantBuffer += chunk;
            aiDiv.innerHTML  = sanitize(assistantBuffer);
            messages.scrollTop = messages.scrollHeight;

            const i = chatHistory.length - 1;
            if(i >= 0 && chatHistory[i].role === 'assistant'){
              chatHistory[i].content = aiDiv.innerHTML;
              if(useHistory) localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
            }
          }
        }

        // Stream ended – if no plain text but tool output exists, show it
        if(!assistantBuffer && lastToolResult && showToolResults){
          try{
            const obj = JSON.parse(lastToolResult);
            aiDiv.innerHTML = `<pre>${sanitize(JSON.stringify(obj, null, 2))}</pre>`;
          }catch{
            aiDiv.innerHTML = sanitize(lastToolResult);
          }
        }

      }else{ // Non-stream fallback
        const res = await fetch(`${BACKEND}/chat`, {
          method:'POST', mode:'cors', headers,
          body:JSON.stringify({message:msg, previous_response_id:prevID}),
          signal,
        });
        if(!res.ok) throw new Error('Network error');
        const j = await res.json();
        addDebug('← response', JSON.stringify(j));

        stopDots(aiDiv);
        const reply = j.response || j.error || 'No response';

        if(looksLikeToolPayload(reply)){
          if(showToolResults){
            try{
              const obj = JSON.parse(reply);
              aiDiv.innerHTML = `<pre>${sanitize(JSON.stringify(obj, null, 2))}</pre>`;
            }catch{
              aiDiv.innerHTML = sanitize(reply);
            }
          }
        }else{
          aiDiv.innerHTML = sanitize(reply);
        }

        const i = chatHistory.length - 1;
        if(i >= 0 && chatHistory[i].role === 'assistant'){
          chatHistory[i].content = aiDiv.innerHTML;
          if(useHistory) localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
        }
        if(j.prev_id){
          prevID = j.prev_id;
          if(useHistory) localStorage.setItem('aiw-prevID', prevID);
        }
      }

    }catch(err){
      if(err.name === 'AbortError'){ aiDiv.remove(); return; }
      console.error('[aiw]', err);
      addDebug('! error', err.message || String(err));
      stopDots(aiDiv);
      aiDiv.textContent = 'Error contacting server';

      const i = chatHistory.length - 1;
      if(i >= 0 && chatHistory[i].role === 'assistant'){
        chatHistory[i].content = 'Error contacting server';
        if(useHistory) localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
      }

    }finally{
      chatBusy = false;
      aiDiv.classList.remove('typing');
      currentFetchController = currentReader = null;
      textarea.focus();
    }
  }

  function abortActiveRequest(){
    if(currentFetchController){ currentFetchController.abort(); currentFetchController = null; }
    if(currentReader){ currentReader.cancel(); currentReader = null; }
  }

  // 10. INITIAL SETUP ───────────────────────────────────────────────────────
  if(chatHistory.length === 0) addDefaultMessages(); else restoreHistory();

  // Auto-open after delay on first visit
  setTimeout(()=>{
    if(box.style.display!=='flex' && !userOpenedChat){
      box.style.display = 'flex';
      setCompact();
      box.classList.add('aiw-auto-open');
    }
  }, 10_000);
  box.addEventListener('animationend', e=>{
    if(e.animationName === 'aiwBoxPop') box.classList.remove('aiw-auto-open');
  });

  // Ensure debug panel starts closed
  setDebugPanel(false);

  // Apply theme on init if enabled
  try{ applyThemeMode(); }catch{}
  // Adaptive updates only when enabled to avoid any baseline color flicker
  function bindAdaptive(){
    window.addEventListener('scroll',  onAdaptiveTick, {passive:true});
    window.addEventListener('resize',  onAdaptiveTick);
    bindAdaptive._mo = new MutationObserver((muts)=>{
      const hasLayoutChange = muts.some(m=> m.type==='childList' || (m.type==='attributes' && (m.attributeName==='style' || m.attributeName==='class')));
      if(hasLayoutChange) onAdaptiveTick();
    });
    bindAdaptive._mo.observe(document.body, {attributes:true, attributeFilter:['style','class'], childList:true, subtree:true});
  }
  function unbindAdaptive(){
    window.removeEventListener('scroll', onAdaptiveTick);
    window.removeEventListener('resize', onAdaptiveTick);
    if(bindAdaptive._mo){ try{ bindAdaptive._mo.disconnect(); }catch{} bindAdaptive._mo = null; }
  }
  if(autoThemeEnabled){ bindAdaptive(); }

})();
