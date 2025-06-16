/* ============================================================================
   AI Compliance Chat Widget – embeddable with runtime settings v2.9
============================================================================ */
(() => {
  'use strict';

  // 1. CONFIG
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
  let isExpanded = false;
  let chatBusy   = false;
  let debugPanelOpen = false;

  // Settings toggles (persisted in localStorage)
  let showChunkDebug   = (localStorage.getItem('aiw-show-chunks-debug') ?? 'true') === 'true';
  let showToolResults  = (localStorage.getItem('aiw-show-tool-results') ?? 'true') === 'true';

  // Request control
  let currentFetchController = null;
  let currentReader          = null;

  // History settings
  const toggleStateStr = localStorage.getItem('aiw-toggle-history');
  let useHistory = (toggleStateStr === null) ? true : (toggleStateStr === 'true');
  let prevID = useHistory ? localStorage.getItem('aiw-prevID') || null : null;

  // Stored chat
  let chatHistory = [];
  if (useHistory) {
    try { chatHistory = JSON.parse(localStorage.getItem('aiw-chat-history') || '[]'); }
    catch { chatHistory = []; }
  }

  // Track manual open
  let userOpenedChat = false;

  // 2. STYLES
  const css = /* css */`
:root{
  --aiw-primary:#2563eb;
  --aiw-primary-light:#3b82f6;
  --aiw-primary-dark:#1e40af;
  --aiw-text-on-primary:#ffffff;
  --aiw-user-bg:#f3f4f6;
  --aiw-user-text:#111827;
  --aiw-assistant-bg:#dcfce7;
  --aiw-assistant-text:#065f46;
  --aiw-radius:12px;
  --aiw-shadow:0 4px 12px rgba(0,0,0,.12);
  --aiw-font:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{
    --aiw-primary:#3b82f6;
    --aiw-primary-light:#60a5fa;
    --aiw-primary-dark:#1d4ed8;
    --aiw-user-bg:#1f2937;
    --aiw-user-text:#e5e7eb;
    --aiw-assistant-bg:#064e3b;
    --aiw-assistant-text:#d1fae5;
  }
}
.aiw-launcher{
  position:fixed; bottom:24px; right:24px;
  display:flex; align-items:center; justify-content:center;
  width:56px; height:56px;
  background:var(--aiw-primary); color:var(--aiw-text-on-primary);
  border:none; border-radius:50%;
  font-size:24px; cursor:pointer;
  box-shadow:var(--aiw-shadow);
  animation:aiwPulse 2.8s ease-in-out infinite;
  z-index:100000;
}
.aiw-launcher:focus-visible{outline:3px solid var(--aiw-primary-light);}
.aiw-box{
  position:fixed; bottom:90px; right:24px;
  display:none; flex-direction:row;
  width:${COMPACT_W}px; height:${COMPACT_H}px;
  background:#fff; border:1px solid #e5e7eb;
  border-radius:var(--aiw-radius); box-shadow:var(--aiw-shadow);
  font-family:var(--aiw-font); overflow:hidden;
  transition:width .25s, height .25s;
  scroll-behavior:smooth;
}
@media(max-width:480px){
  .aiw-box{right:12px; width:calc(100vw - 24px);}
}
.aiw-chat-main {
  display: flex; flex-direction: column; flex: 1 1 0%;
  min-width:0; min-height:0;
}
.aiw-chat-main header{
  display:flex; align-items:center; gap:10px;
  background:var(--aiw-primary); color:var(--aiw-text-on-primary);
  padding:.75rem 1rem; font-size:15px; font-weight:600;
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
.aiw-chat-main .message{
  max-width:90%; padding:.75rem; border-radius:10px;
  line-height:1.45; font-size:14px;
}
.aiw-chat-main .message.user{background:var(--aiw-user-bg); color:var(--aiw-user-text); margin-left:auto;}
.aiw-chat-main .message.assistant{background:var(--aiw-assistant-bg); color:var(--aiw-assistant-text); margin-right:auto;}
.aiw-chat-main .message.legal-note{background:var(--aiw-user-bg); color:#6b7280; margin-right:auto; font-size:13px;}
.aiw-chat-main .message.assistant.typing{opacity:0.8;}
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
.aiw-box { flex-direction: row !important; }
.aiw-debug-panel { order: 0; }
.aiw-chat-main { order: 1; flex: 1 1 0%; display: flex; flex-direction: column; min-width:0; min-height:0;}
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

  document.head.appendChild(Object.assign(document.createElement('style'), {textContent: css}));

  // 3. DOM CREATION
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

  // 4. REFERENCES
  const toggleBtn      = chatMain.querySelector('.toggle');
  const settingsBtn    = chatMain.querySelector('.settings');
  const historyToggle  = chatMain.querySelector('.toggle-history input');
  const resetChatBtn   = chatMain.querySelector('.reset-chat');
  const messages       = chatMain.querySelector('.messages');
  const textarea       = chatMain.querySelector('textarea');
  const sendBtn        = chatMain.querySelector('.send');

  // Debug refs
  const dbgClose  = debugPanel.querySelector('.close');
  const dbgClear  = debugPanel.querySelector('.clear');
  const dbgOutput = debugPanel.querySelector('pre');

  // Modal refs
  const modalForm         = modal.querySelector('form');
  const modelSelect       = modal.querySelector('select[name=model]');
  const instrTextarea     = modal.querySelector('textarea[name=instructions]');
  const showDebugInput    = modal.querySelector('input[name=showdebug]');
  const showChunksInput   = modal.querySelector('input[name=showchunks]');
  const showToolResultsInput = modal.querySelector('input[name=showtoolresults]');
  const modalCancelBtn    = modal.querySelector('.cancel');

  // 5. DEBUG UTILITIES
  const debugLog = [];
  const MAX_LOG  = 200;

  // Add entry to debug log, always show tool/function results, respect chunk toggle for plain chunks
  function addDebug(direction, payload){
    // Always log errors and meta
    if (direction.startsWith('!') || direction === '→ request' || direction === '← response' || direction === '← meta') {
      doLog();
      return;
    }
    // Always log tool/function call results (payload is JSON with .tool or .output)
    if (direction === '← chunk' && looksLikeToolPayload(payload)) {
      doLog();
      return;
    }
    // For plain chunks, respect the chunk debug toggle
    if (direction === '← chunk') {
      if (showChunkDebug) doLog();
      return;
    }
    // Default: log everything else
    doLog();

    function doLog() {
      const ts = new Date().toISOString().split('T')[1].split('Z')[0];
      debugLog.push(`[${ts}] ${direction}: ${payload}`);
      if(debugLog.length>MAX_LOG) debugLog.shift();
      if(debugPanel.classList.contains('open')){
        dbgOutput.textContent = debugLog.join('\n');
        dbgOutput.scrollTop   = dbgOutput.scrollHeight;
      }
    }
  }

  // Check if text is a tool/function result (JSON with .tool/.output)
  function looksLikeToolPayload(txt){
    if(!txt) return false;
    try{
      const obj = JSON.parse(txt);
      return obj && typeof obj==='object' && ('tool' in obj || 'output' in obj);
    }catch{ return false; }
  }

  // 6. SETTINGS LOGIC
  async function openSettings(){
    try{
      const res=await fetch(`${BACKEND}/config`,{method:'GET',mode:'cors'});
      if(!res.ok) throw new Error('Cannot fetch /config');
      const cfg=await res.json();

      const defaultModels=['gpt-4o-mini','gpt-4o','gpt-4o-turbo','gpt-4-turbo','gpt-3.5-turbo-0125'];
      const models=(cfg.available_models||defaultModels).concat(cfg.model||[])
                   .filter((v,i,a)=>a.indexOf(v)===i);

      modelSelect.innerHTML=models.map(m=>`<option value="${m}" ${m===cfg.model?'selected':''}>${m}</option>`).join('');
      instrTextarea.value = cfg.instructions || '';
      showDebugInput.checked    = debugPanelOpen;
      showChunksInput.checked   = showChunkDebug;
      showToolResultsInput.checked = showToolResults;
      modal.classList.add('open');
    }catch(err){
      console.error('[aiw] settings:',err);
      alert('Error loading settings');
    }
  }
  async function saveSettings(e){
    e.preventDefault();
    try{
      // Save backend model/settings
      const res=await fetch(`${BACKEND}/config`,{
        method:'POST',mode:'cors',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({model:modelSelect.value.trim(),instructions:instrTextarea.value.trim()})
      });
      const j=await res.json();
      if(!res.ok||j.error) throw new Error(j.error||'Save failed');
      setDebugPanel(showDebugInput.checked);
      showChunkDebug  = showChunksInput.checked;
      showToolResults = showToolResultsInput.checked;
      localStorage.setItem('aiw-show-chunks-debug', showChunkDebug ? 'true' : 'false');
      localStorage.setItem('aiw-show-tool-results', showToolResults ? 'true' : 'false');
      modal.classList.remove('open');
    }catch(err){
      console.error('[aiw] save settings:',err);
      alert(err.message||'Cannot save settings');
    }
  }

  // Show/hide debug panel and resize chat
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

  // 7. HELPER FUNCTIONS
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
    const div=document.createElement('div');
    div.className=`message ${cls}`;
    div.innerHTML=sanitize(raw);
    messages.appendChild(div);
    messages.scrollTop=messages.scrollHeight;

    chatHistory.push({role:cls,content:div.innerHTML});
    if(useHistory) localStorage.setItem('aiw-chat-history',JSON.stringify(chatHistory));
    return div;
  }
  function startDots(el){el.classList.add('typing');}
  function stopDots (el){el.classList.remove('typing');}

  function restoreHistory(){
    for(const m of chatHistory){
      const d=document.createElement('div');
      d.className=`message ${m.role}`;
      d.innerHTML=m.content;
      messages.appendChild(d);
    }
    messages.scrollTop=messages.scrollHeight;
  }

  // 8. UI EVENTS
  function resizeBox(){
    let baseWidth = isExpanded ? EXPANDED_W : COMPACT_W;
    let totalWidth = baseWidth + (debugPanelOpen ? DEBUG_W : 0);
    Object.assign(box.style,{width:`${totalWidth}px`,height:isExpanded?`80vh`:`${COMPACT_H}px`});
  }
  function setCompact(){isExpanded=false;resizeBox();}
  function setExpanded(){isExpanded=true;resizeBox();}

  toggleBtn   .addEventListener('click',()=>{isExpanded?setCompact():setExpanded();});
  settingsBtn .addEventListener('click',openSettings);
  modalCancelBtn.addEventListener('click',()=>modal.classList.remove('open'));
  modalForm  .addEventListener('submit',saveSettings);

  launcher.addEventListener('click',()=>{
    const open=box.style.display==='flex';
    box.style.display=open?'none':'flex';
    if(!open){setCompact();userOpenedChat=true;textarea.focus();}
  });

  dbgClose.addEventListener('click',()=>setDebugPanel(false));
  dbgClear.addEventListener('click',()=>{debugLog.length=0;dbgOutput.textContent='';});

  resetChatBtn.addEventListener('click',()=>{
    abortActiveRequest(); chatBusy=false; textarea.disabled=false; sendBtn.disabled=false;
    messages.innerHTML=''; chatHistory=[]; if(useHistory) localStorage.removeItem('aiw-chat-history');
    prevID=null; if(useHistory) localStorage.removeItem('aiw-prevID');
    addDefaultMessages();
  });

  sendBtn.addEventListener('click',sendMessage);
  textarea.addEventListener('keydown',e=>{
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();}
  });

  messages.addEventListener('click',e=>{
    const q=e.target.closest('.quick-replies button');
    if(q){textarea.value="I’d like to talk about: "+q.textContent.trim();sendMessage();}
    if(e.target.classList.contains('getInTouch')) window.location.href='mailto:contact@telecom-ai.com';
  });

  historyToggle.addEventListener('change',e=>{
    useHistory=e.target.checked; localStorage.setItem('aiw-toggle-history',useHistory.toString());
    if(!useHistory){
      ['aiw-prevID','aiw-chat-history'].forEach(k=>localStorage.removeItem(k));
    }else{
      if(prevID) localStorage.setItem('aiw-prevID',prevID);
      if(chatHistory.length) localStorage.setItem('aiw-chat-history',JSON.stringify(chatHistory));
    }
  });

  // 9. CORE SEND LOGIC
  async function sendMessage(){
    const msg=textarea.value.trim();
    if(!msg||chatBusy) return;

    abortActiveRequest(); chatBusy=true; textarea.value='';
    addMsg(msg,'user');
    const aiDiv=addMsg('','assistant'); startDots(aiDiv);

    currentFetchController=new AbortController();
    const signal=currentFetchController.signal;
    const headers={'Content-Type':'application/json','X-Widget-Key':WIDGET_KEY};

    // Buffer for assistant response (only plain text chunks for chat window)
    let assistantBuffer = '';
    // Buffer for last tool result, if present
    let lastToolResult = null;

    addDebug('→ request',JSON.stringify({message:msg,prevID}));

    try{
      if(USE_STREAM){
        const res=await fetch(`${BACKEND}/chat/stream`,{
          method:'POST',mode:'cors',headers,
          body:JSON.stringify({message:msg,previous_response_id:prevID}),signal
        });
        if(!res.ok||!res.body) throw new Error('Network error');
        currentReader=res.body.getReader();
        const dec=new TextDecoder(); let buf='';

        while(true){
          const {value,done}=await currentReader.read();
          if(done) break;
          buf+=dec.decode(value,{stream:true});
          const evts=buf.split('\n\n'); buf=evts.pop();
          for(const ev of evts){
            if(ev.startsWith('event: done')){currentReader.cancel();break;}
            if(ev.startsWith('event: meta')){
              const m=JSON.parse(ev.split('\n')[1].slice(6));
              addDebug('← meta',JSON.stringify(m));
              if(m.prev_id){prevID=m.prev_id;if(useHistory) localStorage.setItem('aiw-prevID',prevID);}
              continue;
            }
            const chunk=ev.split('\n').filter(l=>l.startsWith('data:')).map(l=>l.slice(6)).join('\n');
            if(!chunk) continue;
            addDebug('← chunk',chunk);
            if(looksLikeToolPayload(chunk)){
              lastToolResult = chunk;
              continue;
            }
            // Show only normal text chunks in assistant's chat message
            if(aiDiv.classList.contains('typing')){stopDots(aiDiv);aiDiv.innerHTML='';}
            assistantBuffer += chunk;
            aiDiv.innerHTML = sanitize(assistantBuffer);
            messages.scrollTop=messages.scrollHeight;
            const i=chatHistory.length-1;
            if(i>=0&&chatHistory[i].role==='assistant'){
              chatHistory[i].content=aiDiv.innerHTML;
              if(useHistory) localStorage.setItem('aiw-chat-history',JSON.stringify(chatHistory));
            }
          }
        }
        // After streaming, if no normal message, but tool result is present, render it if allowed
        if(!assistantBuffer && lastToolResult && showToolResults){
          try {
            const obj = JSON.parse(lastToolResult);
            aiDiv.innerHTML = `<pre>${sanitize(JSON.stringify(obj, null, 2))}</pre>`;
          } catch {
            aiDiv.innerHTML = sanitize(lastToolResult);
          }
        }
      }else{
        const res=await fetch(`${BACKEND}/chat`,{
          method:'POST',mode:'cors',headers,
          body:JSON.stringify({message:msg,previous_response_id:prevID}),signal
        });
        if(!res.ok) throw new Error('Network error');
        const j=await res.json(); addDebug('← response',JSON.stringify(j));
        stopDots(aiDiv);
        const reply=j.response||j.error||'No response';
        if(looksLikeToolPayload(reply)){
          if(showToolResults){
            try {
              const obj = JSON.parse(reply);
              aiDiv.innerHTML = `<pre>${sanitize(JSON.stringify(obj, null, 2))}</pre>`;
            } catch {
              aiDiv.innerHTML = sanitize(reply);
            }
          }
        } else {
          aiDiv.innerHTML = sanitize(reply);
        }
        const i=chatHistory.length-1;
        if(i>=0&&chatHistory[i].role==='assistant'){
          chatHistory[i].content=aiDiv.innerHTML;
          if(useHistory) localStorage.setItem('aiw-chat-history',JSON.stringify(chatHistory));
        }
        if(j.prev_id){prevID=j.prev_id;if(useHistory) localStorage.setItem('aiw-prevID',prevID);}
      }

    }catch(err){
      if(err.name==='AbortError'){aiDiv.remove();return;}
      console.error('[aiw]',err); addDebug('! error',err.message||String(err));
      stopDots(aiDiv); aiDiv.textContent='Error contacting server';
      const i=chatHistory.length-1;
      if(i>=0&&chatHistory[i].role==='assistant'){
        chatHistory[i].content='Error contacting server';
        if(useHistory) localStorage.setItem('aiw-chat-history',JSON.stringify(chatHistory));
      }

    }finally{
      chatBusy=false; aiDiv.classList.remove('typing');
      currentFetchController=currentReader=null; textarea.focus();
    }
  }

  function abortActiveRequest(){
    if(currentFetchController){currentFetchController.abort();currentFetchController=null;}
    if(currentReader){currentReader.cancel();currentReader=null;}
  }

  // 10. INITIAL SETUP
  if(chatHistory.length===0) addDefaultMessages(); else restoreHistory();

  setTimeout(()=>{
    if(box.style.display!=='flex'&&!userOpenedChat){
      box.style.display='flex'; setCompact(); box.classList.add('aiw-auto-open');
    }
  },10_000);
  box.addEventListener('animationend',e=>{
    if(e.animationName==='aiwBoxPop') box.classList.remove('aiw-auto-open');
  });

  setDebugPanel(false);

})();
