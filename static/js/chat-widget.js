/* ============================================================================
   AI Compliance Chat Widget – embeddable with revised toggle v2.2
   ----------------------------------------------------------------------------
   • Toggle switch (History ON/OFF) to store prevID across reloads.
   • Stores the entire chat conversation when History = ON.
   • "Refresh Chat" button to reset chat + re-add default welcome messages.
   • Forces chatBusy=false on refresh, ensuring user can send new messages.
   • Cancels any ongoing fetch/SSE on refresh so old answers won't appear.
   • Auto-opens chat after 10s if user hasn't opened it themselves yet.
   • With a gentle "pop" animation on auto-open to attract attention.
   • FIX: removed all duplicate declarations to avoid "Cannot redeclare" errors.
============================================================================ */
(() => {
  'use strict';

  /* ───────────────────────── 1. CONFIG ──────────────────────────── */
  const el         = document.currentScript;
  const BACKEND    = (el.dataset.backend || '').replace(/\/$/, '');
  const WIDGET_KEY = el.dataset.key || '';
  const USE_STREAM = el.dataset.stream !== 'false'; // default = true

  if (!BACKEND || !WIDGET_KEY) {
    console.error('[aiw] backend URL and data-key are required');
    return;
  }

  // Sizes for compact/expanded view
  const COMPACT_W = 340, EXPANDED_W = 600, COMPACT_H = 420;

  // Track if chat is expanded or not
  let isExpanded = false;

  // Track if chat is "busy" (request in progress)
  let chatBusy = false;

  // For canceling ongoing requests
  let currentFetchController = null;
  let currentReader = null;

  // Toggle for storing local history
  const toggleStateStr = localStorage.getItem('aiw-toggle-history');
  let useHistory = (toggleStateStr === null) ? true : (toggleStateStr === 'true');

  // If useHistory is ON, load prevID from localStorage, else null
  let prevID = null;
  if (useHistory) {
    prevID = localStorage.getItem('aiw-prevID') || null;
  }

  // If useHistory is ON, load chatHistory from localStorage
  let chatHistory = [];
  if (useHistory) {
    const storedStr = localStorage.getItem('aiw-chat-history');
    if (storedStr) {
      try {
        chatHistory = JSON.parse(storedStr) || [];
      } catch (err) {
        console.error('[aiw] Error parsing aiw-chat-history:', err);
        chatHistory = [];
      }
    }
  }

  // Track if user manually opened the chat
  let userOpenedChat = false;

  /* ───────────────────────── 2. STYLES ──────────────────────────── */
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

/* Launcher button */
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

/* Widget container */
.aiw-box{
  position:fixed; bottom:90px; right:24px;
  display:none; flex-direction:column;
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

/* Gentle "pop" animation for auto-open */
.aiw-box.aiw-auto-open {
  animation: aiwBoxPop 0.8s ease-out;
}
@keyframes aiwBoxPop {
  0%   {transform: scale(0.85) translateY(10%); opacity:0;}
  60%  {transform: scale(1.02) translateY(-2%); opacity:1;}
  100% {transform: scale(1) translateY(0);}
}

/* Header */
.aiw-box header{
  display:flex; align-items:center; gap:10px;
  background:var(--aiw-primary); color:var(--aiw-text-on-primary);
  padding:.75rem 1rem; font-size:15px; font-weight:600;
}
.aiw-box header img{
  width:28px; height:28px; border-radius:50%; background:#fff; padding:2px;
}
.aiw-box header .toggle{
  background:none; border:none; color:inherit;
  font-size:1.1rem; cursor:pointer; line-height:1;
  transition:transform .15s;
}
.aiw-box header .toggle:hover{transform:scale(1.15);}

/* Refresh chat button */
.aiw-box header .reset-chat{
  background:none; border:none; color:inherit;
  font-size:1rem; cursor:pointer; line-height:1;
  transition:transform .15s; margin-left:-2px;
}
.aiw-box header .reset-chat:hover{transform:scale(1.15);}

/* History toggle switch (slider) */
.history-wrapper{
  display:flex; align-items:center;
  margin-left:auto; margin-right:4px;
}
.history-label{
  margin-right:6px; font-size:13px; opacity:0.9;
}
.toggle-history{
  position:relative; display:inline-block; width:36px; height:18px;
  margin-right:8px;
}
.toggle-history input{
  opacity:0; width:0; height:0;
}
.slider{
  position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0;
  background-color:rgba(255,255,255,0.6); border-radius:34px;
  transition:.2s;
}
.slider:before{
  position:absolute; content:"";
  height:14px; width:14px; left:2px; bottom:2px;
  background-color:var(--aiw-primary-dark);
  border-radius:50%; transition:.2s;
}
.toggle-history input:checked + .slider{
  background-color:rgba(255,255,255,0.85);
}
.toggle-history input:checked + .slider:before{
  transform:translateX(18px);
}

/* Message area */
.aiw-box .messages{
  flex:1; min-height:0; padding:1rem;
  display:flex; flex-direction:column; gap:.75rem; overflow-y:auto;
}

/* Footer */
.aiw-box footer{
  display:flex; gap:.5rem; padding:.75rem 1rem;
  border-top:1px solid #f3f4f6;
}
.aiw-box textarea{
  flex:1; font:inherit; font-size:14px;
  padding:.5rem; border:1px solid #d1d5db; border-radius:8px; resize:vertical;
}
.aiw-box textarea:focus-visible{outline:2px solid var(--aiw-primary-light);}
.aiw-box button.send{
  background:var(--aiw-primary); color:var(--aiw-text-on-primary);
  border:none; padding:.5rem 1rem; border-radius:8px; cursor:pointer;
  transition:background .2s;
}
.aiw-box button.send:hover{background:var(--aiw-primary-dark);}

/* Messages */
.aiw-box .message{
  max-width:90%; padding:.75rem; border-radius:10px;
  line-height:1.45; font-size:14px;
}
.aiw-box .message.user{
  background:var(--aiw-user-bg); color:var(--aiw-user-text); margin-left:auto;
}
.aiw-box .message.assistant{
  background:var(--aiw-assistant-bg); color:var(--aiw-assistant-text); margin-right:auto;
}
.aiw-box .message.legal-note{
  background:var(--aiw-user-bg); color:#6b7280; margin-right:auto; font-size:13px;
}

/* Typing indicator */
.aiw-box .message.assistant.typing{opacity:0.8;}
.aiw-box .typing::after{
  content:"."; animation:aiwDots 1.2s steps(3,end) infinite;
}
@keyframes aiwDots{
  0%{content:"."} 33%{content:".."} 66%{content:"..."}
}

/* Quick replies */
.aiw-box .quick-replies{
  display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.5rem;
}
.aiw-box .quick-replies button{
  background:#eef2ff; color:#1e3a8a; border:none; border-radius:6px;
  padding:.4rem .75rem; font-size:0.85rem; cursor:pointer;
  transition:background .2s;
}
.aiw-box .quick-replies button:hover{background:#e0e7ff;}

/* CTA button */
.aiw-box .getInTouch{
  background:var(--aiw-primary); color:var(--aiw-text-on-primary);
  border:none; padding:.5rem 1rem; border-radius:8px;
  font-size:0.85rem; cursor:pointer; transition:background .2s;
}
.aiw-box .getInTouch:hover{background:var(--aiw-primary-dark);}

/* Animations */
@keyframes aiwPulse{
  0%{transform:rotate(0) scale(1); box-shadow:0 0 0 rgba(37,99,235,0.4);}
  50%{transform:rotate(-5deg) scale(1.08); box-shadow:0 0 12px rgba(37,99,235,0.6);}
  100%{transform:rotate(0) scale(1); box-shadow:0 0 0 rgba(37,99,235,0.4);}
}
`;
  // Inject CSS into <head>
  document.head.appendChild(Object.assign(document.createElement('style'), {
    textContent: css
  }));

  /* ───────────────────────── 3. DOM CREATION ───────────────────── */

  // The floating launcher button
  const launcher = Object.assign(document.createElement('button'), {
    className:'aiw-launcher',
    title:'Chat',
    'aria-label':'Open chat widget',
    textContent:'💬'
  });
  document.body.appendChild(launcher);

  // Main chat box container
  const box = document.createElement('div');
  box.className = 'aiw-box';
  box.innerHTML = `
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
  <button title="Expand" class="toggle">⛶</button>
</header>
<div class="messages"></div>
<footer>
  <textarea rows="2" placeholder="Type your message…"></textarea>
  <button class="send">➤</button>
</footer>
`;
  document.body.appendChild(box);

  // Some references
  const toggleBtn     = box.querySelector('.toggle');
  const historyToggle = box.querySelector('.toggle-history input');
  const resetChatBtn  = box.querySelector('.reset-chat');
  const messages      = box.querySelector('.messages');
  const textarea      = box.querySelector('textarea');
  const sendBtn       = box.querySelector('.send');

  /* ───────────────────────── 4. HELPER FUNCTIONS ───────────────── */

  // Basic sanitize to remove <script> tags
  function sanitize(s){
    return s.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi,'');
  }

  // Add default welcome messages if chat is empty
  function addDefaultMessages(){
    addMsg(
      `How can I assist you with telecommunications compliance?
       <div class="quick-replies">
         <button>STIR/SHAKEN</button>
         <button>FCC 911 Rules</button>
         <button>CPNI</button>
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

  // Add a message to the DOM and the chatHistory array
  function addMsg(rawText, cls){
    const safeContent = sanitize(rawText);
    const div = document.createElement('div');
    div.className = `message ${cls}`;
    div.innerHTML = safeContent;

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;

    chatHistory.push({role:cls, content: safeContent});
    if(useHistory){
      localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
    }
    return div;
  }

  // Start/stop "typing..." dots
  function startDots(el){ el.classList.add('typing'); }
  function stopDots(el){ el.classList.remove('typing'); }

  // Restore chat from localStorage if any
  function restoreHistory(){
    for(const msgObj of chatHistory){
      const div = document.createElement('div');
      div.className = `message ${msgObj.role}`;
      div.innerHTML = msgObj.content;
      messages.appendChild(div);
    }
    messages.scrollTop = messages.scrollHeight;
  }

  /* ───────────────────────── 5. UI EVENTS ─────────────────────── */

  // Switch between compact/expanded chat
  function setCompact(){
    Object.assign(box.style,{width:`${COMPACT_W}px`,height:`${COMPACT_H}px`});
    isExpanded=false;
  }
  function setExpanded(){
    Object.assign(box.style,{width:`${EXPANDED_W}px`,height:`80vh`});
    isExpanded=true;
  }

  toggleBtn.addEventListener('click', () => {
    if(isExpanded){ setCompact(); } else { setExpanded(); }
  });

  launcher.addEventListener('click', () => {
    const isOpen = (box.style.display==='flex');
    box.style.display = isOpen ? 'none' : 'flex';
    if(!isOpen){
      setCompact();
      userOpenedChat=true;
      textarea.focus();
    }
  });

  // "Refresh Chat" => resets everything
  resetChatBtn.addEventListener('click', () => {
    abortActiveRequest();
    chatBusy=false;
    textarea.disabled=false;
    sendBtn.disabled=false;

    messages.innerHTML='';
    chatHistory=[];
    if(useHistory){ localStorage.removeItem('aiw-chat-history'); }

    prevID=null;
    if(useHistory){ localStorage.removeItem('aiw-prevID'); }

    addDefaultMessages();
  });

  // Send message events
  sendBtn.addEventListener('click',sendMessage);
  textarea.addEventListener('keydown', e => {
    if(e.key==='Enter' && !e.shiftKey){
      e.preventDefault();
      sendMessage();
    }
  });

  // Quick replies + CTA
  messages.addEventListener('click', e => {
    // If user clicked on one of the quick-replies buttons
    const clickedBtn = e.target.closest('.quick-replies button');
    if(clickedBtn){
      textarea.value = "I’d like to talk about: " + clickedBtn.textContent.trim();
      sendMessage();
    }
    if(e.target.classList.contains('getInTouch')){
      window.location.href='mailto:contact@telecom-ai.com';
    }
  });

  // Toggle storing local history
  historyToggle.addEventListener('change', e => {
    useHistory = e.target.checked;
    localStorage.setItem('aiw-toggle-history',useHistory.toString());

    if(!useHistory){
      localStorage.removeItem('aiw-prevID');
      localStorage.removeItem('aiw-chat-history');
    } else {
      if(prevID){
        localStorage.setItem('aiw-prevID',prevID);
      }
      if(chatHistory.length>0){
        localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
      }
    }
  });

  /* ───────────────────────── 6. CORE SEND LOGIC ───────────────── */

  async function sendMessage(){
    const msg = textarea.value.trim();
    if(!msg || chatBusy) return;

    abortActiveRequest();
    chatBusy=true;
    textarea.value='';

    // Add user message
    addMsg(msg,'user');

    // Prepare an empty assistant message
    const aiDiv = addMsg('','assistant');
    startDots(aiDiv);

    const headers={
      'Content-Type':'application/json',
      'X-Widget-Key':WIDGET_KEY
    };

    // SSE or normal fetch
    currentFetchController = new AbortController();
    const signal=currentFetchController.signal;

    // Helper to append chunks
    function push(chunk){
      if(aiDiv.classList.contains('typing')){
        stopDots(aiDiv);
        aiDiv.innerHTML='';
      }
      const safeChunk=sanitize(chunk);
      aiDiv.innerHTML+=safeChunk;
      messages.scrollTop=messages.scrollHeight;

      // Update last assistant message in chatHistory
      const lastIndex=chatHistory.length-1;
      if(lastIndex>=0 && chatHistory[lastIndex].role==='assistant'){
        chatHistory[lastIndex].content=aiDiv.innerHTML;
        if(useHistory){
          localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
        }
      }
    }

    try{
      if(USE_STREAM){
        // SSE approach
        const res=await fetch(`${BACKEND}/chat/stream`,{
          method:'POST',
          mode:'cors',
          headers,
          body:JSON.stringify({message:msg, previous_response_id:prevID}),
          signal
        });
        if(!res.ok || !res.body) throw new Error('Network error');

        currentReader=res.body.getReader();
        const dec=new TextDecoder();
        let buf='';

        while(true){
          const {value,done}=await currentReader.read();
          if(done) break;
          buf+=dec.decode(value,{stream:true});
          const evts=buf.split('\n\n');
          buf=evts.pop();

          for(const ev of evts){
            if(ev.startsWith('event: done')){
              currentReader.cancel();
              break;
            }
            if(ev.startsWith('event: meta')){
              const data=JSON.parse(ev.split('\n')[1].slice(6));
              if(data.prev_id){
                prevID=data.prev_id;
                if(useHistory){
                  localStorage.setItem('aiw-prevID',prevID);
                }
              }
              continue;
            }
            const chunk=ev
              .split('\n')
              .filter(l=>l.startsWith('data:'))
              .map(l=>l.slice(6))
              .join('\n');
            if(chunk) push(chunk);
          }
        }
      } else {
        // Normal fetch
        const res=await fetch(`${BACKEND}/chat`,{
          method:'POST',
          mode:'cors',
          headers,
          body:JSON.stringify({message:msg, previous_response_id:prevID}),
          signal
        });
        if(!res.ok) throw new Error('Network error');

        const j=await res.json();
        stopDots(aiDiv);

        const safeResp=sanitize(j.response||j.error||'No response');
        aiDiv.innerHTML=safeResp;

        const lastIndex=chatHistory.length-1;
        if(lastIndex>=0 && chatHistory[lastIndex].role==='assistant'){
          chatHistory[lastIndex].content=safeResp;
          if(useHistory){
            localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
          }
        }

        if(j.prev_id){
          prevID=j.prev_id;
          if(useHistory){
            localStorage.setItem('aiw-prevID',prevID);
          }
        }
      }

    }catch(err){
      if(err.name==='AbortError'){
        console.warn('[aiw] Active request was aborted');
        aiDiv.remove();
        return;
      }
      console.error('[aiw]',err);
      stopDots(aiDiv);
      aiDiv.textContent='Error contacting server';

      const lastIndex=chatHistory.length-1;
      if(lastIndex>=0 && chatHistory[lastIndex].role==='assistant'){
        chatHistory[lastIndex].content='Error contacting server';
        if(useHistory){
          localStorage.setItem('aiw-chat-history', JSON.stringify(chatHistory));
        }
      }

    }finally{
      chatBusy=false;
      aiDiv.classList.remove('typing');
      currentFetchController=null;
      currentReader=null;
      textarea.focus();
    }
  }

  function abortActiveRequest(){
    if(currentFetchController){
      currentFetchController.abort();
      currentFetchController=null;
    }
    if(currentReader){
      currentReader.cancel();
      currentReader=null;
    }
  }

  /* ───────────────────────── 7. INITIAL SETUP ──────────────────── */

  // If chatHistory is empty, add default welcome messages
  if(chatHistory.length===0){
    addDefaultMessages();
  } else {
    restoreHistory();
  }

  // Auto-open chat after 10s if not opened manually
  setTimeout(()=>{
    const isOpen=(box.style.display==='flex');
    if(!isOpen && !userOpenedChat){
      // Show chat in compact mode + "pop" animation
      box.style.display='flex';
      setCompact();
      box.classList.add('aiw-auto-open');
    }
  },10000);

  // Remove "pop" animation class after it finishes
  box.addEventListener('animationend', e=>{
    if(e.animationName==='aiwBoxPop'){
      box.classList.remove('aiw-auto-open');
    }
  });

})();
