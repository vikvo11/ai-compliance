/* main.js */
/* Comments in English as requested. -------------------------------------------
   – Drag-&-drop CSV upload
   – Invoice editing + toast notifications
   – Fancy demo animations / crack-glass effect
   – Light/night theme toggle
   – Streaming chat (SSE) with:
       • user bubbles (grey) / assistant bubbles (light-green)
       • chatBusy flag blocks double-send
       • animated “…” while GPT thinks
       • compact ↔ expanded view
   – FIX 1: chat-box has a fixed height (420 px) in compact mode so the
            textarea/footer always stays at the bottom.
   – FIX 2: full SSE payload is now reconstructed on the client and
            new-line characters are preserved (white-space: pre-wrap).
------------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------------
   CONSTANTS
--------------------------------------------------------------------------- */
const COMPACT_HEIGHT = '420px';       // fixed height when folded
const USE_STREAM     = true;          // switch to `/chat` if false

/* ---------------------------------------------------------------------------
   DRAG-AND-DROP FOR CSV
--------------------------------------------------------------------------- */
let dragCounter = 0;
function handleDragOver(e){ e.preventDefault(); }
function handleDragLeave(e){
  e.preventDefault(); dragCounter--;
  if(!dragCounter) document.getElementById('drop-area').style.display = 'none';
}
document.addEventListener('dragenter',e=>{
  e.preventDefault(); dragCounter++;
  document.getElementById('drop-area').style.display = 'flex';
});
function handleDrop(e){
  e.preventDefault(); dragCounter = 0;
  document.getElementById('drop-area').style.display = 'none';
  const f = e.dataTransfer.files;
  if(f.length && f[0].type === 'text/csv'){
    const fd = new FormData(); fd.append('csv_file', f[0]);
    fetch('/', { method:'POST', body:fd })
      .then(r => { if(!r.ok) throw new Error(); location.reload(); })
      .catch(() => alert('Upload failed'));
  }else alert('Please drop a valid CSV file.');
}
document.addEventListener('dragover',handleDragOver);
document.addEventListener('dragleave',handleDragLeave);
document.addEventListener('drop',handleDrop);

function toggleInvoices(){
  const t = document.getElementById('invoiceTable');
  t.style.display = (t.style.display === 'none') ? 'block' : 'none';
}

/* ---------------------------------------------------------------------------
   FADE-IN ON SCROLL
--------------------------------------------------------------------------- */
document.querySelectorAll('.fade').forEach(el=>{
  const io = new IntersectionObserver(e=>{
    if(e[0].isIntersecting){ el.classList.add('show'); io.unobserve(el); }
  },{threshold:0.3});
  io.observe(el);
});

/* ---------------------------------------------------------------------------
   TOAST
--------------------------------------------------------------------------- */
function showToast(msg){
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(()=>{
    t.style.opacity='0';
    setTimeout(()=>c.removeChild(t),800);
  },3000);
}

/* ---------------------------------------------------------------------------
   SAVE INVOICE
--------------------------------------------------------------------------- */
async function saveInvoice(e,id){
  e.preventDefault();
  try{
    const fd = new FormData(e.target);
    const r  = await fetch(`/edit/${id}`,{ method:'POST', body:fd });
    if(!r.ok) throw new Error();
    const d = await r.json();
    document.getElementById(`invoice-${id}-client`).textContent    = d.client_name;
    document.getElementById(`invoice-${id}-invoiceID`).textContent = d.invoice_id;
    document.getElementById(`invoice-${id}-amount`).textContent    = `$${(+d.amount).toFixed(2)}`;
    document.getElementById(`invoice-${id}-due`).textContent       = d.date_due;
    document.getElementById(`invoice-${id}-status`).textContent    = d.status;
    document.getElementById(`edit-row-${id}`).style.display = 'none';
    showToast('Invoice updated successfully!');
  }catch{
    showToast('Error updating invoice.');
  }
}
const toggleEdit=id=>{
  const row=document.getElementById(`edit-row-${id}`);
  row.style.display=(row.style.display==='none') ? 'table-row' : 'none';
};

/* ---------------------------------------------------------------------------
   TYPE EFFECT & TERMINAL DEMOS
--------------------------------------------------------------------------- */
function typeInto(el,txt,speed=60,cb){
  let i=0;
  (function t(){
    if(i<txt.length){
      ('placeholder' in el)
        ? el.placeholder = txt.slice(0, ++i)
        : el.value      += txt[i++ - 1];
      setTimeout(t,speed);
    }else cb?.();
  })();
}
function printLines(id,lines,delay=900){
  const el=document.getElementById(id);
  let i=0;
  (function n(){
    if(i<lines.length){
      el.innerHTML += lines[i++] + '<br>';
      setTimeout(n,delay);
    }else{
      el.innerHTML += '<span class="cursor"></span>';
    }
  })();
}
const onceVisible=(sel,cb)=>{
  const el=document.querySelector(sel);
  if(!el) return;
  const io=new IntersectionObserver(e=>{
    if(e[0].isIntersecting){ cb(); io.disconnect(); }
  },{threshold:0.5});
  io.observe(el);
};

/* Demo triggers ------------------------------------------------------------- */
onceVisible('#demo',()=>{
  typeInto(companyField,'Acme Corporation',60,()=>{
    setTimeout(()=>typeInto(reportField,'Q4 Revenue Report'),300);
    setTimeout(()=>typeInto(dateField,'31 Jan 2025'),800);
  });
  printLines('terminal-main',[
    '> AI.extractData("report.pdf")',
    '→ Filling form fields…',
    '→ Uploading to portal…',
    '→ Status: ✅ Filed Successfully!'
  ]);
});
onceVisible('#actions',()=>{
  printLines('term-extract',[
    '> AI.parse("invoices.zip")',
    '→ 124 invoices processed',
    '→ Data normalized ✅'
  ]);
  setTimeout(()=>printLines('term-search',[
    '> AI.vectorSearch("FCC Part 69")',
    '→ 5 relevant clauses found',
    '→ Mapping to form fields… ✅'
  ]),800);
  setTimeout(()=>printLines('term-submit',[
    '> AI.RPA.submit("FCC-499A")',
    '→ Authenticating…',
    '→ Uploading PDF…',
    '→ Submission ID: #8842 ✅'
  ]),1600);
});

/* ---------------------------------------------------------------------------
   CRACK-GLASS ANIMATION
--------------------------------------------------------------------------- */
function crackGlass(){
  const block=document.getElementById('ai-extract-block');
  const cv=document.getElementById('crack-effect');
  if(!cv) return;
  const ctx=cv.getContext('2d');
  if(!ctx) return;

  cv.style.display='block';
  cv.style.opacity='1';
  cv.style.transition='';
  cv.width=300;
  cv.height=200;
  ctx.clearRect(0,0,cv.width,cv.height);

  navigator.vibrate?.([100,50,100]);
  block.classList.add('vibrate');
  setTimeout(()=>block.classList.remove('vibrate'),700);

  const [cx,cy] = [cv.width/2, cv.height/2];
  for(let i=0;i<24;i++){
    const ang=i*Math.PI*2/24;
    const len=50 + Math.random()*100;
    ctx.beginPath();
    ctx.moveTo(cx,cy);
    ctx.lineTo(cx + Math.cos(ang)*len, cy + Math.sin(ang)*len);
    ctx.strokeStyle=`rgba(100,100,100,${0.5+Math.random()*0.5})`;
    ctx.lineWidth=0.5 + Math.random()*2;
    ctx.stroke();
  }
  for(let r=20;r<=100;r+=20){
    ctx.beginPath();
    ctx.arc(cx,cy,r + Math.random()*5,0,Math.PI*2);
    ctx.strokeStyle=`rgba(120,120,120,${Math.random()*0.4+0.3})`;
    ctx.lineWidth=0.3 + Math.random();
    ctx.stroke();
  }
  setTimeout(()=>{
    cv.style.transition='opacity .8s';
    cv.style.opacity='0';
    setTimeout(()=>{
      cv.style.display='none';
      cv.style.transition='';
    },800);
  },1300);
}

/* ---------------------------------------------------------------------------
   DOM-READY SETUP
--------------------------------------------------------------------------- */
const chatBox = document.getElementById('chat-box'); // used in many places

document.addEventListener('DOMContentLoaded',()=>{
  /* initial chat-box state */
  chatBox.style.display = 'none';
  chatBox.style.width   = '340px';
  chatBox.style.height  = COMPACT_HEIGHT;                // fixed height
  chatBox.querySelector('.messages').style.maxHeight='300px';

  /* expand / collapse */
  document.getElementById('expand-chat')?.addEventListener('click',()=>{
    if(chatBox.style.width === '600px'){                 /* collapse */
      chatBox.style.width  = '340px';
      chatBox.style.height = COMPACT_HEIGHT;
      chatBox.querySelector('.messages').style.maxHeight='300px';
    }else{                                               /* expand */
      chatBox.style.display = 'flex';
      chatBox.style.flexDirection = 'column';
      chatBox.style.width  = '600px';
      chatBox.style.height = '80vh';
      chatBox.querySelector('.messages').style.maxHeight='';
    }
  });

  document.getElementById('ai-extract-block')
    ?.addEventListener('click',crackGlass);
});

/* ---------------------------------------------------------------------------
   CHAT
--------------------------------------------------------------------------- */
const launcher  = document.getElementById('chat-launcher');
const chatInput = chatBox.querySelector('textarea');
const sendBtn   = chatBox.querySelector('button.send');

let chatBusy = false;

/* helper – create bubble */
function addMessage(txt, cls='message'){
  const d = document.createElement('div');
  d.className = cls;
  d.textContent = txt;
  /* keep line breaks – fallback in case CSS is missing */
  d.style.whiteSpace = 'pre-wrap';
  d.style.wordBreak  = 'break-word';

  const pane = chatBox.querySelector('.messages');
  pane.appendChild(d);
  pane.scrollTop = pane.scrollHeight;
  return d;
}

/* launcher toggle */
launcher.addEventListener('click',()=>{
  if(chatBox.style.display === 'flex'){
    chatBox.style.display = 'none';
    return;
  }
  chatBox.style.display = 'flex';
  chatBox.style.flexDirection = 'column';
  chatInput.disabled = false;
  sendBtn.disabled   = false;
  chatInput.focus();
});

/* ellipsis animation ------------------------------------ */
const startDots = el=>{
  let dots=1;
  el.textContent='.';
  el._timer=setInterval(()=>{
    dots = (dots % 3) + 1;
    el.textContent='.'.repeat(dots);
  },400);
};
const stopDots = el=>{
  clearInterval(el._timer);
  delete el._timer;
};

/* send --------------------------------------------------- */
async function sendMessage(){
  const msg = chatInput.value.trim();
  if(!msg || chatBusy) return;

  chatBusy = true;
  chatInput.disabled = true;
  sendBtn.disabled   = true;

  addMessage(msg, 'message user');
  chatInput.value = '';

  const aiDiv = addMessage('', 'message assistant typing');
  startDots(aiDiv);

  /* push received chunk into bubble */
  const push = chunk=>{
    if(aiDiv.classList.contains('typing')){
      stopDots(aiDiv);
      aiDiv.textContent = '';
      aiDiv.classList.remove('typing');
    }
    aiDiv.textContent += chunk;
    chatBox.querySelector('.messages').scrollTop =
      chatBox.querySelector('.messages').scrollHeight;
  };

  try{
    if(USE_STREAM){
      const res = await fetch('/chat/stream',{
        method : 'POST',
        headers: {'Content-Type':'application/json'},
        body   : JSON.stringify({message:msg})
      });
      if(!res.ok || !res.body) throw new Error('Network error');

      const rdr = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while(true){
        const {value,done} = await rdr.read();
        if(done) break;
        buf += dec.decode(value,{stream:true});

        /* each SSE event ends with "\n\n" */
        const events = buf.split('\n\n');
        buf = events.pop();               // keep incomplete tail

        for(const ev of events){
          if(ev.startsWith('event: done')){
            rdr.cancel();                 // server signalled end
            break;
          }
          /* collect ALL "data: " lines, join with \n */
          const chunk = ev
            .split('\n')
            .filter(line => line.startsWith('data:'))
            .map(line   => line.slice(6))
            .join('\n');

          if(chunk) push(chunk);
        }
      }
    }else{
      /* fallback: single-shot endpoint */
      const res = await fetch('/chat',{
        method : 'POST',
        headers: {'Content-Type':'application/json'},
        body   : JSON.stringify({message:msg})
      });
      const d = await res.json();
      stopDots(aiDiv);
      aiDiv.textContent = d.response || d.error || 'No response';
    }
  }catch(err){
    console.error(err);
    stopDots(aiDiv);
    aiDiv.textContent='Error contacting server';
  }finally{
    chatBusy = false;
    chatInput.disabled = false;
    sendBtn.disabled   = false;
    aiDiv.classList.remove('typing');
    chatInput.focus();
  }
}

sendBtn.addEventListener('click',sendMessage);
chatInput.addEventListener('keydown',e=>{
  if(e.key === 'Enter' && !e.shiftKey){
    e.preventDefault();
    sendMessage();
  }
});

/* ---------------------------------------------------------------------------
   AUTO THEME
--------------------------------------------------------------------------- */
(()=>{
  const hr = new Date().getHours();
  if(hr >= 19 || hr < 6){
    document.body.classList.add('night-theme');
    document.getElementById('star-background')
      ?.style.setProperty('display','block');
  }
})();
document.getElementById('theme-toggle')?.addEventListener('click',()=>{
  const body  = document.body;
  const stars = document.getElementById('star-background');
  const night = body.classList.toggle('night-theme');
  if(stars) stars.style.display = night ? 'block' : 'none';
  localStorage.setItem('theme', night ? 'night' : 'light');
});

/* ---------------------------------------------------------------------------
   NOTE FOR STYLESHEET
--------------------------------------------------------------------------- */
/*
  .message {               // user + assistant bubbles
    white-space: pre-wrap; // PRESERVE \n COMING FROM SSE
    word-break : break-word;
  }

  Provide appropriate background colors for .user / .assistant classes
  (e.g. grey vs light-green) in your CSS.
*/
