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
       • **context continuity via localStorage (no cookies needed)**
   – FIX: in compact mode the chat-box now has a fixed height (420 px) so the
          textarea/footer always stays at the bottom.
------------------------------------------------------------------------------- */

/* --------------------------------
   1) CONSTANTS
----------------------------------- */
const COMPACT_HEIGHT   = '420px';          // folded height
const USE_STREAM       = true;             // switch to /chat if false
const PREV_ID_KEY      = 'prevID';         // localStorage entry for context

/* Previous response ID is held on the client, survives reloads. */
let prevID = localStorage.getItem(PREV_ID_KEY) || null;

/* --------------------------------
   2) HELPER: Simple sanitize for HTML
----------------------------------- */
function sanitizeHTML(str) {
  return str.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
}

/* --------------------------------
   3) DRAG-AND-DROP FOR CSV
----------------------------------- */
let dragCounter = 0;
function handleDragOver(e) { e.preventDefault(); }
function handleDragLeave(e) {
  e.preventDefault();
  if (!--dragCounter) document.getElementById('drop-area').style.display = 'none';
}
document.addEventListener('dragenter', e => {
  e.preventDefault();
  dragCounter++;
  document.getElementById('drop-area').style.display = 'flex';
});
function handleDrop(e) {
  e.preventDefault();
  dragCounter = 0;
  document.getElementById('drop-area').style.display = 'none';
  const f = e.dataTransfer.files;
  if (f.length && f[0].type === 'text/csv') {
    const fd = new FormData();
    fd.append('csv_file', f[0]);
    fetch('/', { method: 'POST', body: fd })
      .then(r => { if (!r.ok) throw new Error(); location.reload(); })
      .catch(() => alert('Upload failed'));
  } else alert('Please drop a valid CSV file.');
}
document.addEventListener('dragover', handleDragOver);
document.addEventListener('dragleave', handleDragLeave);
document.addEventListener('drop', handleDrop);

/* --------------------------------
   4) FADE-IN ON SCROLL
----------------------------------- */
document.querySelectorAll('.fade').forEach(el => {
  const io = new IntersectionObserver(e => {
    if (e[0].isIntersecting) { el.classList.add('show'); io.unobserve(el); }
  }, { threshold: 0.3 });
  io.observe(el);
});

/* --------------------------------
   5) TOAST
----------------------------------- */
function showToast(msg) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => {
    t.style.opacity = '0';
    setTimeout(() => c.removeChild(t), 800);
  }, 3000);
}

/* --------------------------------
   6) SAVE INVOICE
----------------------------------- */
async function saveInvoice(e, id) {
  e.preventDefault();
  try {
    const fd = new FormData(e.target);
    const r  = await fetch(`/edit/${id}`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error();
    const d = await r.json();
    document.getElementById(`invoice-${id}-client`).textContent    = d.client_name;
    document.getElementById(`invoice-${id}-invoiceID`).textContent = d.invoice_id;
    document.getElementById(`invoice-${id}-amount`).textContent    = `$${(+d.amount).toFixed(2)}`;
    document.getElementById(`invoice-${id}-due`).textContent       = d.date_due;
    document.getElementById(`invoice-${id}-status`).textContent    = d.status;
    document.getElementById(`edit-row-${id}`).style.display        = 'none';
    showToast('Invoice updated successfully!');
  } catch { showToast('Error updating invoice.'); }
}
const toggleEdit = id => {
  const row = document.getElementById(`edit-row-${id}`);
  row.style.display = (row.style.display === 'none') ? 'table-row' : 'none';
};

/* --------------------------------
   7) TYPE EFFECT & TERMINAL DEMOS
----------------------------------- */
function typeInto(el, txt, speed = 60, cb) {
  let i = 0; (function t() {
    if (i < txt.length) {
      ('placeholder' in el) ? (el.placeholder = txt.slice(0, ++i))
                            : (el.value += txt[i++ - 1]);
      setTimeout(t, speed);
    } else cb?.();
  })();
}
function printLines(id, lines, delay = 900) {
  const el = document.getElementById(id); let i = 0;
  (function n() {
    if (i < lines.length) { el.innerHTML += lines[i++] + '<br>'; setTimeout(n, delay); }
    else el.innerHTML += '<span class="cursor"></span>';
  })();
}
const onceVisible = (sel, cb) => {
  const el = document.querySelector(sel); if (!el) return;
  const io = new IntersectionObserver(e => { if (e[0].isIntersecting) { cb(); io.disconnect(); } },
                                      { threshold: 0.5 });
  io.observe(el);
};
/* Demo triggers */
onceVisible('#demo', () => {
  typeInto(companyField, 'Acme Corporation', 60, () => {
    setTimeout(() => typeInto(reportField, 'Q4 Revenue Report'), 300);
    setTimeout(() => typeInto(dateField, '31 Jan 2025'), 800);
  });
  printLines('terminal-main', [
    '> AI.extractData("report.pdf")',
    '→ Filling form fields…',
    '→ Uploading to portal…',
    '→ Status: ✅ Filed Successfully!'
  ]);
});
onceVisible('#actions', () => {
  printLines('term-extract', [
    '> AI.parse("invoices.zip")',
    '→ 124 invoices processed',
    '→ Data normalized ✅'
  ]);
  setTimeout(() => printLines('term-search', [
    '> AI.vectorSearch("FCC Part 69")',
    '→ 5 relevant clauses found',
    '→ Mapping to form fields… ✅'
  ]), 800);
  setTimeout(() => printLines('term-submit', [
    '> AI.RPA.submit("FCC-499A")',
    '→ Authenticating…',
    '→ Uploading PDF…',
    '→ Submission ID: #8842 ✅'
  ]), 1600);
});

/* --------------------------------
   8) CRACK-GLASS ANIMATION
----------------------------------- */
function crackGlass() {
  const block = document.getElementById('ai-extract-block');
  const cv    = document.getElementById('crack-effect');
  if (!cv) return;
  const ctx = cv.getContext('2d'); if (!ctx) return;

  cv.style.display = 'block'; cv.style.opacity = '1'; cv.style.transition = '';
  cv.width = 300; cv.height = 200; ctx.clearRect(0,0,cv.width,cv.height);

  navigator.vibrate?.([100,50,100]);
  block.classList.add('vibrate');
  setTimeout(() => block.classList.remove('vibrate'), 700);

  const [cx, cy] = [cv.width/2, cv.height/2];
  for (let i = 0; i < 24; i++) {
    const ang = i * Math.PI*2/24, len = 50 + Math.random()*100;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+Math.cos(ang)*len, cy+Math.sin(ang)*len);
    ctx.strokeStyle = `rgba(100,100,100,${0.5+Math.random()*0.5})`;
    ctx.lineWidth = 0.5+Math.random()*2; ctx.stroke();
  }
  for (let r = 20; r<=100; r+=20) {
    ctx.beginPath(); ctx.arc(cx,cy,r+Math.random()*5,0,Math.PI*2);
    ctx.strokeStyle = `rgba(120,120,120,${Math.random()*0.4+0.3})`;
    ctx.lineWidth = 0.3+Math.random(); ctx.stroke();
  }
  setTimeout(()=>{cv.style.transition='opacity .8s'; cv.style.opacity='0';
    setTimeout(()=>{cv.style.display='none'; cv.style.transition='';},800);
  },1300);
}

/* --------------------------------
   9) DOM-READY SETUP
----------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  chatBox.style.display='none';
  chatBox.style.width='340px';
  chatBox.style.height=COMPACT_HEIGHT;
  chatBox.querySelector('.messages').style.maxHeight='300px';

  document.getElementById('expand-chat')?.addEventListener('click', () => {
    if (chatBox.style.width==='600px') {        /* collapse */
      chatBox.style.width='340px';
      chatBox.style.height=COMPACT_HEIGHT;
      chatBox.querySelector('.messages').style.maxHeight='300px';
    } else {                                    /* expand */
      chatBox.style.display='flex';
      chatBox.style.flexDirection='column';
      chatBox.style.width='600px';
      chatBox.style.height='80vh';
      chatBox.querySelector('.messages').style.maxHeight='';
    }
  });
  document.getElementById('ai-extract-block')?.addEventListener('click', crackGlass);
});

/* --------------------------------
   10) CHAT
----------------------------------- */
const launcher  = document.getElementById('chat-launcher');
const chatBox   = document.getElementById('chat-box');
const chatInput = chatBox.querySelector('textarea');
const sendBtn   = chatBox.querySelector('button.send');

let chatBusy = false;

/* Create message bubble */
function addMessage(txt, cls='message') {
  const d = document.createElement('div'); d.className=cls;
  d.innerHTML = sanitizeHTML(txt);
  const pane = chatBox.querySelector('.messages');
  pane.appendChild(d); pane.scrollTop = pane.scrollHeight;
  return d;
}

/* Chat launcher */
launcher.addEventListener('click', ()=>{
  if (chatBox.style.display==='flex') { chatBox.style.display='none'; return; }
  chatBox.style.display='flex'; chatBox.style.flexDirection='column';
  chatInput.disabled=false; sendBtn.disabled=false; chatInput.focus();
});

/* Dots animation */
const startDots = el => { let dots=1;
  el.textContent='.'; el._timer=setInterval(()=>{dots=(dots%3)+1; el.textContent='.'.repeat(dots);},400);
};
const stopDots = el => { clearInterval(el._timer); delete el._timer; };

/* Send */
// --- constants used by the chat widget -----------------------------------
const PREV_ID_KEY = 'prevID';                             // localStorage key
let   prevID      = localStorage.getItem(PREV_ID_KEY);    // last response_id

async function sendMessage() {
  const msg = chatInput.value.trim();
  if (!msg || chatBusy) return;

  /* UI-state ──────────────────────────────────────────────────────────── */
  chatBusy = true;
  chatInput.disabled = true;
  sendBtn.disabled   = true;

  addMessage(msg, 'message user');            // user bubble
  chatInput.value = '';

  const aiDiv = addMessage('', 'message assistant typing');
  startDots(aiDiv);                           // animated “…”

  /* helper to append assistant chunks */
  const push = (chunk) => {
    if (aiDiv.classList.contains('typing')) {
      stopDots(aiDiv);
      aiDiv.textContent = '';                 // clear dots
      aiDiv.classList.remove('typing');
    }
    aiDiv.innerHTML += sanitizeHTML(chunk);
    const pane = chatBox.querySelector('.messages');
    pane.scrollTop = pane.scrollHeight;
  };

  try {
    /* 1) ───────────────────── streaming branch ───────────────────────── */
    if (USE_STREAM) {
      const res = await fetch('/chat/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: msg, previous_response_id: prevID }),
        credentials: 'include',               // send cookies if any
      });
      if (!res.ok || !res.body) throw new Error('Network error');

      const rdr = res.body.getReader();
      const dec = new TextDecoder();
      let   buf = '';

      while (true) {
        const { value, done } = await rdr.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });

        /* split by SSE-frame separator */
        let frameEnd;
        while ((frameEnd = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, frameEnd);
          buf = buf.slice(frameEnd + 2);

          /* parse event type */
          let eventType = 'message';
          let dataLines = [];
          for (const line of frame.split('\n')) {
            if (line.startsWith('event:')) eventType = line.slice(6).trim();
            if (line.startsWith('data:'))  dataLines.push(line.slice(5));
          }
          if (!dataLines.length) continue;

          if (eventType === 'done') {
            rdr.cancel();
            break;
          }

          if (eventType === 'meta') {
            /* {"prev_id": "..."} from server */
            try {
              const meta = JSON.parse(dataLines.join('\n'));
              if (meta.prev_id) {
                prevID = meta.prev_id;
                localStorage.setItem(PREV_ID_KEY, prevID);
              }
            } catch { /* ignore malformed meta */ }
            continue;                       // do not show meta in chat
          }

          /* default: assistant token */
          push(dataLines.join('\n'));
        }
      }

    /* 2) ─────────────────── fallback (non-stream) ────────────────────── */
    } else {
      const res = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message: msg, previous_response_id: prevID }),
      });
      const d = await res.json();
      stopDots(aiDiv);
      aiDiv.innerHTML = sanitizeHTML(d.response || d.error || 'No response');

      if (d.prev_id) {
        prevID = d.prev_id;
        localStorage.setItem(PREV_ID_KEY, prevID);
      }
    }
  } catch (err) {
    console.error(err);
    stopDots(aiDiv);
    aiDiv.innerHTML = sanitizeHTML('Error contacting server');
  } finally {
    chatBusy = false;
    chatInput.disabled = false;
    sendBtn.disabled   = false;
    aiDiv.classList.remove('typing');
    chatInput.focus();
  }
}

/* Enter / button */
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', e=>{
  if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendMessage(); }
});

/* --------------------------------
   11) AUTO THEME
----------------------------------- */
(() => {
  const hr=new Date().getHours();
  if(hr>=19||hr<6){ document.body.classList.add('night-theme');
    document.getElementById('star-background')?.style.setProperty('display','block'); }
})();
document.getElementById('theme-toggle')?.addEventListener('click',()=>{
  const body=document.body, stars=document.getElementById('star-background');
  const night=body.classList.toggle('night-theme');
  if(stars) stars.style.display=night?'block':'none';
  localStorage.setItem('theme', night?'night':'light');
});
