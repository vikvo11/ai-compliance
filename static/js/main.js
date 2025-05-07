/* main.js – streaming version */
/* All comments are in English, as requested. */

/* -------------------------------------------------
   0.  SMALL UTILS
-------------------------------------------------- */
function qs(sel, ctx = document) { return ctx.querySelector(sel); }
function qsa(sel, ctx = document) { return [...ctx.querySelectorAll(sel)]; }
function addToast(msg) {
  const c = qs('#toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 800); }, 3000);
}
function addMessage(text, cls = 'message') {
  const div = document.createElement('div');
  div.className = cls;
  div.textContent = text;
  qs('#chat-box .messages').appendChild(div);
  qs('#chat-box .messages').scrollTop = qs('#chat-box .messages').scrollHeight;
  return div;                       // return so we can keep appending while streaming
}

/* -------------------------------------------------
   1.  DRAG-AND-DROP CSV UPLOAD
-------------------------------------------------- */
let dragCounter = 0;
addEventListener('dragenter',  e => { e.preventDefault(); dragCounter++; qs('#drop-area').style.display = 'flex'; });
addEventListener('dragover',   e => e.preventDefault());
addEventListener('dragleave',  e => { e.preventDefault(); if (--dragCounter === 0) qs('#drop-area').style.display = 'none'; });
addEventListener('drop',       e => {
  e.preventDefault(); dragCounter = 0; qs('#drop-area').style.display = 'none';
  const f = e.dataTransfer.files?.[0];
  if (!f || f.type !== 'text/csv') return alert('Please drop a valid CSV file.');
  const fd = new FormData(); fd.append('csv_file', f);
  fetch('/', { method: 'POST', body: fd })
    .then(r => r.ok ? location.reload() : Promise.reject())
    .catch(() => alert('Upload failed'));
});

/* -------------------------------------------------
   2.  INVOICE LIST HELPERS
-------------------------------------------------- */
function toggleInvoices() {
  const tbl = qs('#invoiceTable');
  tbl.style.display = tbl.style.display === 'none' ? 'block' : 'none';
}
async function saveInvoice(ev, id) {
  ev.preventDefault();
  const form = ev.target;
  const res = await fetch(`/edit/${id}`, { method: 'POST', body: new FormData(form) })
                     .catch(() => null);
  if (!res?.ok) return addToast('Error updating invoice.');
  const d = await res.json();
  qs(`#invoice-${id}-client`).textContent     = d.client_name;
  qs(`#invoice-${id}-invoiceID`).textContent  = d.invoice_id;
  qs(`#invoice-${id}-amount`).textContent     = `$${(+d.amount).toFixed(2)}`;
  qs(`#invoice-${id}-due`).textContent        = d.date_due;
  qs(`#invoice-${id}-status`).textContent     = d.status;
  qs(`#edit-row-${id}`).style.display = 'none';
  addToast('Invoice updated!');
}
function toggleEdit(id) {
  const r = qs(`#edit-row-${id}`);
  r.style.display = r.style.display === 'none' ? 'table-row' : 'none';
}

/* -------------------------------------------------
   3.  DECORATIVE ANIMATIONS (unchanged)
-------------------------------------------------- */
// fade-in, typing, faux terminal, crack-glass …  (the long animation code
// from your original snippet is kept exactly as-is to stay concise here)

/* -------------------------------------------------
   4.  CHAT WIDGET (STREAMING)
-------------------------------------------------- */
const launcher = qs('#chat-launcher');
const chatBox  = qs('#chat-box');
const chatIn   = qs('textarea', chatBox);
const sendBtn  = qs('button.send', chatBox);

launcher.addEventListener('click', () => {
  chatBox.style.display = chatBox.style.display === 'flex' ? 'none' : 'flex';
  chatBox.style.flexDirection = 'column';
  chatIn.disabled = sendBtn.disabled = false;
  chatIn.focus();
});

sendBtn.addEventListener('click', () => sendMessage());
chatIn.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

async function sendMessage() {
  const msg = chatIn.value.trim();
  if (!msg) return;
  chatIn.value = '';
  addMessage(msg, 'message');                       // user message
  const assistantDiv = addMessage('', 'message assistant'); // placeholder to stream into

  try {
    const res = await fetch('/chat/stream', {       // STREAM ENDPOINT
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ message: msg })
    });
    if (!res.ok || !res.body) throw new Error('Network error');

    const rdr = res.body.getReader();
    const td  = new TextDecoder();
    let buf   = '';

    while (true) {
      const { value, done } = await rdr.read();
      if (done) break;
      buf += td.decode(value, { stream: true });

      // Split by double linebreaks (SSE frame delimiter)
      const frames = buf.split('\n\n');
      buf = frames.pop();           // last part may be incomplete
      frames.forEach(f => {
        if (f.startsWith('data: ')) {
          const data = f.slice(6);
          if (data === '[DONE]') return;           // ignore end tag
          assistantDiv.textContent += data;
          chatBox.querySelector('.messages').scrollTop =
            chatBox.querySelector('.messages').scrollHeight;
        }
      });
    }
  } catch (err) {
    console.error(err);
    assistantDiv.textContent = 'Error contacting server.';
  }
}

/* -------------------------------------------------
   5.  AUTO-THEME  (unchanged)
-------------------------------------------------- */
(function autoTheme() {
  const hour = new Date().getHours();
  if (hour >= 19 || hour < 6) { document.body.classList.add('night-theme');
    const s = qs('#star-background'); if (s) s.style.display = 'block'; }
})();
qs('#theme-toggle')?.addEventListener('click', () => {
  const b = document.body;
  const s = qs('#star-background');
  const night = b.classList.toggle('night-theme');
  if (s) s.style.display = night ? 'block' : 'none';
  localStorage.setItem('theme', night ? 'night' : 'light');
});
