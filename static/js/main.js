/* main.js */
/* Comments in English as requested. --------------------------------------------------
   NEW  ➜ Added real-time streaming chat support (/chat/stream, Server-Sent Events)
   Everything else from the previous file is preserved.
------------------------------------------------------------------------------- */

/* --------------------------------
   DRAG-AND-DROP FOR CSV FILES
----------------------------------- */
let dragCounter = 0;
function handleDragOver(e) {
  e.preventDefault();
}
function handleDragLeave(e) {
  e.preventDefault();
  dragCounter--;
  if (dragCounter === 0) {
    document.getElementById('drop-area').style.display = 'none';
  }
}
document.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter++;
  document.getElementById('drop-area').style.display = 'flex';
});
function handleDrop(e) {
  e.preventDefault();
  dragCounter = 0;
  document.getElementById('drop-area').style.display = 'none';

  const files = e.dataTransfer.files;
  if (files.length && files[0].type === 'text/csv') {
    const formData = new FormData();
    formData.append('csv_file', files[0]);
    fetch('/', { method: 'POST', body: formData })
      .then((resp) => { if (!resp.ok) throw new Error('Upload failed'); location.reload(); })
      .catch(() => alert('Upload failed'));
  } else { alert('Please drop a valid CSV file.'); }
}

function toggleInvoices() {
  const tbl = document.getElementById('invoiceTable');
  tbl.style.display = (tbl.style.display === 'none') ? 'block' : 'none';
}

/* --------------------------------
   FADE-IN ANIMATION ON SCROLL
----------------------------------- */
document.querySelectorAll('.fade').forEach(el => {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add('show'); io.unobserve(entry.target); }
    });
  }, { threshold: 0.3 });
  io.observe(el);
});

/* --------------------------------
   TOAST NOTIFICATIONS
----------------------------------- */
function showToast(message) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.classList.add('toast');
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => container.removeChild(toast), 800); }, 3000);
}

/* --------------------------------
   SAVE INVOICE
----------------------------------- */
async function saveInvoice(event, invoiceId) {
  event.preventDefault();
  try {
    const formData = new FormData(event.target);
    const res = await fetch(`/edit/${invoiceId}`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Network response was not ok');

    const d = await res.json();
    document.getElementById(`invoice-${invoiceId}-client`).textContent   = d.client_name;
    document.getElementById(`invoice-${invoiceId}-invoiceID`).textContent = d.invoice_id;
    document.getElementById(`invoice-${invoiceId}-amount`).textContent    = `$${parseFloat(d.amount).toFixed(2)}`;
    document.getElementById(`invoice-${invoiceId}-due`).textContent       = d.date_due;
    document.getElementById(`invoice-${invoiceId}-status`).textContent    = d.status;
    document.getElementById(`edit-row-${invoiceId}`).style.display = 'none';
    showToast('Invoice updated successfully!');
  } catch (err) { console.error(err); showToast('Error updating invoice.'); }
}
function toggleEdit(id) {
  const row = document.getElementById(`edit-row-${id}`);
  row.style.display = (row.style.display === 'none') ? 'table-row' : 'none';
}

/* --------------------------------
   TYPE EFFECT & TERMINAL SIMULATIONS
----------------------------------- */
function typeInto(el, text, speed = 60, cb) {
  let i = 0;
  (function t() {
    if (i < text.length) {
      if (el.placeholder) el.placeholder = text.slice(0, i + 1);
      else el.value += text[i];
      i++; setTimeout(t, speed);
    } else if (cb) cb();
  })();
}
function printLines(id, lines, delay = 900) {
  const t = document.getElementById(id);
  let i = 0;
  (function n() {
    if (i < lines.length) { t.innerHTML += lines[i] + '<br>'; i++; setTimeout(n, delay); }
    else t.innerHTML += '<span class="cursor"></span>';
  })();
}
const obs = (sel, cb) => {
  const el = document.querySelector(sel); if (!el) return;
  const io = new IntersectionObserver(e => { if (e[0].isIntersecting) { cb(); io.disconnect(); } }, { threshold: 0.5 });
  io.observe(el);
};

/* Trigger demo animations once in view */
obs('#demo', () => {
  const companyField = document.getElementById('companyField');
  const reportField  = document.getElementById('reportField');
  const dateField    = document.getElementById('dateField');
  typeInto(companyField, 'Acme Corporation', 60, () => {
    setTimeout(() => typeInto(reportField, 'Q4 Revenue Report'), 300);
    setTimeout(() => typeInto(dateField,  '31 Jan 2025'),       800);
  });
  printLines('terminal-main', [
    '> AI.extractData("report.pdf")',
    '→ Filling form fields…',
    '→ Uploading to portal…',
    '→ Status: ✅ Filed Successfully!'
  ]);
});
obs('#actions', () => {
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
   CRACK GLASS ANIMATION
----------------------------------- */
function crackGlass() {
  const block  = document.getElementById('ai-extract-block');
  const canvas = document.getElementById('crack-effect');
  if (!canvas) return;
  const ctx = canvas.getContext('2d'); if (!ctx) return;

  canvas.style.transition = ''; canvas.style.opacity = '1'; canvas.style.display = 'block';
  canvas.width = 300; canvas.height = 200; ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (navigator.vibrate) navigator.vibrate([100, 50, 100, 30, 100]);
  block.classList.add('vibrate'); setTimeout(() => block.classList.remove('vibrate'), 700);

  const cx = canvas.width / 2, cy = canvas.height / 2;
  for (let i = 0; i < 24; i++) {
    const angle = (Math.PI * 2 * i) / 24, len = 50 + Math.random() * 100;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(angle) * len, cy + Math.sin(angle) * len);
    ctx.strokeStyle = `rgba(100,100,100,${0.5 + Math.random() * 0.5})`;
    ctx.lineWidth   = 0.5 + Math.random() * 2; ctx.stroke();
  }
  for (let r = 20; r <= 100; r += 20) {
    ctx.beginPath(); ctx.arc(cx, cy, r + Math.random() * 5, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(120,120,120,${Math.random() * 0.4 + 0.3})`;
    ctx.lineWidth   = 0.3 + Math.random(); ctx.stroke();
  }
  setTimeout(() => { canvas.style.transition = 'opacity 0.8s ease'; canvas.style.opacity = '0';
    setTimeout(() => { canvas.style.display = 'none'; canvas.style.opacity = '1'; canvas.style.transition = ''; }, 800);
  }, 1300);
}

document.addEventListener('DOMContentLoaded', () => {
  chatBox.style.display = 'none';
  chatBox.style.width   = '340px';
  chatBox.style.height  = 'auto';
  chatBox.querySelector('.messages').style.maxHeight = '300px';

  const expandBtn = document.getElementById('expand-chat');
  if (expandBtn) expandBtn.addEventListener('click', () => {
    if (chatBox.style.width === '600px') {
      chatBox.style.width = '340px'; chatBox.style.height = 'auto';
      chatBox.querySelector('.messages').style.maxHeight = '300px';
    } else {
      chatBox.style.display = 'flex'; chatBox.style.flexDirection = 'column';
      chatBox.style.width   = '600px'; chatBox.style.height = '80vh';
      chatBox.querySelector('.messages').style.maxHeight = '';
    }
  });

  const aiCard = document.getElementById('ai-extract-block');
  if (aiCard) aiCard.addEventListener('click', crackGlass);
});

/* --------------------------------
   CHAT LAUNCHER & SEND BUTTON
----------------------------------- */
const launcher  = document.getElementById('chat-launcher');
const chatBox   = document.getElementById('chat-box');
const chatInput = chatBox.querySelector('textarea');
const sendBtn   = chatBox.querySelector('button.send');

const USE_STREAM = true;   // Toggle SSE streaming

/* Append a message to chat; returns the created element */
function addMessage(text, className = 'message') {
  const div = document.createElement('div');
  div.className = className;
  div.textContent = text;
  chatBox.querySelector('.messages').appendChild(div);
  chatBox.querySelector('.messages').scrollTop =
    chatBox.querySelector('.messages').scrollHeight;
  return div;
}

/* Toggle chat box visibility */
launcher.addEventListener('click', () => {
  chatBox.style.display = chatBox.style.display === 'flex' ? 'none' : 'flex';
  chatBox.style.flexDirection = 'column';
  chatInput.disabled = false; sendBtn.disabled = false; chatInput.focus();
});

/* Send message (click or Enter) */
async function sendMessage() {
  const message = chatInput.value.trim();
  if (!message) return;

  addMessage(message, 'message');          // user bubble
  chatInput.value = '';
  chatInput.focus();

  // Assistant placeholder bubble
  let assistantDiv = addMessage('', 'message typing');

  // Helper to stream chunks into the bubble
  function pushChunk(chunk) {
    assistantDiv.textContent += chunk;
    chatBox.querySelector('.messages').scrollTop =
      chatBox.querySelector('.messages').scrollHeight;
  }

  try {
    if (USE_STREAM) {
      /* -------- STREAMING IMPLEMENTATION (SSE via fetch) -------- */
      const res = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      if (!res.ok || !res.body) throw new Error('Network error');

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Split on double LF (end of SSE block)
        const parts = buffer.split('\n\n');
        buffer = parts.pop();               // keep last incomplete
        for (let part of parts) {
          if (part.startsWith('event: done')) { reader.cancel(); break; }
          const dataLine = part.split('\n').find(l => l.startsWith('data:'));
          if (dataLine) pushChunk(dataLine.slice(6));
        }
      }
    } else {
      /* -------- Classic single-shot endpoint -------- */
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const d = await res.json();
      assistantDiv.textContent = d.response || d.error || 'No response';
    }
  } catch (err) {
    console.error(err);
    assistantDiv.textContent = 'Error contacting server';
  } finally {
    assistantDiv.classList.remove('typing');
  }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

/* --------------------------------
   AUTO THEME (LIGHT / NIGHT)
----------------------------------- */
(function autoTheme() {
  const hour = new Date().getHours();
  if (hour >= 19 || hour < 6) {
    document.body.classList.add('night-theme');
    const stars = document.getElementById('star-background');
    if (stars) stars.style.display = 'block';
  }
})();
const themeBtn = document.getElementById('theme-toggle');
if (themeBtn) themeBtn.addEventListener('click', () => {
  const body = document.body;
  const stars = document.getElementById('star-background');
  const isNight = body.classList.toggle('night-theme');
  if (stars) stars.style.display = isNight ? 'block' : 'none';
  localStorage.setItem('theme', isNight ? 'night' : 'light');
});
