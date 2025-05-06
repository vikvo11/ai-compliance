/* JavaScript logic migrated from templates/index.html */

// Drag-and-drop for CSV files
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
    fetch('/', {
      method: 'POST',
      body: formData
    })
    .then((resp) => {
      if (!resp.ok) throw new Error('Upload failed');
      location.reload();
    })
    .catch(() => alert('Upload failed'));
  } else {
    alert('Please drop a valid CSV file.');
  }
}

function toggleInvoices() {
  const tbl = document.getElementById('invoiceTable');
  tbl.style.display = (tbl.style.display === 'none') ? 'block' : 'none';
}

document.querySelectorAll('.fade').forEach(el => {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('show');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });
  io.observe(el);
});

function showToast(message) {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.classList.add('toast');
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => container.removeChild(toast), 800);
  }, 3000);
}

async function saveInvoice(event, invoiceId) {
  event.preventDefault();
  try {
    const form = event.target;
    const formData = new FormData(form);
    const response = await fetch(`/edit/${invoiceId}`, {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Network response was not ok');
    const data = await response.json();
    document.getElementById(`invoice-${invoiceId}-client`).textContent = data.client_name;
    document.getElementById(`invoice-${invoiceId}-invoiceID`).textContent = data.invoice_id;
    document.getElementById(`invoice-${invoiceId}-amount`).textContent = `$${parseFloat(data.amount).toFixed(2)}`;
    document.getElementById(`invoice-${invoiceId}-due`).textContent = data.date_due;
    document.getElementById(`invoice-${invoiceId}-status`).textContent = data.status;
    document.getElementById(`edit-row-${invoiceId}`).style.display = 'none';
    showToast('Invoice updated successfully!');
  } catch (err) {
    console.error(err);
    showToast('Error updating invoice.');
  }
}

function toggleEdit(id) {
  const row = document.getElementById(`edit-row-${id}`);
  row.style.display = (row.style.display === 'none') ? 'table-row' : 'none';
}

function typeInto(el, text, speed = 60, cb) {
  let i = 0;
  (function t() {
    if (i < text.length) {
      el.placeholder ? el.placeholder = text.slice(0, i + 1) : el.value += text[i];
      i++;
      setTimeout(t, speed);
    } else if (cb) cb();
  })();
}

function printLines(id, lines, delay = 900) {
  const t = document.getElementById(id);
  let i = 0;
  (function n() {
    if (i < lines.length) {
      t.innerHTML += lines[i] + '<br>';
      i++;
      setTimeout(n, delay);
    } else {
      t.innerHTML += '<span class="cursor"></span>';
    }
  })();
}

const obs = (sel, cb) => {
  const el = document.querySelector(sel);
  if (!el) return;
  const io = new IntersectionObserver(e => {
    if (e[0].isIntersecting) {
      cb();
      io.disconnect();
    }
  }, { threshold: 0.5 });
  io.observe(el);
};

obs('#demo', () => {
  const companyField = document.getElementById('companyField');
  const reportField = document.getElementById('reportField');
  const dateField = document.getElementById('dateField');
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

function crackGlass() {
  const block = document.getElementById('ai-extract-block');
  const canvas = document.getElementById('crack-effect');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  canvas.style.transition = '';
  canvas.style.opacity = '1';
  canvas.style.display = 'block';
  canvas.width = 300;
  canvas.height = 200;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (navigator.vibrate) navigator.vibrate([100, 50, 100, 30, 100]);
  block.classList.add('vibrate');
  setTimeout(() => block.classList.remove('vibrate'), 700);
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  for (let i = 0; i < 24; i++) {
    const angle = (Math.PI * 2 * i) / 24;
    const length = 50 + Math.random() * 100;
    const x = centerX + Math.cos(angle) * length;
    const y = centerY + Math.sin(angle) * length;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = `rgba(100,100,100,${0.5 + Math.random() * 0.5})`;
    ctx.lineWidth = 0.5 + Math.random() * 2;
    ctx.stroke();
  }
  for (let r = 20; r <= 100; r += 20) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, r + Math.random() * 5, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(120,120,120,${Math.random() * 0.4 + 0.3})`;
    ctx.lineWidth = 0.3 + Math.random();
    ctx.stroke();
  }
  setTimeout(() => {
    canvas.style.transition = 'opacity 0.8s ease';
    canvas.style.opacity = '0';
    setTimeout(() => {
      canvas.style.display = 'none';
      canvas.style.opacity = '1';
      canvas.style.transition = '';
    }, 800);
  }, 1300);
}

document.addEventListener('DOMContentLoaded', () => {
  const aiCard = document.getElementById('ai-extract-block');
  if (aiCard) aiCard.addEventListener('click', crackGlass);
});

const launcher = document.getElementById('chat-launcher');
const chatBox = document.getElementById('chat-box');
const chatInput = chatBox.querySelector('input[type=text]');
const sendBtn = chatBox.querySelector('button.send');
launcher.addEventListener('click', () => {
  chatBox.style.display = chatBox.style.display === 'flex' ? 'none' : 'flex';
  chatBox.style.flexDirection = 'column';
  chatInput.disabled = false;
  sendBtn.disabled = false;
  sendBtn.addEventListener('click', async () => {
    const message = chatInput.value.trim();
    if (!message) return;

    const messagesContainer = chatBox.querySelector('.messages');
    const userDiv = document.createElement('div');
    userDiv.className = 'message';
    userDiv.textContent = message;
    messagesContainer.appendChild(userDiv);
    chatInput.value = '';

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      const data = await res.json();
      const botDiv = document.createElement('div');
      botDiv.className = 'message';
      botDiv.textContent = data.response || data.error || 'No response';
      messagesContainer.appendChild(botDiv);
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    } catch (err) {
      const errDiv = document.createElement('div');
      errDiv.className = 'message';
      errDiv.textContent = 'Error contacting server';
      messagesContainer.appendChild(errDiv);
    }
  });

  chatInput.focus();
});

(function autoTheme() {
  const hour = new Date().getHours();
  if (hour >= 19 || hour < 6) {
    document.body.classList.add('night-theme');
    document.getElementById('star-background').style.display = 'block';
  }
})();

const themeBtn = document.getElementById('theme-toggle');
if (themeBtn) {
  themeBtn.addEventListener('click', () => {
    const body = document.body;
    const stars = document.getElementById('star-background');
    const isNight = body.classList.toggle('night-theme');
    stars.style.display = isNight ? 'block' : 'none';
    localStorage.setItem('theme', isNight ? 'night' : 'light');
  });
}