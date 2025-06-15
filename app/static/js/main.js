/* main.js — slim version
   -------------------------------------------------------------
   • Drag-&-drop CSV upload
   • Invoice edit + toast notifications
   • Fade-in, demo typing + crack-glass animation
   • Auto light/night theme + single toggle
   ----------------------------------------------------------- */

/* 1) HELPERS
------------------------------------------------------------------------*/
const sanitizeHTML = s => s.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');

/* 2) DRAG-AND-DROP CSV
------------------------------------------------------------------------*/
let dragCounter = 0;
function handleDragOver(e) { e.preventDefault(); }
function handleDragLeave(e) {
  e.preventDefault();
  if (!--dragCounter) document.getElementById('drop-area').style.display = 'none';
}
function handleDrop(e) {
  e.preventDefault();
  dragCounter = 0;
  document.getElementById('drop-area').style.display = 'none';

  const f = e.dataTransfer.files;
  if (!(f.length && f[0].type === 'text/csv')) {
    alert('Please drop a valid CSV file.'); return;
  }

  const fd = new FormData(); fd.append('csv_file', f[0]);
  fetch('/', {method:'POST', body:fd})
    .then(r => { if (!r.ok) throw new Error(); location.reload(); })
    .catch(() => alert('Upload failed'));
}

document.addEventListener('dragenter', e => {
  e.preventDefault(); dragCounter++;
  document.getElementById('drop-area').style.display = 'flex';
});
document.addEventListener('dragover',  handleDragOver);
document.addEventListener('dragleave', handleDragLeave);
document.addEventListener('drop',      handleDrop);

/* 3) TOGGLE INVOICE TABLE
------------------------------------------------------------------------*/
function toggleInvoices() {
  const t = document.getElementById('invoiceTable');
  t.style.display = (t.style.display === 'none') ? 'block' : 'none';
}

/* 4) FADE-IN ON SCROLL
------------------------------------------------------------------------*/
document.querySelectorAll('.fade').forEach(el => {
  const io = new IntersectionObserver(e => {
    if (e[0].isIntersecting) { el.classList.add('show'); io.unobserve(el); }
  }, {threshold:0.3});
  io.observe(el);
});

/* 5) TOAST NOTIFICATIONS
------------------------------------------------------------------------*/
function showToast(msg) {
  const c = document.getElementById('toast-container');
  const t = Object.assign(document.createElement('div'), {className:'toast', textContent:msg});
  c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => c.removeChild(t), 800); }, 3000);
}

/* 6) SAVE INVOICE (inline form)
------------------------------------------------------------------------*/
async function saveInvoice(e, id) {
  e.preventDefault();
  try {
    const fd = new FormData(e.target);
    const r  = await fetch(`/edit/${id}`, {method:'POST', body:fd});
    if (!r.ok) throw new Error();
    const d  = await r.json();
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

/* 7) DEMO TYPE-EFFECT + TERMINAL PRINT
------------------------------------------------------------------------*/
function typeInto(el, txt, speed = 60, cb) {
  let i = 0; (function t() {
    if (i < txt.length) {
      ('placeholder' in el) ? el.placeholder = txt.slice(0, ++i) : el.value += txt[i++ - 1];
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
  const io = new IntersectionObserver(e => { if (e[0].isIntersecting) { cb(); io.disconnect(); } }, {threshold:0.5});
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

/* 8) CRACK-GLASS DEMO (card click)
------------------------------------------------------------------------*/
function crackGlass() {
  const block = document.getElementById('ai-extract-block');
  const cv    = document.getElementById('crack-effect');
  if (!cv) return;
  const ctx = cv.getContext('2d'); if (!ctx) return;

  cv.style.display='block'; cv.style.opacity='1'; cv.style.transition='';
  cv.width=300; cv.height=200; ctx.clearRect(0,0,cv.width,cv.height);

  navigator.vibrate?.([100,50,100]);
  block.classList.add('vibrate');
  setTimeout(()=>block.classList.remove('vibrate'), 700);

  const [cx, cy] = [cv.width/2, cv.height/2];
  for(let i=0;i<24;i++){
    const ang=i*Math.PI*2/24,len=50+Math.random()*100;
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx+Math.cos(ang)*len, cy+Math.sin(ang)*len);
    ctx.strokeStyle=`rgba(100,100,100,${0.5+Math.random()*0.5})`;
    ctx.lineWidth=0.5+Math.random()*2; ctx.stroke();
  }
  for(let r=20;r<=100;r+=20){
    ctx.beginPath(); ctx.arc(cx,cy,r+Math.random()*5,0,Math.PI*2);
    ctx.strokeStyle=`rgba(120,120,120,${Math.random()*0.4+0.3})`;
    ctx.lineWidth=0.3+Math.random(); ctx.stroke();
  }
  setTimeout(()=>{cv.style.transition='opacity .8s';cv.style.opacity='0';
    setTimeout(()=>{cv.style.display='none';cv.style.transition='';},800);
  },1300);
}
document.getElementById('ai-extract-block')?.addEventListener('click', crackGlass);

/* 9) LIGHT / NIGHT THEME
------------------------------------------------------------------------*/
(() => {
  const hr = new Date().getHours();
  if (hr >= 19 || hr < 6) {
    document.body.classList.add('night-theme');
    document.getElementById('star-background')?.style.setProperty('display','block');
  }
})();
document.getElementById('theme-toggle')?.addEventListener('click', () => {
  const body  = document.body;
  const stars = document.getElementById('star-background');
  const night = body.classList.toggle('night-theme');
  if (stars) stars.style.display = night ? 'block' : 'none';
  localStorage.setItem('theme', night ? 'night' : 'light');
});
