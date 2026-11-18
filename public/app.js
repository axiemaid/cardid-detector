// app.js — Static frontend for Card ID Detector
// All processing in a Web Worker. No server, no upload.

const dropzone = document.getElementById('dropzone');
const fileinput = document.getElementById('fileinput');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const loadingSteps = document.getElementById('loading-steps');
const preview = document.getElementById('preview');
const origimg = document.getElementById('origimg');
const cropimg = document.getElementById('cropimg');
const cropCol = document.getElementById('crop-col');
const resultsDiv = document.getElementById('results');
const verdictDiv = document.getElementById('verdict');
const supported = document.getElementById('supported');
const supportedText = document.getElementById('supported-text');
const supportedCodes = document.getElementById('supported-codes');

let worker = null;
let refsData = null;

// ── Load refs.json ────────────────────────────────────────
fetch('refs.json')
  .then(r => r.json())
  .then(data => {
    refsData = data.cards;
    supported.style.display = 'block';
    supportedText.textContent = `${data.count} of 42 Hyper Battle cards are in the database. A "no confident match" may mean the card hasn't been added yet.`;
    supportedCodes.innerHTML = refsData.map(c => `<span class="code-badge">${c.code}</span>`).join('');
  })
  .catch(() => {
    supported.style.display = 'block';
    supportedText.textContent = 'Could not load reference database.';
  });

// ── Worker setup ───────────────────────────────────────────
function getWorker() {
  if (!worker) {
    worker = new Worker('worker.js', { type: 'module' });
  }
  return worker;
}

// ── File handling ──────────────────────────────────────────
dropzone.addEventListener('click', () => fileinput.click());

dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileinput.addEventListener('change', e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function setStep(state, text) {
  loadingSteps.innerHTML = `<div class="step ${state}">${text}</div>`;
}

function setLoading(visible, text) {
  loading.style.display = visible ? 'block' : 'none';
  if (text) loadingText.textContent = text;
  if (!visible) loadingSteps.innerHTML = '';
}

async function handleFile(file) {
  // Show original preview
  const reader = new FileReader();
  reader.onload = e => {
    origimg.src = e.target.result;
    cropimg.src = '';
    cropCol.style.display = 'none';
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);

  // Reset UI
  resultsDiv.innerHTML = '';
  verdictDiv.style.display = 'none';
  setLoading(true, 'Processing...');

  // Send to worker
  const w = getWorker();

  // Convert file to bitmap for transfer
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  w.postMessage({
    type: 'match',
    imageData: imageData.data,
    width: canvas.width,
    height: canvas.height,
    refs: refsData,
  }, [imageData.data.buffer]);

  w.onmessage = (e) => {
    const { type, ...data } = e.data;
    if (type === 'progress') {
      setStep('active', data.message);
    } else if (type === 'step_done') {
      setStep('done', data.message);
    } else if (type === 'crop_preview') {
      // Show cropped image
      const blob = new Blob([data.data], { type: 'image/png' });
      cropimg.src = URL.createObjectURL(blob);
      cropCol.style.display = 'flex';
    } else if (type === 'result') {
      setLoading(false);
      displayResults(data.results);
      displayVerdict(data.verdict);
    } else if (type === 'error') {
      setLoading(false);
      resultsDiv.innerHTML = `<div style="color: var(--red); padding: 12px;">⚠ ${data.message}</div>`;
    }
  };
}

function displayResults(results) {
  if (!results.length) return;
  const html = results.slice(0, 10).map((r, i) => {
    const isBest = i === 0;
    const pct = Math.round(r.score * 100);
    const cls = pct > 70 ? 'high' : pct > 40 ? 'med' : 'low';
    return `
      <div class="result-row ${isBest ? 'best' : ''}">
        <div class="result-code">${r.code}</div>
        <div class="result-bar-wrap">
          <div class="result-bar ${cls}" style="width: ${pct}%"></div>
        </div>
        <div class="result-stats">H${r.hamming} · hist ${r.hist} · SSIM ${r.ssim}</div>
      </div>
    `;
  }).join('');
  resultsDiv.innerHTML = html;
}

function displayVerdict(v) {
  let cls, text;
  if (v.verdict === 'match') {
    cls = 'match';
    text = `✅ Match: ${v.code} (Hamming ${v.hamming})`;
  } else if (v.verdict === 'likely') {
    cls = 'likely';
    text = `🟡 Likely: ${v.code} (Hamming ${v.hamming}) — verify manually`;
  } else {
    cls = 'nomatch';
    text = v.code
      ? `❌ No confident match (closest: ${v.code} @ Hamming ${v.hamming})`
      : `❌ No match found`;
  }
  verdictDiv.className = 'verdict ' + cls;
  verdictDiv.textContent = text;
  verdictDiv.style.display = 'block';
}
