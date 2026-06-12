const state = {
  k: 5,
  train: [],
  test: [],
  datasetSummary: null,
  previewDataUrl: null,
  selectedImageFile: null,
  trainPage: 1,
  testPage: 1,
  pageSize: 8,
  featureChart: null,
  kChart: null
};

function switchTab(panelId, tabElement) {
  document.querySelectorAll('.tab').forEach((tab) => tab.classList.remove('active'));
  document.querySelectorAll('.panel').forEach((panel) => panel.classList.remove('active'));
  tabElement.classList.add('active');
  document.getElementById(`panel-${panelId}`).classList.add('active');
  if (panelId === 'dataset') {
    fetchDataset();
  }
  if (panelId === 'eval') {
    runEval(state.k);
  }
}

function setK(value) {
  state.k = value;
  document.getElementById('st-k').textContent = value;
  document.querySelectorAll('#k-selector .k-btn').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.textContent.replace(/\D/g, '')) === value);
  });
  document.querySelectorAll('#panel-eval .k-btn').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.textContent.replace(/\D/g, '')) === value);
  });
  document.getElementById('btn-upload-predict').textContent = `🔍 Analisis Gambar (K=${value})`;
  document.getElementById('btn-upload-predict').style.display = state.selectedImageFile ? 'inline-flex' : 'none';
  if (document.getElementById('panel-eval').classList.contains('active')) {
    runEval(value);
  }
}

function syncInput(name, sliderId) {
  const value = document.getElementById(sliderId).value;
  document.getElementById(`num-${name}`).value = value;
}

function syncSlider(name, numId) {
  const value = document.getElementById(numId).value;
  document.getElementById(`sl-${name}`).value = value;
}

function dragOver(event) {
  event.preventDefault();
  document.getElementById('upload-area').classList.add('dragover');
}

function dragLeave() {
  document.getElementById('upload-area').classList.remove('dragover');
}

function dropFile(event) {
  event.preventDefault();
  document.getElementById('upload-area').classList.remove('dragover');
  const file = event.dataTransfer.files[0];
  if (file) {
    handleFile(file);
  }
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Mohon unggah file gambar JPG atau PNG.');
    return;
  }
  state.selectedImageFile = file;
  const reader = new FileReader();
  reader.onload = (evt) => {
    state.previewDataUrl = evt.target.result;
    const preview = document.getElementById('preview-img');
    preview.src = state.previewDataUrl;
    preview.style.display = 'block';
    document.getElementById('upload-inner').style.display = 'none';
    document.getElementById('btn-upload-predict').style.display = 'inline-flex';
    document.getElementById('btn-reset').style.display = 'inline-flex';
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  state.selectedImageFile = null;
  state.previewDataUrl = null;
  document.getElementById('preview-img').style.display = 'none';
  document.getElementById('preview-img').src = '';
  document.getElementById('upload-inner').style.display = 'block';
  document.getElementById('file-input').value = '';
  document.getElementById('btn-upload-predict').style.display = 'none';
  document.getElementById('btn-reset').style.display = 'none';
  document.getElementById('result-box').style.display = 'none';
}

function formatNumber(value, digits = 4) {
  return typeof value === 'number' ? value.toFixed(digits) : String(value);
}

function showAppNotice(message) {
  const notice = document.getElementById('app-notice');
  notice.textContent = message;
  notice.style.display = 'block';
}

function clearAppNotice() {
  const notice = document.getElementById('app-notice');
  notice.textContent = '';
  notice.style.display = 'none';
}

function showResult(data) {
  const box = document.getElementById('result-box');
  const emojiElement = document.getElementById('result-emoji');
  const labelElement = document.getElementById('result-label');
  const confElement = document.getElementById('result-conf');
  const featsElement = document.getElementById('result-feats');
  const resultImgWrap = document.getElementById('result-img-wrap');

  box.classList.remove('ripe', 'unripe');
  box.classList.add(data.label === 'ripe' ? 'ripe' : 'unripe');
  emojiElement.textContent = data.label === 'ripe' ? '🍅' : '🟢';
  labelElement.textContent = data.label === 'ripe' ? 'Ripe (Matang)' : 'Unripe (Belum Matang)';
  confElement.textContent = `Confidence: ${Math.round(data.confidence * 100)}%`;
  featsElement.innerHTML = `Contrast: ${formatNumber(data.features.Contrast)} · Correlation: ${formatNumber(data.features.Correlation, 5)} · Energy: ${formatNumber(data.features.Energy, 5)} · Homogeneity: ${formatNumber(data.features.Homogeneity, 5)}`;

  if (state.previewDataUrl) {
    resultImgWrap.innerHTML = `<img src="${state.previewDataUrl}" style="max-width: 220px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);">`;
  } else {
    resultImgWrap.innerHTML = '';
  }

  const neighbors = data.nearest.map((item, idx) => `#${idx + 1}: ${item.label} (jarak ${item.distance.toFixed(3)})`).join('<br>');
  document.getElementById('result-neighbors').innerHTML = `<strong>Tetangga terdekat:</strong><br>${neighbors}`;
  box.style.display = 'block';
}

async function predictFromImage() {
  if (!state.selectedImageFile) {
    alert('Silakan unggah gambar tomat terlebih dahulu.');
    return;
  }
  const form = new FormData();
  form.append('image', state.selectedImageFile);
  form.append('k', state.k);
  try {
    clearAppNotice();
    const response = await fetch('/api/predict/image', { method: 'POST', body: form });
    const result = await response.json();
    if (!result.success) {
      showAppNotice(result.error || 'Terjadi kesalahan saat memprediksi gambar.');
      return;
    }
    showResult(result.result);
  } catch (error) {
    showAppNotice('Tidak dapat menghubungi server. Pastikan Flask server berjalan.');
    console.error(error);
  }
}

async function predictManual() {
  const contrast = parseFloat(document.getElementById('num-contrast').value);
  const correlation = parseFloat(document.getElementById('num-correlation').value);
  const energy = parseFloat(document.getElementById('num-energy').value);
  const homogeneity = parseFloat(document.getElementById('num-homogeneity').value);
  if (Number.isNaN(contrast) || Number.isNaN(correlation) || Number.isNaN(energy) || Number.isNaN(homogeneity)) {
    alert('Isi semua nilai fitur GLCM terlebih dahulu.');
    return;
  }
  try {
    clearAppNotice();
    const response = await fetch('/api/predict/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        Contrast: contrast,
        Correlation: correlation,
        Energy: energy,
        Homogeneity: homogeneity,
        k: state.k
      })
    });
    const result = await response.json();
    if (!result.success) {
      showAppNotice(result.error || 'Terjadi kesalahan saat memprediksi input manual.');
      return;
    }
    state.previewDataUrl = null;
    showResult(result.result);
  } catch (error) {
    showAppNotice('Tidak dapat menghubungi server. Pastikan Flask server berjalan.');
    console.error(error);
  }
}

function loadExample(type) {
  const examples = {
    ripe1: { Contrast: 14.2, Correlation: 0.9978, Energy: 0.0321, Homogeneity: 0.4552 },
    ripe2: { Contrast: 10.1, Correlation: 0.9983, Energy: 0.0358, Homogeneity: 0.4821 },
    unripe1: { Contrast: 42.3, Correlation: 0.9920, Energy: 0.0214, Homogeneity: 0.3450 },
    unripe2: { Contrast: 33.8, Correlation: 0.9942, Energy: 0.0252, Homogeneity: 0.3704 }
  };
  const example = examples[type];
  if (!example) return;
  document.getElementById('sl-contrast').value = example.Contrast;
  document.getElementById('num-contrast').value = example.Contrast;
  document.getElementById('sl-correlation').value = example.Correlation;
  document.getElementById('num-correlation').value = example.Correlation;
  document.getElementById('sl-energy').value = example.Energy;
  document.getElementById('num-energy').value = example.Energy;
  document.getElementById('sl-homogeneity').value = example.Homogeneity;
  document.getElementById('num-homogeneity').value = example.Homogeneity;
  state.previewDataUrl = null;
  predictManual();
}

function renderPagination(containerId, currentPage, totalPages, onPage) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  if (totalPages <= 1) return;
  for (let idx = 1; idx <= totalPages; idx += 1) {
    const btn = document.createElement('button');
    btn.className = 'page-btn' + (idx === currentPage ? ' active' : '');
    btn.textContent = idx;
    btn.onclick = () => onPage(idx);
    container.appendChild(btn);
  }
}

function filterRows(rows, searchTerm, filterValue, type) {
  const normalized = searchTerm.trim().toLowerCase();
  return rows.filter((row) => {
    const rowText = `${row.Contrast} ${row.Correlation} ${row.Energy} ${row.Homogeneity} ${row.Label || ''}`.toLowerCase();
    if (normalized && !rowText.includes(normalized)) return false;
    if (!filterValue || filterValue === 'all') return true;
    if (type === 'train') return row.Label === filterValue;
    if (type === 'test') {
      return filterValue === 'correct' ? row.status === 'correct' : filterValue === 'wrong' ? row.status === 'wrong' : true;
    }
    return true;
  });
}

function renderTrainTable() {
  const searchTerm = document.getElementById('search-train').value;
  const filter = document.getElementById('filter-train').value;
  const filtered = filterRows(state.train, searchTerm, filter, 'train');
  const totalPages = Math.max(1, Math.ceil(filtered.length / state.pageSize));
  state.trainPage = Math.min(state.trainPage, totalPages);
  const start = (state.trainPage - 1) * state.pageSize;
  const pageRows = filtered.slice(start, start + state.pageSize);
  const tbody = document.getElementById('train-tbody');
  tbody.innerHTML = pageRows.map((row, index) => `
    <tr>
      <td>${start + index + 1}</td>
      <td>${formatNumber(row.Contrast)}</td>
      <td>${formatNumber(row.Correlation, 5)}</td>
      <td>${formatNumber(row.Energy, 5)}</td>
      <td>${formatNumber(row.Homogeneity, 5)}</td>
      <td><span class="chip ${row.Label}">${row.Label}</span></td>
    </tr>
  `).join('');
  renderPagination('train-pagination', state.trainPage, totalPages, (page) => {
    state.trainPage = page;
    renderTrainTable();
  });
}

function renderTestTable() {
  const searchTerm = document.getElementById('search-test').value;
  const filter = document.getElementById('filter-test').value;
  const filtered = filterRows(state.test, searchTerm, filter, 'test');
  const totalPages = Math.max(1, Math.ceil(filtered.length / state.pageSize));
  state.testPage = Math.min(state.testPage, totalPages);
  const start = (state.testPage - 1) * state.pageSize;
  const pageRows = filtered.slice(start, start + state.pageSize);
  const tbody = document.getElementById('test-tbody');
  tbody.innerHTML = pageRows.map((row, index) => `
    <tr>
      <td>${start + index + 1}</td>
      <td>${formatNumber(row.Contrast)}</td>
      <td>${formatNumber(row.Correlation, 5)}</td>
      <td>${formatNumber(row.Energy, 5)}</td>
      <td>${formatNumber(row.Homogeneity, 5)}</td>
      <td><span class="chip ${row.Label}">${row.Label}</span></td>
      <td><span class="chip ${row.prediction}">${row.prediction}</span></td>
      <td><span class="chip ${row.status}">${row.status === 'correct' ? 'Benar' : 'Salah'}</span></td>
    </tr>
  `).join('');
  renderPagination('test-pagination', state.testPage, totalPages, (page) => {
    state.testPage = page;
    renderTestTable();
  });
}

function updateDatasetOverview(summary) {
  const trainTotal = document.getElementById('train-total-val');
  const testTotal = document.getElementById('test-total-val');
  const trainRipeBar = document.getElementById('train-ripe-bar');
  const trainUnripeBar = document.getElementById('train-unripe-bar');
  const testRipeBar = document.getElementById('test-ripe-bar');
  const testUnripeBar = document.getElementById('test-unripe-bar');
  const trainRipeVal = document.getElementById('train-ripe-val');
  const trainUnripeVal = document.getElementById('train-unripe-val');
  const testRipeVal = document.getElementById('test-ripe-val');
  const testUnripeVal = document.getElementById('test-unripe-val');

  trainTotal.textContent = summary.total_train;
  testTotal.textContent = summary.total_test;

  const trainRipePct = summary.train_distribution.ripe === 0 ? 0 : Math.round((summary.train_distribution.ripe / summary.total_train) * 100);
  const trainUnripePct = summary.train_distribution.unripe === 0 ? 0 : Math.round((summary.train_distribution.unripe / summary.total_train) * 100);
  const testRipePct = summary.test_distribution.ripe === 0 ? 0 : Math.round((summary.test_distribution.ripe / summary.total_test) * 100);
  const testUnripePct = summary.test_distribution.unripe === 0 ? 0 : Math.round((summary.test_distribution.unripe / summary.total_test) * 100);

  trainRipeBar.style.width = `${trainRipePct}%`;
  trainUnripeBar.style.width = `${trainUnripePct}%`;
  testRipeBar.style.width = `${testRipePct}%`;
  testUnripeBar.style.width = `${testUnripePct}%`;
  trainRipeVal.textContent = summary.train_distribution.ripe;
  trainUnripeVal.textContent = summary.train_distribution.unripe;
  testRipeVal.textContent = summary.test_distribution.ripe;
  testUnripeVal.textContent = summary.test_distribution.unripe;
}

function buildConfusionMatrixHTML(cm) {
  return `
    <div class="cm-header"></div>
    <div class="cm-header">Prediksi Ripe</div>
    <div class="cm-header">Prediksi Unripe</div>
    <div class="cm-label">Actual Ripe</div>
    <div class="cm-cell cm-tp"><div class="cm-val">${cm[0][0]}</div><div class="cm-pct">Benar</div></div>
    <div class="cm-cell cm-fn"><div class="cm-val">${cm[0][1]}</div><div class="cm-pct">Salah</div></div>
    <div class="cm-label">Actual Unripe</div>
    <div class="cm-cell cm-fp"><div class="cm-val">${cm[1][0]}</div><div class="cm-pct">Salah</div></div>
    <div class="cm-cell cm-tn"><div class="cm-val">${cm[1][1]}</div><div class="cm-pct">Benar</div></div>
  `;
}

function renderMetrics(metrics) {
  document.getElementById('metric-list').innerHTML = `
    <li><span class="metric-name">Akurasi</span><span class="metric-val">${metrics.accuracy.toFixed(2)}%</span></li>
    <li><span class="metric-name">Precision Ripe</span><span class="metric-val">${metrics.precision_ripe.toFixed(2)}%</span></li>
    <li><span class="metric-name">Recall Ripe</span><span class="metric-val">${metrics.recall_ripe.toFixed(2)}%</span></li>
    <li><span class="metric-name">Precision Unripe</span><span class="metric-val">${metrics.precision_unripe.toFixed(2)}%</span></li>
    <li><span class="metric-name">Recall Unripe</span><span class="metric-val">${metrics.recall_unripe.toFixed(2)}%</span></li>
  `;
}

function createFeatureChart(summary) {
  const ctx = document.getElementById('featureChart').getContext('2d');
  const labels = Object.keys(summary.ripe);
  const ripeValues = labels.map((key) => summary.ripe[key]);
  const unripeValues = labels.map((key) => summary.unripe[key]);
  if (state.featureChart) state.featureChart.destroy();
  state.featureChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Ripe',
          data: ripeValues,
          backgroundColor: 'rgba(63,185,80,0.7)'
        },
        {
          label: 'Unripe',
          data: unripeValues,
          backgroundColor: 'rgba(248,81,73,0.7)'
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#c9d1d9' } } },
      scales: {
        x: { ticks: { color: '#c9d1d9' } },
        y: { beginAtZero: true, ticks: { color: '#c9d1d9' } }
      }
    }
  });
}

function createKChart(kCurve) {
  const ctx = document.getElementById('kChart').getContext('2d');
  const labels = kCurve.map((item) => `K=${item.k}`);
  const data = kCurve.map((item) => item.accuracy.toFixed(2));
  if (state.kChart) state.kChart.destroy();
  state.kChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Akurasi (%)',
        data,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.2)',
        fill: true,
        tension: 0.3,
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#c9d1d9' } } },
      scales: {
        x: { ticks: { color: '#c9d1d9' } },
        y: { beginAtZero: true, ticks: { color: '#c9d1d9' }, suggestedMax: 100 }
      }
    }
  });
}

async function fetchDataset() {
  if (state.train.length && state.test.length) return;
  try {
    clearAppNotice();
    const response = await fetch('/api/dataset');
    const data = await response.json();
    state.datasetSummary = data.summary;
    state.train = data.train;
    state.test = data.test.map((row) => ({ ...row, prediction: row.Label, status: 'correct' }));
    if (data.predictions) {
      document.getElementById('st-acc').textContent = `${data.predictions.accuracy.toFixed(2)}%`;
    }
    updateDatasetOverview(data.summary);
    renderTrainTable();
    renderTestTable();
  } catch (error) {
    showAppNotice('Tidak dapat memuat dataset. Pastikan aplikasi dijalankan lewat server Flask.');
    console.error(error);
  }
}

async function runEval(k) {
  setK(k);
  try {
    clearAppNotice();
    const response = await fetch(`/api/eval?k=${k}`);
    const data = await response.json();
    if (!data.success) {
      showAppNotice(data.error || 'Gagal mengambil data evaluasi.');
      return;
    }
  const evalData = data.evaluation;
  document.getElementById('eval-acc').textContent = `${evalData.accuracy.toFixed(2)}`;
  document.getElementById('st-acc').textContent = `${evalData.accuracy.toFixed(2)}%`;
  document.getElementById('st-k').textContent = evalData.k;
  document.getElementById('cm-grid').innerHTML = buildConfusionMatrixHTML(evalData.confusion_matrix);
  renderMetrics(evalData.metrics);
  createFeatureChart(evalData.feature_summary);
  createKChart(evalData.k_curve);
  } catch (error) {
    showAppNotice('Tidak dapat mengambil data evaluasi. Pastikan Flask server sedang berjalan.');
    console.error(error);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  setK(5);
  if (window.location.protocol === 'file:') {
    showAppNotice('Halaman dibuka langsung dari file. Jalankan Flask server dan buka lewat http://127.0.0.1:5000');
    return;
  }
  fetchDataset();
  runEval(5);
});
