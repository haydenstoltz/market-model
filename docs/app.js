async function loadPayload() {
  const response = await fetch('./data/market.json', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load market.json (${response.status})`);
  }
  return response.json();
}

function fmt(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '—';
    const abs = Math.abs(value);
    if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    if (abs >= 1) return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
    return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }
  return String(value);
}

function titleize(key) {
  return String(key).replace(/_/g, ' ');
}

function makeKvCard(title, record) {
  const card = document.createElement('article');
  card.className = 'card';
  const h3 = document.createElement('h3');
  h3.textContent = title;
  card.appendChild(h3);

  if (!record || Object.keys(record).length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No data.';
    card.appendChild(empty);
    return card;
  }

  const dl = document.createElement('div');
  dl.className = 'kv';
  Object.entries(record).forEach(([key, value]) => {
    const k = document.createElement('div');
    k.className = 'k';
    k.textContent = titleize(key);
    const v = document.createElement('div');
    v.textContent = fmt(value);
    dl.appendChild(k);
    dl.appendChild(v);
  });
  card.appendChild(dl);
  return card;
}

function renderTable(targetId, rows) {
  const root = document.getElementById(targetId);
  root.innerHTML = '';

  if (!rows || rows.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No data.';
    root.appendChild(empty);
    return;
  }

  const columns = Array.from(rows.reduce((set, row) => {
    Object.keys(row).forEach((k) => set.add(k));
    return set;
  }, new Set()));

  const wrap = document.createElement('div');
  wrap.className = 'table-wrap';
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const tbody = document.createElement('tbody');

  const headerRow = document.createElement('tr');
  columns.forEach((col) => {
    const th = document.createElement('th');
    th.textContent = titleize(col);
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement('tr');
    columns.forEach((col) => {
      const td = document.createElement('td');
      td.textContent = fmt(row[col]);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  wrap.appendChild(table);
  root.appendChild(wrap);
}

function renderHero(payload) {
  const root = document.getElementById('hero-meta');
  root.innerHTML = '';
  root.appendChild(makeKvCard('Payload status', {
    status: payload.status,
    generated_at_utc: payload.generated_at_utc,
    sources_loaded: (payload.available_sources || []).length,
    warnings: (payload.warnings || []).length,
  }));

  const messageCard = document.createElement('article');
  messageCard.className = 'meta-card';
  const h3 = document.createElement('h3');
  h3.textContent = 'Message';
  const p = document.createElement('p');
  p.className = 'meta-line';
  p.textContent = payload.message || '—';
  messageCard.appendChild(h3);
  messageCard.appendChild(p);
  root.appendChild(messageCard);
}

function renderPredictionCards(payload) {
  const root = document.getElementById('prediction-cards');
  root.innerHTML = '';
  const cards = payload.latest_predictions_by_horizon || [];
  if (cards.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No predictions exported yet.';
    root.appendChild(empty);
    return;
  }
  cards.forEach((record) => {
    const horizon = record.horizon === undefined ? 'unknown' : record.horizon;
    root.appendChild(makeKvCard(`Horizon ${horizon}`, record));
  });
}

function renderPositionCards(payload) {
  const root = document.getElementById('position-cards');
  root.innerHTML = '';
  const positions = payload.latest_positions || {};
  Object.entries(positions).forEach(([name, record]) => {
    root.appendChild(makeKvCard(titleize(name), record));
  });
  if (root.children.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No position files exported yet.';
    root.appendChild(empty);
  }
}

function renderCharts(payload) {
  const root = document.getElementById('chart-grid');
  root.innerHTML = '';
  const charts = payload.charts || [];
  if (charts.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = 'No chart assets copied yet.';
    root.appendChild(empty);
    return;
  }

  charts.forEach((chart) => {
    const card = document.createElement('article');
    card.className = 'chart-card';
    const h3 = document.createElement('h3');
    h3.textContent = chart.title || chart.path;
    const img = document.createElement('img');
    img.src = `./${chart.path}`;
    img.alt = chart.title || chart.path;
    card.appendChild(h3);
    card.appendChild(img);
    root.appendChild(card);
  });
}

async function main() {
  try {
    const payload = await loadPayload();
    renderHero(payload);
    renderPredictionCards(payload);
    renderPositionCards(payload);
    renderCharts(payload);

    const tables = payload.tables || {};
    renderTable('strategy-run-summary', tables.strategy_run_summary || []);
    renderTable('baseline-horizon-summary', tables.strategy_baseline_horizon_summary || []);
    renderTable('h3-confirmation-summary', tables.strategy_h3_confirmation_summary || []);
    renderTable('ridge-coef-summary', tables.ridge_coef_summary || []);
    renderTable('recent-predictions', tables.recent_predictions || []);
  } catch (error) {
    const root = document.getElementById('hero-meta');
    root.innerHTML = '';
    root.appendChild(makeKvCard('Load error', { error: error.message }));
  }
}

main();
