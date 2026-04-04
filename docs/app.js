const PALETTE = [
  '#73b7ff',
  '#2ec27e',
  '#f2b84b',
  '#ff7f9f',
  '#9a8cff',
  '#6de5ff',
  '#ffa95e',
];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  const num = Number(value);
  if (!Number.isFinite(num)) return 'N/A';
  return num.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function fmtPercent(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'N/A';
  const num = Number(value) * 100;
  return `${num.toFixed(digits)}%`;
}

function fmtDate(value) {
  if (!value) return 'N/A';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toISOString().slice(0, 10);
}

function titleize(value) {
  return String(value || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

async function loadPayload() {
  const response = await fetch('./data/market.json', { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to load market.json (${response.status})`);
  }
  return response.json();
}

function renderHeroMeta(payload) {
  const root = document.getElementById('hero-meta');
  root.innerHTML = '';

  const meta = payload.snapshot_metadata || {};
  const statusCard = el('article', 'hero-meta-card');
  statusCard.appendChild(el('h3', null, 'Payload Status'));
  [
    ['Status', payload.status || 'unknown'],
    ['Generated', fmtDate(payload.generated_at_utc)],
    ['Sources', fmtNumber((payload.available_sources || []).length, 0)],
    ['Warnings', fmtNumber((payload.warnings || []).length, 0)],
  ].forEach(([k, v]) => {
    const row = el('div', 'hero-meta-row');
    row.appendChild(el('span', 'label', k));
    row.appendChild(el('span', null, String(v)));
    statusCard.appendChild(row);
  });
  root.appendChild(statusCard);

  const modelCard = el('article', 'hero-meta-card');
  modelCard.appendChild(el('h3', null, 'Model Snapshot'));
  [
    ['Version', meta.model_version || 'N/A'],
    ['Payload', meta.payload_version || 'N/A'],
    ['Message', payload.message || 'N/A'],
  ].forEach(([k, v]) => {
    const row = el('div', 'hero-meta-row');
    row.appendChild(el('span', 'label', k));
    row.appendChild(el('span', null, String(v)));
    modelCard.appendChild(row);
  });
  root.appendChild(modelCard);
}

function kvCard(title, rows) {
  const card = el('article', 'snapshot-card');
  card.appendChild(el('h3', null, title));

  const grid = el('div', 'kv-grid');
  rows.forEach(([k, v, kind]) => {
    const row = el('div', 'kv-row');
    row.appendChild(el('span', 'key', k));
    if (kind === 'pill') {
      const pill = el('span', `pill ${v === 'INVEST' || v === 'PASS' ? 'good' : v === 'CASH' || v === 'FAIL' ? 'bad' : ''}`, String(v));
      row.appendChild(pill);
    } else {
      row.appendChild(el('span', 'value', String(v)));
    }
    grid.appendChild(row);
  });
  card.appendChild(grid);
  return card;
}

function renderSnapshots(payload) {
  const root = document.getElementById('snapshot-cards');
  root.innerHTML = '';

  const meta = payload.snapshot_metadata || {};
  const latestPred = meta.latest_prediction_state || {};
  const best = meta.best_strategy || {};
  const positions = payload.latest_positions || {};

  root.appendChild(
    kvCard('Latest Prediction State', [
      ['Horizon', latestPred.horizon !== undefined ? `h${latestPred.horizon}` : 'N/A'],
      ['Date', fmtDate(latestPred.date)],
      ['y_pred', fmtNumber(latestPred.y_pred, 6)],
      ['y_true', fmtNumber(latestPred.y_true, 6)],
      ['Signal', Number(latestPred.signal || 0) > 0 ? 'INVEST' : 'CASH', 'pill'],
    ])
  );

  const bestRisk = best.best_risk_adjusted || {};
  root.appendChild(
    kvCard('Best Risk-Adjusted', [
      ['Strategy', bestRisk.strategy || bestRisk.strategy_type || (bestRisk.horizon ? `h${bestRisk.horizon}` : 'N/A')],
      ['Sharpe', fmtNumber(bestRisk.Sharpe_strat ?? bestRisk.Sharpe, 3)],
      ['CAGR', fmtPercent(bestRisk.CAGR_strat ?? bestRisk.CAGR)],
      ['Max DD', fmtPercent(bestRisk.max_drawdown_strat ?? bestRisk.max_drawdown)],
    ])
  );

  const bestGrowth = best.best_growth || {};
  root.appendChild(
    kvCard('Best Growth', [
      ['Strategy', bestGrowth.strategy || bestGrowth.strategy_type || (bestGrowth.horizon ? `h${bestGrowth.horizon}` : 'N/A')],
      ['Final Equity', fmtNumber(bestGrowth.final_equity_strat ?? bestGrowth.final_equity, 3)],
      ['CAGR', fmtPercent(bestGrowth.CAGR_strat ?? bestGrowth.CAGR)],
      ['Sharpe', fmtNumber(bestGrowth.Sharpe_strat ?? bestGrowth.Sharpe, 3)],
    ])
  );

  const h3Position = positions.strategy_h1 || {};
  root.appendChild(
    kvCard('Latest Baseline Position', [
      ['Date', fmtDate(h3Position.date)],
      ['Weight', fmtNumber(h3Position.weight, 3)],
      ['Turnover', fmtNumber(h3Position.turnover, 3)],
      ['Target', h3Position.target_state || 'N/A', 'pill'],
    ])
  );
}

function createEmpty(container, message) {
  container.innerHTML = '';
  const box = el('div', 'chart-empty', message);
  container.appendChild(box);
}

function parseSeries(series) {
  const points = [];
  (series || []).forEach((row) => {
    const xRaw = row.date || row.x || row.timestamp;
    const x = new Date(xRaw).getTime();
    const y = Number(row.value ?? row.y ?? row.rolling_hit_rate ?? row.y_pred ?? row.y_true);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      points.push({ x, y, label: fmtDate(xRaw) });
    }
  });
  return points.sort((a, b) => a.x - b.x);
}

function buildPath(points, sx, sy) {
  if (!points.length) return '';
  return points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`)
    .join(' ');
}

function renderLineChart(targetId, options) {
  const host = document.getElementById(targetId);
  if (!host) return;
  host.innerHTML = '';

  const series = (options.series || [])
    .map((s, idx) => ({
      label: s.label || `Series ${idx + 1}`,
      color: s.color || PALETTE[idx % PALETTE.length],
      points: parseSeries(s.data),
    }))
    .filter((s) => s.points.length > 0);

  if (!series.length) {
    createEmpty(host, 'No chart data available.');
    return;
  }

  const width = Math.max(host.clientWidth || 920, 480);
  const height = options.height || 420;
  const margin = { top: 20, right: 20, bottom: 40, left: 60 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const all = series.flatMap((s) => s.points);
  const xMin = Math.min(...all.map((p) => p.x));
  const xMax = Math.max(...all.map((p) => p.x));
  let yMin = Math.min(...all.map((p) => p.y));
  let yMax = Math.max(...all.map((p) => p.y));

  if (options.includeZero) {
    yMin = Math.min(yMin, 0);
    yMax = Math.max(yMax, 0);
  }
  if (yMin === yMax) {
    const pad = Math.abs(yMin) < 1 ? 1 : Math.abs(yMin) * 0.2;
    yMin -= pad;
    yMax += pad;
  }

  const sx = (x) => margin.left + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const sy = (y) => margin.top + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'svg-chart');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', options.ariaLabel || 'time series chart');

  const yTicks = 5;
  for (let i = 0; i <= yTicks; i += 1) {
    const yVal = yMin + ((yMax - yMin) * i) / yTicks;
    const yPx = sy(yVal);
    const grid = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    grid.setAttribute('x1', margin.left);
    grid.setAttribute('x2', width - margin.right);
    grid.setAttribute('y1', yPx);
    grid.setAttribute('y2', yPx);
    grid.setAttribute('class', 'grid-line');
    svg.appendChild(grid);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', margin.left - 8);
    label.setAttribute('y', yPx + 4);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('class', 'axis-text');
    label.textContent = options.yFormat ? options.yFormat(yVal) : fmtNumber(yVal, 3);
    svg.appendChild(label);
  }

  const xTicks = 6;
  for (let i = 0; i <= xTicks; i += 1) {
    const xVal = xMin + ((xMax - xMin) * i) / xTicks;
    const xPx = sx(xVal);
    const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    tick.setAttribute('x1', xPx);
    tick.setAttribute('x2', xPx);
    tick.setAttribute('y1', height - margin.bottom);
    tick.setAttribute('y2', height - margin.bottom + 5);
    tick.setAttribute('class', 'axis-line');
    svg.appendChild(tick);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', xPx);
    label.setAttribute('y', height - margin.bottom + 18);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('class', 'axis-text');
    label.textContent = new Date(xVal).toISOString().slice(0, 7);
    svg.appendChild(label);
  }

  const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  xAxis.setAttribute('x1', margin.left);
  xAxis.setAttribute('x2', width - margin.right);
  xAxis.setAttribute('y1', height - margin.bottom);
  xAxis.setAttribute('y2', height - margin.bottom);
  xAxis.setAttribute('class', 'axis-line');
  svg.appendChild(xAxis);

  const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  yAxis.setAttribute('x1', margin.left);
  yAxis.setAttribute('x2', margin.left);
  yAxis.setAttribute('y1', margin.top);
  yAxis.setAttribute('y2', height - margin.bottom);
  yAxis.setAttribute('class', 'axis-line');
  svg.appendChild(yAxis);

  series.forEach((s) => {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', buildPath(s.points, sx, sy));
    path.setAttribute('stroke', s.color);
    path.setAttribute('class', 'series-line');
    svg.appendChild(path);
  });

  host.appendChild(svg);

  const legend = el('div', 'legend-wrap');
  series.forEach((s) => {
    const item = el('span', 'legend-item');
    const swatch = el('span', 'legend-swatch');
    swatch.style.background = s.color;
    item.appendChild(swatch);
    item.appendChild(el('span', null, s.label));
    legend.appendChild(item);
  });
  host.appendChild(legend);
}

function renderCoefficientBars(targetId, rows) {
  const host = document.getElementById(targetId);
  if (!host) return;
  host.innerHTML = '';

  const data = (rows || [])
    .map((r) => ({
      feature: r.feature_name || r.feature || 'unknown',
      rank: Number(r.mean_abs_coef ?? 0),
      signed: Number(r.mean_coef ?? r.mean_abs_coef ?? 0),
    }))
    .filter((r) => Number.isFinite(r.rank) && Number.isFinite(r.signed))
    .sort((a, b) => b.rank - a.rank)
    .slice(0, 15);

  if (!data.length) {
    createEmpty(host, 'No coefficient data available.');
    return;
  }

  const width = Math.max(host.clientWidth || 820, 420);
  const height = Math.max(360, data.length * 28 + 40);
  const margin = { top: 12, right: 18, bottom: 20, left: 220 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const rowH = innerH / data.length;
  const maxAbs = Math.max(...data.map((d) => Math.abs(d.signed))) || 1;
  const x = (v) => margin.left + ((v + maxAbs) / (maxAbs * 2)) * innerW;
  const zeroX = x(0);

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'svg-chart');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('preserveAspectRatio', 'none');

  const zeroLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  zeroLine.setAttribute('x1', zeroX);
  zeroLine.setAttribute('x2', zeroX);
  zeroLine.setAttribute('y1', margin.top);
  zeroLine.setAttribute('y2', height - margin.bottom);
  zeroLine.setAttribute('class', 'axis-line');
  svg.appendChild(zeroLine);

  data.forEach((d, i) => {
    const y = margin.top + i * rowH + rowH * 0.2;
    const barH = rowH * 0.6;
    const x1 = x(Math.min(0, d.signed));
    const x2 = x(Math.max(0, d.signed));
    const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bar.setAttribute('x', Math.min(x1, x2));
    bar.setAttribute('y', y);
    bar.setAttribute('width', Math.max(1, Math.abs(x2 - x1)));
    bar.setAttribute('height', barH);
    bar.setAttribute('fill', d.signed >= 0 ? '#2ec27e' : '#ff6b7a');
    bar.setAttribute('opacity', '0.9');
    svg.appendChild(bar);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', margin.left - 8);
    label.setAttribute('y', y + barH * 0.74);
    label.setAttribute('text-anchor', 'end');
    label.setAttribute('class', 'axis-text');
    label.textContent = d.feature;
    svg.appendChild(label);
  });

  host.appendChild(svg);
}

function renderTable(targetId, rows) {
  const root = document.getElementById(targetId);
  if (!root) return;
  root.innerHTML = '';

  if (!rows || !rows.length) {
    createEmpty(root, 'No table rows available.');
    return;
  }

  const columns = Array.from(
    rows.reduce((set, row) => {
      Object.keys(row || {}).forEach((k) => set.add(k));
      return set;
    }, new Set())
  );

  const wrap = el('div', 'table-wrap');
  const table = el('table');
  const thead = el('thead');
  const tbody = el('tbody');

  const trHead = el('tr');
  columns.forEach((col) => trHead.appendChild(el('th', null, titleize(col))));
  thead.appendChild(trHead);

  rows.forEach((row) => {
    const tr = el('tr');
    columns.forEach((col) => {
      const td = el('td');
      const value = row[col];
      if (typeof value === 'number') {
        td.textContent = fmtNumber(value, Math.abs(value) < 1 ? 6 : 4);
      } else {
        td.textContent = value === null || value === undefined || value === '' ? 'N/A' : String(value);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  wrap.appendChild(table);
  root.appendChild(wrap);
}

function mountTradingViewWidget(targetId, widgetCfg) {
  const host = document.getElementById(targetId);
  if (!host) return;
  host.innerHTML = '';

  if (!widgetCfg || !widgetCfg.script_url || !widgetCfg.config) {
    createEmpty(host, 'Widget configuration is missing.');
    return;
  }

  const wrap = el('div', 'tv-wrap');
  const container = el('div', 'tradingview-widget-container');
  const widget = el('div', 'tradingview-widget-container__widget');
  container.appendChild(widget);

  const script = document.createElement('script');
  script.type = 'text/javascript';
  script.async = true;
  script.src = widgetCfg.script_url;
  script.text = JSON.stringify(widgetCfg.config);
  container.appendChild(script);

  wrap.appendChild(container);
  host.appendChild(wrap);
}

function renderLiveWidgets(payload) {
  const widgets = payload.live_widgets || {};
  mountTradingViewWidget('live-ticker-tape', widgets.ticker_tape);
  mountTradingViewWidget('live-main-chart', widgets.main_chart);
  mountTradingViewWidget('live-market-overview', widgets.market_overview);
}

function renderCharts(payload) {
  const chartSeries = payload.chart_series || {};
  const byHorizon = chartSeries.predictions_by_horizon || {};

  const predSeries = Object.entries(byHorizon).map(([h, rows], idx) => ({
    label: h.toUpperCase(),
    color: PALETTE[idx % PALETTE.length],
    data: (rows || []).map((r) => ({ date: r.date, value: r.y_pred })),
  }));
  renderLineChart('chart-predictions-by-horizon', {
    series: predSeries,
    includeZero: true,
    yFormat: (v) => fmtPercent(v, 1),
    height: 470,
    ariaLabel: 'prediction history by horizon',
  });

  const h1Rows = byHorizon.h1 || [];
  renderLineChart('chart-pred-vs-actual-h1', {
    series: [
      { label: 'Predicted (H1)', color: '#73b7ff', data: h1Rows.map((r) => ({ date: r.date, value: r.y_pred })) },
      { label: 'Actual (H1)', color: '#f2b84b', data: h1Rows.map((r) => ({ date: r.date, value: r.y_true })) },
    ],
    includeZero: true,
    yFormat: (v) => fmtPercent(v, 1),
    height: 380,
    ariaLabel: 'predicted versus actual horizon one',
  });

  const rolling = chartSeries.rolling_hit_rate_by_horizon || {};
  const rollingSeries = Object.entries(rolling.horizons || {}).map(([h, rows], idx) => ({
    label: `${h.toUpperCase()} Hit Rate`,
    color: PALETTE[idx % PALETTE.length],
    data: (rows || []).map((r) => ({ date: r.date, value: r.rolling_hit_rate })),
  }));
  renderLineChart('chart-rolling-hit-rate', {
    series: rollingSeries,
    includeZero: true,
    yFormat: (v) => fmtPercent(v, 0),
    height: 380,
    ariaLabel: 'rolling hit rate by horizon',
  });

  const curves = (chartSeries.strategy_curves || {}).series || [];
  renderLineChart('chart-strategy-curves', {
    series: curves.map((s, idx) => ({
      label: s.label || s.strategy,
      color: s.is_benchmark ? '#9caecb' : PALETTE[idx % PALETTE.length],
      data: s.data,
    })),
    includeZero: false,
    yFormat: (v) => fmtNumber(v, 2),
    height: 470,
    ariaLabel: 'strategy equity curves',
  });

  const drawdowns = (chartSeries.strategy_drawdowns || {}).series || [];
  renderLineChart('chart-strategy-drawdowns', {
    series: drawdowns.map((s, idx) => ({
      label: s.label || s.strategy,
      color: s.is_benchmark ? '#9caecb' : PALETTE[idx % PALETTE.length],
      data: s.data,
    })),
    includeZero: true,
    yFormat: (v) => fmtPercent(v, 1),
    height: 470,
    ariaLabel: 'strategy drawdowns',
  });

  const weights = (chartSeries.strategy_weights || {}).series || [];
  renderLineChart('chart-strategy-weights', {
    series: weights.map((s, idx) => ({
      label: s.label || s.strategy,
      color: PALETTE[idx % PALETTE.length],
      data: s.data,
    })),
    includeZero: true,
    yFormat: (v) => fmtNumber(v, 2),
    height: 470,
    ariaLabel: 'strategy exposure weights',
  });

  renderCoefficientBars('chart-top-coefficients', chartSeries.top_coefficients || []);
}

function renderTables(payload) {
  const tables = payload.tables || {};
  renderTable('table-strategy-run-summary', tables.strategy_run_summary || []);
  renderTable('table-baseline-horizon', tables.strategy_baseline_horizon_summary || []);
  renderTable('table-h3-confirmation', tables.strategy_h3_confirmation_summary || []);
  renderTable('table-ridge-coefs', tables.ridge_coef_summary || []);
  renderTable('table-recent-predictions', tables.recent_predictions || []);
}

function renderError(error) {
  const root = document.getElementById('hero-meta');
  root.innerHTML = '';
  const card = el('article', 'hero-meta-card');
  card.appendChild(el('h3', null, 'Dashboard Load Error'));
  const line = el('div', 'hero-meta-row');
  line.appendChild(el('span', 'label', 'Error'));
  line.appendChild(el('span', null, error.message || String(error)));
  card.appendChild(line);
  root.appendChild(card);
}

async function main() {
  try {
    const payload = await loadPayload();
    renderHeroMeta(payload);
    renderSnapshots(payload);
    renderLiveWidgets(payload);
    renderCharts(payload);
    renderTables(payload);
  } catch (error) {
    renderError(error);
  }
}

main();
