const PROVIDER = 'finnhub';
const FINNHUB_BASE_URL = 'https://finnhub.io/api/v1';
const DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'DIA', 'IWM', 'VIX', 'AAPL', 'MSFT', 'NVDA'];

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.end(JSON.stringify(payload));
}

function handleCors(req, res) {
  if (req.method && req.method.toUpperCase() === 'OPTIONS') {
    sendJson(res, 204, {});
    return true;
  }
  return false;
}

function getApiKey() {
  return (process.env.MARKET_DATA_API_KEY || process.env.FINNHUB_API_KEY || '').trim();
}

function parseSymbols(symbolsRaw) {
  const raw = typeof symbolsRaw === 'string' ? symbolsRaw : '';
  const list = (raw || DEFAULT_SYMBOLS.join(','))
    .split(',')
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
  return Array.from(new Set(list)).slice(0, 30);
}

function parseChartRange(rangeRaw) {
  const normalized = String(rangeRaw || '1D').trim().toUpperCase();
  const nowSec = Math.floor(Date.now() / 1000);

  const map = {
    '1D': { seconds: 24 * 60 * 60, resolution: '5', interval: '5m' },
    '5D': { seconds: 5 * 24 * 60 * 60, resolution: '15', interval: '15m' },
    '1M': { seconds: 30 * 24 * 60 * 60, resolution: '60', interval: '60m' },
    '3M': { seconds: 90 * 24 * 60 * 60, resolution: 'D', interval: '1d' },
    '6M': { seconds: 180 * 24 * 60 * 60, resolution: 'D', interval: '1d' },
    '1Y': { seconds: 365 * 24 * 60 * 60, resolution: 'W', interval: '1w' },
  };
  const config = map[normalized] || map['1D'];
  return {
    range: normalized in map ? normalized : '1D',
    resolution: config.resolution,
    interval: config.interval,
    fromSec: nowSec - config.seconds,
    toSec: nowSec,
  };
}

async function fetchProviderJson(url) {
  const response = await fetch(url, {
    headers: {
      Accept: 'application/json',
      'User-Agent': 'market-model-dashboard/1.0',
    },
  });
  if (!response.ok) {
    throw new Error(`Provider HTTP ${response.status}`);
  }
  return response.json();
}

function normalizeQuote(symbol, data) {
  return {
    symbol,
    price: Number.isFinite(Number(data.c)) ? Number(data.c) : null,
    change: Number.isFinite(Number(data.d)) ? Number(data.d) : null,
    changePct: Number.isFinite(Number(data.dp)) ? Number(data.dp) : null,
    open: Number.isFinite(Number(data.o)) ? Number(data.o) : null,
    high: Number.isFinite(Number(data.h)) ? Number(data.h) : null,
    low: Number.isFinite(Number(data.l)) ? Number(data.l) : null,
    prevClose: Number.isFinite(Number(data.pc)) ? Number(data.pc) : null,
    timestamp: Number.isFinite(Number(data.t))
      ? new Date(Number(data.t) * 1000).toISOString()
      : new Date().toISOString(),
  };
}

async function fetchQuotes(symbols, apiKey) {
  const results = await Promise.all(
    symbols.map(async (symbol) => {
      try {
        const url = `${FINNHUB_BASE_URL}/quote?symbol=${encodeURIComponent(symbol)}&token=${encodeURIComponent(apiKey)}`;
        const data = await fetchProviderJson(url);
        return { ok: true, quote: normalizeQuote(symbol, data) };
      } catch (error) {
        return { ok: false, symbol, error: error.message || String(error) };
      }
    })
  );

  return {
    quotes: results.filter((r) => r.ok).map((r) => r.quote),
    errors: results
      .filter((r) => !r.ok)
      .map((r) => ({ symbol: r.symbol, message: r.error })),
  };
}

async function fetchChart(symbol, range, apiKey) {
  const cfg = parseChartRange(range);
  const sym = String(symbol || 'SPY').trim().toUpperCase();
  const url = `${FINNHUB_BASE_URL}/stock/candle?symbol=${encodeURIComponent(sym)}&resolution=${encodeURIComponent(cfg.resolution)}&from=${cfg.fromSec}&to=${cfg.toSec}&token=${encodeURIComponent(apiKey)}`;
  const data = await fetchProviderJson(url);

  if (!data || data.s !== 'ok' || !Array.isArray(data.t) || !Array.isArray(data.c)) {
    return {
      symbol: sym,
      range: cfg.range,
      interval: cfg.interval,
      points: [],
      providerStatus: data && data.s ? data.s : 'unknown',
    };
  }

  const points = [];
  for (let i = 0; i < data.t.length; i += 1) {
    const t = Number(data.t[i]);
    const c = Number(data.c[i]);
    if (!Number.isFinite(t) || !Number.isFinite(c)) continue;
    points.push({
      t: new Date(t * 1000).toISOString(),
      o: Number.isFinite(Number(data.o && data.o[i])) ? Number(data.o[i]) : null,
      h: Number.isFinite(Number(data.h && data.h[i])) ? Number(data.h[i]) : null,
      l: Number.isFinite(Number(data.l && data.l[i])) ? Number(data.l[i]) : null,
      c,
      v: Number.isFinite(Number(data.v && data.v[i])) ? Number(data.v[i]) : null,
    });
  }

  return {
    symbol: sym,
    range: cfg.range,
    interval: cfg.interval,
    points,
    providerStatus: 'ok',
  };
}

module.exports = {
  PROVIDER,
  DEFAULT_SYMBOLS,
  sendJson,
  handleCors,
  getApiKey,
  parseSymbols,
  fetchQuotes,
  fetchChart,
};
