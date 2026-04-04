const {
  PROVIDER,
  handleCors,
  sendJson,
  getApiKey,
  fetchChart,
} = require('./_marketProvider');

module.exports = async (req, res) => {
  if (handleCors(req, res)) return;

  if ((req.method || 'GET').toUpperCase() !== 'GET') {
    sendJson(res, 405, {
      ok: false,
      provider: PROVIDER,
      error: 'Method not allowed. Use GET.',
    });
    return;
  }

  const apiKey = getApiKey();
  if (!apiKey) {
    sendJson(res, 500, {
      ok: false,
      provider: PROVIDER,
      error: 'Server misconfigured: MARKET_DATA_API_KEY is not set.',
    });
    return;
  }

  const symbol = (req.query && req.query.symbol) || 'SPY';
  const range = (req.query && req.query.range) || '1D';

  try {
    const payload = await fetchChart(symbol, range, apiKey);
    sendJson(res, 200, {
      ok: Array.isArray(payload.points) && payload.points.length > 0,
      provider: PROVIDER,
      asOf: new Date().toISOString(),
      symbol: payload.symbol,
      range: payload.range,
      interval: payload.interval,
      points: payload.points,
      providerStatus: payload.providerStatus,
    });
  } catch (error) {
    sendJson(res, 502, {
      ok: false,
      provider: PROVIDER,
      asOf: new Date().toISOString(),
      symbol: String(symbol).toUpperCase(),
      range: String(range).toUpperCase(),
      interval: null,
      points: [],
      error: error.message || String(error),
    });
  }
};
