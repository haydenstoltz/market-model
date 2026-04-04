const {
  PROVIDER,
  handleCors,
  sendJson,
  getApiKey,
  parseSymbols,
  fetchQuotes,
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

  const symbols = parseSymbols(req.query && req.query.symbols);
  if (!symbols.length) {
    sendJson(res, 400, {
      ok: false,
      provider: PROVIDER,
      error: 'No symbols provided.',
    });
    return;
  }

  try {
    const { quotes, errors } = await fetchQuotes(symbols, apiKey);
    sendJson(res, 200, {
      ok: quotes.length > 0,
      provider: PROVIDER,
      asOf: new Date().toISOString(),
      requestedSymbols: symbols,
      quotes,
      errors,
    });
  } catch (error) {
    sendJson(res, 502, {
      ok: false,
      provider: PROVIDER,
      asOf: new Date().toISOString(),
      requestedSymbols: symbols,
      quotes: [],
      errors: [{ symbol: null, message: error.message || String(error) }],
    });
  }
};
