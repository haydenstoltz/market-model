# Live Market API (Serverless Target: Vercel)

This directory provides a minimal backend for the static dashboard.

## Endpoints
- `/api/quotes?symbols=SPY,QQQ,DIA,IWM,VIX,AAPL,MSFT,NVDA`
- `/api/chart?symbol=SPY&range=1D`

## Provider
- Finnhub proxy (normalized response shape).

## Environment Variables (server-side only)
- `MARKET_DATA_API_KEY` (preferred)
- `FINNHUB_API_KEY` (fallback)

Do not expose either key in `docs/` files.

## Deployment
Deploy this repository (or just this `api/` directory) to Vercel.  
Set `MARKET_DATA_API_KEY` in Vercel project settings.

Then set exporter env var before generating dashboard payload:

```bash
export MARKET_LIVE_API_BASE_URL="https://<your-vercel-project>.vercel.app"
python scripts/export_market_site.py
```
