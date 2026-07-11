# hagan-sabr

The **SABR** stochastic-volatility model — the Hagan et al. (2002) implied-volatility expansions (**lognormal** and **normal**), the **Obłój** correction, and **(α, ρ, ν) calibration** to a market smile. Zero dependencies.

```
npm install hagan-sabr
```

## Why

SABR is the market-standard model for interest-rate swaption and FX volatility smiles — but there was no focused JavaScript/TypeScript implementation. `@quantlib/ql` has it buried in a 28 MB WASM bundle behind a C++-style API. This is the Hagan formulas as a handful of typed functions, and its lognormal output **matches QuantLib's `sabrVolatility` to 1e-9** across 315 test cases.

```ts
import { lognormalVol, calibrate } from "hagan-sabr";

const p = { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 };
lognormalVol(100, 100, 1, p);   // ATM Black vol ≈ 0.0202
lognormalVol(100, 80, 1, p);    // the smile at a lower strike

// Fit (α, ρ, ν) to a market smile (β fixed):
const fit = calibrate(100, 2, [
  { strike: 80, vol: 0.245 },
  { strike: 100, vol: 0.202 },
  { strike: 120, vol: 0.199 },
  // …
], { beta: 0.5 });
fit.alpha; fit.rho; fit.nu; fit.rmse;
```

## API

Parameters: forward `F`, strike `K`, expiry `T` (years), and `{ alpha, beta, rho, nu }` — α > 0, 0 ≤ β ≤ 1, −1 < ρ < 1, ν ≥ 0. Vols are decimals.

- **`lognormalVol(F, K, T, params)`** — Hagan Black/lognormal implied vol (matches QuantLib).
- **`normalVol(F, K, T, params)`** — Hagan Bachelier/normal implied vol (eq. 2.18).
- **`oblojVol(F, K, T, params)`** — the Obłój (2008) refinement that fixes the small-strike / long-maturity leading term.
- **`calibrate(F, T, quotes, opts?)`** — least-squares fit of (α, ρ, ν) with β fixed (default 0.5), via Nelder–Mead; returns the params plus fit `rmse`.

Invalid inputs throw `RangeError`.

## Correctness

The lognormal expansion is validated against **QuantLib's `sabrVolatility`** — 315 fixtures spanning β ∈ {0, 0.3, 0.5, 0.7, 1}, forwards from 0.02 to 100, expiries to 5y, and strikes from deep ITM to deep OTM — all matching to **1e-9** ([`test/generate-fixtures.py`](test/generate-fixtures.py)). The normal and Obłój variants are cross-checked against an independent reference implementation, and calibration is verified to recover known parameters from a synthetic smile.

## Related

Pairs with [`svi-vol-surface`](https://github.com/moshejs/svi-vol-surface) (arbitrage-free surface parametrization). Part of a quant/fixed-income toolkit: [`compounded-sofr`](https://github.com/moshejs/compounded-sofr) · [`day-count`](https://github.com/moshejs/day-count) · [`instrument-identifiers`](https://github.com/moshejs/instrument-identifiers) · [`newyorkfed`](https://github.com/moshejs/newyorkfed).

## Author

Built by **[Moshe Malka](https://moshemalka.com)** — engineering leader in New York City. Studio work at [Quentin.Code](https://www.quentin.software/).

MIT © Moshe Malka
