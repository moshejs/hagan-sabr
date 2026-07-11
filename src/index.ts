/**
 * hagan-sabr — the SABR stochastic-volatility model.
 *
 * The Hagan et al. (2002) implied-volatility expansions — lognormal (Black)
 * and normal (Bachelier) — with the Obłój (2008) refinement, plus
 * least-squares calibration of (α, ρ, ν) to a market smile.
 *
 * Parameters:
 *   F     forward
 *   K     strike
 *   T     time to expiry (years)
 *   alpha instantaneous vol level (α > 0)
 *   beta  CEV exponent (0 ≤ β ≤ 1)
 *   rho   correlation (−1 < ρ < 1)
 *   nu    vol-of-vol (ν ≥ 0)
 *
 * Volatilities are returned as decimals (0.20 = 20%).
 */

export interface SabrParams {
  alpha: number;
  beta: number;
  rho: number;
  nu: number;
}

function validate(F: number, K: number, T: number, p: SabrParams): void {
  if (!(F > 0)) throw new RangeError(`forward must be positive, got ${F}`);
  if (!(K > 0)) throw new RangeError(`strike must be positive, got ${K}`);
  if (!(T > 0)) throw new RangeError(`expiry must be positive, got ${T}`);
  if (!(p.alpha > 0)) throw new RangeError(`alpha must be positive, got ${p.alpha}`);
  if (p.beta < 0 || p.beta > 1) throw new RangeError(`beta must be in [0,1], got ${p.beta}`);
  if (p.rho <= -1 || p.rho >= 1) throw new RangeError(`rho must be in (-1,1), got ${p.rho}`);
  if (p.nu < 0) throw new RangeError(`nu must be non-negative, got ${p.nu}`);
}

/**
 * z / x(z), the SABR correction factor, evaluated stably near z = 0 where it
 * tends to 1.
 */
function zOverX(z: number, rho: number): number {
  if (Math.abs(z) < 1e-12) {
    // Series: z/x(z) ≈ 1 − ρz/2 + (2−3ρ²)z²/12 …
    return 1 - (rho * z) / 2 + ((2 - 3 * rho * rho) * z * z) / 12;
  }
  const x = Math.log((Math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho));
  return z / x;
}

/**
 * Hagan (2002) lognormal (Black) implied volatility. This matches
 * QuantLib's `sabrVolatility`.
 *
 * ```ts
 * lognormalVol(100, 100, 1, { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 });
 * // ATM ≈ 0.0202 (= alpha / F^(1-beta) × …)
 * ```
 */
export function lognormalVol(F: number, K: number, T: number, p: SabrParams): number {
  validate(F, K, T, p);
  const { alpha, beta, rho, nu } = p;
  const oneMinusBeta = 1 - beta;
  const logFK = Math.log(F / K);

  // Second-order time correction, common to ATM and non-ATM.
  const FKbeta = Math.pow(F * K, oneMinusBeta / 2);
  const term1 = (oneMinusBeta * oneMinusBeta * alpha * alpha) / (24 * FKbeta * FKbeta);
  const term2 = (rho * beta * nu * alpha) / (4 * FKbeta);
  const term3 = ((2 - 3 * rho * rho) * nu * nu) / 24;
  const timeFactor = 1 + (term1 + term2 + term3) * T;

  if (Math.abs(logFK) < 1e-12) {
    // ATM
    return (alpha / Math.pow(F, oneMinusBeta)) * timeFactor;
  }

  const logFK2 = logFK * logFK;
  const logFK4 = logFK2 * logFK2;
  const denom =
    FKbeta *
    (1 +
      (oneMinusBeta * oneMinusBeta * logFK2) / 24 +
      (oneMinusBeta * oneMinusBeta * oneMinusBeta * oneMinusBeta * logFK4) / 1920);
  const z = (nu / alpha) * FKbeta * logFK;
  return (alpha / denom) * zOverX(z, rho) * timeFactor;
}

/**
 * Hagan (2002) normal (Bachelier) implied volatility (eq. 2.18), in the same
 * units as the forward. Parallels the lognormal formula: the same z and
 * z/x(z), a `(FK)^{β/2}` level, and a normal-specific time correction.
 */
export function normalVol(F: number, K: number, T: number, p: SabrParams): number {
  validate(F, K, T, p);
  const { alpha, beta, rho, nu } = p;
  const oneMinusBeta = 1 - beta;

  // Time correction (common to ATM and non-ATM), uses the −β(2−β) normal term.
  const FKpow1mb = Math.pow(F * K, oneMinusBeta); // (FK)^{1-β}
  const FKhalf = Math.pow(F * K, oneMinusBeta / 2); // (FK)^{(1-β)/2}
  const c1 = (-beta * (2 - beta) * alpha * alpha) / (24 * FKpow1mb);
  const c2 = (rho * beta * nu * alpha) / (4 * FKhalf);
  const c3 = ((2 - 3 * rho * rho) * nu * nu) / 24;
  const timeFactor = 1 + (c1 + c2 + c3) * T;

  const level = alpha * Math.pow(F * K, beta / 2);

  if (Math.abs(F - K) < 1e-12 * Math.max(1, F)) {
    return level * timeFactor; // ATM: log terms and z/x(z) → 1
  }

  const logFK = Math.log(F / K);
  const logFK2 = logFK * logFK;
  const logFK4 = logFK2 * logFK2;
  const numerator = 1 + logFK2 / 24 + logFK4 / 1920;
  const denominator =
    1 +
    (oneMinusBeta * oneMinusBeta * logFK2) / 24 +
    (oneMinusBeta * oneMinusBeta * oneMinusBeta * oneMinusBeta * logFK4) / 1920;
  const z = (nu / alpha) * FKhalf * logFK;
  return level * (numerator / denominator) * zOverX(z, rho) * timeFactor;
}

/**
 * Lognormal vol with the Obłój (2008) correction, which fixes an
 * inconsistency in the original expansion for small strikes / long maturities.
 * Differs from {@link lognormalVol} only in the leading (zeroth-order) term.
 */
export function oblojVol(F: number, K: number, T: number, p: SabrParams): number {
  validate(F, K, T, p);
  const { alpha, beta, rho, nu } = p;
  const oneMinusBeta = 1 - beta;
  const logFK = Math.log(F / K);

  const FKbeta = Math.pow(F * K, oneMinusBeta / 2);
  const term1 = (oneMinusBeta * oneMinusBeta * alpha * alpha) / (24 * FKbeta * FKbeta);
  const term2 = (rho * beta * nu * alpha) / (4 * FKbeta);
  const term3 = ((2 - 3 * rho * rho) * nu * nu) / 24;
  const timeFactor = 1 + (term1 + term2 + term3) * T;

  // Obłój leading term uses the exact z with the CEV-consistent z definition.
  let leading: number;
  if (Math.abs(logFK) < 1e-12) {
    leading = alpha / Math.pow(F, oneMinusBeta);
  } else if (oneMinusBeta < 1e-8) {
    const z = (nu / alpha) * logFK;
    leading = (nu * logFK) / (z === 0 ? 1 : Math.log((Math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho)));
  } else {
    const z = (nu / alpha) * (Math.pow(F, oneMinusBeta) - Math.pow(K, oneMinusBeta)) / oneMinusBeta;
    const xz = Math.log((Math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho));
    leading = (nu * logFK) / (xz === 0 ? 1 : xz);
  }
  return leading * timeFactor;
}

export interface MarketQuote {
  strike: number;
  /** Observed lognormal (Black) implied vol as a decimal. */
  vol: number;
  /** Optional weight for the least-squares fit (default 1). */
  weight?: number;
}

export interface CalibrationOptions {
  /** Fixed β (SABR β is usually fixed, not fitted). Default 0.5. */
  beta?: number;
  /** Initial guesses. */
  initial?: Partial<Pick<SabrParams, "alpha" | "rho" | "nu">>;
  /** Max Nelder–Mead iterations. Default 2000. */
  maxIter?: number;
}

export interface CalibrationResult extends SabrParams {
  /** Root-mean-squared vol error at the optimum. */
  rmse: number;
  iterations: number;
}

/**
 * Calibrate (α, ρ, ν) to a market smile by least squares (β fixed), using a
 * Nelder–Mead simplex. Returns the fitted SABR parameters and fit RMSE.
 */
export function calibrate(
  F: number,
  T: number,
  quotes: readonly MarketQuote[],
  opts: CalibrationOptions = {},
): CalibrationResult {
  if (!(F > 0)) throw new RangeError(`forward must be positive, got ${F}`);
  if (!(T > 0)) throw new RangeError(`expiry must be positive, got ${T}`);
  if (quotes.length < 3) throw new RangeError("calibration needs at least 3 quotes");
  const beta = opts.beta ?? 0.5;

  // ATM vol seeds alpha: σ_ATM ≈ α / F^{1-β}.
  const atm = [...quotes].sort((a, b) => Math.abs(a.strike - F) - Math.abs(b.strike - F))[0]!;
  const alpha0 = opts.initial?.alpha ?? atm.vol * Math.pow(F, 1 - beta);
  const rho0 = opts.initial?.rho ?? 0;
  const nu0 = opts.initial?.nu ?? 0.3;

  // Optimize in an unconstrained space: alpha=exp(a), rho=tanh(r), nu=exp(n).
  const objective = (v: number[]): number => {
    const alpha = Math.exp(v[0]!);
    const rho = Math.tanh(v[1]!);
    const nu = Math.exp(v[2]!);
    let sse = 0;
    for (const q of quotes) {
      const model = lognormalVol(F, q.strike, T, { alpha, beta, rho, nu });
      const w = q.weight ?? 1;
      const e = model - q.vol;
      sse += w * e * e;
    }
    return sse;
  };

  const start: number[] = [Math.log(alpha0), Math.atanh(clamp(rho0, -0.999, 0.999)), Math.log(Math.max(nu0, 1e-6))];
  const { point, iterations } = nelderMead(objective, start, opts.maxIter ?? 2000);
  const alpha = Math.exp(point[0]!);
  const rho = Math.tanh(point[1]!);
  const nu = Math.exp(point[2]!);
  const totalW = quotes.reduce((s, q) => s + (q.weight ?? 1), 0);
  const rmse = Math.sqrt(objective(point) / totalW);
  return { alpha, beta, rho, nu, rmse, iterations };
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/** Minimal Nelder–Mead simplex minimizer. */
function nelderMead(
  f: (x: number[]) => number,
  x0: number[],
  maxIter: number,
): { point: number[]; value: number; iterations: number } {
  const n = x0.length;
  const alpha = 1;
  const gamma = 2;
  const rho = 0.5;
  const sigma = 0.5;

  const simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const p = x0.slice();
    const xi = p[i]!;
    p[i] = xi + (xi !== 0 ? 0.1 * xi : 0.1);
    simplex.push(p);
  }
  let fvals = simplex.map(f);
  let iter = 0;

  for (; iter < maxIter; iter++) {
    const order = fvals.map((v, i) => i).sort((a, b) => fvals[a]! - fvals[b]!);
    const s = order.map((i) => simplex[i]!);
    const fs = order.map((i) => fvals[i]!);
    for (let i = 0; i < s.length; i++) {
      simplex[i] = s[i]!;
      fvals[i] = fs[i]!;
    }
    if (Math.abs(fvals[n]! - fvals[0]!) < 1e-14) break;

    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j]! += simplex[i]![j]!;
    }
    for (let j = 0; j < n; j++) centroid[j]! /= n;

    const worst = simplex[n]!;
    const reflected = centroid.map((c, j) => c + alpha * (c - worst[j]!));
    const fr = f(reflected);

    if (fr < fvals[0]!) {
      const expanded = centroid.map((c, j) => c + gamma * (reflected[j]! - c));
      const fe = f(expanded);
      if (fe < fr) {
        simplex[n] = expanded;
        fvals[n] = fe;
      } else {
        simplex[n] = reflected;
        fvals[n] = fr;
      }
    } else if (fr < fvals[n - 1]!) {
      simplex[n] = reflected;
      fvals[n] = fr;
    } else {
      const contracted = centroid.map((c, j) => c + rho * (worst[j]! - c));
      const fc = f(contracted);
      if (fc < fvals[n]!) {
        simplex[n] = contracted;
        fvals[n] = fc;
      } else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[0]!.map((b, j) => b + sigma * (simplex[i]![j]! - b));
          fvals[i] = f(simplex[i]!);
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < fvals.length; i++) if (fvals[i]! < fvals[best]!) best = i;
  return { point: simplex[best]!, value: fvals[best]!, iterations: iter };
}
