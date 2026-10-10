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
 *   shift optional displacement s ≥ 0 (shifted SABR): the formulas are
 *         applied to F + s and K + s, so negative rates down to −s work
 *
 * Volatilities are returned as decimals (0.20 = 20%).
 */

export interface SabrParams {
  alpha: number;
  beta: number;
  rho: number;
  nu: number;
  /**
   * Shifted-SABR displacement (default 0). Forward and strike are replaced
   * by F + shift and K + shift, so they only need to exceed −shift. The
   * lognormal/Obłój outputs are then *shifted*-lognormal vols; the normal vol
   * is unaffected by the change of variable and is a plain Bachelier vol.
   */
  shift?: number;
}

/** Validates inputs and returns the shifted (F, K) actually used. */
function validate(F: number, K: number, T: number, p: SabrParams): [number, number] {
  const s = p.shift ?? 0;
  if (!(s >= 0 && s < Infinity)) throw new RangeError(`shift must be a finite number ≥ 0, got ${s}`);
  const Fs = F + s;
  const Ks = K + s;
  if (!(Fs > 0 && Fs < Infinity)) {
    throw new RangeError(`forward + shift must be positive and finite, got ${F} + ${s}`);
  }
  if (!(Ks > 0 && Ks < Infinity)) {
    throw new RangeError(`strike + shift must be positive and finite, got ${K} + ${s}`);
  }
  if (!(T > 0 && T < Infinity)) throw new RangeError(`expiry must be positive and finite, got ${T}`);
  if (!(p.alpha > 0 && p.alpha < Infinity)) {
    throw new RangeError(`alpha must be positive and finite, got ${p.alpha}`);
  }
  if (!(p.beta >= 0 && p.beta <= 1)) throw new RangeError(`beta must be in [0,1], got ${p.beta}`);
  if (!(p.rho > -1 && p.rho < 1)) throw new RangeError(`rho must be in (-1,1), got ${p.rho}`);
  if (!(p.nu >= 0 && p.nu < Infinity)) {
    throw new RangeError(`nu must be non-negative and finite, got ${p.nu}`);
  }
  return [Fs, Ks];
}

/**
 * z / x(z), the SABR correction factor, where
 * x(z) = ln((√(1 − 2ρz + z²) + z − ρ) / (1 − ρ)).
 *
 * The textbook form loses all precision near z = 0 (the log argument is
 * 1 + O(z), so x carries an absolute error of ~1e-16 and z/x a relative error
 * of ~1e-16/z — 3e-4 at z ≈ 1e-11). Here x is evaluated without cancellation:
 * with s = √(1 − 2ρz + z²) and P = s + z − ρ (> 0, computed as
 * (1 − ρ²)/(s − z + ρ) when z − ρ < 0), the log argument minus one is
 * u = z(P + 1 − ρ) / ((s + 1)(1 − ρ)), a product of non-cancelling terms, and
 * x = log1p(u).
 */
function zOverX(z: number, rho: number): number {
  if (z === 0) return 1;
  if (Math.abs(z) < 1e-9) {
    // Series: z/x(z) = 1 − ρz/2 + (2 − 3ρ²)z²/12 + O(z³); exact to double here.
    return 1 - (rho * z) / 2 + ((2 - 3 * rho * rho) * z * z) / 12;
  }
  const s = Math.sqrt(1 - 2 * rho * z + z * z);
  const zr = z - rho;
  const P = zr >= 0 ? s + zr : (1 - rho * rho) / (s - zr);
  const u = (z * (P + 1 - rho)) / ((s + 1) * (1 - rho));
  const x = u > -0.5 ? Math.log1p(u) : Math.log(P / (1 - rho));
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
  [F, K] = validate(F, K, T, p);
  const { alpha, beta, rho, nu } = p;
  const oneMinusBeta = 1 - beta;
  const logFK = Math.log(F / K);

  // Second-order time correction, common to ATM and non-ATM.
  const FKbeta = Math.pow(F * K, oneMinusBeta / 2);
  const term1 = (oneMinusBeta * oneMinusBeta * alpha * alpha) / (24 * FKbeta * FKbeta);
  const term2 = (rho * beta * nu * alpha) / (4 * FKbeta);
  const term3 = ((2 - 3 * rho * rho) * nu * nu) / 24;
  const timeFactor = 1 + (term1 + term2 + term3) * T;

  if (logFK === 0) {
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
  [F, K] = validate(F, K, T, p);
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

  const logFK = Math.log(F / K);
  if (logFK === 0) {
    return level * timeFactor; // ATM: log terms and z/x(z) → 1
  }

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
  [F, K] = validate(F, K, T, p);
  const { alpha, beta, rho, nu } = p;
  const oneMinusBeta = 1 - beta;
  const logFK = Math.log(F / K);

  const FKbeta = Math.pow(F * K, oneMinusBeta / 2);
  const term1 = (oneMinusBeta * oneMinusBeta * alpha * alpha) / (24 * FKbeta * FKbeta);
  const term2 = (rho * beta * nu * alpha) / (4 * FKbeta);
  const term3 = ((2 - 3 * rho * rho) * nu * nu) / 24;
  const timeFactor = 1 + (term1 + term2 + term3) * T;

  // Obłój leading term: σ₀ = ν·ln(F/K) / x(z) with the CEV-consistent
  // z = ν·q/α, q = ∫_K^F dx/x^β = (F^{1−β} − K^{1−β})/(1−β) (= ln(F/K) at β=1).
  // Written as α·(ln(F/K)/q)·(z/x(z)) it has the right limits: ν → 0 gives
  // α·ln(F/K)/q, and F → K gives α/F^{1−β}. q is evaluated as
  // K^{1−β}·expm1((1−β)ln(F/K))/(1−β) so the ratio ln(F/K)/q stays accurate
  // near the money.
  const e = oneMinusBeta * logFK;
  const kPow = Math.pow(K, oneMinusBeta);
  const logOverQ = e === 0 ? 1 / kPow : e / (kPow * Math.expm1(e));
  const q = logFK / logOverQ;
  const leading = alpha * logOverQ * zOverX((nu * q) / alpha, rho);
  return leading * timeFactor;
}

export interface MarketQuote {
  strike: number;
  /**
   * Observed implied vol as a decimal: lognormal (Black, or shifted-Black when
   * a shift is set) by default, or normal (Bachelier) with `volType: "normal"`.
   */
  vol: number;
  /** Optional weight for the least-squares fit (default 1). */
  weight?: number;
}

export interface CalibrationOptions {
  /** Fixed β (SABR β is usually fixed, not fitted). Default 0.5. */
  beta?: number;
  /** Shifted-SABR displacement applied to forward and strikes. Default 0. */
  shift?: number;
  /** Whether quote vols are lognormal (default) or normal (Bachelier). */
  volType?: "lognormal" | "normal";
  /** Initial guesses. */
  initial?: Partial<Pick<SabrParams, "alpha" | "rho" | "nu">>;
  /** Max total Nelder–Mead iterations across all starts. Default 6000. */
  maxIter?: number;
}

export interface CalibrationResult extends SabrParams {
  /** Root-mean-squared vol error at the optimum. */
  rmse: number;
  iterations: number;
}

/**
 * Calibrate (α, ρ, ν) to a market smile by least squares (β fixed), using a
 * Nelder–Mead simplex run from several (ρ, ν) seeds (or from `initial`), each
 * restarted until it stops improving.
 * Returns the fitted SABR parameters and fit RMSE.
 */
export function calibrate(
  F: number,
  T: number,
  quotes: readonly MarketQuote[],
  opts: CalibrationOptions = {},
): CalibrationResult {
  const beta = opts.beta ?? 0.5;
  const shift = opts.shift ?? 0;
  const normal = opts.volType === "normal";
  if (opts.volType !== undefined && opts.volType !== "lognormal" && !normal) {
    throw new RangeError(`volType must be "lognormal" or "normal", got ${String(opts.volType)}`);
  }
  if (quotes.length < 3) throw new RangeError("calibration needs at least 3 quotes");
  // Validates F, T, β, shift and every strike up front.
  for (const q of quotes) {
    validate(F, q.strike, T, { alpha: 1, beta, rho: 0, nu: 0, shift });
    if (!(q.vol > 0 && q.vol < Infinity)) throw new RangeError(`quote vol must be positive, got ${q.vol}`);
    const w = q.weight ?? 1;
    if (!(w >= 0 && w < Infinity)) throw new RangeError(`quote weight must be ≥ 0, got ${w}`);
  }
  const model = normal ? normalVol : lognormalVol;
  const Fs = F + shift;

  // ATM vol seeds alpha: σ_ATM ≈ α / F^{1-β} (lognormal) or α·F^β (normal).
  const atm = [...quotes].sort((a, b) => Math.abs(a.strike - F) - Math.abs(b.strike - F))[0]!;
  const alpha0 =
    opts.initial?.alpha ?? (normal ? atm.vol / Math.pow(Fs, beta) : atm.vol * Math.pow(Fs, 1 - beta));
  const rho0 = opts.initial?.rho ?? 0;
  const nu0 = opts.initial?.nu ?? 0.3;
  // Errors are measured relative to the ATM vol so the simplex convergence
  // test behaves the same for 20% Black vols and 50bp normal vols.
  const scale = atm.vol;

  // Optimize in an unconstrained space: alpha=exp(a), rho=tanh(r), nu=exp(n).
  // rho is clamped so tanh rounding to ±1 can never leave the valid domain.
  const toParams = (v: number[]): SabrParams => ({
    alpha: Math.exp(v[0]!),
    beta,
    rho: clamp(Math.tanh(v[1]!), -0.9999, 0.9999),
    nu: Math.exp(v[2]!),
    shift,
  });
  const objective = (v: number[]): number => {
    const p = toParams(v);
    if (!(p.alpha > 0 && p.alpha < Infinity && p.nu < Infinity)) return Infinity;
    let sse = 0;
    for (const q of quotes) {
      const e = (model(F, q.strike, T, p) - q.vol) / scale;
      sse += (q.weight ?? 1) * e * e;
    }
    return Number.isNaN(sse) ? Infinity : sse;
  };

  const maxIter = opts.maxIter ?? 6000;
  // The SABR least-squares surface can have several basins (especially for
  // long expiries and high ν), so unless the caller pins a starting point the
  // search runs from a small grid of (ρ, ν) seeds and keeps the best.
  const seeds: [number, number][] =
    opts.initial?.rho !== undefined || opts.initial?.nu !== undefined
      ? [[rho0, nu0]]
      : [[rho0, nu0], ...[-0.8, 0, 0.8].flatMap((r): [number, number][] => [[r, 0.3], [r, 1.5]])];
  const budget = Math.max(1, Math.floor(maxIter / seeds.length));
  let point: number[] = [];
  let value = Infinity;
  let iterations = 0;
  for (const [r0, n0] of seeds) {
    let x = [Math.log(alpha0), Math.atanh(clamp(r0, -0.999, 0.999)), Math.log(Math.max(n0, 1e-6))];
    let fx = objective(x);
    let used = 0;
    // Nelder–Mead can stall on a degenerate simplex; restarting from the
    // incumbent with a fresh simplex lets it continue.
    for (let restart = 0; restart < 10 && used < budget; restart++) {
      const r = nelderMead(objective, x, budget - used);
      used += r.iterations;
      const improved = r.value < fx - 1e-15 * (1 + fx);
      if (r.value <= fx) {
        x = r.point;
        fx = r.value;
      }
      if (!improved) break;
    }
    iterations += used;
    if (fx < value) {
      point = x;
      value = fx;
    }
  }
  const totalW = quotes.reduce((s, q) => s + (q.weight ?? 1), 0);
  const rmse = scale * Math.sqrt(value / totalW);
  const fit = toParams(point);
  return { alpha: fit.alpha, beta, rho: fit.rho, nu: fit.nu, ...(shift ? { shift } : {}), rmse, iterations };
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
    p[i] = xi + (Math.abs(xi) > 1 ? 0.1 * xi : 0.1);
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
