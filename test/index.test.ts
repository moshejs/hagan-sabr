import { describe, expect, it } from "vitest";
import { calibrate, lognormalVol, normalVol, oblojVol } from "../src/index.js";
import fixtures from "./fixtures.json";

interface Fx {
  F: number;
  K: number;
  T: number;
  alpha: number;
  beta: number;
  rho: number;
  nu: number;
  vol: number;
}

const relErr = (got: number, ref: number) => Math.abs(got - ref) / Math.abs(ref);

describe("lognormal SABR — matches QuantLib's sabrVolatility", () => {
  it(`reproduces all ${fixtures.lognormal.length} QuantLib fixtures`, () => {
    let maxErr = 0;
    for (const f of fixtures.lognormal as Fx[]) {
      const got = lognormalVol(f.F, f.K, f.T, { alpha: f.alpha, beta: f.beta, rho: f.rho, nu: f.nu });
      const err = Math.abs(got - f.vol);
      maxErr = Math.max(maxErr, err);
      expect(err, `F=${f.F} K=${f.K} T=${f.T} β=${f.beta}`).toBeLessThan(1e-9);
    }
    // eslint-disable-next-line no-console
    expect(maxErr).toBeLessThan(1e-9);
  });

  it("ATM equals α / F^(1-β) to leading order", () => {
    // small T so the time correction is tiny
    const v = lognormalVol(100, 100, 1e-6, { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 });
    expect(v).toBeCloseTo(0.2 / Math.sqrt(100), 6); // 0.02
  });

  it("β=1 ATM is exactly α (lognormal)", () => {
    const v = lognormalVol(100, 100, 1e-8, { alpha: 0.25, beta: 1, rho: 0, nu: 0.5 });
    expect(v).toBeCloseTo(0.25, 6);
  });

  it("produces a smile (OTM wings above ATM for these params)", () => {
    const p = { alpha: 0.2, beta: 0.5, rho: 0, nu: 0.6 };
    const atm = lognormalVol(100, 100, 1, p);
    expect(lognormalVol(100, 80, 1, p)).toBeGreaterThan(atm);
    expect(lognormalVol(100, 125, 1, p)).toBeGreaterThan(atm);
  });
});

describe("normal (Bachelier) SABR — matches QuantLib's Normal shiftedSabrVolatility", () => {
  it(`reproduces all ${fixtures.normal.length} QuantLib fixtures`, () => {
    for (const f of fixtures.normal as Fx[]) {
      const got = normalVol(f.F, f.K, f.T, { alpha: f.alpha, beta: f.beta, rho: f.rho, nu: f.nu });
      expect(relErr(got, f.vol), `F=${f.F} K=${f.K}`).toBeLessThan(1e-9);
    }
  });

  it("ATM normal ≈ α·F^β to leading order", () => {
    const v = normalVol(100, 100, 1e-8, { alpha: 0.2, beta: 0.5, rho: 0, nu: 0.4 });
    expect(v).toBeCloseTo(0.2 * Math.pow(100, 0.5), 4); // 2.0
  });
});

describe("Obłój correction", () => {
  it(`reproduces all ${fixtures.obloj.length} reference fixtures (50-digit mpmath)`, () => {
    for (const f of fixtures.obloj as Fx[]) {
      const got = oblojVol(f.F, f.K, f.T, { alpha: f.alpha, beta: f.beta, rho: f.rho, nu: f.nu });
      expect(relErr(got, f.vol), `F=${f.F} K=${f.K} ν=${f.nu}`).toBeLessThan(1e-12);
    }
  });

  it("ν = 0 reduces to the CEV leading term α·ln(F/K)·(1−β)/(F^{1−β} − K^{1−β}), not 0", () => {
    const p = { alpha: 0.2, beta: 0.5, rho: 0, nu: 0 };
    const lead = (0.2 * Math.log(100 / 80) * 0.5) / (Math.sqrt(100) - Math.sqrt(80));
    const tf = 1 + ((0.25 * 0.04) / (24 * Math.sqrt(8000))) * 1;
    expect(oblojVol(100, 80, 1, p)).toBeCloseTo(lead * tf, 14);
    // β = 1, ν = 0 is flat Black: exactly α at every strike
    expect(oblojVol(100, 60, 2, { alpha: 0.3, beta: 1, rho: 0.2, nu: 0 })).toBeCloseTo(0.3, 14);
  });

  it("agrees closely with Hagan lognormal near ATM at short expiry", () => {
    const p = { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 };
    const hagan = lognormalVol(100, 101, 0.1, p);
    const obloj = oblojVol(100, 101, 0.1, p);
    expect(Math.abs(hagan - obloj)).toBeLessThan(2e-3);
  });
});

describe("near-ATM numerical stability", () => {
  type NearAtm = Omit<Fx, "vol"> & { lognormal: number; normal: number; obloj: number };
  it(`matches 50-digit references within 1e-13 of the money (${fixtures.near_atm.length} cases)`, () => {
    for (const f of fixtures.near_atm as NearAtm[]) {
      const msg = `F=${f.F} K=${f.K} β=${f.beta} ρ=${f.rho}`;
      expect(relErr(lognormalVol(f.F, f.K, f.T, f), f.lognormal), msg).toBeLessThan(1e-13);
      expect(relErr(normalVol(f.F, f.K, f.T, f), f.normal), msg).toBeLessThan(1e-13);
      expect(relErr(oblojVol(f.F, f.K, f.T, f), f.obloj), msg).toBeLessThan(1e-13);
    }
  });

  it("is continuous through the forward (regression: x(z) cancellation)", () => {
    // Previously off by 3e-4 here (0.2563188 vs QuantLib/exact 0.2563984).
    const p = { alpha: 0.0075, beta: 0, rho: 0.9, nu: 0.05 };
    expect(lognormalVol(0.03, 0.0299999999997, 10, p)).toBeCloseTo(0.25639843750111635, 13);
    const atm = lognormalVol(0.03, 0.03, 10, p);
    for (const eps of [1e-15, 1e-12, 1e-10, 1e-8]) {
      expect(Math.abs(lognormalVol(0.03, 0.03 * (1 + eps), 10, p) - atm)).toBeLessThan(eps + 1e-15); // |dσ/dlnK| ≈ 0.1 here
    }
  });
});

describe("shifted SABR (negative rates) — matches QuantLib's shiftedSabrVolatility", () => {
  type Shifted = Omit<Fx, "vol"> & { shift: number; lognormal: number; normal: number };
  it(`reproduces all ${fixtures.shifted.length} shifted fixtures, lognormal and normal`, () => {
    for (const f of fixtures.shifted as Shifted[]) {
      const msg = `F=${f.F} K=${f.K} shift=${f.shift}`;
      expect(relErr(lognormalVol(f.F, f.K, f.T, f), f.lognormal), msg).toBeLessThan(1e-9);
      expect(relErr(normalVol(f.F, f.K, f.T, f), f.normal), msg).toBeLessThan(1e-9);
    }
  });

  it("shift = 0 is the unshifted model; F + shift and K + shift must be positive", () => {
    const p = { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 };
    expect(lognormalVol(100, 90, 1, { ...p, shift: 0 })).toBe(lognormalVol(100, 90, 1, p));
    expect(lognormalVol(-0.01, 0.005, 1, { ...p, shift: 0.02 })).toBe(lognormalVol(0.01, 0.025, 1, p));
    expect(() => lognormalVol(-0.01, 0.01, 1, { ...p, shift: 0.01 })).toThrow(RangeError);
    expect(() => normalVol(0.01, -0.03, 1, { ...p, shift: 0.02 })).toThrow(RangeError);
    expect(() => normalVol(0.01, 0.01, 1, { ...p, shift: -0.01 })).toThrow(RangeError);
  });
});

describe("calibration", () => {
  it("recovers SABR parameters from a synthetic smile", () => {
    const F = 100;
    const T = 2;
    const truth = { alpha: 0.22, beta: 0.5, rho: -0.35, nu: 0.55 };
    const strikes = [70, 80, 90, 100, 110, 120, 135];
    const quotes = strikes.map((strike) => ({
      strike,
      vol: lognormalVol(F, strike, T, truth),
    }));

    const fit = calibrate(F, T, quotes, { beta: 0.5 });
    expect(fit.rmse).toBeLessThan(1e-4);
    expect(fit.alpha).toBeCloseTo(truth.alpha, 3);
    expect(fit.rho).toBeCloseTo(truth.rho, 2);
    expect(fit.nu).toBeCloseTo(truth.nu, 2);

    // the fitted surface reprices the quotes
    for (const q of quotes) {
      expect(lognormalVol(F, q.strike, T, fit)).toBeCloseTo(q.vol, 4);
    }
  });

  it("escapes the local minimum a single Nelder–Mead start fell into", () => {
    // Regression: a single start converged to rmse ≈ 0.046 on this smile.
    const F = 0.03;
    const T = 8.18589848558693;
    const truth = { alpha: 0.07865209337822102, beta: 0.7, rho: -0.22638339843898236, nu: 1.111852030019207 };
    const quotes = [0.5, 0.7, 0.85, 0.95, 1, 1.05, 1.15, 1.3, 1.6].map((m) => ({
      strike: F * m,
      vol: lognormalVol(F, F * m, T, truth),
    }));
    const fit = calibrate(F, T, quotes, { beta: 0.7 });
    expect(fit.rmse).toBeLessThan(1e-6);
    expect(fit.rho).toBeCloseTo(truth.rho, 3);
    expect(fit.nu).toBeCloseTo(truth.nu, 3);
  });

  it("calibrates shifted SABR to normal-vol quotes on negative strikes", () => {
    const F = -0.002;
    const T = 5;
    const truth = { alpha: 0.012, beta: 0.3, rho: -0.2, nu: 0.45, shift: 0.02 };
    const quotes = [-0.012, -0.008, -0.004, -0.002, 0, 0.004, 0.01, 0.02].map((strike) => ({
      strike,
      vol: normalVol(F, strike, T, truth),
    }));
    const fit = calibrate(F, T, quotes, { beta: 0.3, shift: 0.02, volType: "normal" });
    expect(fit.rmse).toBeLessThan(1e-8);
    expect(fit.shift).toBe(0.02);
    expect(fit.alpha).toBeCloseTo(truth.alpha, 5);
    expect(fit.rho).toBeCloseTo(truth.rho, 3);
    expect(fit.nu).toBeCloseTo(truth.nu, 3);
    for (const q of quotes) expect(normalVol(F, q.strike, T, fit)).toBeCloseTo(q.vol, 8);
  });

  it("requires at least three valid quotes", () => {
    expect(() => calibrate(100, 1, [{ strike: 100, vol: 0.2 }], {})).toThrow(RangeError);
    const q = [90, 100, 110].map((strike) => ({ strike, vol: 0.2 }));
    expect(() => calibrate(100, 1, [...q, { strike: -5, vol: 0.2 }])).toThrow(RangeError);
    expect(() => calibrate(100, 1, [...q, { strike: 105, vol: Number.NaN }])).toThrow(RangeError);
    expect(() => calibrate(100, 1, q, { beta: 2 })).toThrow(RangeError);
  });
});

describe("validation", () => {
  it("rejects non-positive F/K/T/alpha and out-of-range rho/beta", () => {
    const p = { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 };
    expect(() => lognormalVol(0, 100, 1, p)).toThrow(RangeError);
    expect(() => lognormalVol(100, -1, 1, p)).toThrow(RangeError);
    expect(() => lognormalVol(100, 100, 0, p)).toThrow(RangeError);
    expect(() => lognormalVol(100, 100, 1, { ...p, alpha: 0 })).toThrow(RangeError);
    expect(() => lognormalVol(100, 100, 1, { ...p, beta: 1.5 })).toThrow(RangeError);
    expect(() => lognormalVol(100, 100, 1, { ...p, rho: 1 })).toThrow(RangeError);
  });

  it("rejects NaN parameters instead of returning NaN", () => {
    const p = { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 };
    for (const f of [lognormalVol, normalVol, oblojVol]) {
      expect(() => f(100, 100, 1, { ...p, beta: Number.NaN })).toThrow(RangeError);
      expect(() => f(100, 100, 1, { ...p, rho: Number.NaN })).toThrow(RangeError);
      expect(() => f(100, 100, 1, { ...p, nu: Number.NaN })).toThrow(RangeError);
      expect(() => f(Number.POSITIVE_INFINITY, 100, 1, p)).toThrow(RangeError);
    }
  });
});
