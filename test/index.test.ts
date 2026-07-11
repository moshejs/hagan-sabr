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

describe("normal (Bachelier) SABR — matches the reference formula", () => {
  it(`reproduces all ${fixtures.normal.length} reference fixtures`, () => {
    for (const f of fixtures.normal as Fx[]) {
      const got = normalVol(f.F, f.K, f.T, { alpha: f.alpha, beta: f.beta, rho: f.rho, nu: f.nu });
      expect(Math.abs(got - f.vol), `F=${f.F} K=${f.K}`).toBeLessThan(1e-12 * Math.max(1, f.F));
    }
  });

  it("ATM normal ≈ α·F^β to leading order", () => {
    const v = normalVol(100, 100, 1e-8, { alpha: 0.2, beta: 0.5, rho: 0, nu: 0.4 });
    expect(v).toBeCloseTo(0.2 * Math.pow(100, 0.5), 4); // 2.0
  });
});

describe("Obłój correction", () => {
  it(`reproduces all ${fixtures.obloj.length} reference fixtures`, () => {
    for (const f of fixtures.obloj as Fx[]) {
      const got = oblojVol(f.F, f.K, f.T, { alpha: f.alpha, beta: f.beta, rho: f.rho, nu: f.nu });
      expect(Math.abs(got - f.vol), `F=${f.F} K=${f.K}`).toBeLessThan(1e-12);
    }
  });

  it("agrees closely with Hagan lognormal near ATM at short expiry", () => {
    const p = { alpha: 0.2, beta: 0.5, rho: -0.3, nu: 0.4 };
    const hagan = lognormalVol(100, 101, 0.1, p);
    const obloj = oblojVol(100, 101, 0.1, p);
    expect(Math.abs(hagan - obloj)).toBeLessThan(2e-3);
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

  it("requires at least three quotes", () => {
    expect(() => calibrate(100, 1, [{ strike: 100, vol: 0.2 }], {})).toThrow(RangeError);
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
});
