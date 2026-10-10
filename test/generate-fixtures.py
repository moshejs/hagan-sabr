#!/usr/bin/env python3
"""Reference fixtures for the SABR implied-vol formulas.

- lognormal: QuantLib `sabrVolatility` (Hagan 2002 Black vol).
- normal:    QuantLib `shiftedSabrVolatility(..., shift=0, VolatilityType.Normal)`
             (Hagan 2002 normal vol, same simplified form as `normalVol`).
- shifted:   QuantLib `shiftedSabrVolatility` with a positive shift and negative
             forwards/strikes, in both ShiftedLognormal and Normal flavours.
- obloj:     a 50-digit mpmath transcription of Oblój (2008), "Fine-tune your
             smile: correction to Hagan et al.", eq. for sigma_0 incl. the nu=0
             limit (QuantLib has no Oblój variant).
- near_atm:  50-digit mpmath values of the Hagan lognormal / normal and Oblój
             formulas for strikes within 1e-13..1e-5 of the forward, where
             double-precision evaluation of x(z) cancels catastrophically.
             (QuantLib is not used here: its own two-term series cutover is off
             by up to ~3e-8 relative in this region.)

Usage: python -I test/generate-fixtures.py   (needs QuantLib, mpmath)
Writes test/fixtures.json.
"""
import json

import mpmath as mp
import QuantLib as ql

mp.mp.dps = 50


def xz(z, rho):
    return mp.log((mp.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))


def zox(z, rho):
    return mp.mpf(1) if z == 0 else z / xz(z, rho)


def time_factor_ln(F, K, T, a, b, r, n):
    omb = 1 - b
    fk = (F * K) ** (omb / 2)
    return 1 + (omb**2 * a * a / (24 * fk * fk) + r * b * n * a / (4 * fk) + (2 - 3 * r * r) * n * n / 24) * T


def hagan_ln(F, K, T, a, b, r, n):
    F, K, T, a, b, r, n = map(mp.mpf, (F, K, T, a, b, r, n))
    omb = 1 - b
    L = mp.log(F / K)
    fk = (F * K) ** (omb / 2)
    den = fk * (1 + omb**2 * L**2 / 24 + omb**4 * L**4 / 1920)
    return a / den * zox(n / a * fk * L, r) * time_factor_ln(F, K, T, a, b, r, n)


def hagan_n(F, K, T, a, b, r, n):
    F, K, T, a, b, r, n = map(mp.mpf, (F, K, T, a, b, r, n))
    omb = 1 - b
    L = mp.log(F / K)
    tf = 1 + (-b * (2 - b) * a * a / (24 * (F * K) ** omb) + r * b * n * a / (4 * (F * K) ** (omb / 2))
              + (2 - 3 * r * r) * n * n / 24) * T
    num = 1 + L**2 / 24 + L**4 / 1920
    den = 1 + omb**2 * L**2 / 24 + omb**4 * L**4 / 1920
    return a * (F * K) ** (b / 2) * num / den * zox(n / a * (F * K) ** (omb / 2) * L, r) * tf


def obloj(F, K, T, a, b, r, n):
    F, K, T, a, b, r, n = map(mp.mpf, (F, K, T, a, b, r, n))
    omb = 1 - b
    L = mp.log(F / K)
    if F == K:
        lead = a / F**omb
    else:
        q = L if omb == 0 else (F**omb - K**omb) / omb  # int_K^F x^-beta dx
        lead = a * L / q if n == 0 else n * L / xz(n * q / a, r)
    return lead * time_factor_ln(F, K, T, a, b, r, n)


PARAM_SETS = [
    dict(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4),
    dict(alpha=0.03, beta=0.0, rho=0.1, nu=0.5),
    dict(alpha=0.25, beta=1.0, rho=-0.5, nu=0.6),
    dict(alpha=0.15, beta=0.7, rho=0.0, nu=0.3),
    dict(alpha=0.4, beta=0.3, rho=-0.7, nu=0.8),
    dict(alpha=0.1, beta=0.5, rho=0.4, nu=0.0),  # nu = 0: pure CEV
]
FORWARDS = [100.0, 0.02, 50.0]
T_LIST = [0.25, 1.0, 5.0]

out = {"lognormal": [], "normal": [], "shifted": [], "obloj": [], "near_atm": []}
for p in PARAM_SETS:
    a, b, r, n = p["alpha"], p["beta"], p["rho"], p["nu"]
    for F in FORWARDS:
        for T in T_LIST:
            for m in [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]:
                K = F * m
                row = dict(F=F, K=K, T=T, **p)
                # QuantLib arg order: (strike, forward, expiry, alpha, beta, nu, rho)
                out["lognormal"].append({**row, "vol": ql.sabrVolatility(K, F, T, a, b, n, r)})
                out["normal"].append({**row, "vol": ql.shiftedSabrVolatility(K, F, T, a, b, n, r, 0.0, ql.Normal)})
                out["obloj"].append({**row, "vol": float(obloj(F, K, T, a, b, r, n))})

# Shifted SABR with negative forwards/strikes (EUR/JPY/CHF-style rates).
for p in PARAM_SETS:
    a, b, r, n = p["alpha"], p["beta"], p["rho"], p["nu"]
    a = a * 0.02 ** (1 - b) / 100 ** (1 - b) * 5  # rescale alpha for rate-sized forwards
    for shift in [0.01, 0.03]:
        for F in [-0.005, 0.0, 0.004]:
            for T in [0.5, 5.0]:
                for K in [-0.0075, -0.004, -0.001, 0.0, 0.002, 0.006, 0.015]:
                    if F + shift <= 0 or K + shift <= 0:
                        continue
                    row = dict(F=F, K=K, T=T, alpha=a, beta=b, rho=r, nu=n, shift=shift)
                    out["shifted"].append({
                        **row,
                        "lognormal": ql.shiftedSabrVolatility(K, F, T, a, b, n, r, shift, ql.ShiftedLognormal),
                        "normal": ql.shiftedSabrVolatility(K, F, T, a, b, n, r, shift, ql.Normal),
                    })

for p in PARAM_SETS + [dict(alpha=0.3, beta=0.9, rho=0.9, nu=0.5), dict(alpha=0.05, beta=0.0, rho=-0.95, nu=1.5)]:
    a, b, r, n = p["alpha"], p["beta"], p["rho"], p["nu"]
    for F in [0.03, 1.0, 100.0]:
        for eps in [1e-13, 1e-11, 1e-9, 1e-7, 1e-5]:
            for K in [F * (1 - eps), F * (1 + eps)]:
                aa = a * F ** (1 - b)  # keep the vol level O(a)
                out["near_atm"].append(dict(
                    F=F, K=K, T=10.0, alpha=aa, beta=b, rho=r, nu=n,
                    lognormal=float(hagan_ln(F, K, 10.0, aa, b, r, n)),
                    normal=float(hagan_n(F, K, 10.0, aa, b, r, n)),
                    obloj=float(obloj(F, K, 10.0, aa, b, r, n)),
                ))

out["_meta"] = {
    "generator": "test/generate-fixtures.py",
    "QuantLib": ql.__version__,
    "mpmath": mp.__version__,
}
with open("test/fixtures.json", "w") as f:
    json.dump(out, f)
print({k: len(v) for k, v in out.items() if k != "_meta"})
