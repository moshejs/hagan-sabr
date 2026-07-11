#!/usr/bin/env python3
"""Reference fixtures for the SABR implied-vol formulas.

- LOGNORMAL fixtures come from QuantLib's `sabrVolatility` (the authoritative
  Hagan 2002 Black-vol implementation) — an EXTERNAL gold standard.
- NORMAL and OBLOJ fixtures come from an independent Python transcription of
  the Hagan (2002) formulas below, guarding against TS transcription bugs.

Writes test/fixtures.json.
"""
import json
import math

import QuantLib as ql


def z_over_x(z, rho):
    if abs(z) < 1e-12:
        return 1 - rho * z / 2 + (2 - 3 * rho * rho) * z * z / 12
    x = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    return z / x


def normal_vol(F, K, T, alpha, beta, rho, nu):
    omb = 1 - beta
    FKpow = (F * K) ** omb
    FKhalf = (F * K) ** (omb / 2)
    c1 = -beta * (2 - beta) * alpha * alpha / (24 * FKpow)
    c2 = rho * beta * nu * alpha / (4 * FKhalf)
    c3 = (2 - 3 * rho * rho) * nu * nu / 24
    tf = 1 + (c1 + c2 + c3) * T
    level = alpha * (F * K) ** (beta / 2)
    if abs(F - K) < 1e-12 * max(1, F):
        return level * tf
    logfk = math.log(F / K)
    num = 1 + logfk**2 / 24 + logfk**4 / 1920
    den = 1 + omb**2 * logfk**2 / 24 + omb**4 * logfk**4 / 1920
    z = nu / alpha * FKhalf * logfk
    return level * (num / den) * z_over_x(z, rho) * tf


def obloj_vol(F, K, T, alpha, beta, rho, nu):
    omb = 1 - beta
    logfk = math.log(F / K)
    FKbeta = (F * K) ** (omb / 2)
    t1 = omb**2 * alpha * alpha / (24 * FKbeta * FKbeta)
    t2 = rho * beta * nu * alpha / (4 * FKbeta)
    t3 = (2 - 3 * rho * rho) * nu * nu / 24
    tf = 1 + (t1 + t2 + t3) * T
    if abs(logfk) < 1e-12:
        leading = alpha / F**omb
    elif omb < 1e-8:
        z = nu / alpha * logfk
        xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        leading = nu * logfk / xz
    else:
        z = nu / alpha * (F**omb - K**omb) / omb
        xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
        leading = nu * logfk / xz
    return leading * tf


PARAM_SETS = [
    dict(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4),
    dict(alpha=0.03, beta=0.0, rho=0.1, nu=0.5),
    dict(alpha=0.25, beta=1.0, rho=-0.5, nu=0.6),
    dict(alpha=0.15, beta=0.7, rho=0.0, nu=0.3),
    dict(alpha=0.4, beta=0.3, rho=-0.7, nu=0.8),
]
FORWARDS = [100.0, 0.02, 50.0]
T_LIST = [0.25, 1.0, 5.0]

lognormal, normal, obloj = [], [], []
for p in PARAM_SETS:
    for F in FORWARDS:
        for T in T_LIST:
            for moneyness in [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]:
                K = F * moneyness
                # QuantLib arg order: sabrVolatility(strike, forward, expiry, alpha, beta, nu, rho)
                try:
                    lv = ql.sabrVolatility(K, F, T, p["alpha"], p["beta"], p["nu"], p["rho"])
                except RuntimeError:
                    continue
                row = dict(F=F, K=K, T=T, **p)
                lognormal.append({**row, "vol": lv})
                normal.append({**row, "vol": normal_vol(F, K, T, p["alpha"], p["beta"], p["rho"], p["nu"])})
                obloj.append({**row, "vol": obloj_vol(F, K, T, p["alpha"], p["beta"], p["rho"], p["nu"])})

with open("test/fixtures.json", "w") as f:
    json.dump({"lognormal": lognormal, "normal": normal, "obloj": obloj}, f)
print(f"lognormal={len(lognormal)} (QuantLib) normal={len(normal)} obloj={len(obloj)}")
print("sample ATM lognormal:", round(lognormal[3]["vol"], 8) if len(lognormal) > 3 else None)
