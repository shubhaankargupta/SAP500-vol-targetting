import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
lookback = 5
ann = 252
target_vol = 0.4
leverage_cap = 1
close = yf.download("SPY", start="2015-01-01", end="2025-08-18", auto_adjust=True)["Close"]

ret = close.pct_change().dropna().astype(float)
realized = ret.rolling(lookback).std() * np.sqrt(ann)

pos = (target_vol / realized. replace (0.0, np.nan)).clip(0.0, leverage_cap).shift(1).fillna (0.0)

bh_ret = ret
strat_ret = pos * ret

def equity(r):
    return (1.0 + r).cumprod()

eq_strat = equity(strat_ret)
eq_bh = equity(bh_ret)


def stats(r):
    r = r.dropna()
    years = (r.index[-1] - r.index[0]).days / 365.25
    total = (1.0 + r).prod().item() if hasattr((1.0 + r).prod(), "item") else float((1.0 + r).prod())
    cagr = total**(1.0 / years) - 1.0 if years > 0 else np.nan

    sd = r.std().item()
    mu = r.mean().item()
    sharpe = (mu / sd) * np.sqrt(ann) if sd > 0 else np.nan
    vol = sd * np.sqrt(ann)

    curve = (1.0 + r).cumprod()
    mdd = (curve / curve.cummax() - 1.0).min().item()
    return pd.Series({"CAGR": cagr, "Sharpe": sharpe, "Vol": vol, "MaxDD": mdd})

print(pd.concat({"Vol Target": stats(strat_ret), "Buy & Hold": stats(bh_ret)}, axis=1).round(3))

plt.figure(figsize=(10.5, 4.6))
plt.plot(eq_bh.index, eq_bh.values, label="Buy & Hold (SPY)")
plt.plot(eq_strat.index, eq_strat.values, label=f"Vol Target ({int(target_vol*100)}% cap {leverage_cap}x)")
plt.title("Volatility Targeting vs Buy & Hold")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10.5, 2.8))
plt.plot(pos.index, pos.values)
plt.title("Position Size (TargetVol / RealizedVol, capped)")
plt.tight_layout(); plt.show()