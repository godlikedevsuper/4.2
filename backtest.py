"""日経平均 4.2倍ブル(合成) × 日経平均200日移動平均クロス戦略のバックテスト.

- シグナル: 日経平均(元指数)の終値が200日SMAを上回れば保有、下回れば全額現金。
- 4.2倍ブルは日次リターン×4.2(毎日リバランス)で合成。信託報酬などのコストを日割りで控除。
- 執行ラグ: 終値でシグナル確定 → 翌営業日に発注 → その日の終値(基準価額)で約定。
  つまり t日のシグナルは t+2日のリターンから反映 (lag=2)。比較用に lag=1 も出す。
- 現金部分の金利は0とする。
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEV = 4.2
FEE = 0.01          # 年率コスト (信託報酬など)
TRADE_COST = 0.001  # 売買1回あたり (片道)
MA = 200
START = sys.argv[1] if len(sys.argv) > 1 else "1965-01-01"


def load():
    d = yf.download("^N225", start="1960-01-01", auto_adjust=False, progress=False)
    c = d["Close"].squeeze().dropna()
    return c[c > 0]


def synth_lev(r, lev=LEV, fee=FEE):
    # 1日で指数が -1/lev 以上下落したら基準価額ゼロ (償還)
    lr = (lev * r).clip(lower=-1.0) - fee / 245
    return lr


def run(close, lev, lag, fee=FEE, trade_cost=TRADE_COST, band=0.0):
    r = close.pct_change().fillna(0)
    ma = close.rolling(MA).mean()
    sig = pd.Series(np.nan, index=close.index)
    sig[close > ma * (1 + band)] = 1
    sig[close < ma * (1 - band)] = 0
    sig = sig.ffill().fillna(0)
    pos = sig.shift(lag).fillna(0)
    lr = synth_lev(r, lev, fee) if lev != 1 else r - fee / 245 * 0
    trades = pos.diff().abs().fillna(0)
    strat = pos * lr - trades * trade_cost
    return strat, pos, trades


def stats(ret, pos=None, trades=None):
    ret = ret.dropna()
    eq = (1 + ret).cumprod()
    yrs = len(ret) / 245
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1
    vol = ret.std() * np.sqrt(245)
    dd = (eq / eq.cummax() - 1).min()
    sharpe = ret.mean() / ret.std() * np.sqrt(245) if ret.std() > 0 else np.nan
    out = dict(最終倍率=eq.iloc[-1], CAGR=cagr, 年率Vol=vol, 最大DD=dd, Sharpe=sharpe,
               MAR=cagr / abs(dd) if dd < 0 else np.nan)
    if pos is not None:
        out["保有率"] = pos.loc[ret.index].mean()
        out["売買回数/年"] = trades.loc[ret.index].sum() / yrs
    return out


def main():
    close = load()
    close = close[close.index >= pd.Timestamp(START) - pd.Timedelta(days=400)]
    r = close.pct_change().fillna(0)
    idx = close.index[close.index >= START]
    idx = idx[MA:] if START <= str(close.index[0].date()) else idx

    rows, curves = {}, {}
    def add(name, ret, pos=None, tr=None):
        ret = ret.loc[idx]
        rows[name] = stats(ret, pos, tr)
        curves[name] = (1 + ret).cumprod()

    add("日経平均 B&H", r)
    add(f"{LEV}倍 B&H", synth_lev(r))
    s, p, t = run(close, 1, 2); add("日経 1倍 × 200MA", s, p, t)
    s, p, t = run(close, LEV, 2); add(f"{LEV}倍 × 200MA (lag2)", s, p, t)
    s, p, t = run(close, LEV, 1); add(f"{LEV}倍 × 200MA (lag1,理想)", s, p, t)
    s, p, t = run(close, LEV, 2, band=0.02); add(f"{LEV}倍 × 200MA ±2%バンド", s, p, t)
    for f in (0.0, 0.03):
        s, p, t = run(close, LEV, 2, fee=f); add(f"{LEV}倍 × 200MA 年コスト{f:.0%}", s, p, t)

    df = pd.DataFrame(rows).T
    fmt = df.copy()
    for c in ["CAGR", "年率Vol", "最大DD", "保有率"]:
        fmt[c] = df[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    fmt["最終倍率"] = df["最終倍率"].map(lambda x: f"{x:,.1f}")
    for c in ["Sharpe", "MAR", "売買回数/年"]:
        fmt[c] = df[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(f"期間: {idx[0].date()} 〜 {idx[-1].date()}  ({len(idx)/245:.1f}年)")
    print(fmt.to_markdown())

    # 年別リターン (4.2倍×MA vs 日経)
    s, p, t = run(close, LEV, 2)
    yr = pd.DataFrame({"日経": r.loc[idx], "4.2x×MA": s.loc[idx], "4.2x B&H": synth_lev(r).loc[idx]})
    yr = (1 + yr).groupby(yr.index.year).prod() - 1
    tag = START[:4]
    yr.map(lambda x: f"{x:.1%}").to_markdown(open(f"results/yearly_{tag}.md", "w"))

    # 最悪日 (保有中)
    held = s.loc[idx][p.loc[idx] == 1]
    worst = held.nsmallest(5)
    print("\n保有中の最悪日:")
    for d, v in worst.items():
        print(f"  {d.date()}  戦略 {v:.1%}  (日経 {r[d]:.1%})")

    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for k in ["日経平均 B&H", f"{LEV}倍 B&H", "日経 1倍 × 200MA", f"{LEV}倍 × 200MA (lag2)"]:
        ax[0].plot(curves[k].clip(lower=1e-4), label=k)
        eq = curves[k]
        ax[1].plot(eq / eq.cummax() - 1, label=k)
    ax[0].set_yscale("log"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title(f"Nikkei 225: {LEV}x synthetic vs 200MA timing ({idx[0].year}-{idx[-1].year})")
    ax[1].set_ylabel("Drawdown"); ax[1].grid(alpha=.3)
    ax[0].legend([l.replace("日経平均", "N225").replace("日経", "N225").replace("倍", "x").replace("×", "x") for l in ax[0].get_legend_handles_labels()[1]])
    fig.tight_layout(); fig.savefig(f"results/equity_{tag}.png", dpi=110)


if __name__ == "__main__":
    main()
