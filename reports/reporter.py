"""
Report Generator
================
Generates a beautiful text + HTML performance report for the trading system.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_report(results: dict, strategy_name: str, symbol: str,
                    timeframe: str, df: pd.DataFrame,
                    output_dir: Path = None) -> str:
    """Generate a comprehensive trading report."""
    lines = []
    w = 72  # Width

    def header(title):
        lines.append("=" * w)
        lines.append(f"  {title}")
        lines.append("=" * w)

    def sub(title):
        lines.append(f"\n── {title} {'─' * (w - len(title) - 4)}")

    def row(label, value, highlight=False):
        prefix = "  ★ " if highlight else "    "
        lines.append(f"{prefix}{label:<35} {value}")

    # ── Header ────────────────────────────────────────────────────────
    header(f"CRYPTO TRADING SYSTEM — PERFORMANCE REPORT")
    lines.append(f"  Strategy : {strategy_name}")
    lines.append(f"  Symbol   : {symbol}")
    lines.append(f"  Timeframe: {timeframe}")
    lines.append(f"  Period   : {df.index[0].date()} → {df.index[-1].date()}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * w)

    if 'error' in results:
        lines.append(f"\n  ERROR: {results['error']}")
        return "\n".join(lines)

    metrics = results.get('metrics', results)

    # ── P&L Summary ───────────────────────────────────────────────────
    sub("PROFIT & LOSS SUMMARY")
    ret = metrics.get('total_return_pct', 0)
    ic  = 10000  # Assume $10k
    row("Initial Capital", f"${ic:,.0f}")
    row("Final Capital",   f"${metrics.get('final_capital', 0):,.0f}")
    row("Total Return",    f"{ret:+.2f}%", highlight=ret > 0)
    row("Total Fees Paid", f"${metrics.get('total_fees', 0):,.2f}")
    row("Net P&L",         f"${metrics.get('final_capital', ic) - ic:+,.2f}", highlight=True)

    # ── Risk Metrics ──────────────────────────────────────────────────
    sub("RISK METRICS")
    sharpe = metrics.get('sharpe', 0)
    dd     = metrics.get('max_drawdown_pct', 0)
    pf     = metrics.get('profit_factor', 0)
    row("Sharpe Ratio",   f"{sharpe:.3f}  {'✅ Good' if sharpe > 1 else '⚠️  Below 1' if sharpe > 0 else '❌ Negative'}")
    row("Calmar Ratio",   f"{metrics.get('calmar', 0):.3f}")
    row("Profit Factor",  f"{pf:.3f}   {'✅ > 1.5' if pf > 1.5 else '⚠️  Marginal' if pf > 1 else '❌ < 1'}", highlight=pf > 1.5)
    row("Max Drawdown",   f"{dd:.2f}%  {'✅ < 15%' if dd < 15 else '⚠️  < 25%' if dd < 25 else '❌ High'}")
    row("Expectancy",     f"${metrics.get('expectancy', 0):.2f} per trade")

    # ── Trade Statistics ──────────────────────────────────────────────
    sub("TRADE STATISTICS")
    wr  = metrics.get('win_rate', 0)
    n   = metrics.get('total_trades', 0)
    row("Total Trades",   f"{n}")
    row("Win Rate",       f"{wr:.1f}%  {'✅' if wr > 55 else '⚠️ '}", highlight=wr > 55)
    row("Avg Win",        f"${metrics.get('avg_win_usd', 0):.2f}")
    row("Avg Loss",       f"${metrics.get('avg_loss_usd', 0):.2f}")
    row("Win/Loss Ratio", f"{metrics.get('win_loss_ratio', 0):.3f}")
    row("Avg Trade %",    f"{metrics.get('avg_trade_pct', 0):+.4f}%")
    row("Avg Duration",   f"{metrics.get('avg_duration_h', 0):.1f} hours")

    # ── Exit Analysis ─────────────────────────────────────────────────
    sub("EXIT ANALYSIS")
    tp  = metrics.get('tp_exits', 0)
    sl  = metrics.get('sl_exits', 0)
    sig = metrics.get('signal_exits', 0)
    total_exits = tp + sl + sig or 1
    row("Take-Profit Exits", f"{tp} ({tp/total_exits*100:.0f}%)")
    row("Stop-Loss Exits",   f"{sl} ({sl/total_exits*100:.0f}%)")
    row("Signal Exits",      f"{sig} ({sig/total_exits*100:.0f}%)")

    # ── Walk-Forward Summary ──────────────────────────────────────────
    if 'fold_metrics' in results:
        sub("WALK-FORWARD VALIDATION")
        row("Number of Folds",    f"{results.get('n_folds', 0)}")
        row("Avg Win Rate",       f"{results.get('avg_win_rate', 0):.1f}%")
        row("Avg Profit Factor",  f"{results.get('avg_profit_factor', 0):.3f}")
        row("Avg Sharpe",         f"{results.get('avg_sharpe', 0):.3f}")
        row("Avg Max Drawdown",   f"{results.get('avg_max_dd', 0):.2f}%")

        lines.append("\n  Fold-by-Fold Results:")
        lines.append(f"  {'Fold':>4}  {'Trades':>6}  {'WinRate':>7}  {'PF':>6}  {'Return%':>8}  {'Sharpe':>7}  {'MaxDD%':>7}")
        lines.append(f"  {'─'*4}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*7}")
        for fm in results['fold_metrics']:
            lines.append(
                f"  {fm['fold']:>4}  {fm['total_trades']:>6}  "
                f"{fm['win_rate']:>6.1f}%  {fm['profit_factor']:>6.3f}  "
                f"{fm['total_return_pct']:>+8.2f}%  {fm['sharpe']:>7.3f}  "
                f"{fm['max_drawdown_pct']:>6.2f}%"
            )

    # ── Overall Assessment ────────────────────────────────────────────
    sub("OVERALL ASSESSMENT")
    score = 0
    checks = []
    if wr > 50:    score += 1; checks.append("✅ Win rate > 50%")
    else:          checks.append("❌ Win rate < 50%")
    if pf > 1.5:   score += 1; checks.append("✅ Profit factor > 1.5")
    elif pf > 1:   checks.append("⚠️  Profit factor 1-1.5")
    else:          checks.append("❌ Profit factor < 1")
    if sharpe > 1: score += 1; checks.append("✅ Sharpe > 1.0")
    elif sharpe > 0: checks.append("⚠️  Sharpe 0-1")
    else:          checks.append("❌ Negative Sharpe")
    if dd < 20:    score += 1; checks.append("✅ Max drawdown < 20%")
    else:          checks.append("⚠️  Max drawdown > 20%")
    if ret > 0:    score += 1; checks.append("✅ Positive ROI")
    else:          checks.append("❌ Negative ROI")

    for c in checks:
        lines.append(f"  {c}")

    verdict = {5: "🏆 EXCELLENT", 4: "✅ GOOD", 3: "📊 ACCEPTABLE", 2: "⚠️  NEEDS WORK", 1: "❌ POOR", 0: "❌ FAILING"}
    lines.append(f"\n  Overall Score: {score}/5 — {verdict.get(score, '?')}")
    lines.append("\n" + "=" * w)
    lines.append("  ⚠️  DISCLAIMER: Past performance does not guarantee future results.")
    lines.append("  Always paper-trade first. Never risk money you cannot afford to lose.")
    lines.append("=" * w)

    report = "\n".join(lines)

    if output_dir:
        out_file = output_dir / f"report_{strategy_name}_{symbol.replace('/', '_')}.txt"
        out_file.write_text(report)
        print(f"[Report] Saved to {out_file}")

    return report


def compare_strategies(results: dict) -> str:
    """Generate a comparison table for multiple strategies."""
    lines = []
    w = 88
    lines.append("=" * w)
    lines.append("  STRATEGY COMPARISON")
    lines.append("=" * w)
    lines.append(f"  {'Strategy':<30} {'Return%':>8} {'WinRate':>8} {'PF':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Trades':>7}")
    lines.append(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for name, result in results.items():
        m = result.get('metrics', {})
        if 'fold_metrics' in result:  # Walk-forward result
            m = {
                'total_return_pct': sum(f['total_return_pct'] for f in result['fold_metrics']),
                'win_rate': result.get('avg_win_rate', 0),
                'profit_factor': result.get('avg_profit_factor', 0),
                'sharpe': result.get('avg_sharpe', 0),
                'max_drawdown_pct': result.get('avg_max_dd', 0),
                'total_trades': result.get('total_trades', 0),
            }
        if 'error' in m:
            lines.append(f"  {name:<30} {'ERROR':>8}")
            continue
        lines.append(
            f"  {name:<30} "
            f"{m.get('total_return_pct', 0):>+7.2f}% "
            f"{m.get('win_rate', 0):>7.1f}% "
            f"{m.get('profit_factor', 0):>7.3f} "
            f"{m.get('sharpe', 0):>7.3f} "
            f"{m.get('max_drawdown_pct', 0):>6.2f}% "
            f"{m.get('total_trades', 0):>7}"
        )

    lines.append("=" * w)
    return "\n".join(lines)
