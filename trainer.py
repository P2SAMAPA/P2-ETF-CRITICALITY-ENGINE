import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from criticality import compute_criticality

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Criticality Engine) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": [], "criticality_results": {}}
            continue

        universe_crit = {}
        for win in config.WINDOWS:
            if len(returns) < win:
                continue
            crit = compute_criticality(returns, win)
            if crit is None:
                continue
            universe_crit[f"window_{win}"] = crit
            gamma = crit['gamma']
            if gamma is not None:
                print(f"  Window {win}d: gamma={gamma:.3f}, cash_alloc={crit['cash_allocation']:.0%}")
            else:
                print(f"  Window {win}d: gamma insufficient data, cash_alloc={crit['cash_allocation']:.0%}")
            # print top ETFs for this window
            top = crit['top_etfs']
            if top:
                print(f"    Top ETFs: {[e['ticker'] for e in top]}")

        # For dashboard, choose the largest window that succeeded
        best_win = None
        for win in sorted(config.WINDOWS, reverse=True):
            if f"window_{win}" in universe_crit:
                best_win = win
                break
        if best_win:
            all_results[universe_name] = {
                "criticality_results": universe_crit,
                "top_etfs": universe_crit[f"window_{best_win}"]["top_etfs"],
                "cash_allocation": universe_crit[f"window_{best_win}"]["cash_allocation"],
                "run_date": today
            }
        else:
            all_results[universe_name] = {"top_etfs": [], "criticality_results": {}}

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/criticality_{today}.json")
    with open(local_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Neuronal Avalanche Criticality Engine complete ===")

if __name__ == "__main__":
    main()
