"""社交導航模擬執行入口。

範例：
  python run.py                       # 預設 crossing 情境，開即時畫面
  python run.py --scenario hallway    # 指定情境
  python run.py --no-render           # 無畫面快速跑，印出評分
  python run.py --plot out.png        # 另存軌跡圖
  python run.py --all --no-render     # 一次跑完所有情境並比較分數
"""
from __future__ import annotations

import argparse
import json

from socnav.scenarios import SCENARIOS
from socnav.simulator import Simulator


def run_one(name: str, render: bool, plot: str | None, max_time: float) -> dict:
    world = SCENARIOS[name]()
    sim = Simulator(world, max_time=max_time)
    result = sim.run_render() if render else sim.run_headless()
    if plot:
        sim.save_plot(plot)
        print(f"[plot] 已存軌跡圖：{plot}")
    return result


def main():
    ap = argparse.ArgumentParser(description="輕量社交導航模擬 (SocNavBench 風格)")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="crossing")
    ap.add_argument("--no-render", action="store_true", help="不開畫面，快速跑")
    ap.add_argument("--plot", default=None, help="輸出軌跡 PNG 路徑")
    ap.add_argument("--all", action="store_true", help="跑完所有情境 (隱含 --no-render)")
    ap.add_argument("--max-time", type=float, default=60.0)
    args = ap.parse_args()

    if args.all:
        rows = {}
        for name in SCENARIOS:
            res = run_one(name, render=False, plot=None, max_time=args.max_time)
            rows[name] = res
            print(f"\n=== {name} ===")
            print(json.dumps(res, ensure_ascii=False, indent=2))
        print("\n=== 總覽 ===")
        for name, r in rows.items():
            ok = "OK " if r["reached_goal"] else "FAIL"
            print(f"{name:10s} {ok}  score={r['score']:7.1f}  "
                  f"min_ped={r['min_ped_dist_m']}m  collisions={r['obstacle_collisions']}")
        return

    res = run_one(args.scenario, render=not args.no_render,
                  plot=args.plot, max_time=args.max_time)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
