# TO CHECK

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runner = os.path.join(root, "recode", "runner", "agent_comparison.py")
    out_dir = os.path.join(root, "recode", "runner", "sigma_sweep_figs")
    os.makedirs(out_dir, exist_ok=True)

    sigmas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    for sigma in sigmas:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["SAVE_FIGS_DIR"] = out_dir
        env["SAVE_FIGS_PREFIX"] = f"sigma_{sigma:.2f}"
        cmd = [
            sys.executable,
            runner,
            "--sigma",
            str(sigma),
            "--n-iter",
            "5",
        ]
        print(f"[run] sigma={sigma}")
        proc = subprocess.run(cmd, env=env, cwd=root, check=False)
        if proc.returncode != 0:
            print(f"[warn] run failed for sigma={sigma} with code {proc.returncode}")
    print(f"[done] figures saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
