from __future__ import annotations

import argparse
from typing import List, Tuple

from functions import build_h_action_function
from design_optimizer import jeffreys_optimizer_multi_bo

def make_demo_events() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    Eff_ = [(-1.0, 10.0), (-0.5, 40.0), (-0.8, 120.0)]
    Rew_ = [(1.5, 20.0), (1.0, 60.0), (2.0, 150.0)]
    A = [(Eff_[0], Rew_[0]), (Eff_[1], Rew_[1]), (Eff_[2], Rew_[2])]
    return Eff_, Rew_, A


def build_six_models():
    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            f = build_h_action_function(kernel=kernel, update=update)
            models.append(f)
    return models


def main():
    parser = argparse.ArgumentParser(description="Multi-model BO design optimization (Jeffreys)")
    parser.add_argument("--N_t", type=int, default=10, help="number of measurement times")
    parser.add_argument("--n-calls", type=int, default=40, help="number of BO evaluations")
    parser.add_argument("--n-init", type=int, default=10, help="number of BO initial random evaluations")
    parser.add_argument("--sigma", type=float, default=0.1, help="obs noise stddev")
    parser.add_argument("--outer", type=int, default=20, help="K_outer Monte Carlo samples")
    parser.add_argument("--inner", type=int, default=20, help="K_inner Monte Carlo samples per model")
    args = parser.parse_args()

    print("[info] Building models…")
    models = build_six_models()
    Eff_, Rew_, A = make_demo_events()

    fixed = {
        'Eff_': ('fixed', Eff_),
        'Rew_': ('fixed', Rew_),
        'A': ('fixed', A),
    }

    print(f"[info] Running BO with N_t={args.N_t}, n_calls={args.n_calls}, n_init={args.n_init}")
    res = jeffreys_optimizer_multi_bo(
        models,
        N_t=args.N_t,
        t_min=0.0,
        t_max=600.0,
        min_gap=5.0,
        fixed_input=fixed,
        sigma=args.sigma,
        K_outer=args.outer,
        K_inner=args.inner,
        n_calls=args.n_calls,
        n_initial_points=args.n_init,
        rng=123,
        objective='maximin',
    )

    print("[result] Best utility (maximin):", res['utility'])
    print("[result] Best t:", res['inputs']['t'])
    print("[result] U matrix:")
    for row in res['U']:
        print([round(x, 3) for x in row])


if __name__ == "__main__":
    main()

