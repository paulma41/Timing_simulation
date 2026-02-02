import math
import os
import sys
from typing import List

# Allow running as a script from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from recode.design.build import build_design_v5, build_fixed_times
from recode.design_optimizer.chernoff import _pairwise_chernoff_matrix, maximin_chernoff
from recode.design_optimizer.laplace import laplace_predictive
from recode.design_optimizer.laplace_chernoff import laplace_chernoff_risk
from recode.models.h_actions import build_h_action_function


def build_six_models(observation: str = "identity"):
    models = []
    for kernel in ["event_weighted", "action_avg"]:
        for update in ["continuous", "event", "action"]:
            f = build_h_action_function(kernel=kernel, update=update, observation=observation)
            models.append(f)
    return models


def main() -> None:
    models = build_six_models(observation="identity")

    t_fixed = build_fixed_times(0.0, 300.0, 30.0)
    action_types = [(0.5, 1.0), (0.25, 0.5), (0.125, 0.25)]

    actions_idx_times = []
    for i in range(len(t_fixed) - 1):
        start = t_fixed[i]
        end = t_fixed[i + 1]
        t_rew = min(end - 1.0, start + 10.0)
        t_eff = max(start + 1.0, t_rew - 5.0)
        type_idx = i % len(action_types)
        actions_idx_times.append((type_idx, start, end, t_eff, t_rew))

    design = build_design_v5(
        action_types=action_types,
        actions_idx_times=actions_idx_times,
        bonuses_t=[],
        rew_bonus_vals=[1.5, 1.0, 2.0],
        t_fixed=t_fixed,
        meas_times=t_fixed,
        meas_sources=None,
    )

    pred = laplace_predictive(models, design, sigma=0.1, n_jobs=1, return_lowrank=False)
    mu_list = pred["mu"]
    Vy_list = pred["Vy"]

    C, _ = _pairwise_chernoff_matrix(mu_list, Vy_list)
    score_pairwise = maximin_chernoff(C)

    risk_lc = laplace_chernoff_risk(models, design, sigma=0.1)

    if not math.isfinite(score_pairwise):
        raise AssertionError("pairwise Chernoff score is not finite")
    if not math.isfinite(risk_lc):
        raise AssertionError("laplace_chernoff risk is not finite")

    print("pairwise_maximin:", float(score_pairwise))
    print("laplace_chernoff_risk:", float(risk_lc))


if __name__ == "__main__":
    main()
