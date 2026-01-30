import numpy as np

from functions import build_h_action_function
from design_optimizer.base import numerical_jacobian


def check(update: str) -> None:
    f = build_h_action_function(kernel="action_avg", update=update)

    inputs = {
        "t": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0],
        "Eff_": [(0.5, 2.0), (0.25, 7.0)],
        "Rew_": [(1.0, 8.0), (0.5, 14.0)],
        "A": [((0.5, 2.0), (1.0, 8.0)), ((0.25, 7.0), (0.5, 14.0))],
        "A_typed": [(0, (0.5, 2.0), (1.0, 8.0)), (1, (0.25, 7.0), (0.5, 14.0))],
        "K_types": 3,
    }
    params = dict(f.parameters)

    J_ana = np.asarray(f.jacobian(inputs, params))
    J_num = np.asarray(numerical_jacobian(f, inputs, params, eps=1e-5))

    diff = np.abs(J_ana - J_num)
    print(update, "max abs diff", float(diff.max()), "mean", float(diff.mean()))


check("event")
check("action")
