from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    HAVE_NUMPY = False

from .function_spec import Function


Number = Union[int, float]
Event = Tuple[float, float]  # (value, time)
Action = Tuple[Event, Event]  # ((e_val, t_e), (r_val, t_r)) with t_e < t_r
# Typed variant: (type_idx, (e_val, t_e), (r_val, t_r))
ActionTyped = Tuple[int, Event, Event]


def _snap_times(t: Sequence[float], marks: Sequence[float]) -> List[float]:
    if not marks:
        return list(t)
    marks_sorted = sorted(marks)
    snapped: List[float] = []
    j = 0
    for ti in t:
        while j + 1 < len(marks_sorted) and marks_sorted[j + 1] <= ti:
            j += 1
        if marks_sorted[j] <= ti:
            snapped.append(marks_sorted[j])
        else:
            snapped.append(ti)
    return snapped


def _snap_times_or_default(
    t: Sequence[float],
    marks: Sequence[float],
    default_value: float,
) -> List[float]:
    if not marks:
        return [default_value for _ in t]
    marks_sorted = sorted(marks)
    snapped: List[float] = []
    j = 0
    for ti in t:
        while j + 1 < len(marks_sorted) and marks_sorted[j + 1] <= ti:
            j += 1
        if marks_sorted[j] <= ti:
            snapped.append(marks_sorted[j])
        else:
            snapped.append(default_value)
    return snapped


def _eval_event_weighted(
    t: Sequence[float], Eff_: Sequence[Event], Rew_: Sequence[Event], params: Mapping[str, float]
) -> List[float]:
    h0 = float(params.get("h0", 0.0))
    gamma = float(params.get("gamma", 0.95))
    w_eff = float(params.get("W_effort", -1.0))
    w_rew = float(params.get("W_reward", 1.0))
    w_time = float(params.get("W_time", 0.0))

    if gamma < 0:
        gamma = 0.0
    if gamma > 1.0:
        gamma = 1.0

    if HAVE_NUMPY:
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        out = np.full_like(t_arr, h0, dtype=float)
        if Eff_:
            e_vals = np.asarray([ev[0] for ev in Eff_], dtype=float)
            e_times = np.asarray([ev[1] for ev in Eff_], dtype=float)
            mask = e_times[None, :] <= t_arr[:, None]
            dt = np.maximum(t_arr[:, None] - e_times[None, :], 0.0)
            if gamma == 0.0:
                dec = (dt == 0.0).astype(float)
            else:
                x = dt * math.log(gamma if gamma > 0 else 1e-12)
                x = np.clip(x, -745.0, 0.0)
                dec = np.exp(x)
            contrib = w_eff * e_vals[None, :] * dec * mask
            out += contrib.sum(axis=1)
        if Rew_:
            r_vals = np.asarray([rv[0] for rv in Rew_], dtype=float)
            r_times = np.asarray([rv[1] for rv in Rew_], dtype=float)
            mask = r_times[None, :] <= t_arr[:, None]
            dt = np.maximum(t_arr[:, None] - r_times[None, :], 0.0)
            if gamma == 0.0:
                dec = (dt == 0.0).astype(float)
            else:
                x = dt * math.log(gamma if gamma > 0 else 1e-12)
                x = np.clip(x, -745.0, 0.0)
                dec = np.exp(x)
            contrib = w_rew * r_vals[None, :] * dec * mask
            out += contrib.sum(axis=1)
        out = out + w_time * t_arr
        return out.tolist()

    res: List[float] = []
    ln_g = math.log(gamma if gamma > 0 else 1e-12)
    for ti in t:
        v = h0
        for val, te in Eff_:
            if te <= ti:
                dt = ti - te
                x = dt * ln_g
                x = max(min(x, 0.0), -745.0)
                v += w_eff * val * math.exp(x)
        for val, tr in Rew_:
            if tr <= ti:
                dt = ti - tr
                x = dt * ln_g
                x = max(min(x, 0.0), -745.0)
                v += w_rew * val * math.exp(x)
        v += w_time * ti
        res.append(v)
    return res


def build_h_action_function(
    *,
    kernel: Literal["event_weighted", "action_avg"] = "event_weighted",
    update: Literal["continuous", "event", "action"] = "continuous",
    observation: Literal["identity", "sigmoid"] = "identity",
    parameters: Optional[Dict[str, float]] = None,
) -> Function:
    """
    Build a parametrized h(t) function (2 kernels x 3 updates).
    - kernel: 'event_weighted' or 'action_avg'
    - update: 'continuous' | 'event' | 'action'
    - observation: 'identity' or 'sigmoid' (applies after h(t))
    Defaults: h0=0, gamma=0.95, W_effort=-1.0, W_reward=1.0
    """
    default_params = {
        "h0": 0.0,
        "gamma": 0.95,
        "W_effort": -1.0,
        "W_reward": 1.0,
        "W_time": 0.0,
        "K_types": 3,
        "obs_temp": 1.0,
        "obs_bias": 0.0,
    }
    params = {**default_params, **(parameters or {})}

    sim_priors: Dict[str, Dict[str, Any]] = {
        "h0": {"dist": "Normal", "mu": 0.3020, "sigma": 0.07},
        "gamma": {"dist": "Normal", "mu": 0.3315, "sigma": 0.2},
        "W_effort": {"dist": "Normal", "mu": -0.0051, "sigma": 0.008},
        "W_reward": {"dist": "Normal", "mu": 0.2536, "sigma": 0.05},
        "W_time": {"dist": "Normal", "mu": -0.0010, "sigma": 0.0003},
        "K_types": {"dist": "Normal", "mu": 3.0, "sigma": 0.0},
        "obs_temp": {"dist": "Normal", "mu": 1.0, "sigma": 0.0},
        "obs_bias": {"dist": "Normal", "mu": 0.0, "sigma": 0.0},
    }

    from typing import Sequence as TSequence, Union as TUnion
    input_spec: Dict[str, Dict[str, Any]] = {
        "t": {
            "desc": "evaluation times (R+)",
            "py_type": TUnion[float, TSequence[float]],
            "required": True,
            "range": {"min": 0.0, "max": None, "include_min": True, "include_max": True, "elementwise": True},
        },
        "Eff_": {
            "desc": "efforts (value<=0, time>=0)",
            "py_type": TSequence[TSequence[float]],
            "required": True,
        },
        "Rew_": {
            "desc": "rewards (value>=0, time>=0)",
            "py_type": TSequence[TSequence[float]],
            "required": True,
        },
        "A": {
            "desc": "actions [((e,t_e),(r,t_r))]",
            "py_type": TSequence[TSequence[TSequence[float]]],
            "required": True,
        },
        "A_typed": {
            "desc": "typed actions [(k, (e,t_e), (r,t_r))]",
            "py_type": Any,
            "required": False,
        },
        "K": {
            "desc": "number of actions (optional)",
            "py_type": TUnion[int, None],
            "required": False,
        },
    }
    output_spec: Dict[str, Dict[str, Any]] = {
        "h": {"desc": "h(t) value", "py_type": TUnion[float, List[float]]}
    }

    def _sigmoid_scalar(val: float, temp: float, bias: float) -> float:
        x = temp * val + bias
        x = max(min(x, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-x))

    def _sigmoid_array(arr: "np.ndarray", temp: float, bias: float) -> "np.ndarray":
        x = temp * arr + bias
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _apply_observation_vals(vals: Any, obs_temp: float, obs_bias: float) -> Any:
        if observation != "sigmoid":
            return vals
        if HAVE_NUMPY and isinstance(vals, np.ndarray):
            return _sigmoid_array(vals, obs_temp, obs_bias)
        if isinstance(vals, list):
            return [_sigmoid_scalar(float(v), obs_temp, obs_bias) for v in vals]
        return _sigmoid_scalar(float(vals), obs_temp, obs_bias)

    def _apply_observation_jac(Jm: Any, h_vals: Any, obs_temp: float, obs_bias: float) -> Any:
        if observation != "sigmoid":
            return Jm
        if HAVE_NUMPY and isinstance(Jm, np.ndarray):
            h_arr = np.asarray(h_vals, dtype=float).reshape(-1)
            s = _sigmoid_array(h_arr, obs_temp, obs_bias)
            scale = (obs_temp * s * (1.0 - s)).reshape(-1, 1)
            return Jm * scale
        scales = [_sigmoid_scalar(float(v), obs_temp, obs_bias) for v in h_vals]
        scales = [obs_temp * v * (1.0 - v) for v in scales]
        for i, scale in enumerate(scales):
            row = Jm[i]
            for j in range(len(row)):
                row[j] *= scale
        return Jm

    def evaluator_adapter(t_any: Any, Eff_in_any: Any, Rew_in_any: Any, params_local: Dict[str, float]) -> Any:
        A_local: Sequence[Action] = evaluator_adapter._A  # type: ignore[attr-defined]
        A_typed_local: Sequence[ActionTyped] = evaluator_adapter._A_typed  # type: ignore[attr-defined]
        upd: str = evaluator_adapter._update  # type: ignore[attr-defined]
        ker: str = evaluator_adapter._kernel  # type: ignore[attr-defined]

        Eff_seq: Sequence[Event] = list(Eff_in_any or [])
        Rew_seq: Sequence[Event] = list(Rew_in_any or [])
        if isinstance(t_any, (int, float)):
            t_list = [float(t_any)]
        else:
            t_list = [float(x) for x in t_any]

        h0 = float(params_local.get("h0", 0.0))
        gamma = float(params_local.get("gamma", 0.95))
        w_eff = float(params_local.get("W_effort", -1.0))
        w_rew = float(params_local.get("W_reward", 1.0))
        w_time = float(params_local.get("W_time", 0.0))
        obs_temp = float(params_local.get("obs_temp", 1.0))
        obs_bias = float(params_local.get("obs_bias", 0.0))
        if gamma < 0:
            gamma = 0.0
        if gamma > 1.0:
            gamma = 1.0

        precomp_eval = getattr(evaluator_adapter, "_precomp_eval", None)
        if HAVE_NUMPY and precomp_eval is not None and precomp_eval.get("kernel") == ker:
            if ker == "event_weighted":
                t_arr = precomp_eval["t_arr"]
                eff = precomp_eval["eff"]
                rew = precomp_eval["rew"]
                out = np.full_like(t_arr, h0, dtype=float)
                if eff["vals"].size:
                    gdt = np.power(gamma, eff["dt"]) * eff["mask"]
                    out += w_eff * (eff["vals"][None, :] * gdt).sum(axis=1)
                if rew["vals"].size:
                    gdt = np.power(gamma, rew["dt"]) * rew["mask"]
                    out += w_rew * (rew["vals"][None, :] * gdt).sum(axis=1)
                out = out + w_time * t_arr
                out = _apply_observation_vals(out, obs_temp, obs_bias)
                return out[0] if len(t_list) == 1 else out.tolist()

            t_base = precomp_eval["t_base"]
            out = np.full_like(t_base, h0, dtype=float)
            h_sum = np.zeros_like(t_base)
            for pre_k in precomp_eval["per_type"]:
                contrib = np.zeros_like(t_base)
                eff = pre_k["eff"]
                if eff["vals"].size:
                    gdt = np.power(gamma, eff["dt"]) * eff["mask"]
                    contrib += w_eff * (eff["vals"][None, :] * gdt).sum(axis=1)
                rew = pre_k["rew"]
                if rew["vals"].size:
                    gdt = np.power(gamma, rew["dt"]) * rew["mask"]
                    contrib += w_rew * (rew["vals"][None, :] * gdt).sum(axis=1)
                h_sum += contrib
            out += h_sum / float(precomp_eval["K_types"])
            out = out + w_time * t_base
            out = _apply_observation_vals(out, obs_temp, obs_bias)
            return out[0] if len(t_list) == 1 else out.tolist()

        t_snap_by_type = None
        if upd == "event":
            marks_global = [et for _, et in Eff_seq] + [rt for _, rt in Rew_seq]
            if ker == "event_weighted":
                t_eval = _snap_times(t_list, marks_global)
            else:
                k_types_local = max(1, int(params_local.get("K_types", 1)))
                marks_per_type: List[List[float]] = [[] for _ in range(k_types_local)]
                if A_typed_local:
                    for k, (e, te), (r, tr) in A_typed_local:
                        idx = min(max(int(k), 0), len(marks_per_type) - 1)
                        marks_per_type[idx].append(te)
                        marks_per_type[idx].append(tr)
                else:
                    marks_per_type[0].extend(marks_global)
                t_snap_by_type = [
                    _snap_times(t_list, marks_per_type[k]) if marks_per_type[k] else t_list
                    for k in range(len(marks_per_type))
                ]
                t_eval = t_list
        elif upd == "action":
            if ker == "event_weighted":
                if A_typed_local:
                    marks = [tr for _, (_, _), (_, tr) in A_typed_local]
                else:
                    marks = [tr for (_, _), (_, tr) in A_local]
                t_eval = _snap_times(t_list, marks) if marks else t_list
            else:
                k_types_local = max(1, int(params_local.get("K_types", 1)))
                marks_per_type = [[] for _ in range(k_types_local)]
                if A_typed_local:
                    for k, (_, _), (_, tr) in A_typed_local:
                        idx = min(max(int(k), 0), len(marks_per_type) - 1)
                        marks_per_type[idx].append(tr)
                else:
                    for (_, _), (_, tr) in A_local:
                        marks_per_type[0].append(tr)
                default_snap = (min(t_list) - 1.0) if t_list else -1.0
                t_snap_by_type = [
                    _snap_times_or_default(t_list, marks_per_type[k], default_snap)
                    for k in range(len(marks_per_type))
                ]
                t_eval = t_list
        else:
            t_eval = t_list

        if ker == "event_weighted":
            vals = _eval_event_weighted(t_eval, Eff_seq, Rew_seq, params_local)
        else:
            vals = _eval_action_avg(t_eval, A_local, params_local, A_typed=A_typed_local, t_snap_by_type=t_snap_by_type)

        vals = _apply_observation_vals(vals, obs_temp, obs_bias)
        return vals[0] if len(t_list) == 1 else [float(v) for v in vals]

    evaluator_adapter._A = []  # type: ignore[attr-defined]
    evaluator_adapter._A_typed = []  # type: ignore[attr-defined]
    evaluator_adapter._update = update  # type: ignore[attr-defined]
    evaluator_adapter._kernel = kernel  # type: ignore[attr-defined]
    evaluator_adapter._precomp_eval = None  # type: ignore[attr-defined]
    evaluator_adapter._precomp_jac = None  # type: ignore[attr-defined]
    evaluator_adapter._precomp_key = None  # type: ignore[attr-defined]

    f = Function(
        name=f"h_actions_{kernel}_{update}",
        parameters=params,
        input=input_spec,
        output=output_spec,
        sim_priors=sim_priors,
        _evaluator=evaluator_adapter,
    )

    original_eval = f.eval

    def _jacobian(inputs: Dict[str, Any], params_local: Dict[str, float]) -> List[List[float]]:
        t_any = inputs.get("t", [])
        Eff_in = inputs.get("Eff_", [])
        Rew_in = inputs.get("Rew_", [])
        A_local = inputs.get("A", [])
        A_typed = inputs.get("A_typed", inputs.get("A_types", []))

        if isinstance(t_any, (int, float)):
            t_list = [float(t_any)]
        else:
            t_list = [float(x) for x in t_any]

        if kernel == "event_weighted":
            if update == "event":
                marks = [et for _, et in Eff_in] + [rt for _, rt in Rew_in]
                t_eval = _snap_times(t_list, marks)
            elif update == "action":
                if A_typed:
                    marks = [tr for _, (_, _), (r, tr) in A_typed]
                else:
                    marks = [tr for (_, _), (r, tr) in A_local]
                t_eval = _snap_times(t_list, marks)
            else:
                t_eval = t_list
        else:
            t_eval = t_list

        h0 = float(params_local.get("h0", 0.0))
        gamma = float(params_local.get("gamma", 0.95))
        W_eff = float(params_local.get("W_effort", -1.0))
        W_rew = float(params_local.get("W_reward", 1.0))
        W_time = float(params_local.get("W_time", 0.0))
        obs_temp = float(params_local.get("obs_temp", 1.0))
        obs_bias = float(params_local.get("obs_bias", 0.0))

        typed_actions: List[ActionTyped] = []
        if A_typed:
            for entry in A_typed:
                try:
                    k, eff, rew = entry
                    e, te = eff
                    r, tr = rew
                    typed_actions.append((int(k), (float(e), float(te)), (float(r), float(tr))))
                except Exception:
                    continue
        else:
            for entry in A_local:
                try:
                    if len(entry) == 3 and isinstance(entry[0], (int, float)):
                        k = int(entry[0])
                        (e, te), (r, tr) = entry[1], entry[2]
                    else:
                        k = 0
                        (e, te), (r, tr) = entry  # type: ignore[misc]
                    typed_actions.append((k, (float(e), float(te)), (float(r), float(tr))))
                except Exception:
                    continue
        if typed_actions:
            k_from_data = 1 + max(max(k, 0) for k, _, _ in typed_actions)
        else:
            k_from_data = len(A_local) if A_local else 1
        K_types = float(params_local.get("K_types", k_from_data))
        if K_types <= 0:
            K_types = 1.0
        k_types_int = max(1, int(K_types))

        def _build_t_snap_by_type() -> List[List[float]]:
            if update == "event":
                marks_per_type: List[List[float]] = [[] for _ in range(k_types_int)]
                if A_typed:
                    for k, (e, te), (r, tr) in typed_actions:
                        idx = min(max(int(k), 0), k_types_int - 1)
                        marks_per_type[idx].append(te)
                        marks_per_type[idx].append(tr)
                else:
                    marks_global = [et for _, et in Eff_in] + [rt for _, rt in Rew_in]
                    marks_per_type[0].extend(marks_global)
                return [
                    _snap_times(t_list, marks_per_type[k]) if marks_per_type[k] else t_list
                    for k in range(k_types_int)
                ]
            if update == "action":
                marks_per_type = [[] for _ in range(k_types_int)]
                if A_typed:
                    for k, (_, _), (_, tr) in typed_actions:
                        idx = min(max(int(k), 0), k_types_int - 1)
                        marks_per_type[idx].append(tr)
                else:
                    for (_, _), (_, tr) in A_local:
                        marks_per_type[0].append(tr)
                default_snap = (min(t_list) - 1.0) if t_list else -1.0
                return [
                    _snap_times_or_default(t_list, marks_per_type[k], default_snap)
                    for k in range(k_types_int)
                ]
            return [t_list for _ in range(k_types_int)]

        pnames = list(params_local.keys())
        n_params = len(pnames)

        def _idx(name: str) -> int:
            try:
                return pnames.index(name)
            except ValueError:
                return -1

        idx_h0 = _idx("h0")
        idx_gamma = _idx("gamma")
        idx_weff = _idx("W_effort")
        idx_wrew = _idx("W_reward")
        idx_wtime = _idx("W_time")

        Eff_list = list(Eff_in) if Eff_in else []
        Rew_list = list(Rew_in) if Rew_in else []

        if HAVE_NUMPY:
            precomp_jac = getattr(evaluator_adapter, "_precomp_jac", None)
            precomp_eval = getattr(evaluator_adapter, "_precomp_eval", None)
            precomp_key = getattr(evaluator_adapter, "_precomp_key", None)
            if (
                kernel == "event_weighted"
                and precomp_jac is not None
                and precomp_key == id(inputs)
                and precomp_jac.get("kernel") == kernel
            ):
                t_arr = precomp_jac["t_arr"]
                Jm = np.zeros((t_arr.shape[0], n_params), dtype=float)
                if idx_h0 >= 0:
                    Jm[:, idx_h0] = 1.0
                if idx_wtime >= 0:
                    Jm[:, idx_wtime] = t_arr

                inv_gamma = (1.0 / gamma) if gamma > 0.0 else 0.0
                eff = precomp_jac["eff"]
                rew = precomp_jac["rew"]

                if eff["vals"].size:
                    gdt_eff = np.power(gamma, eff["dt"]) * eff["mask"]
                    contrib_eff = (eff["vals"][None, :] * gdt_eff).sum(axis=1)
                    dgamma_eff = (eff["vals"][None, :] * gdt_eff * (eff["dt"] * inv_gamma)).sum(axis=1)
                else:
                    contrib_eff = np.zeros_like(t_arr)
                    dgamma_eff = np.zeros_like(t_arr)

                if rew["vals"].size:
                    gdt_rew = np.power(gamma, rew["dt"]) * rew["mask"]
                    contrib_rew = (rew["vals"][None, :] * gdt_rew).sum(axis=1)
                    dgamma_rew = (rew["vals"][None, :] * gdt_rew * (rew["dt"] * inv_gamma)).sum(axis=1)
                else:
                    contrib_rew = np.zeros_like(t_arr)
                    dgamma_rew = np.zeros_like(t_arr)

                if idx_weff >= 0:
                    Jm[:, idx_weff] = contrib_eff
                if idx_wrew >= 0:
                    Jm[:, idx_wrew] = contrib_rew
                if idx_gamma >= 0:
                    Jm[:, idx_gamma] = W_eff * dgamma_eff + W_rew * dgamma_rew
                h_vals = h0 + (W_eff * contrib_eff) + (W_rew * contrib_rew) + (W_time * t_arr)
                Jm = _apply_observation_jac(Jm, h_vals, obs_temp, obs_bias)
                return Jm.tolist()
            if (
                kernel != "event_weighted"
                and precomp_eval is not None
                and precomp_key == id(inputs)
                and precomp_eval.get("kernel") == kernel
                and "per_type" in precomp_eval
            ):
                t_base = np.asarray(precomp_eval["t_base"], dtype=float).reshape(-1)
                Jm = np.zeros((t_base.shape[0], n_params), dtype=float)
                if idx_h0 >= 0:
                    Jm[:, idx_h0] = 1.0
                if idx_wtime >= 0:
                    Jm[:, idx_wtime] = t_base

                inv_gamma = (1.0 / gamma) if gamma > 0.0 else 0.0
                total_eff = np.zeros_like(t_base)
                total_rew = np.zeros_like(t_base)
                total_dgamma_eff = np.zeros_like(t_base)
                total_dgamma_rew = np.zeros_like(t_base)

                for pre_k in precomp_eval["per_type"]:
                    eff = pre_k["eff"]
                    rew = pre_k["rew"]
                    if eff["vals"].size:
                        gdt_eff = np.power(gamma, eff["dt"]) * eff["mask"]
                        total_eff += (eff["vals"][None, :] * gdt_eff).sum(axis=1)
                        total_dgamma_eff += (eff["vals"][None, :] * gdt_eff * (eff["dt"] * inv_gamma)).sum(axis=1)
                    if rew["vals"].size:
                        gdt_rew = np.power(gamma, rew["dt"]) * rew["mask"]
                        total_rew += (rew["vals"][None, :] * gdt_rew).sum(axis=1)
                        total_dgamma_rew += (rew["vals"][None, :] * gdt_rew * (rew["dt"] * inv_gamma)).sum(axis=1)

                scale = 1.0 / K_types
                if idx_weff >= 0:
                    Jm[:, idx_weff] = scale * total_eff
                if idx_wrew >= 0:
                    Jm[:, idx_wrew] = scale * total_rew
                if idx_gamma >= 0:
                    Jm[:, idx_gamma] = scale * (W_eff * total_dgamma_eff + W_rew * total_dgamma_rew)
                h_vals = h0 + scale * (W_eff * total_eff + W_rew * total_rew) + (W_time * t_base)
                Jm = _apply_observation_jac(Jm, h_vals, obs_temp, obs_bias)
                return Jm.tolist()

            t_arr = np.asarray(t_eval, dtype=float).reshape(-1)
            Jm = np.zeros((t_arr.shape[0], n_params), dtype=float)
            if idx_h0 >= 0:
                Jm[:, idx_h0] = 1.0
            if idx_wtime >= 0:
                Jm[:, idx_wtime] = t_arr

            inv_gamma = (1.0 / gamma) if gamma > 0.0 else 0.0

            if kernel == "event_weighted":
                if Eff_list:
                    e_vals = np.asarray([val for val, _ in Eff_list], dtype=float)
                    e_times = np.asarray([te for _, te in Eff_list], dtype=float)
                    dt = t_arr[:, None] - e_times[None, :]
                    mask = dt >= 0.0
                    dt = np.where(mask, dt, 0.0)
                    if gamma > 0.0:
                        gdt = np.power(gamma, dt) * mask
                        contrib_eff = (e_vals[None, :] * gdt).sum(axis=1)
                        dgamma_eff = (e_vals[None, :] * gdt * (dt * inv_gamma)).sum(axis=1)
                    else:
                        contrib_eff = np.zeros_like(t_arr)
                        dgamma_eff = np.zeros_like(t_arr)
                else:
                    contrib_eff = np.zeros_like(t_arr)
                    dgamma_eff = np.zeros_like(t_arr)

                if Rew_list:
                    r_vals = np.asarray([val for val, _ in Rew_list], dtype=float)
                    r_times = np.asarray([tr for _, tr in Rew_list], dtype=float)
                    dt = t_arr[:, None] - r_times[None, :]
                    mask = dt >= 0.0
                    dt = np.where(mask, dt, 0.0)
                    if gamma > 0.0:
                        gdt = np.power(gamma, dt) * mask
                        contrib_rew = (r_vals[None, :] * gdt).sum(axis=1)
                        dgamma_rew = (r_vals[None, :] * gdt * (dt * inv_gamma)).sum(axis=1)
                    else:
                        contrib_rew = np.zeros_like(t_arr)
                        dgamma_rew = np.zeros_like(t_arr)
                else:
                    contrib_rew = np.zeros_like(t_arr)
                    dgamma_rew = np.zeros_like(t_arr)

                if idx_weff >= 0:
                    Jm[:, idx_weff] = contrib_eff
                if idx_wrew >= 0:
                    Jm[:, idx_wrew] = contrib_rew
                if idx_gamma >= 0:
                    Jm[:, idx_gamma] = W_eff * dgamma_eff + W_rew * dgamma_rew

            else:
                scale = 1.0 / K_types
                total_eff = np.zeros_like(t_arr)
                total_rew = np.zeros_like(t_arr)
                total_dgamma_eff = np.zeros_like(t_arr)
                total_dgamma_rew = np.zeros_like(t_arr)
                if typed_actions:
                    t_snap_by_type = _build_t_snap_by_type()
                    eff_vals_by_type: List[List[float]] = [[] for _ in range(k_types_int)]
                    eff_times_by_type: List[List[float]] = [[] for _ in range(k_types_int)]
                    rew_vals_by_type: List[List[float]] = [[] for _ in range(k_types_int)]
                    rew_times_by_type: List[List[float]] = [[] for _ in range(k_types_int)]
                    for k, (e, te), (r, tr) in typed_actions:
                        idx = min(max(int(k), 0), k_types_int - 1)
                        eff_vals_by_type[idx].append(e)
                        eff_times_by_type[idx].append(te)
                        rew_vals_by_type[idx].append(r)
                        rew_times_by_type[idx].append(tr)

                    for k in range(k_types_int):
                        t_k = np.asarray(t_snap_by_type[k], dtype=float).reshape(-1)
                        if eff_vals_by_type[k]:
                            e_vals = np.asarray(eff_vals_by_type[k], dtype=float)
                            e_times = np.asarray(eff_times_by_type[k], dtype=float)
                            dt_e = t_k[:, None] - e_times[None, :]
                            mask_e = dt_e >= 0.0
                            dt_e = np.where(mask_e, dt_e, 0.0)
                            if gamma > 0.0:
                                gdt_e = np.power(gamma, dt_e) * mask_e
                                total_eff += (e_vals[None, :] * gdt_e).sum(axis=1)
                                total_dgamma_eff += (e_vals[None, :] * gdt_e * (dt_e * inv_gamma)).sum(axis=1)
                        if rew_vals_by_type[k]:
                            r_vals = np.asarray(rew_vals_by_type[k], dtype=float)
                            r_times = np.asarray(rew_times_by_type[k], dtype=float)
                            dt_r = t_k[:, None] - r_times[None, :]
                            mask_r = dt_r >= 0.0
                            dt_r = np.where(mask_r, dt_r, 0.0)
                            if gamma > 0.0:
                                gdt_r = np.power(gamma, dt_r) * mask_r
                                total_rew += (r_vals[None, :] * gdt_r).sum(axis=1)
                                total_dgamma_rew += (r_vals[None, :] * gdt_r * (dt_r * inv_gamma)).sum(axis=1)

                if idx_weff >= 0:
                    Jm[:, idx_weff] = scale * total_eff
                if idx_wrew >= 0:
                    Jm[:, idx_wrew] = scale * total_rew
                if idx_gamma >= 0:
                    Jm[:, idx_gamma] = scale * (W_eff * total_dgamma_eff + W_rew * total_dgamma_rew)

            if kernel == "event_weighted":
                h_vals = h0 + (W_eff * contrib_eff) + (W_rew * contrib_rew) + (W_time * t_arr)
            else:
                h_vals = h0 + scale * (W_eff * total_eff + W_rew * total_rew) + (W_time * t_arr)
            Jm = _apply_observation_jac(Jm, h_vals, obs_temp, obs_bias)
            return Jm.tolist()

        n_t = len(t_eval)
        J = [[0.0 for _ in range(n_params)] for _ in range(n_t)]
        t_base = t_eval
        t_snap_by_type = _build_t_snap_by_type() if kernel != "event_weighted" else []
        h_vals = [0.0 for _ in range(n_t)]
        for it, t_base_val in enumerate(t_base):
            if idx_h0 >= 0:
                J[it][idx_h0] = 1.0
            if idx_wtime >= 0:
                J[it][idx_wtime] = t_base_val

            contrib_eff = 0.0
            contrib_rew = 0.0
            dgamma_eff = 0.0
            dgamma_rew = 0.0

            if kernel == "event_weighted":
                tt = t_base_val
                for val, te in Eff_list:
                    if te <= tt:
                        dt = tt - te
                        if gamma > 0:
                            gdt = gamma ** dt
                            contrib_eff += val * gdt
                            dgamma_eff += val * gdt * (dt / gamma)
                for val, tr in Rew_list:
                    if tr <= tt:
                        dt = tt - tr
                        if gamma > 0:
                            gdt = gamma ** dt
                            contrib_rew += val * gdt
                            dgamma_rew += val * gdt * (dt / gamma)
                if idx_weff >= 0:
                    J[it][idx_weff] = contrib_eff
                if idx_wrew >= 0:
                    J[it][idx_wrew] = contrib_rew
                if idx_gamma >= 0:
                    J[it][idx_gamma] = W_eff * dgamma_eff + W_rew * dgamma_rew
                h_vals[it] = h0 + (W_eff * contrib_eff) + (W_rew * contrib_rew) + (W_time * t_base_val)

            else:
                scale = 1.0 / K_types
                for _k, (e, te), (r, tr) in typed_actions:
                    k_idx = min(max(int(_k), 0), k_types_int - 1)
                    tt = t_snap_by_type[k_idx][it] if t_snap_by_type else t_base_val
                    if te <= tt and gamma > 0:
                        dt = tt - te
                        gdt = gamma ** dt
                        contrib_eff += e * gdt
                        dgamma_eff += e * gdt * (dt / gamma)
                    if tr <= tt and gamma > 0:
                        dt = tt - tr
                        gdt = gamma ** dt
                        contrib_rew += r * gdt
                        dgamma_rew += r * gdt * (dt / gamma)
                if idx_weff >= 0:
                    J[it][idx_weff] = scale * contrib_eff
                if idx_wrew >= 0:
                    J[it][idx_wrew] = scale * contrib_rew
                if idx_gamma >= 0:
                    J[it][idx_gamma] = scale * (W_eff * dgamma_eff + W_rew * dgamma_rew)
                h_vals[it] = h0 + scale * (W_eff * contrib_eff + W_rew * contrib_rew) + (W_time * t_base_val)

        J = _apply_observation_jac(J, h_vals, obs_temp, obs_bias)
        return J

    def eval_with_A(inputs: Dict[str, Any]) -> Any:
        A_local = inputs.get("A", [])
        A_typed_local = inputs.get("A_typed", inputs.get("A_types", []))
        evaluator_adapter._A = A_local  # type: ignore[attr-defined]
        evaluator_adapter._A_typed = A_typed_local  # type: ignore[attr-defined]

        k_old = f.parameters.get("K_types") if "K_types" in f.parameters else None
        if "K_types" in inputs:
            try:
                f.parameters["K_types"] = int(inputs["K_types"])
            except Exception:
                f.parameters["K_types"] = inputs["K_types"]

        precomp_eval = None
        precomp_jac = None
        if HAVE_NUMPY:
            t_any = inputs.get("t", [])
            if isinstance(t_any, (int, float)):
                t_list = [float(t_any)]
            else:
                t_list = [float(x) for x in t_any]
            Eff_seq = inputs.get("Eff_", [])
            Rew_seq = inputs.get("Rew_", [])

            typed_actions: List[ActionTyped] = []
            if A_typed_local:
                for entry in A_typed_local:
                    try:
                        k, eff, rew = entry
                        e, te = eff
                        r, tr = rew
                        typed_actions.append((int(k), (float(e), float(te)), (float(r), float(tr))))
                    except Exception:
                        continue
            else:
                for entry in A_local:
                    try:
                        if len(entry) == 3 and isinstance(entry[0], (int, float)):
                            k = int(entry[0])
                            (e, te), (r, tr) = entry[1], entry[2]
                        else:
                            k = 0
                            (e, te), (r, tr) = entry  # type: ignore[misc]
                        typed_actions.append((k, (float(e), float(te)), (float(r), float(tr))))
                    except Exception:
                        continue

            if typed_actions:
                k_from_data = 1 + max(max(k, 0) for k, _, _ in typed_actions)
            else:
                k_from_data = len(A_local) if A_local else 1
            K_types_local = int(f.parameters.get("K_types", k_from_data))
            if K_types_local <= 0:
                K_types_local = 1

            if update == "event":
                marks = [et for _, et in Eff_seq] + [rt for _, rt in Rew_seq]
                t_eval_jac = _snap_times(t_list, marks) if marks else t_list
            elif update == "action":
                if A_typed_local:
                    marks = [tr for _, (_, _), (_, tr) in A_typed_local]
                else:
                    marks = [tr for (_, _), (_, tr) in A_local]
                t_eval_jac = _snap_times(t_list, marks) if marks else t_list
            else:
                t_eval_jac = t_list

            def _build_dt_mask(t_arr: "np.ndarray", times_arr: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
                if times_arr.size == 0:
                    dt = np.zeros((t_arr.shape[0], 0), dtype=float)
                    mask = np.zeros((t_arr.shape[0], 0), dtype=bool)
                    return dt, mask
                dt = t_arr[:, None] - times_arr[None, :]
                mask = dt >= 0.0
                dt = np.where(mask, dt, 0.0)
                return dt, mask

            def _precomp_for_vals(t_arr: "np.ndarray", vals: Sequence[float], times: Sequence[float]) -> Dict[str, "np.ndarray"]:
                vals_arr = np.asarray(list(vals), dtype=float)
                times_arr = np.asarray(list(times), dtype=float)
                dt, mask = _build_dt_mask(t_arr, times_arr)
                return {"vals": vals_arr, "dt": dt, "mask": mask}

            t_arr_jac = np.asarray(t_eval_jac, dtype=float).reshape(-1)
            if kernel == "event_weighted":
                eff_vals = [val for val, _ in Eff_seq]
                eff_times = [te for _, te in Eff_seq]
                rew_vals = [val for val, _ in Rew_seq]
                rew_times = [tr for _, tr in Rew_seq]
                precomp_jac = {
                    "kernel": kernel,
                    "t_arr": t_arr_jac,
                    "eff": _precomp_for_vals(t_arr_jac, eff_vals, eff_times),
                    "rew": _precomp_for_vals(t_arr_jac, rew_vals, rew_times),
                }
                precomp_eval = precomp_jac
            else:
                eff_vals = [e for _, (e, _), (_, __) in typed_actions]
                eff_times = [te for _, (_, te), (_, __) in typed_actions]
                rew_vals = [r for _, (_, __), (r, _) in typed_actions]
                rew_times = [tr for _, (_, __), (_, tr) in typed_actions]
                precomp_jac = {
                    "kernel": kernel,
                    "t_arr": t_arr_jac,
                    "eff": _precomp_for_vals(t_arr_jac, eff_vals, eff_times),
                    "rew": _precomp_for_vals(t_arr_jac, rew_vals, rew_times),
                }

                if update == "event":
                    marks_per_type = [[] for _ in range(K_types_local)]
                    if typed_actions:
                        for k, (e, te), (r, tr) in typed_actions:
                            idx = min(max(int(k), 0), len(marks_per_type) - 1)
                            marks_per_type[idx].append(te)
                            marks_per_type[idx].append(tr)
                    else:
                        marks_per_type[0].extend([et for _, et in Eff_seq] + [rt for _, rt in Rew_seq])
                    t_snap_by_type = [
                        _snap_times(t_list, marks_per_type[k]) if marks_per_type[k] else t_list
                        for k in range(len(marks_per_type))
                    ]
                elif update == "action":
                    marks_per_type = [[] for _ in range(K_types_local)]
                    if typed_actions:
                        for k, (_, _), (_, tr) in typed_actions:
                            idx = min(max(int(k), 0), len(marks_per_type) - 1)
                            marks_per_type[idx].append(tr)
                    else:
                        for (_, _), (_, tr) in A_local:
                            marks_per_type[0].append(tr)
                    default_snap = (min(t_list) - 1.0) if t_list else -1.0
                    t_snap_by_type = [
                        _snap_times_or_default(t_list, marks_per_type[k], default_snap)
                        for k in range(len(marks_per_type))
                    ]
                else:
                    t_snap_by_type = [t_list for _ in range(K_types_local)]

                per_type = []
                for k in range(K_types_local):
                    t_k = np.asarray(t_snap_by_type[k], dtype=float).reshape(-1)
                    eff_vals_k: List[float] = []
                    eff_times_k: List[float] = []
                    rew_vals_k: List[float] = []
                    rew_times_k: List[float] = []
                    for kk, (e, te), (r, tr) in typed_actions:
                        idx = min(max(int(kk), 0), K_types_local - 1)
                        if idx != k:
                            continue
                        eff_vals_k.append(e)
                        eff_times_k.append(te)
                        rew_vals_k.append(r)
                        rew_times_k.append(tr)
                    per_type.append({
                        "t_arr": t_k,
                        "eff": _precomp_for_vals(t_k, eff_vals_k, eff_times_k),
                        "rew": _precomp_for_vals(t_k, rew_vals_k, rew_times_k),
                    })

                precomp_eval = {
                    "kernel": kernel,
                    "t_base": np.asarray(t_list, dtype=float).reshape(-1),
                    "per_type": per_type,
                    "K_types": K_types_local,
                }

        evaluator_adapter._precomp_eval = precomp_eval  # type: ignore[attr-defined]
        evaluator_adapter._precomp_jac = precomp_jac  # type: ignore[attr-defined]
        evaluator_adapter._precomp_key = id(inputs)  # type: ignore[attr-defined]

        routed = dict(inputs)
        if "S1" not in routed and "Eff_" in routed:
            routed["S1"] = routed["Eff_"]
        if "S2" not in routed and "Rew_" in routed:
            routed["S2"] = routed["Rew_"]
        try:
            return original_eval(routed)
        finally:
            if k_old is None and "K_types" in f.parameters:
                f.parameters.pop("K_types", None)
            elif k_old is not None:
                f.parameters["K_types"] = k_old

    f.eval = eval_with_A  # type: ignore[assignment]
    f.jacobian = _jacobian  # type: ignore[attr-defined]
    return f


def _eval_action_avg(
    t: Sequence[float],
    A: Sequence[Action],
    params: Mapping[str, float],
    *,
    A_typed: Optional[Sequence[ActionTyped]] = None,
    t_snap_by_type: Optional[List[List[float]]] = None,
) -> List[float]:
    h0 = float(params.get("h0", 0.0))
    gamma = float(params.get("gamma", 0.95))
    w_eff = float(params.get("W_effort", -1.0))
    w_rew = float(params.get("W_reward", 1.0))
    w_time = float(params.get("W_time", 0.0))
    if gamma < 0:
        gamma = 0.0
    if gamma > 1.0:
        gamma = 1.0

    typed: List[ActionTyped] = []
    if A_typed:
        for entry in A_typed:
            try:
                k, eff, rew = entry
                e, te = eff
                r, tr = rew
                typed.append((int(k), (float(e), float(te)), (float(r), float(tr))))
            except Exception:
                continue
    elif A:
        for entry in A:
            try:
                if len(entry) == 3 and isinstance(entry[0], (int, float)):
                    k = int(entry[0])
                    (e, te), (r, tr) = entry[1], entry[2]
                    typed.append((k, (float(e), float(te)), (float(r), float(tr))))
                else:
                    (e, te), (r, tr) = entry  # type: ignore[misc]
                    typed.append((0, (float(e), float(te)), (float(r), float(tr))))
            except Exception:
                continue

    if typed:
        k_from_data = 1 + max(max(k, 0) for k, _, _ in typed)
    else:
        k_from_data = len(A) if A else 1
    K_types = max(1, int(params.get("K_types", k_from_data)))

    t_base = [float(x) for x in t]
    if t_snap_by_type is None or len(t_snap_by_type) != K_types:
        t_snap_by_type = [t_base for _ in range(K_types)]

    if HAVE_NUMPY:
        t_arr = np.asarray(t_base, dtype=float).reshape(-1)
        out = np.full_like(t_arr, h0, dtype=float)

        h_types = np.zeros((K_types, t_arr.shape[0]), dtype=float)
        for k, (e, te), (r, tr) in typed:
            k_idx = min(max(int(k), 0), K_types - 1)
            t_k = np.asarray(t_snap_by_type[k_idx], dtype=float).reshape(-1)
            times = np.asarray([te, tr], dtype=float)
            vals = np.asarray([w_eff * e, w_rew * r], dtype=float)
            mask = times[None, :] <= t_k[:, None]
            dt = np.maximum(t_k[:, None] - times[None, :], 0.0)
            if gamma == 0.0:
                dec = (dt == 0.0).astype(float)
            else:
                x = dt * math.log(gamma if gamma > 0 else 1e-12)
                x = np.clip(x, -745.0, 0.0)
                dec = np.exp(x)
            contrib = (vals[None, :] * dec * mask).sum(axis=1)
            h_types[k_idx] += contrib

        out += h_types.sum(axis=0) / float(K_types)
        out = out + w_time * t_arr
        return out.tolist()

    res: List[float] = []
    ln_g = math.log(gamma if gamma > 0 else 1e-12)
    for idx, ti in enumerate(t_base):
        v = h0
        h_sum = 0.0
        for k, (e, te), (r, tr) in typed:
            k_idx = min(max(int(k), 0), K_types - 1)
            ti_snap = t_snap_by_type[k_idx][idx] if idx < len(t_snap_by_type[k_idx]) else ti
            if te <= ti_snap:
                dt = ti_snap - te
                x = dt * ln_g
                x = max(min(x, 0.0), -745.0)
                h_sum += w_eff * e * math.exp(x)
            if tr <= ti_snap:
                dt = ti_snap - tr
                x = dt * ln_g
                x = max(min(x, 0.0), -745.0)
                h_sum += w_rew * r * math.exp(x)
        v += h_sum / float(K_types)
        v += w_time * ti
        res.append(v)
    return res
