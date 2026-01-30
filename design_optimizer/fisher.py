from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from functions import Function
from .base import (
    BaseDesignOptimizer,
    OptimizeResult,
    ensure_rng,
    normalize_fixed_input,
    generate_candidate_inputs,
    numerical_jacobian,
    jtj,
    logdet_psd,
    trace_matrix,
)

try:
    import numpy as np  # type: ignore
    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    HAVE_NUMPY = False


class FisherOptimizer(BaseDesignOptimizer):
    name: str = "fisher_optimizer"

    def __init__(
        self,
        objective: str = "logdet",  # 'logdet' or 'trace'
        budget: int = 200,
        rng: Optional[Union[int, random.Random]] = None,
        param_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.objective = objective
        self.budget = int(budget)
        self.rng = ensure_rng(rng)
        self.param_weights = dict(param_weights) if param_weights else None

    def _score(self, J: List[List[float]]) -> Tuple[float, List[List[float]]]:
        F = jtj(J)  # assume noise variance = 1
        if self.objective == "trace":
            return trace_matrix(F), F
        # default: logdet
        return logdet_psd(F), F

    def optimize(
        self,
        f: Function,
        N: int,
        fixed_input: Optional[Mapping[str, Any]] = None,
        *,
        budget: Optional[int] = None,
        eps: float = 1e-5,
        param_weights: Optional[Mapping[str, float]] = None,
        debug: bool = False,
        debug_print: bool = False,
        debug_max: int = 10,
    ) -> OptimizeResult:
        budget = int(budget if budget is not None else self.budget)
        best_score = float('-inf')
        best_inputs: Optional[Dict[str, Any]] = None
        best_F: Optional[List[List[float]]] = None
        dbg: Dict[str, Any] = {
            'attempts': 0,
            'eval_failures': 0,
            'eval_failure_examples': [],
            'scores': [],
            'neg_inf_scores': 0,
            'jacobian_ranks': [],
            'candidate_examples': [],
        }

        # Ensure required inputs exist per spec
        req = [k for k, s in f.input.items() if s.get("required", True)]

        # Normalize fixed_input into richer spec with counts
        norm_spec = normalize_fixed_input(f, N, fixed_input)

        # Parameter weights
        pnames = list(f.parameters.keys())
        weights = dict(self.param_weights) if self.param_weights else {}
        if param_weights:
            weights.update(param_weights)
        # Convert to column scaling factors sqrt(w)
        col_scale = [float(weights.get(p, 1.0)) ** 0.5 for p in pnames]

        for _ in range(budget):
            dbg['attempts'] += 1
            cand = generate_candidate_inputs(f, N, norm_spec, self.rng)
            # Fill missing required keys if any (robustness)
            for k in req:
                if k not in cand:
                    # Try default
                    if k == "t":
                        cand[k] = list(range(1, N + 1))
                    else:
                        cand[k] = []

            # Validate via f.eval type/range checks
            try:
                _ = f.eval(cand)
            except Exception as e:
                # skip invalid candidate
                if debug and len(dbg['eval_failure_examples']) < debug_max:
                    csum = {
                        't_len': (len(cand.get('t', [])) if isinstance(cand.get('t'), list) else None),
                        'S1_len': (len(cand.get('S1', [])) if isinstance(cand.get('S1'), list) else None),
                        'S2_len': (len(cand.get('S2', [])) if isinstance(cand.get('S2'), list) else None),
                    }
                    # capture key parameter snapshot for debugging
                    psnap = {
                        'C': f.parameters.get('C'),
                        'B': f.parameters.get('B'),
                        'A1': f.parameters.get('A1'),
                        'lambda1': f.parameters.get('lambda1'),
                        'A2': f.parameters.get('A2'),
                        'lambda2': f.parameters.get('lambda2'),
                    }
                    dbg['eval_failure_examples'].append({'error': repr(e), 'candidate_summary': csum, 'params': psnap})
                dbg['eval_failures'] += 1
                continue

            # Jacobian and score
            J = numerical_jacobian(f, cand, f.parameters, eps=eps)
            # Apply parameter weights to columns
            if any(w != 1.0 for w in col_scale):
                if HAVE_NUMPY:
                    Jm = np.asarray(J, dtype=float)
                    Jm = Jm * np.asarray(col_scale, dtype=float)[None, :]
                    J = Jm.tolist()
                else:
                    for i in range(len(J)):
                        for j in range(len(J[i])):
                            J[i][j] *= col_scale[j]
            # rank diagnostic
            if debug and HAVE_NUMPY:
                try:
                    rnk = int(np.linalg.matrix_rank(np.asarray(J, dtype=float)))
                    if len(dbg['jacobian_ranks']) < debug_max:
                        dbg['jacobian_ranks'].append(rnk)
                except Exception:
                    pass
            score, F = self._score(J)
            if debug:
                if not (score > float('-inf')):
                    dbg['neg_inf_scores'] += 1
                if len(dbg['scores']) < debug_max:
                    dbg['scores'].append(float(score))
                if len(dbg['candidate_examples']) < debug_max:
                    def _summ(v):
                        if isinstance(v, list) and v:
                            return {'len': len(v), 'min': min(v), 'max': max(v)}
                        if isinstance(v, list):
                            return {'len': 0}
                        return type(v).__name__
                    dbg['candidate_examples'].append({
                        't': _summ(cand.get('t', [])),
                        'S1': _summ(cand.get('S1', [])),
                        'S2': _summ(cand.get('S2', [])),
                    })
            if score > best_score:
                best_score = score
                best_inputs = cand
                best_F = F

        # Debug summary on failure
        if best_inputs is None and debug:
            msg = (
                f"[FisherOptimizer DEBUG] No valid design. Attempts={dbg['attempts']}, "
                f"eval_failures={dbg['eval_failures']}, neg_inf_scores={dbg['neg_inf_scores']}, "
                f"sample_scores={dbg['scores']}"
            )
            if debug_print:
                print(msg)
                print("[FisherOptimizer DEBUG] eval_failure_examples:")
                for ex in dbg['eval_failure_examples']:
                    print("  -", ex)
                if dbg['jacobian_ranks']:
                    print("[FisherOptimizer DEBUG] jacobian ranks (samples):", dbg['jacobian_ranks'])
        if best_inputs is None:
            raise RuntimeError("Aucun design valide trouvé avec les contraintes fournies.")

        return OptimizeResult(
            inputs=best_inputs,
            score=best_score,
            criterion=self.objective,
            fim=best_F,
            meta={"evaluations": budget, "debug": dbg if debug else None},
        )


def fisher_optimizer(
    f: Function,
    N: int,
    fixed_input: Optional[Mapping[str, Any]] = None,
    *,
    objective: str = "logdet",
    budget: int = 200,
    rng: Optional[Union[int, random.Random]] = None,
    eps: float = 1e-5,
    param_weights: Optional[Mapping[str, float]] = None,
    debug: bool = False,
    debug_print: bool = False,
    debug_max: int = 10,
) -> OptimizeResult:
    opt = FisherOptimizer(objective=objective, budget=budget, rng=rng, param_weights=param_weights)
    return opt.optimize(
        f, N=N, fixed_input=fixed_input, budget=budget, eps=eps,
        param_weights=param_weights, debug=debug, debug_print=debug_print, debug_max=debug_max,
    )


# Back-compat alias using requested name
def Fisher_info_optimizer(
    f: Function,
    N: int,
    fixed_input: Optional[Mapping[str, Any]] = None,
    *,
    objective: str = "logdet",
    budget: int = 200,
    rng: Optional[Union[int, random.Random]] = None,
    eps: float = 1e-5,
    param_weights: Optional[Mapping[str, float]] = None,
    debug: bool = False,
    debug_print: bool = False,
    debug_max: int = 10,
) -> OptimizeResult:
    return fisher_optimizer(
        f=f,
        N=N,
        fixed_input=fixed_input,
        objective=objective,
        budget=budget,
        rng=rng,
        eps=eps,
        param_weights=param_weights,
        debug=debug,
        debug_print=debug_print,
        debug_max=debug_max,
    )
