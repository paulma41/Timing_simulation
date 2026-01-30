from __future__ import annotations

from dataclasses import dataclass
import random
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # Optional numpy acceleration
    import numpy as np
    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAVE_NUMPY = False

from functions import Function


SpecRange = Dict[str, Any]
FixedInputSpec = Mapping[str, Tuple[str, Any]]

# New richer spec structure (normalized)
NormalizedInputSpec = Dict[str, Dict[str, Any]]  # per key -> {'mode','count', 'value' or 'range'}


RngLike = Union[int, random.Random]


@dataclass
class OptimizeResult:
    inputs: Dict[str, Any]
    score: float
    criterion: str
    fim: Optional[list[list[float]]] = None
    meta: Optional[Dict[str, Any]] = None


class BaseDesignOptimizer:
    name: str = "base"

    def optimize(
        self,
        f: Function,
        N: int,
        fixed_input: Optional[FixedInputSpec] = None,
        **kwargs: Any,
    ) -> OptimizeResult:  # pragma: no cover - abstract
        raise NotImplementedError


# ----------------------------
# Generic helper functionality
# ----------------------------

def ensure_rng(seed_or_rng: Optional[RngLike]) -> random.Random:
    if seed_or_rng is None:
        return random.Random()
    if isinstance(seed_or_rng, random.Random):
        return seed_or_rng
    return random.Random(seed_or_rng)


def get_default_range_for_key(f: Function, key: str) -> Optional[Dict[str, Any]]:
    spec = f.input.get(key, {})
    r = spec.get("range")
    return r if isinstance(r, dict) else None


def sample_from_range_dict(rng: random.Random, r: Mapping[str, Any], n: int) -> List[float]:
    rmin = r.get("min", 0.0)
    rmax = r.get("max", 1.0)
    if rmin is None and rmax is None:
        rmin, rmax = 0.0, float(n)
    if rmin is None:
        rmin = 0.0
    if rmax is None:
        rmax = float(n)
    rmin = float(rmin)
    rmax = float(rmax)
    if rmax < rmin:
        rmin, rmax = rmax, rmin
    return sorted(rng.uniform(rmin, rmax) for _ in range(max(0, int(n))))


def normalize_fixed_input(
    f: Function,
    N: int,
    fixed_input: Optional[Mapping[str, Any]],
    *,
    default_event_count: Optional[int] = None,
) -> NormalizedInputSpec:
    """
    Normalise la spécification fixed_input en un dict par clé:
      { key: {'mode': 'fixed'|'var', 'count': int|None, 'value': Any? , 'range': Any?} }

    Formats acceptés en entrée pour chaque clé:
      - Tuple (mode, X) ou (mode, X, count)
      - Dict {'mode': ..., 'value': ...} (fixed) ou {'mode': ..., 'range': ..., 'count': ...} (var)
      - Si absent: par défaut ('var','Default')

    count: nombre d'éléments à générer pour cette entrée
      - par défaut: 't' -> N; autres (événements) -> default_event_count ou aléatoire lors de la génération
    """
    spec: NormalizedInputSpec = {}

    # Initialiser avec var/Default par défaut
    for key in f.input.keys():
        spec[key] = {"mode": "var", "range": "Default", "count": (N if key == "t" else default_event_count)}

    if fixed_input:
        for key, v in fixed_input.items():
            if key not in f.input:
                raise KeyError(f"fixed_input contient une clé inconnue '{key}' (attendu dans {list(f.input.keys())})")

            mode: Optional[str] = None
            val: Any = None
            count: Optional[int] = None

            if isinstance(v, tuple):
                if len(v) not in (2, 3):
                    raise ValueError(f"Tuple spec pour '{key}' doit avoir 2 ou 3 éléments")
                mode = str(v[0])
                val = v[1]
                if len(v) == 3:
                    count = int(v[2])
            elif isinstance(v, dict):
                mode = str(v.get("mode", "var"))
                if mode == "fixed":
                    if "value" not in v:
                        raise ValueError(f"Spec dict pour '{key}': 'value' requis en mode fixed")
                    val = v.get("value")
                else:
                    val = v.get("range", "Default")
                if "count" in v and v["count"] is not None:
                    count = int(v["count"])
            else:
                raise TypeError(f"Spec pour '{key}' doit être tuple ou dict")

            if mode not in ("fixed", "var"):
                raise ValueError(f"Spec pour '{key}': 'mode' doit être 'fixed' ou 'var'")

            # Appliquer
            if mode == "fixed":
                spec[key] = {"mode": "fixed", "value": val, "count": count}
            else:
                spec[key] = {"mode": "var", "range": val, "count": count}

    return spec


def generate_candidate_inputs(
    f: Function,
    N: int,
    spec: NormalizedInputSpec,
    rng: random.Random,
) -> Dict[str, Any]:
    """Génère un design candidat en respectant la spec normalisée."""
    inputs: Dict[str, Any] = {}

    for key, s in spec.items():
        mode = s.get("mode", "var")
        count = s.get("count", None)
        if key == "t" and (count is None or count <= 0):
            count = N

        if mode == "fixed":
            inputs[key] = s.get("value")
            continue

        # mode == 'var'
        rng_spec = s.get("range", "Default")
        # If this key is optional in the function spec and range is Default with no count,
        # skip generating it so it remains absent from inputs.
        finfo = f.input.get(key, {})
        is_required = bool(finfo.get("required", True))
        if (not is_required) and (rng_spec == "Default") and (s.get("count", None) in (None, 0)) and (s.get("mode", "var") == "var"):
            continue
        if isinstance(rng_spec, str) and rng_spec.lower() == "default":
            rdict = get_default_range_for_key(f, key)
            if rdict is None:
                # fallback simples
                if key == "t":
                    inputs[key] = list(range(1, N + 1))
                else:
                    # par défaut aucun évènement si pas de range
                    m = int(count) if isinstance(count, int) and count is not None else 0
                    inputs[key] = [] if m <= 0 else sorted(rng.sample(list(range(1, N + 1)), k=min(m, N)))
                continue
            rng_spec = rdict

        # Dict -> continu
        if isinstance(rng_spec, Mapping):
            m = int(count) if isinstance(count, int) and count is not None else (N if key == "t" else rng.randint(0, max(0, N)))
            vals = sample_from_range_dict(rng, rng_spec, m)
            if key == "t":
                # Enforce minimal spacing (e.g. 5s) between measurement times
                vals = sorted(vals)
                spaced: List[float] = []
                min_gap = 5.0
                last = None
                for v in vals:
                    if last is None or v - last >= min_gap:
                        spaced.append(v)
                        last = v
                vals = spaced
            inputs[key] = vals
            continue

        # Discret/itérable
        if isinstance(rng_spec, (list, tuple, range)):
            values = list(rng_spec)
            m = int(count) if isinstance(count, int) and count is not None else (N if key == "t" else rng.randint(0, max(0, N)))
            if not values:
                inputs[key] = []
                continue
            if m <= 0:
                inputs[key] = []
                continue
            if key == "t":
                if len(values) >= m:
                    picks = rng.sample(values, m)
                else:
                    picks = [rng.choice(values) for _ in range(m)]
                picks = sorted(picks)
                # Enforce minimal spacing between measurement times
                spaced: List[float] = []
                min_gap = 5.0
                last = None
                for v in picks:
                    if last is None or v - last >= min_gap:
                        spaced.append(v)
                        last = v
                inputs[key] = spaced
            else:
                picks = rng.choices(values, k=m)
                inputs[key] = sorted(picks)
            continue

        # scalaire
        inputs[key] = rng_spec

    return inputs


def numerical_jacobian(
    f: Function,
    inputs: Dict[str, Any],
    params: Mapping[str, float],
    eps: float = 1e-5,
) -> List[List[float]]:
    y0 = f.eval(inputs)
    y_list: List[float] = y0 if isinstance(y0, list) else [float(y0)]

    pnames = list(params.keys())

    if HAVE_NUMPY:
        J = np.zeros((len(y_list), len(pnames)), dtype=float)
    else:
        J = [[0.0 for _ in pnames] for _ in y_list]  # type: ignore[assignment]

    for j, pname in enumerate(pnames):
        p0 = params[pname]
        delta = eps * max(1.0, abs(float(p0)))
        params_plus = dict(params)
        params_plus[pname] = float(p0) + delta
        old_params = dict(f.parameters)
        try:
            f.parameters.update(params_plus)
            y_plus = f.eval(inputs)
        finally:
            f.parameters.update(old_params)
        y_plus_list = y_plus if isinstance(y_plus, list) else [float(y_plus)]

        params_minus = dict(params)
        params_minus[pname] = float(p0) - delta
        old_params = dict(f.parameters)
        try:
            f.parameters.update(params_minus)
            y_minus = f.eval(inputs)
        finally:
            f.parameters.update(old_params)
        y_minus_list = y_minus if isinstance(y_minus, list) else [float(y_minus)]

        if HAVE_NUMPY:
            J[:, j] = (np.array(y_plus_list) - np.array(y_minus_list)) / (2.0 * delta)
        else:
            for i in range(len(y_list)):
                J[i][j] = (y_plus_list[i] - y_minus_list[i]) / (2.0 * delta)  # type: ignore[index]

    if HAVE_NUMPY:
        return J.tolist()
    return J  # type: ignore[return-value]


def jtj(J: List[List[float]]) -> List[List[float]]:
    if HAVE_NUMPY:
        Jm = np.asarray(J, dtype=float)
        M = Jm.T @ Jm
        return M.tolist()
    n = len(J)
    k = len(J[0]) if n else 0
    M = [[0.0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            s = 0.0
            for r in range(n):
                s += J[r][i] * J[r][j]
            M[i][j] = s
    return M


def logdet_psd(M: List[List[float]], ridge: float = 1e-12) -> float:
    if HAVE_NUMPY:
        A = np.asarray(M, dtype=float)
        if ridge is not None and ridge > 0:
            A = A + ridge * np.eye(A.shape[0])
        sign, ld = np.linalg.slogdet(A)
        if sign <= 0:
            return float('-inf')
        return float(ld)
    # Fallback pure Python
    k = len(M)
    A = [[M[i][j] + (ridge if i == j else 0.0) for j in range(k)] for i in range(k)]
    logdet = 0.0
    for i in range(k):
        pivot = i
        maxabs = abs(A[i][i])
        for r in range(i + 1, k):
            if abs(A[r][i]) > maxabs:
                maxabs = abs(A[r][i])
                pivot = r
        if maxabs <= 0.0:
            return float('-inf')
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
        pivot_val = A[i][i]
        logdet += math.log(abs(pivot_val))
        for r in range(i + 1, k):
            if A[r][i] == 0:
                continue
            factor = A[r][i] / pivot_val
            for c in range(i, k):
                A[r][c] -= factor * A[i][c]
    return logdet


def trace_matrix(M: List[List[float]]) -> float:
    if HAVE_NUMPY:
        return float(np.trace(np.asarray(M, dtype=float)))
    return sum(M[i][i] for i in range(len(M)))


__all__ = [
    "BaseDesignOptimizer",
    "OptimizeResult",
    "ensure_rng",
    "get_default_range_for_key",
    "sample_from_range_dict",
    "normalize_fixed_input",
    "generate_candidate_inputs",
    "numerical_jacobian",
    "jtj",
    "logdet_psd",
    "trace_matrix",
]
