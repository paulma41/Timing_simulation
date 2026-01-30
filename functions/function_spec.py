from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, get_args, get_origin

try:  # Optional numpy to support ndarray inputs
    import numpy as np
    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAVE_NUMPY = False


@dataclass
class Function:
    """
    Classe générique de fonction pour simulation/évaluation.

    Attributs (clés demandées):
    - eval: méthode qui évalue la fonction (f.eval(inputs))
    - input: spécifications des entrées, incluant types Python et descriptions
      Exemple par entrée: {
        'desc': str,
        'py_type': type | typing object (ex: float | Sequence[float]),
        'required': bool (défaut True),
        'range': {'min': float|None, 'max': float|None, 'include_min': bool, 'include_max': bool, 'elementwise': bool}
      }
    - output: spécifications de la sortie avec 'desc' et 'py_type'
    - parameters: dictionnaire des paramètres numériquement définis
    - sim_priors: dictionnaire décrivant des lois a priori pour la simulation

    L'implémentation concrète est fournie par `_evaluator` (callable interne).
    """

    name: str
    parameters: Dict[str, float]
    input: Dict[str, Dict[str, Any]]
    output: Dict[str, Dict[str, Any]]
    sim_priors: Dict[str, Any]
    _evaluator: Callable[[Any, Any, Any, Dict[str, float]], Any] = field(repr=False)

    def _check_type(self, value: Any, expected: Any) -> bool:
        """Vérifie value contre un type Python ou un type typing (Union, Sequence[float], list[float], ...)."""
        # Autoriser int comme float
        def _is_float_like(v: Any) -> bool:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return True
            if HAVE_NUMPY and isinstance(v, (np.floating, np.integer)):
                return True
            return False

        # typing.Any : toujours valide
        try:
            from typing import Any as TypingAny  # type: ignore
            if expected is TypingAny:
                return True
        except Exception:
            pass

        origin = get_origin(expected)
        if origin is None:
            # Cas d'un type concret
            if expected is float:
                return _is_float_like(value)
            return isinstance(value, expected)

        # Gestion Union[...] (|)
        if origin is getattr(__import__('typing'), 'Union') or origin is getattr(__import__('types'), 'UnionType', None):
            return any(self._check_type(value, arg) for arg in get_args(expected))

        # Gestion des séquences paramétrées (list[float], Sequence[float], tuple[float, ...])
        args = get_args(expected)
        try:
            import collections.abc as cabc
        except Exception:  # pragma: no cover
            cabc = None  # type: ignore

        if origin in (list, tuple):
            if not isinstance(value, origin):
                return False
            if not args:
                return True
            el_type = args[0]
            return all(self._check_type(v, el_type) for v in value)

        # Sequence[...] ou collections.abc.Sequence
        if cabc and (origin is cabc.Sequence):
            # Accept numpy arrays as sequences of numbers if numpy is available
            if HAVE_NUMPY and isinstance(value, np.ndarray):
                if not args:
                    return True
                el_type = args[0]
                # For numeric expectations (float), accept numeric dtype
                if el_type is float:
                    return np.issubdtype(value.dtype, np.number)
                # fallback: elementwise check for first few elements
                try:
                    it = value.flat if hasattr(value, 'flat') else value
                    for idx, vv in enumerate(it):
                        if idx >= 8:  # limit check
                            break
                        if not self._check_type(vv, el_type):
                            return False
                    return True
                except Exception:
                    return False

            if not isinstance(value, cabc.Sequence) or isinstance(value, (str, bytes)):
                return False
            if not args:
                return True
            el_type = args[0]
            return all(self._check_type(v, el_type) for v in value)

        # Fallback: tenter isinstance avec l'origine
        try:
            return isinstance(value, origin)  # type: ignore[arg-type]
        except Exception:
            return False

    def _check_range(self, value: Any, spec: Mapping[str, Any]) -> bool:
        """Vérifie la contrainte de 'range' si fournie. Retourne True si ok ou non spécifié."""
        if not spec:
            return True
        rmin: Optional[float] = spec.get('min')
        rmax: Optional[float] = spec.get('max')
        inc_min: bool = spec.get('include_min', True)
        inc_max: bool = spec.get('include_max', True)
        elementwise: bool = spec.get('elementwise', True)

        def _ok_one(x: Any) -> bool:
            try:
                xv = float(x)
            except Exception:
                return False
            if rmin is not None:
                if inc_min and not (xv >= rmin):
                    return False
                if not inc_min and not (xv > rmin):
                    return False
            if rmax is not None:
                if inc_max and not (xv <= rmax):
                    return False
                if not inc_max and not (xv < rmax):
                    return False
            return True

        # Séquences: vérifier élément par élément si elementwise, sinon ne pas appliquer
        if isinstance(value, (list, tuple)):
            if not elementwise:
                return True
            return all(_ok_one(v) for v in value)
        return _ok_one(value)

    def eval(self, inputs: Dict[str, Any]) -> Any:
        """
        Évalue la fonction à partir d'un dictionnaire d'entrées.

        Les spécifications d'entrée sont décrites dans `self.input` par clé
        (desc, py_type, required, range). Des vérifications de type et de range
        sont effectuées si renseignées.
        """
        if not isinstance(inputs, dict):
            raise TypeError("inputs doit être un dict contenant les clés attendues.")

        # Vérification des champs requis
        required_keys = [k for k, s in self.input.items() if s.get('required', True)]
        missing = [k for k in required_keys if k not in inputs]
        if missing:
            raise KeyError(f"Entrées manquantes: {missing} (requis: {required_keys})")

        # Vérification types + ranges si fournis
        for key, spec in self.input.items():
            if key not in inputs:
                continue
            val = inputs[key]
            expected = spec.get('py_type')
            if expected is not None and not self._check_type(val, expected):
                raise TypeError(f"Type invalide pour '{key}': {type(val)}; attendu {expected}")
            r_spec = spec.get('range')
            if r_spec and not self._check_range(val, r_spec):
                raise ValueError(f"Valeur hors range pour '{key}': {val} not in {r_spec}")

        t = inputs.get("t")
        S1 = inputs.get("S1", [])
        S2 = inputs.get("S2", [])
        return self._evaluator(t, S1, S2, self.parameters)
