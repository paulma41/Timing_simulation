from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal

import numpy as np

Number = Union[int, float]
Event = Tuple[float, float]          # (value, time)
Action= Tuple[int, Event, Event]  # (type_idx, (e_val, t_e), (r_val, t_r))
