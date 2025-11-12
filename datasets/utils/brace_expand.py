from typing import List, Optional
import braceexpand


def expand_brace_patterns(
    patterns: List[str],
    *,
    fraction: Optional[float] = None,
    count: Optional[int] = None,
) -> List[str]:
    out: List[str] = []
    for p in patterns:
        expanded = list(braceexpand.braceexpand(p))
        if len(expanded) > 1:
            if fraction is not None and fraction < 1.0:
                n = max(1, int(len(expanded) * fraction))
                expanded = expanded[:n]
            if count is not None:
                expanded = expanded[:count]

        out.extend(expanded)
    return out
