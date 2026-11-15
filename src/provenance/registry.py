"""Resolve a numeric claim to the artefact that produced it.

Six defects in this project were caught by asking whether a number could be right, not by a test
failing, and two of them were transcription errors between an artefact and a table. So no number
in the manuscript is written by hand. Each is registered as a *location* -- a result file, an
accessor into it, and a format -- and rendered from that file at build time. The registry cannot
disagree with the artefact, because it never stores a value.

Accessor grammar, applied left to right against parsed JSON or a DataFrame:

* ``a.b`` mapping key
* ``a[2]`` list index
* ``a[space=T_betti]`` first list element whose ``space`` field equals ``T_betti``
* ``a[variant=size_only,model=lightgbm]`` conjunction of field matches

For a CSV the first step selects rows and the last names a column::

    "[variant=size_only]pr_auc_mean"

Formats are ``.4f``-style specs, or ``x`` for an integer with thousands separators, so that the
rendered string is exactly what appears in the prose.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

_STEP = re.compile(r"""
    (?:^|\.|(?<=\]))(?P<key>[A-Za-z_][\w]*)   # .key, or a column right after ]
  | \['(?P<quoted>[^']*)'\]              # ['2']  -- for keys that are not identifiers
  | \[(?P<index>-?\d+)\]                 # [3]
  | \[(?P<match>[^\]]*=[^\]]*)\]         # [field=value,other=value]
""", re.VERBOSE)


class ClaimError(RuntimeError):
    """Raised when an accessor cannot be resolved, so a bad path fails loudly."""


@dataclass(frozen=True)
class Claim:
    """One registered number: where it lives and how it is written."""

    claim_id: str
    source: str
    #: An accessor into the artefact, or empty when ``expr`` carries the arithmetic instead.
    accessor: str = ""
    #: An arithmetic expression over accessors, for quantities the prose derives rather than quotes
    #: -- a ratio of two stored counts, a percentage, a unit change that is more than a factor.
    expr: str = ""
    fmt: str = ".4f"
    #: Multiplier applied before formatting, for stored seconds written as milliseconds.
    scale: float = 1.0
    note: str = ""
    #: Claims that are context rather than evidence (a configured radius, a dataset size) are
    #: exempt from the "must appear in the manuscript" check but still verified against source.
    context_only: bool = False

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.source


def _steps(accessor: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    position = 0
    for match in _STEP.finditer(accessor):
        if match.start() > position:
            raise ClaimError(f"cannot parse accessor {accessor!r} at offset {position}")
        position = match.end()
        for kind in ("key", "quoted", "index", "match"):
            if match.group(kind) is not None:
                out.append((kind, match.group(kind)))
                break
    if position != len(accessor):
        raise ClaimError(f"trailing text in accessor {accessor!r}")
    if not out:
        raise ClaimError(f"empty accessor {accessor!r}")
    return out


def _coerce(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _matches(row: Any, spec: str) -> bool:
    for clause in spec.split(","):
        field, _, wanted = clause.partition("=")
        field, wanted = field.strip(), wanted.strip()
        try:
            actual = row[field]
        except (KeyError, IndexError, TypeError):
            return False
        if str(actual) != wanted and _coerce(wanted) != actual:
            return False
    return True


def _walk(obj: Any, steps: list[tuple[str, str]], accessor: str) -> Any:
    current = obj
    for kind, token in steps:
        if kind in ("key", "quoted"):
            if not isinstance(current, dict) or token not in current:
                keys = sorted(current)[:8] if isinstance(current, dict) else type(current).__name__
                raise ClaimError(f"{accessor}: no key {token!r}; have {keys}")
            current = current[token]
        elif kind == "index":
            try:
                current = current[int(token)]
            except (TypeError, IndexError, KeyError) as exc:
                raise ClaimError(f"{accessor}: index {token} failed ({exc})") from exc
        else:
            if not isinstance(current, list):
                raise ClaimError(f"{accessor}: [{token}] needs a list, got "
                                 f"{type(current).__name__}")
            found = [row for row in current if _matches(row, token)]
            if not found:
                raise ClaimError(f"{accessor}: no element matching [{token}] among "
                                 f"{len(current)} elements")
            if len(found) > 1:
                raise ClaimError(f"{accessor}: [{token}] matched {len(found)} elements; "
                                 "an accessor must identify exactly one")
            current = found[0]
    return current


_AGGREGATE = re.compile(r"^(?P<fn>mean|sd|median|sum|min|max|count)\((?P<inner>.+)\)$")

#: What a results table actually reports is almost never a stored cell. A family's PR-AUC is the
#: mean over the seeds, so an accessor has to be able to say so rather than demanding a unique row.
_AGGREGATORS = {
    "mean": lambda series: series.mean(),
    "sd": lambda series: series.std(ddof=1),
    "median": lambda series: series.median(),
    "sum": lambda series: series.sum(),
    "min": lambda series: series.min(),
    "max": lambda series: series.max(),
    "count": lambda series: float(series.size),
}


def resolve(claim: Claim) -> Any:
    """Read the raw value a claim points at. Raises ClaimError rather than returning a default."""
    if not claim.path.exists():
        raise ClaimError(f"{claim.claim_id}: missing source {claim.source}")
    if claim.path.suffix == ".json":
        return _walk(json.loads(claim.path.read_text(encoding="utf-8")),
                     _steps(claim.accessor), claim.accessor)

    frame = pd.read_csv(claim.path)
    accessor = claim.accessor
    aggregate = _AGGREGATE.match(accessor)
    if aggregate:
        accessor = aggregate.group("inner")
    steps = _steps(accessor)
    if steps[-1][0] != "key":
        raise ClaimError(f"{claim.claim_id}: a CSV accessor must end in a column name")
    _ = accessor
    column = steps[-1][1]
    selected = frame
    for kind, token in steps[:-1]:
        if kind != "match":
            raise ClaimError(f"{claim.claim_id}: CSV row selection uses [field=value] only")
        mask = pd.Series(True, index=selected.index)
        for clause in token.split(","):
            field, _, wanted = clause.partition("=")
            field, wanted = field.strip(), wanted.strip()
            if field not in selected.columns:
                raise ClaimError(f"{claim.claim_id}: no column {field!r} in {claim.source}; "
                                 f"have {list(selected.columns)[:10]}")
            mask &= selected[field].astype(str) == wanted
        selected = selected[mask]
    if column not in selected.columns:
        raise ClaimError(f"{claim.claim_id}: no column {column!r} in {claim.source}")
    if aggregate:
        values = pd.to_numeric(selected[column], errors="coerce").dropna()
        if values.empty:
            raise ClaimError(f"{claim.claim_id}: nothing numeric to aggregate in {column!r}")
        return _AGGREGATORS[aggregate.group("fn")](values)
    if len(selected) != 1:
        raise ClaimError(f"{claim.claim_id}: row selection matched {len(selected)} rows, needs 1; "
                         "wrap the accessor in mean(...) to aggregate them deliberately")
    return selected.iloc[0][column]


# An accessor may begin with a row selector rather than a key, because that is how a CSV column is
# addressed: "[axis=radius,stage=topological]ms_per_entity".
_ACCESSOR = re.compile(
    r"(?:[A-Za-z_][\w]*|\[[^\]]*=[^\]]*\])"
    r"(?:(?:\.[A-Za-z_][\w]*)|(?:\['[^']*'\])|(?:\[-?\d+\])|(?:\[[^\]]*=[^\]]*\])"
    r"|(?:(?<=\])[A-Za-z_][\w]*))*"
)
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
)


def evaluate_derivation(source: str, expr: str) -> float:
    """Evaluate an arithmetic expression over accessors into one artefact.

    Used for quantities the prose derives rather than quotes -- a percentage of two stored counts,
    a sum of two edge lists. The expression is parsed and restricted to arithmetic, so a derivation
    cannot become a way to run code, and every accessor in it is resolved against the artefact. A
    derivation that no longer resolves raises instead of silently passing.
    """
    path = REPO_ROOT / source
    if not path.exists():
        raise ClaimError(f"derivation source missing: {source}")

    values: dict[str, float] = {}
    substituted = expr
    # Right to left, so replacing one accessor does not shift the offsets of those before it.
    for position, match in enumerate(reversed(list(_ACCESSOR.finditer(expr)))):
        accessor = match.group(0)
        name = f"_v{position}"
        # Resolved through the ordinary accessor path, so a derivation works over a CSV -- where
        # the cost tables live -- as readily as over JSON.
        raw = resolve(Claim(claim_id=f"derivation:{accessor}", source=source, accessor=accessor))
        try:
            values[name] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ClaimError(f"{accessor} is not numeric: {raw!r}") from exc
        substituted = substituted[:match.start()] + name + substituted[match.end():]

    tree = ast.parse(substituted, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ClaimError(f"expression uses {type(node).__name__}, which is not arithmetic")
    return float(eval(compile(tree, "<derivation>", "eval"), {"__builtins__": {}}, values))


def format_value(value: float, fmt: str) -> str:
    """Render a number the way the prose writes it."""
    if fmt == "x":
        return f"{int(round(float(value))):,}"
    if fmt == "d":
        return str(int(round(float(value))))
    if fmt == "raw":
        return str(value)
    return format(float(value), fmt)


def render(claim: Claim) -> str:
    """The exact string this claim must appear as in the prose."""
    if claim.expr:
        if claim.accessor:
            raise ClaimError(f"{claim.claim_id}: give either an accessor or an expr, not both")
        value = evaluate_derivation(claim.source, claim.expr)
    else:
        if not claim.accessor:
            raise ClaimError(f"{claim.claim_id}: needs an accessor or an expr")
        value = resolve(claim)
    if claim.scale != 1.0:
        value = float(value) * claim.scale
    return format_value(value, claim.fmt)
