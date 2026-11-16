"""Translate LlamaIndex `MetadataFilters` into TopK filter expressions."""

import datetime
import logging
import re
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from topk_sdk.data import f64_list
from topk_sdk.query import LogicalExpr, field, match, not_

logger = logging.getLogger(__name__)

REF_DOC_ID_KEYS = ("ref_doc_id", "doc_id", "document_id")
"""Filter keys that LlamaIndex stores use for the source document ID."""

_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1

# `TextExpr` (returned by `match()` and `should()`) is not importable from
# `topk_sdk.query` in every SDK version, so look it up from an instance.
TextExpr: type = next(
    (c for c in type(match("x")).__mro__ if c.__name__ == "TextExpr"),
    type(match("x")),
)

FilterLike = Union[MetadataFilters, MetadataFilter, LogicalExpr, Any]


def _and(exprs: Sequence[Any]) -> Any:
    return reduce(lambda left, right: left.and_(right), exprs)


def _or(exprs: Sequence[Any]) -> Any:
    return reduce(lambda left, right: left.or_(right), exprs)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _list_value(values: Sequence[Any]) -> Any:
    """
    Convert a Python list into a TopK list value.

    The SDK sends a Python list of floats as 32-bit floats, which never equal
    the 64-bit floats stored for scalar metadata (4.7 as f32 is not 4.7 as
    f64). Lists that contain a float are therefore sent as 64-bit float lists,
    and mixed int/float lists become float lists.
    """
    items = list(values)
    if items and all(_is_number(v) for v in items):
        if any(isinstance(v, float) for v in items):
            return f64_list([float(v) for v in items])
    return items


def _as_list(value: Any, op: FilterOperator, key: str) -> List[Any]:
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    elif value is None:
        values = []
    else:
        values = [value]
    if not values:
        raise ValueError(
            f"Filter operator {op.value!r} on {key!r} needs a non-empty list of values."
        )
    return values


def _translate_filter(
    f: MetadataFilter, metadata_field: str, ref_doc_id_field: str
) -> LogicalExpr:
    if not f.key:
        raise ValueError("Metadata filter keys must be non-empty strings.")
    if f.key in REF_DOC_ID_KEYS:
        # LlamaIndex exposes the source document ID under these keys; TopK
        # stores it in a top-level field.
        expr = field(ref_doc_id_field)
    else:
        expr = field(f"{metadata_field}.{f.key}")
    op = FilterOperator(f.operator)
    value = f.value

    if op == FilterOperator.IS_EMPTY:
        # `None`, empty strings and empty lists are not stored in the
        # filterable metadata (see `metadata_for_filtering`), so "empty"
        # means "null".
        return expr.is_null()
    if op in (FilterOperator.IN, FilterOperator.NIN):
        clause = expr.in_(_list_value(_as_list(value, op, f.key)))
        return clause if op == FilterOperator.IN else not_(clause)
    if op == FilterOperator.ANY:
        return _or([expr.contains(v) for v in _as_list(value, op, f.key)])
    if op == FilterOperator.ALL:
        return _and([expr.contains(v) for v in _as_list(value, op, f.key)])

    if value is None:
        raise ValueError(
            f"Filter operator {op.value!r} on {f.key!r} needs a value. Use "
            "FilterOperator.IS_EMPTY to match missing values."
        )
    if isinstance(value, (list, tuple)):
        raise ValueError(
            f"Filter operator {op.value!r} on {f.key!r} takes a single value, got "
            f"a list. Use FilterOperator.IN, ANY or ALL for lists."
        )
    if op == FilterOperator.EQ:
        return expr.eq(value)
    if op == FilterOperator.NE:
        # LlamaIndex's in-memory store matches `!=` when the key is missing;
        # say so explicitly, as TopK's `ne` on a missing string is not true.
        return expr.ne(value).or_(expr.is_null())
    if op == FilterOperator.GT:
        return expr.gt(value)
    if op == FilterOperator.GTE:
        return expr.gte(value)
    if op == FilterOperator.LT:
        return expr.lt(value)
    if op == FilterOperator.LTE:
        return expr.lte(value)
    if op == FilterOperator.CONTAINS:
        return expr.contains(value)
    if op == FilterOperator.TEXT_MATCH:
        # Case-sensitive substring match, without tokenization or stemming.
        return expr.contains(str(value))
    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        return expr.regexp_match(re.escape(str(value)), "i")
    raise ValueError(f"Unsupported filter operator: {op.value!r}.")  # pragma: no cover


def _translate_filters(
    filters: MetadataFilters, metadata_field: str, ref_doc_id_field: str
) -> Optional[Any]:
    exprs = []
    for f in filters.filters:
        if isinstance(f, MetadataFilters):
            sub = _translate_filters(f, metadata_field, ref_doc_id_field)
            if sub is not None:
                exprs.append(sub)
        else:
            exprs.append(_translate_filter(f, metadata_field, ref_doc_id_field))
    if not exprs:
        return None
    condition = FilterCondition(filters.condition or FilterCondition.AND)
    if condition == FilterCondition.AND:
        return _and(exprs)
    if condition == FilterCondition.OR:
        return _or(exprs)
    if condition == FilterCondition.NOT:
        # Same semantics as LlamaIndex's in-memory store: match when none of
        # the filters match. TopK, like that store, treats a comparison on a
        # missing key as not matching, so NOT keeps documents without the key.
        return not_(_or(exprs))
    raise ValueError(
        f"Unsupported filter condition: {condition!r}."
    )  # pragma: no cover


def to_topk_filter(
    filters: Optional[FilterLike],
    metadata_field: str = "metadata",
    ref_doc_id_field: str = "ref_doc_id",
) -> Optional[Any]:
    """
    Build a TopK filter expression.

    Args:
        filters: LlamaIndex `MetadataFilters` (any operator and condition,
            nested), a single `MetadataFilter`, or a TopK expression from
            `topk_sdk.query` (for example `field("metadata.year") >= 2020` or
            `match("rust")`), which is passed through unchanged.
        metadata_field: Struct field that stores node metadata. Filter keys
            are resolved inside it, so the key `year` becomes `metadata.year`.
            The keys `ref_doc_id`, `doc_id` and `document_id` refer to the
            source document ID, as in other LlamaIndex vector stores.
        ref_doc_id_field: Top-level field that stores the source document ID.

    Returns:
        A TopK expression, or `None` if there is nothing to filter on.

    """
    if filters is None:
        return None
    if isinstance(filters, MetadataFilters):
        return _translate_filters(filters, metadata_field, ref_doc_id_field)
    if isinstance(filters, MetadataFilter):
        return _translate_filter(filters, metadata_field, ref_doc_id_field)
    if isinstance(filters, (LogicalExpr, TextExpr)):
        return filters
    raise ValueError(
        "Filters must be LlamaIndex MetadataFilters or a TopK expression from "
        f"`topk_sdk.query`, got {type(filters).__name__}."
    )


def _filter_value(value: Any) -> Tuple[bool, Any]:
    """Return `(keep, value)` for one metadata value in the filterable copy."""
    if value is None or value == "":
        return False, None
    if isinstance(value, bool) or isinstance(value, (str, float)):
        return True, value
    if isinstance(value, int):
        return _I64_MIN <= value <= _I64_MAX, value
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        # ISO 8601 strings compare chronologically (for a single timezone).
        return True, value.isoformat()
    if isinstance(value, dict):
        nested = metadata_for_filtering(value)
        return bool(nested), nested
    if isinstance(value, (list, tuple)):
        items = list(value)
        if not items:
            return False, None
        if all(isinstance(v, str) for v in items):
            return True, items
        if all(_is_number(v) for v in items):
            if all(isinstance(v, int) for v in items):
                return all(_I64_MIN <= v <= _I64_MAX for v in items), items
            return True, _list_value(items)
        return False, None
    return False, None


def metadata_for_filtering(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the copy of node metadata that is stored for filtering.

    The full metadata is always kept in the serialized node; this copy only
    holds values TopK can filter on:

    * `None`, empty strings and empty lists are dropped, so `IS_EMPTY` is
      `is_null()`.
    * Dates and datetimes become ISO 8601 strings.
    * Lists that contain a float become 64-bit float lists.
    * Values TopK can't store (mixed-type lists, lists of objects, integers
      outside 64 bits, other objects) are left out, with a warning.
    """
    out: Dict[str, Any] = {}
    for key, value in metadata.items():
        keep, converted = _filter_value(value)
        if keep:
            out[key] = converted
        elif value is not None and value != "" and value != [] and value != {}:
            logger.warning(
                "Metadata key %r has a value TopK can't filter on (%s); it is kept "
                "in the node but can't be used in filters.",
                key,
                type(value).__name__,
            )
    return out
