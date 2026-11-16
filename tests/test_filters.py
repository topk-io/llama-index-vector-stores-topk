import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from topk_sdk.data import f64_list
from topk_sdk.query import field, match, not_

from llama_index.vector_stores.topk import to_topk_filter
from llama_index.vector_stores.topk.filters import metadata_for_filtering


def f(key, value, op=FilterOperator.EQ):
    return MetadataFilter(key=key, value=value, operator=op)


def same(a, b) -> None:
    assert repr(a) == repr(b)


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        (FilterOperator.EQ, "a", field("metadata.k").eq("a")),
        (
            FilterOperator.NE,
            "a",
            field("metadata.k").ne("a").or_(field("metadata.k").is_null()),
        ),
        (FilterOperator.GT, 1, field("metadata.k").gt(1)),
        (FilterOperator.GTE, 1.5, field("metadata.k").gte(1.5)),
        (FilterOperator.LT, 1, field("metadata.k").lt(1)),
        (FilterOperator.LTE, 1, field("metadata.k").lte(1)),
        (FilterOperator.IN, ["a", "b"], field("metadata.k").in_(["a", "b"])),
        (FilterOperator.IN, [1, 2.5], field("metadata.k").in_(f64_list([1.0, 2.5]))),
        (FilterOperator.NIN, ["a"], not_(field("metadata.k").in_(["a"]))),
        (FilterOperator.CONTAINS, "x", field("metadata.k").contains("x")),
        (FilterOperator.TEXT_MATCH, "Foo", field("metadata.k").contains("Foo")),
        (
            FilterOperator.TEXT_MATCH_INSENSITIVE,
            "a.b",
            field("metadata.k").regexp_match(r"a\.b", "i"),
        ),
        (FilterOperator.IS_EMPTY, None, field("metadata.k").is_null()),
        (
            FilterOperator.ANY,
            ["x", "y"],
            field("metadata.k").contains("x").or_(field("metadata.k").contains("y")),
        ),
        (
            FilterOperator.ALL,
            ["x", "y"],
            field("metadata.k").contains("x").and_(field("metadata.k").contains("y")),
        ),
    ],
)
def test_every_operator(op, value, expected) -> None:
    same(to_topk_filter(MetadataFilters(filters=[f("k", value, op)])), expected)


def test_every_operator_is_covered() -> None:
    for op in FilterOperator:
        value = None if op == FilterOperator.IS_EMPTY else ["a"]
        if op not in (
            FilterOperator.IN,
            FilterOperator.NIN,
            FilterOperator.ANY,
            FilterOperator.ALL,
            FilterOperator.IS_EMPTY,
        ):
            value = "a"
        assert to_topk_filter(MetadataFilters(filters=[f("k", value, op)])) is not None


def test_conditions_and_nesting() -> None:
    filters = MetadataFilters(
        filters=[
            f("a", 1),
            MetadataFilters(
                filters=[f("b", "x"), f("c", 2, FilterOperator.GT)],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    expected = (
        field("metadata.a")
        .eq(1)
        .and_(field("metadata.b").eq("x").or_(field("metadata.c").gt(2)))
    )
    same(to_topk_filter(filters), expected)


def test_not_means_none_match() -> None:
    filters = MetadataFilters(
        filters=[f("a", 1), f("b", 2)], condition=FilterCondition.NOT
    )
    same(
        to_topk_filter(filters),
        not_(field("metadata.a").eq(1).or_(field("metadata.b").eq(2))),
    )


def test_custom_metadata_field_and_passthrough() -> None:
    same(
        to_topk_filter(MetadataFilters(filters=[f("a", 1)]), "meta"),
        field("meta.a").eq(1),
    )
    expr = field("metadata.year") >= 2020
    assert to_topk_filter(expr) is expr
    text = match("rust")
    assert to_topk_filter(text) is text
    assert to_topk_filter(None) is None
    assert to_topk_filter(MetadataFilters(filters=[])) is None


@pytest.mark.parametrize(
    "bad",
    [
        f("k", ["a"], FilterOperator.EQ),
        f("k", [], FilterOperator.IN),
        f("k", None, FilterOperator.GT),
    ],
)
def test_bad_values(bad) -> None:
    with pytest.raises(ValueError):
        to_topk_filter(MetadataFilters(filters=[bad]))


def test_bad_type() -> None:
    with pytest.raises(ValueError, match="MetadataFilters"):
        to_topk_filter({"a": 1})


def test_metadata_for_filtering() -> None:
    import datetime

    out = metadata_for_filtering(
        {
            "none": None,
            "empty_list": [],
            "empty_str": "",
            "floats": [1, 2.0],
            "ints": [1, 2],
            "strs": ["a", "b"],
            "s": "x",
            "b": True,
            "when": datetime.datetime(2024, 5, 1, 12, 0),
            "day": datetime.date(2024, 5, 1),
            "mixed": ["x", 1],
            "objs": [{"a": 1}],
            "huge": 2**70,
            "nested": {"a": 1, "b": None},
            "obj": object(),
        }
    )
    assert str(out.pop("floats")) == "List(F64([1.0, 2.0]))"
    assert out == {
        "ints": [1, 2],
        "strs": ["a", "b"],
        "s": "x",
        "b": True,
        "when": "2024-05-01T12:00:00",
        "day": "2024-05-01",
        "nested": {"a": 1},
    }


def test_ref_doc_id_keys_map_to_top_level_field() -> None:
    for key in ("ref_doc_id", "doc_id", "document_id"):
        same(
            to_topk_filter(MetadataFilters(filters=[f(key, "d1")])),
            field("ref_doc_id").eq("d1"),
        )
    same(
        to_topk_filter(MetadataFilters(filters=[f("doc_id", "d1")]), "m", "src"),
        field("src").eq("d1"),
    )
