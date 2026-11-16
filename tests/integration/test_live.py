"""Live tests against TopK. They only delete collections they created."""

import datetime
import hashlib
import math
import re
import time
from typing import Callable, List

import pytest
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from topk_sdk.query import field, fn, select
from topk_sdk.schema import semantic_index, text

from llama_index.vector_stores.topk import (
    TopKServerSideEmbedding,
    TopKVectorStore,
)

DOCS = [
    (
        "rust",
        "Rust guarantees memory safety without a garbage collector.",
        {
            "lang": "rust",
            "year": 2015,
            "tags": ["systems", "safety"],
            "stars": 4.9,
            "ratios": [0.1, 0.7],
            "note": "",
            "released": datetime.date(2015, 5, 15),
        },
    ),
    (
        "python",
        "Python is known for readable syntax and scientific libraries.",
        {
            "lang": "python",
            "year": 1991,
            "tags": ["scripting", "data"],
            "stars": 4.7,
            "paradigm": "multi",
            "released": datetime.date(1991, 2, 20),
        },
    ),
    (
        "go",
        "Go has a garbage collector and built-in goroutines for concurrency.",
        {"lang": "go", "year": 2012, "tags": ["systems", "concurrency"], "stars": 4.5},
    ),
    (
        "haskell",
        "Haskell is a purely functional language with lazy evaluation.",
        {"lang": "haskell", "year": 1990, "tags": ["functional"], "stars": 4.1},
    ),
    (
        "sql",
        "SQL queries relational databases with declarative statements.",
        {"lang": "sql", "year": 1974, "tags": ["data"], "stars": 4.0, "note": None},
    ),
]


def nodes() -> List[TextNode]:
    out = []
    for node_id, body, meta in DOCS:
        out.append(TextNode(id_=node_id, text=body, metadata=dict(meta)))
    return out


def documents() -> List[Document]:
    return [Document(id_=i, text=t, metadata=dict(m)) for i, t, m in DOCS]


class HashEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embedding, so dense tests need no API key."""

    dim: int = 64

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z]+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._vec(text)


def node_ids(result) -> List[str]:
    return [n.node_id for n in result.nodes]


@pytest.fixture(scope="module")
def store(new_collection_name: Callable[[str], str], region: str) -> TopKVectorStore:
    s = TopKVectorStore(collection_name=new_collection_name("semantic"), region=region)
    s.add(nodes())
    return s


def q(text: str, k: int = 5, **kwargs) -> VectorStoreQuery:
    return VectorStoreQuery(query_str=text, similarity_top_k=k, **kwargs)


# --- Server-side embeddings ------------------------------------------------


def test_vector_store_index_without_embedding_model(
    new_collection_name: Callable[[str], str], region: str
) -> None:
    vs = TopKVectorStore(collection_name=new_collection_name("index"), region=region)
    index = VectorStoreIndex.from_documents(
        documents(),
        storage_context=StorageContext.from_defaults(vector_store=vs),
        embed_model=TopKServerSideEmbedding(),
    )
    results = index.as_retriever(similarity_top_k=2).retrieve(
        "memory safe systems programming"
    )
    assert results[0].node.ref_doc_id == "rust"
    assert results[0].node.metadata["lang"] == "rust"
    assert results[0].score is not None

    # Reopen the same collection from scratch and query it.
    reopened = VectorStoreIndex.from_vector_store(
        TopKVectorStore(collection_name=vs.collection_name, region=region),
        embed_model=TopKServerSideEmbedding(),
    )
    top = reopened.as_retriever(similarity_top_k=1).retrieve("functional programming")
    assert top[0].node.metadata["lang"] == "haskell"

    vs.delete("haskell")
    left = vs.get_nodes(filters=MetadataFilters(filters=[]))
    assert "haskell" not in {n.ref_doc_id for n in left}


def test_semantic_round_trip(store: TopKVectorStore) -> None:
    result = store.query(q("purely functional lazy language", k=1))
    (node,) = result.nodes
    assert node.node_id == "haskell"
    assert node.get_content() == DOCS[3][1]
    assert node.metadata == DOCS[3][2]
    rust = store.get_nodes(node_ids=["rust"])[0]
    # The node keeps its original metadata (dates as ISO strings after the
    # JSON round trip that every LlamaIndex vector store does).
    assert rust.metadata["released"] == "2015-05-15"
    assert rust.metadata["note"] == ""
    assert isinstance(result.similarities[0], float)


# --- Keyword and hybrid ----------------------------------------------------


def test_keyword_search(store: TopKVectorStore) -> None:
    result = store.query(q("garbage collector", mode=VectorStoreQueryMode.TEXT_SEARCH))
    assert set(node_ids(result)) == {"rust", "go"}
    all_terms = store.query(
        q("garbage goroutines", mode=VectorStoreQueryMode.TEXT_SEARCH),
        keyword_match_all=True,
    )
    assert node_ids(all_terms) == ["go"]


def test_hybrid_search_one_query(store: TopKVectorStore) -> None:
    result = store.query(q("garbage collector", k=5, mode=VectorStoreQueryMode.HYBRID))
    assert len(result.nodes) == 5  # BM25 boosts, does not exclude
    assert set(node_ids(result)[:2]) == {"rust", "go"}
    required = store.query(
        q("garbage collector", mode=VectorStoreQueryMode.HYBRID),
        keyword_required=True,
    )
    assert set(node_ids(required)) == {"rust", "go"}


def test_hybrid_alpha_and_weights(store: TopKVectorStore) -> None:
    sem_only = store.query(
        q("collector of garbage", k=5, mode=VectorStoreQueryMode.HYBRID, alpha=1.0)
    )
    kw_only = store.query(
        q("collector of garbage", k=5, mode=VectorStoreQueryMode.HYBRID, alpha=0.0)
    )
    assert len(sem_only.nodes) == 5
    assert set(node_ids(kw_only)[:2]) == {"rust", "go"}
    weighted = store.query(
        q("garbage collector", mode=VectorStoreQueryMode.HYBRID),
        weights={"semantic": 1.0, "keyword": 0.2},
    )
    assert node_ids(weighted)[0] in {"rust", "go"}
    sh = store.query(q("garbage collector", mode=VectorStoreQueryMode.SEMANTIC_HYBRID))
    assert set(node_ids(sh)[:2]) == {"rust", "go"}


def test_custom_ranking_with_metadata(store: TopKVectorStore) -> None:
    result = store.query(
        q("programming language", k=5),
        ranking=lambda s: s.semantic * 0.0 + field("metadata.year"),
    )
    assert node_ids(result) == ["rust", "go", "python", "haskell", "sql"]
    oldest = store.query(
        q("programming language", k=1),
        ranking=field("metadata.year"),
        ranking_ascending=True,
    )
    assert node_ids(oldest) == ["sql"]


# --- Filters ---------------------------------------------------------------


def mf(key, value, op=FilterOperator.EQ) -> MetadataFilter:
    return MetadataFilter(key=key, value=value, operator=op)


@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        (mf("lang", "rust"), {"rust"}),
        (mf("lang", "rust", FilterOperator.NE), {"python", "go", "haskell", "sql"}),
        (mf("year", 2012, FilterOperator.GT), {"rust"}),
        (mf("year", 2012, FilterOperator.GTE), {"rust", "go"}),
        (mf("year", 1990, FilterOperator.LT), {"sql"}),
        (mf("year", 1990, FilterOperator.LTE), {"sql", "haskell"}),
        (mf("stars", 4.6, FilterOperator.GT), {"rust", "python"}),
        (mf("lang", ["go", "sql"], FilterOperator.IN), {"go", "sql"}),
        (mf("lang", ["go", "sql"], FilterOperator.NIN), {"rust", "python", "haskell"}),
        (mf("year", [1990, 1991], FilterOperator.IN), {"haskell", "python"}),
        (mf("tags", "systems", FilterOperator.CONTAINS), {"rust", "go"}),
        (
            mf("tags", ["functional", "data"], FilterOperator.ANY),
            {"haskell", "python", "sql"},
        ),
        (mf("tags", ["systems", "safety"], FilterOperator.ALL), {"rust"}),
        (mf("lang", "ask", FilterOperator.TEXT_MATCH), {"haskell"}),
        (mf("lang", "PYTH", FilterOperator.TEXT_MATCH_INSENSITIVE), {"python"}),
        (
            mf("note", None, FilterOperator.IS_EMPTY),
            {"rust", "python", "go", "haskell", "sql"},
        ),
        # Floats that are not exact in 32 bits.
        (mf("stars", [4.7, 4.9], FilterOperator.IN), {"python", "rust"}),
        (mf("stars", [4.7], FilterOperator.NIN), {"rust", "go", "haskell", "sql"}),
        (mf("stars", 4.7), {"python"}),
        (mf("ratios", 0.1, FilterOperator.CONTAINS), {"rust"}),
        (mf("ratios", [0.7, 0.3], FilterOperator.ANY), {"rust"}),
        # Keys missing from most documents match NE and NIN, as in LlamaIndex.
        (mf("paradigm", "multi", FilterOperator.NE), {"rust", "go", "haskell", "sql"}),
        (
            mf("paradigm", ["multi"], FilterOperator.NIN),
            {"rust", "go", "haskell", "sql"},
        ),
        (
            mf("paradigm", None, FilterOperator.IS_EMPTY),
            {"rust", "go", "haskell", "sql"},
        ),
        # Dates are stored as ISO 8601 strings.
        (mf("released", "2000-01-01", FilterOperator.GT), {"rust"}),
    ],
)
def test_every_filter_operator(store: TopKVectorStore, flt, expected) -> None:
    filters = MetadataFilters(filters=[flt])
    result = store.query(q("programming", k=10, filters=filters))
    assert set(node_ids(result)) == expected
    nodes_ = store.get_nodes(filters=filters)
    assert {n.node_id for n in nodes_} == expected


def test_filter_conditions(store: TopKVectorStore) -> None:
    nested = MetadataFilters(
        filters=[
            mf("tags", "systems", FilterOperator.CONTAINS),
            MetadataFilters(
                filters=[mf("year", 2014, FilterOperator.GT), mf("lang", "go")],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    assert {n.node_id for n in store.get_nodes(filters=nested)} == {"rust", "go"}
    not_ = MetadataFilters(
        filters=[mf("lang", "rust"), mf("lang", "go")], condition=FilterCondition.NOT
    )
    assert {n.node_id for n in store.get_nodes(filters=not_)} == {
        "python",
        "haskell",
        "sql",
    }


def test_not_keeps_documents_without_the_key(store: TopKVectorStore) -> None:
    not_multi = MetadataFilters(
        filters=[mf("paradigm", "multi")], condition=FilterCondition.NOT
    )
    assert {n.node_id for n in store.get_nodes(filters=not_multi)} == {
        "rust",
        "go",
        "haskell",
        "sql",
    }


def test_filter_on_doc_id(new_collection_name, region) -> None:
    vs = TopKVectorStore(collection_name=new_collection_name("docid"), region=region)
    VectorStoreIndex.from_documents(
        documents(),
        storage_context=StorageContext.from_defaults(vector_store=vs),
        embed_model=TopKServerSideEmbedding(),
    )
    for key in ("doc_id", "ref_doc_id", "document_id"):
        got = vs.get_nodes(filters=MetadataFilters(filters=[mf(key, "go")]))
        assert [n.ref_doc_id for n in got] == ["go"]


def test_doc_and_node_id_restrictions(store: TopKVectorStore) -> None:
    result = store.query(q("language", node_ids=["go", "sql"]))
    assert set(node_ids(result)) == {"go", "sql"}


def test_raw_topk_query(store: TopKVectorStore) -> None:
    rows = store.query_raw(
        select("text", score=fn.semantic_similarity("text", "lazy evaluation"))
        .filter(field("metadata.year") < 2000)
        .sort(field("score"), asc=False)
        .limit(1)
    )
    assert rows[0]["_id"] == "haskell"


# --- Partitions and consistency --------------------------------------------


def test_partitions_isolate_tenants(
    new_collection_name: Callable[[str], str], region: str
) -> None:
    name = new_collection_name("tenants")
    acme = TopKVectorStore(collection_name=name, partition="acme", region=region)
    acme.add([TextNode(id_="a1", text="Acme sells rocket skates.")])
    acme.add(
        [TextNode(id_="g1", text="Globex builds doomsday devices.")], partition="globex"
    )
    assert node_ids(acme.query(q("rocket", k=5))) == ["a1"]
    assert node_ids(acme.query(q("rocket", k=5), partition="globex")) == ["g1"]
    assert set(acme.list_partitions()) >= {"acme", "globex"}
    nobody = acme.query(q("rocket"), partition="initech")
    assert nobody.nodes == []
    assert acme.count() == 1


@pytest.mark.parametrize("consistency", ["balanced", "indexed", "strong"])
def test_consistency_modes(store: TopKVectorStore, consistency: str) -> None:
    # "indexed" reads only compacted files and skips recent writes, so a fresh
    # collection can look empty until compaction catches up.
    deadline = time.monotonic() + (600 if consistency == "indexed" else 0)
    while True:
        result = store.query(q("garbage collector", k=1), consistency=consistency)
        if result.nodes or time.monotonic() > deadline:
            break
        time.sleep(5)
    assert len(result.nodes) == 1


def test_read_your_writes_and_lsn(
    new_collection_name: Callable[[str], str], region: str
) -> None:
    vs = TopKVectorStore(collection_name=new_collection_name("ryw"), region=region)
    vs.add([TextNode(id_="fresh", text="A brand new document about zebras.")])
    lsn = vs.last_lsn
    assert lsn
    assert node_ids(vs.query(q("zebras", k=1))) == ["fresh"]
    other = TopKVectorStore(
        collection_name=vs.collection_name, region=region, read_your_writes=False
    )
    assert node_ids(other.query(q("zebras", k=1), lsn=lsn)) == ["fresh"]
    vs.clear()
    assert vs.count() == 0
    vs.add([TextNode(id_="again", text="Written right after clear().")])
    assert vs.count() == 1


# --- Dense, sparse, multi-vector -------------------------------------------


def test_dense_vectors_through_index(
    new_collection_name: Callable[[str], str], region: str
) -> None:
    embed = HashEmbedding()
    vs = TopKVectorStore(
        collection_name=new_collection_name("dense"), region=region, embed_dim=64
    )
    index = VectorStoreIndex.from_documents(
        documents(),
        storage_context=StorageContext.from_defaults(vector_store=vs),
        embed_model=embed,
    )
    top = index.as_retriever(similarity_top_k=1).retrieve("readable syntax python")
    assert top[0].node.ref_doc_id == "python"
    hybrid = index.as_retriever(
        similarity_top_k=5, vector_store_query_mode="hybrid", alpha=0.5
    ).retrieve("garbage collector")
    assert {hybrid[0].node.ref_doc_id, hybrid[1].node.ref_doc_id} == {"rust", "go"}
    # Reopen: the dimension and metric are read from the collection's schema.
    reopened = TopKVectorStore(collection_name=vs.collection_name, region=region)
    assert reopened.embed_dim == 64
    assert reopened.is_embedding_query is True


def test_euclidean_metric(
    new_collection_name: Callable[[str], str], region: str
) -> None:
    vs = TopKVectorStore(
        collection_name=new_collection_name("l2"),
        region=region,
        embed_dim=2,
        distance_metric="euclidean",
        text_index="none",
    )
    vs.add(
        [
            TextNode(id_="near", text="near", embedding=[0.0, 1.0]),
            TextNode(id_="far", text="far", embedding=[5.0, 5.0]),
        ]
    )
    result = vs.query(VectorStoreQuery(query_embedding=[0.0, 0.9], similarity_top_k=2))
    assert node_ids(result) == ["near", "far"]
    assert result.similarities[0] > result.similarities[1]


def test_sparse_splade(new_collection_name: Callable[[str], str], region: str) -> None:
    sparse_mod = pytest.importorskip("llama_index.sparse_embeddings.fastembed")
    vs = TopKVectorStore(
        collection_name=new_collection_name("sparse"),
        region=region,
        sparse_embedding_model=sparse_mod.FastEmbedSparseEmbedding(
            model_name="prithivida/Splade_PP_en_v1"
        ),
    )
    vs.add(nodes())
    result = vs.query(q("memory safety", k=1, mode=VectorStoreQueryMode.SPARSE))
    assert node_ids(result) == ["rust"]


def test_multi_vector_maxsim(
    new_collection_name: Callable[[str], str], region: str
) -> None:
    pytest.importorskip("fastembed")
    from llama_index.vector_stores.topk import FastEmbedMultiVectorEmbedding

    vs = TopKVectorStore(
        collection_name=new_collection_name("colbert"),
        region=region,
        multi_vector_embedding_model=FastEmbedMultiVectorEmbedding(),
    )
    vs.add(nodes())
    result = vs.query(q("relational database queries", k=1), search_mode="multi_vector")
    assert node_ids(result) == ["sql"]
    hybrid = vs.query(q("lazy functional", k=1, mode=VectorStoreQueryMode.HYBRID))
    assert node_ids(hybrid) == ["haskell"]


def test_existing_collection_written_outside_llamaindex(
    new_collection_name: Callable[[str], str], region: str, topk_client
) -> None:
    name = new_collection_name("external")
    topk_client.collections().create(
        name, schema={"text": text().index(semantic_index())}
    )
    lsn = topk_client.collection(name).upsert(
        [
            {
                "_id": "x1",
                "text": "Penguins live in Antarctica.",
                "metadata": {"src": "sql"},
            }
        ]
    )
    vs = TopKVectorStore(collection_name=name, region=region)
    result = vs.query(q("birds in the cold", k=1), lsn=lsn)
    assert result.nodes[0].get_content() == "Penguins live in Antarctica."
    assert result.nodes[0].metadata == {"src": "sql"}


# --- Async -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_async(new_collection_name: Callable[[str], str], region: str) -> None:
    vs = TopKVectorStore(collection_name=new_collection_name("async"), region=region)
    await vs.async_add(nodes())
    result = await vs.aquery(q("functional programming", k=1))
    assert node_ids(result) == ["haskell"]
    kw = await vs.aquery(q("garbage", mode=VectorStoreQueryMode.TEXT_SEARCH))
    assert set(node_ids(kw)) == {"rust", "go"}
    got = await vs.aget_nodes(node_ids=["sql", "rust"])
    assert [n.node_id for n in got] == ["sql", "rust"]
    await vs.adelete_nodes(node_ids=["sql"])
    assert await vs.aget_nodes(node_ids=["sql"]) == []
    index = VectorStoreIndex.from_vector_store(
        vs, embed_model=TopKServerSideEmbedding()
    )
    answer = await index.as_retriever(similarity_top_k=1).aretrieve("memory safety")
    assert answer[0].node.node_id == "rust"
    await vs.aclear()
    start = time.monotonic()
    while vs.count() and time.monotonic() - start < 30:
        time.sleep(1)
    assert vs.count() == 0
