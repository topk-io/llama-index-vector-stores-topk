"""Unit tests for TopKVectorStore against an in-memory fake client."""

import json
from typing import Dict, List

import pytest
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from topk_sdk.query import field, match
from topk_sdk.schema import f32_vector, semantic_index, text, vector_index

from llama_index.vector_stores.topk import (
    BaseMultiVectorEmbedding,
    TopKSearchMode,
    TopKServerSideEmbedding,
    TopKVectorStore,
)
from tests.fake_topk import FakeAsyncClient, FakeClient


class FakeSparse(BaseSparseEmbedding):
    def _get_query_embedding(self, query: str) -> Dict[int, float]:
        return {1: 0.5, 7: 1.0}

    async def _aget_query_embedding(self, query: str) -> Dict[int, float]:
        return {1: 0.5, 7: 1.0}

    def _get_text_embedding(self, text: str) -> Dict[int, float]:
        return {len(text): 1.0}

    async def _aget_text_embedding(self, text: str) -> Dict[int, float]:
        return {len(text): 1.0}


class FakeMulti(BaseMultiVectorEmbedding):
    def get_text_embedding_batch(self, texts: List[str]) -> List[List[List[float]]]:
        return [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] for _ in texts]

    def get_query_embedding(self, query: str) -> List[List[float]]:
        return [[1.0, 0.0, 0.0]]


@pytest.fixture()
def client() -> FakeClient:
    return FakeClient()


def make_store(client: FakeClient, **kwargs) -> TopKVectorStore:
    return TopKVectorStore(
        "test", client=client, async_client=FakeAsyncClient(client), **kwargs
    )


def node(i: str, text: str = "hello", **meta) -> TextNode:
    n = TextNode(id_=i, text=text, metadata=meta)
    n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=f"doc-{i}")
    return n


def last_query(client: FakeClient):
    return client.last("query")


def q_repr(client: FakeClient) -> str:
    return repr(last_query(client)[2])


# ---------------------------------------------------------------------------
# Construction and schema
# ---------------------------------------------------------------------------


def test_creates_semantic_collection_by_default(client) -> None:
    store = make_store(client)
    _, name, schema = client.last("create")
    assert name == "test"
    assert "SemanticIndex" in str(schema["text"])
    assert set(schema) == {"text"}
    assert store.is_embedding_query is False
    assert store.stores_text is True


def test_creates_dense_sparse_multi_fields(client) -> None:
    store = make_store(
        client,
        embed_dim=3,
        distance_metric="dot_product",
        sparse_embedding_model=FakeSparse(),
        multi_vector_embedding_model=FakeMulti(),
    )
    schema = client.last("create")[2]
    assert "dimension: 3" in str(schema["dense_vector"])
    assert "DotProduct" in str(schema["dense_vector"])
    assert "F32SparseVector" in str(schema["sparse_vector"])
    assert "MultiVectorIndex" in str(schema["multi_vector"])
    assert "dimension: 3" in str(schema["multi_vector"])
    assert store.is_embedding_query is True


def test_reads_existing_schema(client) -> None:
    client.schemas["test"] = {
        "text": text().index(semantic_index()),
        "dense_vector": f32_vector(4).index(vector_index(metric="euclidean")),
    }
    store = make_store(client)
    assert store.embed_dim == 4
    assert store.distance_metric == "euclidean"
    assert store.is_embedding_query is True
    assert not client.all("create")


def test_rejects_conflicting_schema(client) -> None:
    client.schemas["test"] = {
        "text": text().index(semantic_index()),
        "dense_vector": f32_vector(4).index(vector_index(metric="cosine")),
    }
    with pytest.raises(ValueError, match="embed_dim=8"):
        make_store(client, embed_dim=8)
    with pytest.raises(ValueError, match="distance_metric"):
        make_store(client, distance_metric="euclidean")
    with pytest.raises(ValueError, match="text_index"):
        make_store(client, text_index="keyword")
    with pytest.raises(ValueError, match="sparse_embedding_model"):
        make_store(client, sparse_embedding_model=FakeSparse())


def test_no_network_without_create(client) -> None:
    make_store(client, create_collection=False)
    assert client.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance_metric": "manhattan"},
        {"embed_dim": 0},
        {"consistency": "eventual"},
        {"hybrid_weights": {"bogus": 1.0}},
        {"hybrid_weights": {"dense": 1.0}},
        {"batch_size": 0},
    ],
)
def test_invalid_config(client, kwargs) -> None:
    with pytest.raises(ValueError):
        make_store(client, create_collection=False, **kwargs)


def test_missing_credentials(monkeypatch) -> None:
    monkeypatch.delenv("TOPK_API_KEY", raising=False)
    monkeypatch.delenv("TOPK_REGION", raising=False)
    store = TopKVectorStore("x", create_collection=False)
    with pytest.raises(ValueError, match="console.topk.io"):
        _ = store.client
    store = TopKVectorStore("x", create_collection=False, api_key="k")
    with pytest.raises(ValueError, match="TOPK_REGION"):
        _ = store.client


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_add_writes_text_node_content_and_filterable_metadata(client) -> None:
    store = make_store(client)
    n = node("n1", "hello world", year=2024, tags=["a"], empty=[], none=None)
    assert store.add([n]) == ["n1"]
    _, key, docs = client.last("upsert")
    assert key == ("test", None)
    (doc,) = docs
    assert doc["_id"] == "n1"
    assert doc["text"] == "hello world"
    assert doc["ref_doc_id"] == "doc-n1"
    assert doc["metadata"] == {"year": 2024, "tags": ["a"]}
    assert doc["node_type"] == "TextNode"
    content = json.loads(doc["node_content"])
    assert content["text"] == ""
    assert content["metadata"]["none"] is None
    assert "dense_vector" not in doc
    assert store.last_lsn == "1"


def test_add_ignores_client_embeddings_without_embed_dim(client, caplog) -> None:
    store = make_store(client)
    n = node("n1")
    n.embedding = [0.1, 0.2]
    store.add([n])
    assert "dense_vector" not in client.last("upsert")[2][0]
    assert "server-side semantic search" in caplog.text


def test_add_dense_validates_embeddings(client) -> None:
    store = make_store(client, embed_dim=2)
    with pytest.raises(ValueError, match="no embedding"):
        store.add([node("n1")])
    n = node("n2")
    n.embedding = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="3-dimensional"):
        store.add([n])
    n.embedding = [1, 2]
    store.add([n])
    assert client.last("upsert")[2][0]["dense_vector"] == [1.0, 2.0]


def test_add_computes_sparse_and_multi_vectors(client) -> None:
    store = make_store(
        client,
        sparse_embedding_model=FakeSparse(),
        multi_vector_embedding_model=FakeMulti(),
    )
    store.add([node("n1", "abc")])
    doc = client.last("upsert")[2][0]
    assert "sparse_vector" in doc
    assert "multi_vector" in doc


def test_add_rejects_text_too_long_for_semantic_index(client) -> None:
    store = make_store(client)
    with pytest.raises(ValueError, match="SentenceSplitter"):
        store.add([node("n1", "x" * 5000)])
    keyword = make_store(FakeClient(), text_index="keyword")
    keyword.add([node("n1", "x" * 5000)])


def test_add_batches_and_partitions(client) -> None:
    store = make_store(client, batch_size=2)
    store.add([node(str(i)) for i in range(5)], partition="acme")
    upserts = client.all("upsert")
    assert [len(c[2]) for c in upserts] == [2, 2, 1]
    assert all(c[1] == ("test", "acme") for c in upserts)
    assert store.last_lsn is None  # the store's own partition was not written


def test_large_documents_split_into_several_requests(client, monkeypatch) -> None:
    import llama_index.vector_stores.topk.base as base

    monkeypatch.setattr(base, "MAX_REQUEST_BYTES", 2000)
    store = make_store(client)
    store.add([node(str(i), "y" * 1000) for i in range(3)])
    assert len(client.all("upsert")) == 3


def test_delete_by_ref_doc_id(client) -> None:
    store = make_store(client)
    store.add([node("n1")])
    store.delete("doc-n1")
    target = client.last("delete")[2]
    assert repr(target) == repr(field("ref_doc_id") == "doc-n1")


def test_delete_nodes(client) -> None:
    store = make_store(client)
    store.add([node("n1")])
    store.delete_nodes(
        node_ids=["n1"],
        filters=MetadataFilters(filters=[MetadataFilter(key="a", value=1)]),
    )
    target = client.last("delete")[2]
    expected = field("_id").in_(["n1"]).and_(field("metadata.a").eq(1))
    assert repr(target) == repr(expected)
    with pytest.raises(ValueError, match="clear"):
        store.delete_nodes()
    calls = len(client.calls)
    store.delete_nodes(node_ids=[])
    assert len(client.calls) == calls


def test_clear_keeps_partition(client) -> None:
    store = make_store(client)
    store.add([node("n1")])
    store.clear()
    assert repr(client.last("delete")[2]) == repr(field("_id").is_not_null())
    assert not client.all("delete_partition")


def test_delete_on_missing_partition_is_noop(client) -> None:
    store = make_store(client, partition="ghost")
    store.delete("x")
    store.clear()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def test_round_trip_preserves_node(client) -> None:
    store = make_store(client)
    n = node("n1", "hello", year=2024)
    n.excluded_llm_metadata_keys = ["year"]
    store.add([n])
    result = store.query(VectorStoreQuery(query_str="hi", similarity_top_k=2))
    (got,) = result.nodes
    assert got.node_id == "n1"
    assert got.get_content() == "hello"
    assert got.metadata == {"year": 2024}
    assert got.ref_doc_id == "doc-n1"
    assert got.excluded_llm_metadata_keys == ["year"]
    assert result.similarities == [1.0]
    assert result.ids == ["n1"]


def test_rows_written_outside_llamaindex(client) -> None:
    store = make_store(client)
    client.data[("test", None)]["raw"] = {
        "_id": "raw",
        "text": "external",
        "metadata": {"source": "sql"},
    }
    client.query_results = [{"_id": "raw", "text": "external", "semantic_score": 2.0}]
    (got,) = store.query(VectorStoreQuery(query_str="q")).nodes
    assert got.get_content() == "external"
    assert got.metadata == {"source": "sql"}
    assert client.last("get")[3] == ["metadata"]


def test_get_nodes_by_id_keeps_order(client) -> None:
    store = make_store(client)
    store.add([node("a"), node("b")])
    nodes = store.get_nodes(node_ids=["b", "a", "missing"])
    assert [n.node_id for n in nodes] == ["b", "a"]
    assert "Limit { k: 3 }" in q_repr(client)
    assert store.get_nodes(node_ids=[]) == []


def test_get_nodes_by_filter(client) -> None:
    store = make_store(client)
    store.add([node("a")])
    store.get_nodes(
        filters=MetadataFilters(
            filters=[MetadataFilter(key="y", value=1, operator=FilterOperator.GT)]
        ),
        limit=5,
    )
    r = q_repr(client)
    assert "field(metadata.y), op=Gt," in r
    assert "Limit { k: 5 }" in r


def test_query_missing_partition_returns_empty(client) -> None:
    store = make_store(client, partition="nobody")
    result = store.query(VectorStoreQuery(query_str="q"))
    assert result.nodes == []
    assert store.get_nodes(node_ids=["x"]) == []
    assert store.count() == 0


def test_partition_routing_and_helpers(client) -> None:
    store = make_store(client)
    store.add([node("a")], partition="acme")
    store.query(VectorStoreQuery(query_str="q"), partition="acme")
    assert last_query(client)[1] == ("test", "acme")
    assert store.list_partitions() == ["acme"]
    assert store.count(partition="acme") == 1
    store.delete_partition("acme")
    assert client.last("delete_partition") == ("delete_partition", "test", "acme")
    with pytest.raises(ValueError):
        store.delete_partition()


# ---------------------------------------------------------------------------
# Query modes
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_store(client) -> TopKVectorStore:
    store = make_store(
        client,
        embed_dim=3,
        sparse_embedding_model=FakeSparse(),
        multi_vector_embedding_model=FakeMulti(),
    )
    client.query_results = []
    client.data.setdefault(("test", None), {})
    return store


def test_default_without_dense_is_semantic(client) -> None:
    store = make_store(client)
    client.query_results = []
    store.query(VectorStoreQuery(query_str="rust memory", similarity_top_k=7))
    r = q_repr(client)
    assert 'SemanticSimilarity { field: "text", query: "rust memory" }' in r
    assert "field(semantic_score)" in r and "order: Desc" in r
    assert "Limit { k: 7 }" in r


def test_default_with_embedding_is_dense(full_store, client) -> None:
    full_store.query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0], query_str="q"))
    r = q_repr(client)
    assert "VectorScore" in r or "VectorDistance" in r
    assert "SemanticSimilarity" not in r


def test_text_search_is_bm25_with_match(full_store, client) -> None:
    full_store.query(
        VectorStoreQuery(
            query_str="garbage collector", mode=VectorStoreQueryMode.TEXT_SEARCH
        )
    )
    r = q_repr(client)
    assert "KeywordScore" in r
    assert "should: false" in r and '"garbage collector"' in r


def test_keyword_options(full_store, client) -> None:
    full_store.query(
        VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.TEXT_SEARCH),
        keyword_query="exact terms",
        keyword_match_all=True,
    )
    r = q_repr(client)
    assert "all: true" in r and '"exact terms"' in r


def test_sparse_mode(full_store, client) -> None:
    full_store.query(VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.SPARSE))
    r = q_repr(client)
    assert "sparse_vector" in r and "sparse_score" in r


def test_sparse_mode_requires_model(client) -> None:
    store = make_store(client)
    with pytest.raises(ValueError, match="TEXT_SEARCH"):
        store.query(VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.SPARSE))


def test_hybrid_uses_every_signal_in_one_query(full_store, client) -> None:
    full_store.query(
        VectorStoreQuery(
            query_str="q",
            query_embedding=[1.0, 0.0, 0.0],
            mode=VectorStoreQueryMode.HYBRID,
        )
    )
    assert len(client.all("query")) == 1
    r = q_repr(client)
    for alias in (
        "semantic_score",
        "keyword_score",
        "dense_score",
        "sparse_score",
        "multi_vector_score",
    ):
        assert f'"{alias}"' in r
    assert "should: true" in r  # BM25 boosts without excluding


def test_hybrid_alpha_weights(full_store) -> None:
    signals = full_store._signals(TopKSearchMode.HYBRID, None, 0.75)
    assert signals == {
        "semantic": 0.75,
        "keyword": 0.25,
        "dense": 0.75,
        "sparse": 0.25,
        "multi_vector": 0.75,
    }
    assert set(full_store._signals(TopKSearchMode.HYBRID, None, 1.0)) == {
        "semantic",
        "dense",
        "multi_vector",
    }
    with pytest.raises(ValueError, match="alpha"):
        full_store._signals(TopKSearchMode.HYBRID, None, 2.0)


def test_hybrid_weights_override(full_store, client) -> None:
    signals = full_store._signals(
        TopKSearchMode.HYBRID, {"semantic": 2.0, "keyword": 0.5}, None
    )
    # Signals that are not named keep their default weight of 1.0.
    assert signals == {
        "semantic": 2.0,
        "keyword": 0.5,
        "dense": 1.0,
        "sparse": 1.0,
        "multi_vector": 1.0,
    }
    # A weight of 0 drops a signal.
    assert full_store._signals(
        TopKSearchMode.HYBRID,
        {"dense": 0, "sparse": 0, "multi_vector": 0, "semantic": 0},
        None,
    ) == {"keyword": 1.0}
    full_store.query(
        VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.HYBRID),
        weights={"semantic": 1.0, "keyword": 0.2, "dense": 0},
    )
    r = q_repr(client)
    assert "Coalesce" in r and "Float(0.2)" in r


def test_semantic_hybrid_is_semantic_plus_bm25(full_store) -> None:
    query = VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.SEMANTIC_HYBRID)
    mode, restrict = full_store._resolve_mode(query, None)
    assert set(full_store._signals(mode, None, None, restrict)) == {
        "semantic",
        "keyword",
    }


def test_hybrid_scores_combined(full_store, client) -> None:
    client.query_results = [
        {
            "_id": "a",
            "text": "t",
            "node_content": None,
            "semantic_score": 2.0,
            "keyword_score": None,
        }
    ]
    client.data[("test", None)]["a"] = {"_id": "a", "text": "t"}
    result = full_store.query(
        VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.HYBRID),
        weights={"semantic": 0.5, "keyword": 1.0, "dense": 0},
    )
    assert result.similarities == [1.0 + 0.0 + 0.0 + 0.0]


def test_search_mode_multi_vector(full_store, client) -> None:
    full_store.query(VectorStoreQuery(query_str="q"), search_mode="multi_vector")
    r = q_repr(client)
    assert "multi_vector_score" in r
    assert "SemanticSimilarity" not in r


def test_unsupported_mode(full_store) -> None:
    with pytest.raises(ValueError, match="not supported"):
        full_store.query(VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.MMR))


def test_unknown_option(full_store) -> None:
    with pytest.raises(ValueError, match="Unknown query options"):
        full_store.query(VectorStoreQuery(query_str="q"), bogus=1)


def test_top_k_limit(full_store) -> None:
    with pytest.raises(ValueError, match="10000"):
        full_store.query(VectorStoreQuery(query_str="q", similarity_top_k=10_001))


def test_dense_dimension_mismatch(full_store) -> None:
    with pytest.raises(ValueError, match="dimensions"):
        full_store.query(VectorStoreQuery(query_embedding=[1.0], query_str="q"))


def test_euclidean_dense_sorts_ascending_and_reports_similarity(client) -> None:
    store = make_store(client, embed_dim=2, distance_metric="euclidean")
    client.query_results = [{"_id": "a", "text": "t", "dense_score": 1.0}]
    client.data[("test", None)]["a"] = {"_id": "a", "text": "t"}
    result = store.query(VectorStoreQuery(query_embedding=[0.0, 1.0]))
    assert "order: Asc" in q_repr(client)
    assert result.similarities == [0.5]


def test_filters_doc_ids_node_ids_in_same_query(full_store, client) -> None:
    full_store.query(
        VectorStoreQuery(
            query_str="q",
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="year", value=2020, operator=FilterOperator.GTE)
                ]
            ),
            doc_ids=["d1"],
            node_ids=["n1"],
        ),
        topk_filter=match("rust"),
    )
    r = q_repr(client)
    assert "field(metadata.year), op=GtEq" in r
    assert "field(ref_doc_id), op=In" in r
    assert "field(_id), op=In" in r
    assert '"rust"' in r


def test_custom_ranking(full_store, client) -> None:
    full_store.query(
        VectorStoreQuery(
            query_str="q",
            query_embedding=[1.0, 0.0, 0.0],
            mode=VectorStoreQueryMode.HYBRID,
        ),
        ranking=lambda s: s.semantic
        + s.keyword.coalesce(0.0) * 0.1
        + field("metadata.popularity").coalesce(0.0) * 0.01,
        ranking_ascending=False,
    )
    r = q_repr(client)
    assert "field(metadata.popularity)" in r
    with pytest.raises(ValueError, match="ranking"):
        full_store.query(VectorStoreQuery(query_str="q"), ranking=lambda s: 3)


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------


def test_read_your_writes_passes_lsn(client) -> None:
    store = make_store(client)
    store.query(VectorStoreQuery(query_str="q"))
    assert last_query(client)[3] == {}
    store.add([node("a")])
    store.query(VectorStoreQuery(query_str="q"))
    assert last_query(client)[3] == {"lsn": "1"}


def test_consistency_modes(client) -> None:
    store = make_store(client, consistency="strong", read_your_writes=False)
    store.add([node("a")])
    store.query(VectorStoreQuery(query_str="q"))
    assert last_query(client)[3] == {"consistency": "strong"}
    store.query(VectorStoreQuery(query_str="q"), consistency="indexed")
    assert last_query(client)[3] == {"consistency": "indexed"}
    store.query(VectorStoreQuery(query_str="q"), consistency="balanced")
    assert last_query(client)[3] == {}
    store.query(VectorStoreQuery(query_str="q"), lsn="42")
    assert last_query(client)[3] == {"lsn": "42"}
    store.query(VectorStoreQuery(query_str="q"), lsn="42", consistency="strong")
    assert last_query(client)[3] == {"lsn": "42", "consistency": "strong"}
    with pytest.raises(ValueError):
        store.query(VectorStoreQuery(query_str="q"), consistency="eventual")


def test_lsn_is_tracked_per_partition(client) -> None:
    store = make_store(client)
    store.add([node("a")], partition="acme")
    store.add([node("b")])
    store.query(VectorStoreQuery(query_str="q"), partition="acme")
    assert last_query(client)[3] == {"lsn": "1"}
    store.query(VectorStoreQuery(query_str="q"))
    assert last_query(client)[3] == {"lsn": "2"}


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_round_trip(client) -> None:
    store = make_store(client, sparse_embedding_model=FakeSparse())
    assert await store.async_add([node("a", "hello")]) == ["a"]
    result = await store.aquery(
        VectorStoreQuery(query_str="q", mode=VectorStoreQueryMode.HYBRID)
    )
    assert [n.node_id for n in result.nodes] == ["a"]
    nodes = await store.aget_nodes(node_ids=["a"])
    assert nodes[0].get_content() == "hello"
    await store.adelete("doc-a")
    await store.adelete_nodes(node_ids=["a"])
    await store.aclear()
    assert len(client.all("delete")) == 3


# ---------------------------------------------------------------------------
# LlamaIndex integration without an embedding model
# ---------------------------------------------------------------------------


def test_vector_store_index_without_embedding_model(client, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = make_store(client)
    index = VectorStoreIndex.from_documents(
        [Document(text="TopK embeds on the server.", id_="d1")],
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=TopKServerSideEmbedding(),
    )
    doc = client.last("upsert")[2][0]
    assert doc["text"] == "TopK embeds on the server."
    assert doc["ref_doc_id"] == "d1"
    assert "dense_vector" not in doc
    nodes = index.as_retriever(similarity_top_k=3).retrieve("who embeds?")
    assert nodes[0].get_content() == "TopK embeds on the server."
    assert 'query: "who embeds?"' in q_repr(client)
    hybrid = index.as_retriever(
        vector_store_query_mode="hybrid",
        alpha=0.5,
        vector_store_kwargs={"keyword_required": True},
    )
    hybrid.retrieve("embeds")
    assert "KeywordScore" in q_repr(client)


def test_ingestion_pipeline_writes_nodes(client) -> None:
    store = make_store(client)
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=64, chunk_overlap=0),
            TopKServerSideEmbedding(),
        ],
        vector_store=store,
    )
    pipeline.run(documents=[Document(text="One sentence. " * 40)])
    assert sum(len(c[2]) for c in client.all("upsert")) > 1


def test_server_side_embedding_makes_no_vectors() -> None:
    e = TopKServerSideEmbedding()
    assert e.get_text_embedding_batch(["a", "b"]) == [[], []]
    assert e.get_query_embedding("q") == []


# ---------------------------------------------------------------------------
# Regression tests for review findings
# ---------------------------------------------------------------------------


def test_partial_weights_keep_other_signals(client) -> None:
    store = make_store(client, hybrid_weights={"keyword": 0.3})
    client.query_results = []
    store.query(VectorStoreQuery(query_str="hello world", mode="hybrid"))
    r = q_repr(client)
    assert "SemanticSimilarity" in r
    assert "should: true" in r  # BM25 still boosts instead of excluding


def test_to_dict_from_dict_round_trip(client) -> None:
    store = make_store(client, embed_dim=3, partition="acme", consistency="strong")
    data = store.to_dict()
    clone = TopKVectorStore.from_dict(
        {**data, "create_collection": False, "client": client}
    )
    assert clone.collection_name == "test"
    assert clone.embed_dim == 3
    assert clone.partition == "acme"
    assert clone.consistency == "strong"
    assert clone.is_embedding_query is True


def test_store_consistency_kept_with_read_your_writes(client) -> None:
    store = make_store(client, consistency="strong")
    store.add([node("a")])
    store.query(VectorStoreQuery(query_str="q"))
    assert last_query(client)[3] == {"lsn": "1", "consistency": "strong"}


def test_unconvertible_metadata_does_not_fail_ingestion(client, caplog) -> None:
    import datetime

    store = make_store(client)
    n = node("a", mixed=["x", 1], when=datetime.datetime(2024, 1, 1), huge=2**70)
    store.add([n])
    doc = client.last("upsert")[2][0]
    assert doc["metadata"] == {"when": "2024-01-01T00:00:00"}
    assert json.loads(doc["node_content"])["metadata"]["mixed"] == ["x", 1]
    assert "can't filter on" in caplog.text


def test_empty_node_ids_are_no_restriction(client) -> None:
    # VectorStoreIndex.as_retriever() passes node_ids=[] for stores that keep
    # the text, so an empty list must not filter everything out.
    store = make_store(client)
    client.query_results = []
    store.query(VectorStoreQuery(query_str="q", node_ids=[], doc_ids=[]))
    r = q_repr(client)
    assert "field(_id), op=In" not in r and "field(ref_doc_id), op=In" not in r


def test_filter_on_doc_id(client) -> None:
    store = make_store(client)
    client.query_results = []
    store.query(
        VectorStoreQuery(
            query_str="q",
            filters=MetadataFilters(filters=[MetadataFilter(key="doc_id", value="d1")]),
        )
    )
    assert "field(ref_doc_id), op=Eq" in q_repr(client)


def test_async_needs_async_client_when_only_client_given(client, monkeypatch) -> None:
    monkeypatch.delenv("TOPK_API_KEY", raising=False)
    store = TopKVectorStore("test", client=client)
    with pytest.raises(ValueError, match="async_client"):
        _ = store.async_client


def test_lsn_never_goes_backwards(client) -> None:
    store = make_store(client)
    store._record_lsn(None, "5")
    store._record_lsn(None, "3")
    assert store.last_lsn == "5"
    store._record_lsn(None, "7")
    assert store.last_lsn == "7"


def test_validation_happens_before_any_write(client) -> None:
    store = make_store(client, batch_size=1)
    with pytest.raises(ValueError, match="4096"):
        store.add([node("ok"), node("bad", "x" * 5000)])
    assert not client.all("upsert")


def test_get_nodes_splits_large_id_lists(client, monkeypatch) -> None:
    import llama_index.vector_stores.topk.base as base

    monkeypatch.setattr(base, "MAX_QUERY_RESULTS", 2)
    store = make_store(client)
    store.add([node(str(i)) for i in range(5)])
    nodes = store.get_nodes(node_ids=["4", "3", "2", "1", "0"])
    assert [n.node_id for n in nodes] == ["4", "3", "2", "1", "0"]
    assert len(client.all("query")) == 3


def test_explicit_none_partition_means_store_partition(client) -> None:
    store = make_store(client, partition="acme")
    store.add([node("a")], partition=None)
    assert client.last("upsert")[1] == ("test", "acme")
