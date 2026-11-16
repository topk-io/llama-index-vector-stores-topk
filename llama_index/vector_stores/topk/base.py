"""TopK vector store."""

import logging
import os
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from llama_index.core.base.embeddings.base_sparse import BaseSparseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from topk_sdk import AsyncClient, Client
from topk_sdk.data import f32_sparse_vector as sparse_data
from topk_sdk.data import matrix as matrix_data
from topk_sdk.error import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    PartitionNotFoundError,
)
from topk_sdk.query import FunctionExpr, LogicalExpr, Query, field, fn, match, select
from topk_sdk.query import should as should_match
from topk_sdk.schema import (
    f32_sparse_vector,
    f32_vector,
    keyword_index,
    matrix,
    multi_vector_index,
    semantic_index,
    text,
    vector_index,
)

from llama_index.vector_stores.topk.embeddings import (
    BaseMultiVectorEmbedding,
    normalize_multi_vector,
    normalize_sparse,
)
from llama_index.vector_stores.topk.filters import (
    TextExpr,
    metadata_for_filtering,
    to_topk_filter,
)

if TYPE_CHECKING:
    from topk_sdk import AsyncCollectionClient, CollectionClient

logger = logging.getLogger(__name__)

Consistency = Literal["balanced", "indexed", "strong"]
Metric = Literal["cosine", "dot_product", "euclidean"]

MAX_QUERY_RESULTS = 10_000
"""TopK returns at most this many results from a single query."""

MAX_SEMANTIC_TEXT_CHARS = 4096
"""Longest value a `semantic_index()` text field accepts."""

MAX_REQUEST_BYTES = 4 * 1024 * 1024
"""Target upsert payload size, well under TopK's 8MB request limit."""

DEFAULT_REGION_ENV = "TOPK_REGION"
DEFAULT_API_KEY_ENV = "TOPK_API_KEY"

# Score aliases selected in queries. TopK SQL rejects aliases that start with
# an underscore, so none of these do.
_SCORE_ALIAS = {
    "semantic": "semantic_score",
    "keyword": "keyword_score",
    "dense": "dense_score",
    "sparse": "sparse_score",
    "multi_vector": "multi_vector_score",
}
# Signals on the "vector" side of `alpha` (1.0 = vector only) and on the
# "keyword" side (0.0 = keyword only), following LlamaIndex's convention.
_VECTOR_SIGNALS = ("semantic", "dense", "multi_vector")
_LEXICAL_SIGNALS = ("keyword", "sparse")
_METRICS = ("cosine", "dot_product", "euclidean")
_METRIC_NAMES: Dict[str, Metric] = {
    "Cosine": "cosine",
    "DotProduct": "dot_product",
    "Euclidean": "euclidean",
}


class TopKSearchMode(str, Enum):
    """
    A TopK scoring signal, or `hybrid` for all of them in one query.

    LlamaIndex query modes map onto these: `DEFAULT` runs `dense` when a query
    embedding is given and `semantic` otherwise, `TEXT_SEARCH` runs `keyword`,
    `SPARSE` runs `sparse`, `HYBRID` runs `hybrid` and `SEMANTIC_HYBRID` runs
    `semantic` plus `keyword`. Pass `search_mode` in `vector_store_kwargs` to
    pick one directly, for example `multi_vector`.
    """

    SEMANTIC = "semantic"
    """Server-side embeddings and reranking: no embedding model needed."""
    KEYWORD = "keyword"
    """BM25 keyword search over the text field."""
    DENSE = "dense"
    """Vector search with the embeddings LlamaIndex computed."""
    SPARSE = "sparse"
    """Sparse vector search (for example SPLADE), exact with 100% recall."""
    MULTI_VECTOR = "multi_vector"
    """Late-interaction (ColBERT-style) multi-vector search scored with MaxSim."""
    HYBRID = "hybrid"
    """Every configured signal, combined in one query."""


class TopKTextIndex(str, Enum):
    """Index created on the text field of a new collection."""

    SEMANTIC = "semantic"
    """`semantic_index()`: server-side embeddings and BM25 (max 4,096 chars)."""
    KEYWORD = "keyword"
    """`keyword_index()`: BM25 only, no length cap beyond the document size."""
    NONE = "none"
    """No index: text is stored and returned but not searchable."""


class TopKScores:
    """
    Score fields passed to a `ranking` callable.

    Each attribute is a TopK field expression for one signal's score. A signal
    that did not match a document (for example BM25 for a document without a
    query term) has a null score; use `.coalesce(0.0)` when combining scores.
    """

    semantic = field(_SCORE_ALIAS["semantic"])
    keyword = field(_SCORE_ALIAS["keyword"])
    dense = field(_SCORE_ALIAS["dense"])
    sparse = field(_SCORE_ALIAS["sparse"])
    multi_vector = field(_SCORE_ALIAS["multi_vector"])


Ranking = Union[LogicalExpr, FunctionExpr, Callable[[TopKScores], Any]]

_QUERY_OPTIONS = {
    "search_mode",
    "weights",
    "keyword_query",
    "keyword_match_all",
    "keyword_required",
    "ranking",
    "ranking_ascending",
    "topk_filter",
    "partition",
    "lsn",
    "consistency",
}


class TopKVectorStore(BasePydanticVectorStore):
    """
    TopK vector store.

    [TopK](https://topk.io) is a search database with built-in semantic search
    (embeddings and reranking computed on the server), BM25 keyword search,
    dense, sparse and multi-vector search, metadata filtering and custom
    scoring, all available in a single query.

    By default the store uses TopK's server-side semantic search, so no
    embedding model is needed: build the index with
    `embed_model=TopKServerSideEmbedding()`. Pass `embed_dim` to also store
    the dense embeddings LlamaIndex computes.

    Args:
        collection_name: Name of the TopK collection. It is created on first
            use if it does not exist.
        partition: Partition (tenant) to read and write. Partitions are
            isolated sub-collections that share a schema. Each call can
            override it with `partition=`.
        api_key: TopK API key. Defaults to the `TOPK_API_KEY` environment
            variable. Get one at https://console.topk.io.
        region: TopK region, for example `"aws-us-east-1-elastica"`. Defaults
            to the `TOPK_REGION` environment variable.
        client: An existing `topk_sdk.Client` to use instead of `api_key` and
            `region`.
        async_client: An existing `topk_sdk.AsyncClient` for async methods.
            Needed with `client` unless `api_key` (or `TOPK_API_KEY`) is set.
        embed_dim: Dimension of the dense embeddings LlamaIndex computes. When
            set, they are stored in a vector index and used for `DEFAULT`
            queries. Leave it out to use server-side semantic search only.
        distance_metric: Metric of the dense vector index: `"cosine"`
            (default), `"dot_product"` or `"euclidean"`.
        sparse_embedding_model: Optional LlamaIndex sparse embedding model
            (for example SPLADE via `FastEmbedSparseEmbedding`). Enables the
            `SPARSE` query mode.
        multi_vector_embedding_model: Optional late-interaction model (see
            `FastEmbedMultiVectorEmbedding`), scored with MaxSim.
        text_index: Index on the text field of a new collection:
            `"semantic"` (default: server-side embeddings and BM25, values up
            to 4,096 characters), `"keyword"` (BM25 only) or `"none"`.
        hybrid_weights: Default weights for hybrid search, keyed by signal
            (`semantic`, `keyword`, `dense`, `sparse`, `multi_vector`).
            Every configured signal defaults to 1.0; 0 removes a signal.
            Weights multiply TopK's raw scores, which are not normalized, so
            calibrate them on real scores.
        consistency: Default read consistency: `"balanced"` (TopK's default),
            `"indexed"` or `"strong"`.
        read_your_writes: Pass the LSN of this store's latest write in a
            partition to later reads of it, so they always see that write,
            together with the store's `consistency`. Applies to reads that
            set neither `lsn` nor `consistency`.
        create_collection: Create the collection if it does not exist, or
            check an existing collection's schema against these settings.
            With `False`, no network call is made on construction.

    Examples:
        `pip install llama-index-vector-stores-topk`

        ```python
        from llama_index.core import Document, StorageContext, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.topk import (
            TopKServerSideEmbedding,
            TopKVectorStore,
        )

        # Reads TOPK_API_KEY and TOPK_REGION from the environment.
        vector_store = TopKVectorStore(collection_name="llamaindex-docs")
        index = VectorStoreIndex.from_documents(
            [Document(text="TopK embeds text on the server.")],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=TopKServerSideEmbedding(),
            # Semantic text values are capped at 4,096 characters.
            transformations=[SentenceSplitter(chunk_size=512)],
        )
        retriever = index.as_retriever(vector_store_query_mode="hybrid")
        ```

    """

    stores_text: bool = True
    flat_metadata: bool = False
    is_embedding_query: bool = False

    collection_name: str
    partition: Optional[str] = None
    embed_dim: Optional[int] = None
    distance_metric: Metric = "cosine"
    text_index: TopKTextIndex = TopKTextIndex.SEMANTIC
    hybrid_weights: Dict[str, float] = Field(default_factory=dict)
    consistency: Optional[Consistency] = None
    read_your_writes: bool = True
    batch_size: int = 256
    text_field: str = "text"
    metadata_field: str = "metadata"
    node_content_field: str = "node_content"
    node_type_field: str = "node_type"
    ref_doc_id_field: str = "ref_doc_id"
    vector_field: str = "dense_vector"
    sparse_vector_field: str = "sparse_vector"
    multi_vector_field: str = "multi_vector"

    _client: Optional[Client] = PrivateAttr(default=None)
    _async_client: Optional[AsyncClient] = PrivateAttr(default=None)
    _api_key: Optional[str] = PrivateAttr(default=None)
    _region: Optional[str] = PrivateAttr(default=None)
    _host: str = PrivateAttr(default="topk.io")
    _https: bool = PrivateAttr(default=True)
    _sparse_model: Optional[BaseSparseEmbedding] = PrivateAttr(default=None)
    _multi_model: Optional[BaseMultiVectorEmbedding] = PrivateAttr(default=None)
    _requested_text_index: Optional[TopKTextIndex] = PrivateAttr(default=None)
    _requested_metric: Optional[str] = PrivateAttr(default=None)
    _requested_embed_dim: Optional[int] = PrivateAttr(default=None)
    _last_lsn: Dict[Optional[str], str] = PrivateAttr(default_factory=dict)
    _warned_ignored_embeddings: bool = PrivateAttr(default=False)

    def __init__(
        self,
        collection_name: str,
        *,
        partition: Optional[str] = None,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        host: str = "topk.io",
        https: bool = True,
        client: Optional[Client] = None,
        async_client: Optional[AsyncClient] = None,
        embed_dim: Optional[int] = None,
        distance_metric: Optional[Metric] = None,
        sparse_embedding_model: Optional[BaseSparseEmbedding] = None,
        multi_vector_embedding_model: Optional[BaseMultiVectorEmbedding] = None,
        text_index: Optional[Union[TopKTextIndex, str]] = None,
        hybrid_weights: Optional[Mapping[str, float]] = None,
        consistency: Optional[Consistency] = None,
        read_your_writes: bool = True,
        create_collection: bool = True,
        batch_size: int = 256,
        text_field: str = "text",
        metadata_field: str = "metadata",
        node_content_field: str = "node_content",
        node_type_field: str = "node_type",
        ref_doc_id_field: str = "ref_doc_id",
        vector_field: str = "dense_vector",
        sparse_vector_field: str = "sparse_vector",
        multi_vector_field: str = "multi_vector",
        **kwargs: Any,
    ) -> None:
        """Create a TopK vector store."""
        # `to_dict()` includes these computed fields; drop them so that
        # `TopKVectorStore.from_dict(store.to_dict())` works.
        for computed in ("is_embedding_query", "stores_text", "flat_metadata"):
            kwargs.pop(computed, None)
        if not collection_name:
            raise ValueError("`collection_name` is required.")
        if distance_metric is not None and distance_metric not in _METRICS:
            raise ValueError(
                "`distance_metric` must be 'cosine', 'dot_product' or "
                f"'euclidean', got {distance_metric!r}."
            )
        if embed_dim is not None and embed_dim < 1:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}.")
        if consistency is not None and consistency not in (
            "balanced",
            "indexed",
            "strong",
        ):
            raise ValueError(
                "`consistency` must be 'balanced', 'indexed' or 'strong', got "
                f"{consistency!r}."
            )
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        requested_index = TopKTextIndex(text_index) if text_index else None
        super().__init__(  # type: ignore[call-arg]
            collection_name=collection_name,
            partition=partition,
            embed_dim=embed_dim,
            distance_metric=distance_metric or "cosine",
            text_index=requested_index or TopKTextIndex.SEMANTIC,
            hybrid_weights=dict(hybrid_weights or {}),
            consistency=consistency,
            read_your_writes=read_your_writes,
            batch_size=batch_size,
            text_field=text_field,
            metadata_field=metadata_field,
            node_content_field=node_content_field,
            node_type_field=node_type_field,
            ref_doc_id_field=ref_doc_id_field,
            vector_field=vector_field,
            sparse_vector_field=sparse_vector_field,
            multi_vector_field=multi_vector_field,
            is_embedding_query=embed_dim is not None,
            **kwargs,
        )
        self._client = client
        self._async_client = async_client
        self._api_key = api_key
        self._region = region
        self._host = host
        self._https = https
        self._sparse_model = sparse_embedding_model
        self._multi_model = multi_vector_embedding_model
        self._requested_text_index = requested_index
        self._requested_metric = distance_metric
        self._requested_embed_dim = embed_dim
        self._validate_weights(self.hybrid_weights)
        if create_collection:
            self._ensure_collection()
        if self.hybrid_weights:
            # Fail at construction, not at the first hybrid query.
            self._signals(TopKSearchMode.HYBRID, None, None)

    @classmethod
    def class_name(cls) -> str:
        return "TopKVectorStore"

    # ------------------------------------------------------------------
    # Clients and collection management
    # ------------------------------------------------------------------

    def _credentials(self) -> Tuple[str, str]:
        api_key = self._api_key or os.environ.get(DEFAULT_API_KEY_ENV)
        region = self._region or os.environ.get(DEFAULT_REGION_ENV)
        if not api_key:
            raise ValueError(
                "TopK API key not found. Pass `api_key=` or set the TOPK_API_KEY "
                "environment variable. Get a key at https://console.topk.io."
            )
        if not region:
            raise ValueError(
                "TopK region not found. Pass `region=` or set the TOPK_REGION "
                "environment variable, for example 'aws-us-east-1-elastica'. "
                "See https://docs.topk.io/regions."
            )
        return api_key, region

    @property
    def client(self) -> Client:
        """The synchronous `topk_sdk.Client`."""
        if self._client is None:
            api_key, region = self._credentials()
            self._client = Client(
                api_key=api_key, region=region, host=self._host, https=self._https
            )
        return self._client

    @property
    def async_client(self) -> AsyncClient:
        """The asynchronous `topk_sdk.AsyncClient`."""
        if self._async_client is None:
            if self._client is not None and not (
                self._api_key or os.environ.get(DEFAULT_API_KEY_ENV)
            ):
                raise ValueError(
                    "This store was created with `client=` only. Async methods need "
                    "`async_client=topk_sdk.AsyncClient(...)` as well, or an API key "
                    "(`api_key=` or TOPK_API_KEY)."
                )
            api_key, region = self._credentials()
            self._async_client = AsyncClient(
                api_key=api_key, region=region, host=self._host, https=self._https
            )
        return self._async_client

    @property
    def sparse_embedding_model(self) -> Optional[BaseSparseEmbedding]:
        """The sparse embedding model, if any."""
        return self._sparse_model

    @property
    def multi_vector_embedding_model(self) -> Optional[BaseMultiVectorEmbedding]:
        """The multi-vector (late-interaction) embedding model, if any."""
        return self._multi_model

    def _partition(self, kwargs: Mapping[str, Any]) -> Optional[str]:
        partition = kwargs.get("partition")
        return partition if partition is not None else self.partition

    def _collection(self, partition: Optional[str]) -> "CollectionClient":
        return self.client.collection(self.collection_name, partition)

    def _acollection(self, partition: Optional[str]) -> "AsyncCollectionClient":
        return self.async_client.collection(self.collection_name, partition)

    def _schema(self, multi_vector_dim: Optional[int]) -> Dict[str, Any]:
        schema: Dict[str, Any] = {}
        if self.text_index == TopKTextIndex.SEMANTIC:
            schema[self.text_field] = text().index(semantic_index())
        elif self.text_index == TopKTextIndex.KEYWORD:
            schema[self.text_field] = text().index(keyword_index())
        else:
            schema[self.text_field] = text()
        if self.embed_dim is not None:
            schema[self.vector_field] = f32_vector(self.embed_dim).index(
                vector_index(metric=self.distance_metric)
            )
        if self._sparse_model is not None:
            schema[self.sparse_vector_field] = f32_sparse_vector().index(
                vector_index(metric="dot_product")
            )
        if multi_vector_dim is not None:
            schema[self.multi_vector_field] = matrix(
                multi_vector_dim, value_type="f32"
            ).index(multi_vector_index(metric="maxsim"))
        return schema

    def _multi_vector_dim(self) -> Optional[int]:
        if self._multi_model is None:
            return None
        probe = normalize_multi_vector(
            self._multi_model.get_query_embedding("dimension probe")
        )
        return len(probe[0])

    def _ensure_collection(self) -> None:
        collections = self.client.collections()
        try:
            existing = collections.get(self.collection_name)
        except CollectionNotFoundError:
            existing = None
        if existing is not None:
            self._reconcile_schema(existing.schema)
            return
        try:
            collections.create(
                self.collection_name, schema=self._schema(self._multi_vector_dim())
            )
        except CollectionAlreadyExistsError:
            self._reconcile_schema(collections.get(self.collection_name).schema)

    def _reconcile_schema(self, schema: Mapping[str, Any]) -> None:
        """
        Check an existing collection's schema against this store's settings.

        The text index, dense dimension and metric are read from the schema
        (a field's metric is only visible in its text form). A setting that
        contradicts the schema raises `ValueError` instead of silently
        returning wrong results.
        """
        specs = {name: str(spec) for name, spec in schema.items()}
        where = f"collection {self.collection_name!r}"

        text_spec = specs.get(self.text_field, "")
        if "SemanticIndex" in text_spec:
            actual_index = TopKTextIndex.SEMANTIC
        elif "KeywordIndex" in text_spec:
            actual_index = TopKTextIndex.KEYWORD
        else:
            actual_index = TopKTextIndex.NONE
        requested = self._requested_text_index
        if requested is not None and requested != actual_index:
            raise ValueError(
                f"`text_index={requested.value!r}` does not match {where}, whose "
                f"field {self.text_field!r} has text index {actual_index.value!r}."
            )
        self.text_index = actual_index

        def indexed(field_name: str, kind: str, what: str) -> str:
            spec = specs.get(field_name)
            if spec is None or kind not in spec:
                raise ValueError(
                    f"{what} needs an indexed field {field_name!r}, but {where} "
                    f"has {spec or 'no such field'}. Use a new collection, or set "
                    "the field name to an indexed field of this collection."
                )
            return spec

        dense_spec = specs.get(self.vector_field, "")
        if "VectorIndex" in dense_spec and "MultiVectorIndex" not in dense_spec:
            metric = next(
                (
                    m
                    for name, m in _METRIC_NAMES.items()
                    if f"metric: {name}" in dense_spec
                ),
                None,
            )
            if metric is not None:
                if self._requested_metric not in (None, metric):
                    raise ValueError(
                        f"`distance_metric={self._requested_metric!r}` does not "
                        f"match {where}, whose vector index uses {metric!r}."
                    )
                self.distance_metric = metric
            dim = _dimension(dense_spec)
            if dim is not None:
                if self._requested_embed_dim not in (None, dim):
                    raise ValueError(
                        f"`embed_dim={self._requested_embed_dim}` does not match "
                        f"{where}, whose vector field has dimension {dim}."
                    )
                self.embed_dim = dim
                self.is_embedding_query = True
        elif self._requested_embed_dim is not None:
            indexed(self.vector_field, "VectorIndex", "`embed_dim`")
        if self._sparse_model is not None:
            indexed(self.sparse_vector_field, "VectorIndex", "`sparse_embedding_model`")
        if self._multi_model is not None:
            indexed(
                self.multi_vector_field,
                "MultiVectorIndex",
                "`multi_vector_embedding_model`",
            )

    def delete_collection(self) -> None:
        """
        Delete the whole TopK collection, including every partition.

        For about a minute after a delete, writes to a collection recreated
        under the same name can be lost. Use a new name instead.
        """
        self.client.collections().delete(self.collection_name)
        self._last_lsn.clear()

    def delete_partition(self, partition: Optional[str] = None) -> None:
        """
        Delete a partition (tenant) and all of its documents.

        Deletion takes about a minute to apply, and writes to the same
        partition name are rejected for a while afterwards.

        Args:
            partition: Partition to delete. Defaults to this store's partition.

        """
        name = partition or self.partition
        if not name:
            raise ValueError("No partition given and this store is not scoped to one.")
        self.client.collection(self.collection_name).delete_partition(name)
        self._last_lsn.pop(name, None)

    def list_partitions(self) -> List[str]:
        """Return the names of the collection's partitions."""
        return [
            p.name
            for p in self.client.collection(self.collection_name).list_partitions()
        ]

    def count(self, partition: Optional[str] = None, **kwargs: Any) -> int:
        """Return the number of documents in the store's (or given) partition."""
        part = partition if partition is not None else self.partition
        try:
            return self._collection(part).count(
                **self._read_options(part, kwargs.get("lsn"), kwargs.get("consistency"))
            )
        except PartitionNotFoundError:
            return 0

    # ------------------------------------------------------------------
    # Read options
    # ------------------------------------------------------------------

    def _record_lsn(self, partition: Optional[str], lsn: str) -> None:
        """Remember the latest LSN, ignoring older ones from concurrent writes."""
        previous = self._last_lsn.get(partition)
        if previous is not None:
            try:
                if int(lsn) < int(previous):
                    return
            except ValueError:
                pass
        self._last_lsn[partition] = lsn

    @property
    def last_lsn(self) -> Optional[str]:
        """LSN of this store's latest write to its own partition."""
        return self._last_lsn.get(self.partition)

    def _read_options(
        self,
        partition: Optional[str],
        lsn: Optional[str] = None,
        consistency: Optional[Consistency] = None,
    ) -> Dict[str, Any]:
        if lsn is not None:
            options: Dict[str, Any] = {"lsn": lsn}
            if consistency not in (None, "balanced"):
                options["consistency"] = consistency
            return options
        if consistency is not None:
            if consistency not in ("balanced", "indexed", "strong"):
                raise ValueError(
                    "`consistency` must be 'balanced', 'indexed' or 'strong', "
                    f"got {consistency!r}."
                )
            return {} if consistency == "balanced" else {"consistency": consistency}
        last = self._last_lsn.get(partition)
        if self.read_your_writes and last is not None:
            options = {"lsn": last}
            if self.consistency not in (None, "balanced"):
                options["consistency"] = self.consistency
            return options
        if self.consistency not in (None, "balanced"):
            return {"consistency": self.consistency}
        return {}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _node_text(self, node: BaseNode) -> str:
        content = node.get_content(metadata_mode=MetadataMode.NONE)
        if (
            self.text_index == TopKTextIndex.SEMANTIC
            and len(content) > MAX_SEMANTIC_TEXT_CHARS
        ):
            raise ValueError(
                f"Node {node.node_id!r} has {len(content)} characters, but TopK's "
                f"semantic index accepts at most {MAX_SEMANTIC_TEXT_CHARS}. Split "
                "documents into smaller chunks, for example with "
                "`SentenceSplitter(chunk_size=512)`, or create the store with "
                "`text_index='keyword'`."
            )
        return content

    def _node_row(
        self,
        node: BaseNode,
        content: str,
        sparse: Optional[Any],
        multi: Optional[Any],
    ) -> Tuple[Dict[str, Any], int]:
        """Build a TopK document for a node. Also returns its estimated size."""
        node_dict = node_to_metadata_dict(
            node, remove_text=True, flat_metadata=self.flat_metadata
        )
        row: Dict[str, Any] = {
            "_id": node.node_id,
            self.text_field: content,
            self.node_content_field: node_dict["_node_content"],
            self.node_type_field: node_dict["_node_type"],
        }
        if node.ref_doc_id is not None:
            row[self.ref_doc_id_field] = node.ref_doc_id
        filterable = metadata_for_filtering(node.metadata)
        if filterable:
            row[self.metadata_field] = filterable
        size = (
            128
            + len(content.encode("utf-8"))
            + 2 * len(node_dict["_node_content"].encode("utf-8"))
        )
        if self.embed_dim is not None:
            embedding = node.get_embedding() if node.embedding else None
            if not embedding:
                raise ValueError(
                    f"Node {node.node_id!r} has no embedding, but this store "
                    f"stores dense vectors (embed_dim={self.embed_dim}). Build the "
                    "index with an `embed_model` of that dimension."
                )
            if len(embedding) != self.embed_dim:
                raise ValueError(
                    f"Node {node.node_id!r} has a {len(embedding)}-dimensional "
                    f"embedding, but the store expects embed_dim={self.embed_dim}."
                )
            row[self.vector_field] = [float(x) for x in embedding]
            size += 4 * self.embed_dim
        elif node.embedding and not self._warned_ignored_embeddings:
            self._warned_ignored_embeddings = True
            logger.warning(
                "Ignoring client-side node embeddings: this TopK store uses "
                "server-side semantic search. Pass `embed_dim=` to store them, "
                "or use `embed_model=TopKServerSideEmbedding()` to skip "
                "computing them."
            )
        if sparse is not None:
            vec = normalize_sparse(sparse)
            row[self.sparse_vector_field] = sparse_data(vec)
            size += 8 * len(vec)
        if multi is not None:
            rows = normalize_multi_vector(multi)
            row[self.multi_vector_field] = matrix_data(rows, value_type="f32")
            size += 4 * sum(len(r) for r in rows)
        return row, size

    @staticmethod
    def _size_batches(
        rows: List[Dict[str, Any]], sizes: List[int]
    ) -> Iterable[List[Dict[str, Any]]]:
        """Split rows into upsert requests that stay under the request limit."""
        batch: List[Dict[str, Any]] = []
        total = 0
        for row, size in zip(rows, sizes, strict=True):
            if batch and total + size > MAX_REQUEST_BYTES:
                yield batch
                batch, total = [], 0
            batch.append(row)
            total += size
        if batch:
            yield batch

    def _build_rows(
        self,
        nodes: Sequence[BaseNode],
        texts: List[str],
        sparse: Optional[List[Any]],
        multi: Optional[List[Any]],
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        rows, sizes = [], []
        for i, node in enumerate(nodes):
            row, size = self._node_row(
                node,
                texts[i],
                sparse[i] if sparse is not None else None,
                multi[i] if multi is not None else None,
            )
            rows.append(row)
            sizes.append(size)
        return rows, sizes

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """
        Upsert nodes. Nodes with an existing ID are replaced.

        With server-side semantic search no embedding happens on the client.
        Sparse and multi-vector embeddings are computed from the node text
        when the corresponding models are configured.

        Args:
            nodes: Nodes to add.
            **add_kwargs: `partition` to write to another partition (tenant).

        Returns:
            The IDs of the added nodes.

        """
        if not nodes:
            return []
        partition = self._partition(add_kwargs)
        # Validate every node before the first write, so that a bad node
        # doesn't leave a partial write behind.
        all_texts = [self._node_text(n) for n in nodes]
        ids: List[str] = []
        for start in range(0, len(nodes), self.batch_size):
            chunk = list(nodes[start : start + self.batch_size])
            texts = all_texts[start : start + self.batch_size]
            sparse = (
                self._sparse_model.get_text_embedding_batch(texts)
                if self._sparse_model is not None
                else None
            )
            multi = (
                self._multi_model.get_text_embedding_batch(texts)
                if self._multi_model is not None
                else None
            )
            rows, sizes = self._build_rows(chunk, texts, sparse, multi)
            for request in self._size_batches(rows, sizes):
                self._record_lsn(partition, self._collection(partition).upsert(request))
            ids.extend(n.node_id for n in chunk)
        return ids

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Asynchronously upsert nodes. See `add`."""
        if not nodes:
            return []
        partition = self._partition(kwargs)
        all_texts = [self._node_text(n) for n in nodes]
        ids: List[str] = []
        for start in range(0, len(nodes), self.batch_size):
            chunk = list(nodes[start : start + self.batch_size])
            texts = all_texts[start : start + self.batch_size]
            sparse = (
                await self._sparse_model.aget_text_embedding_batch(texts)
                if self._sparse_model is not None
                else None
            )
            multi = (
                await self._multi_model.aget_text_embedding_batch(texts)
                if self._multi_model is not None
                else None
            )
            rows, sizes = self._build_rows(chunk, texts, sparse, multi)
            for request in self._size_batches(rows, sizes):
                self._record_lsn(
                    partition, await self._acollection(partition).upsert(request)
                )
            ids.extend(n.node_id for n in chunk)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node of a source document (`ref_doc_id`)."""
        partition = self._partition(delete_kwargs)
        self._delete(partition, field(self.ref_doc_id_field) == ref_doc_id)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Asynchronously delete every node of a source document."""
        partition = self._partition(delete_kwargs)
        await self._adelete(partition, field(self.ref_doc_id_field) == ref_doc_id)

    def _nodes_target(
        self, node_ids: Optional[List[str]], filters: Optional[MetadataFilters]
    ) -> Optional[Any]:
        clauses = []
        if node_ids is not None:
            if not node_ids:
                return None
            clauses.append(field("_id").in_([str(i) for i in node_ids]))
        expr = to_topk_filter(filters, self.metadata_field, self.ref_doc_id_field)
        if isinstance(expr, TextExpr):
            raise ValueError("Deletes accept metadata filters, not text matches.")
        if expr is not None:
            clauses.append(expr)
        if not clauses:
            raise ValueError(
                "Pass `node_ids` or `filters`. To remove everything, use `clear()`."
            )
        target = clauses[0]
        for clause in clauses[1:]:
            target = target.and_(clause)
        return target

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes by ID, by metadata filter, or both (nodes matching both)."""
        target = self._nodes_target(node_ids, filters)
        if target is not None:
            self._delete(self._partition(delete_kwargs), target)

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete nodes by ID and/or metadata filter."""
        target = self._nodes_target(node_ids, filters)
        if target is not None:
            await self._adelete(self._partition(delete_kwargs), target)

    def clear(self, **kwargs: Any) -> None:
        """
        Delete every node in the store's partition (or `partition=`).

        The collection and partition themselves are kept, so the store can be
        written to again right away.
        """
        self._delete(self._partition(kwargs), field("_id").is_not_null())

    async def aclear(self, **kwargs: Any) -> None:
        """Asynchronously delete every node in the store's partition."""
        await self._adelete(self._partition(kwargs), field("_id").is_not_null())

    def _delete(self, partition: Optional[str], target: Any) -> None:
        try:
            self._record_lsn(partition, self._collection(partition).delete(target))
        except PartitionNotFoundError:
            pass

    async def _adelete(self, partition: Optional[str], target: Any) -> None:
        try:
            self._record_lsn(
                partition, await self._acollection(partition).delete(target)
            )
        except PartitionNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Reads by ID or filter
    # ------------------------------------------------------------------

    def _row_to_node(self, row: Mapping[str, Any]) -> BaseNode:
        content = row.get(self.text_field) or ""
        node_json = row.get(self.node_content_field)
        if node_json:
            node = metadata_dict_to_node(
                {
                    "_node_content": node_json,
                    "_node_type": row.get(self.node_type_field),
                },
                text=content,
            )
        else:
            # Written outside LlamaIndex: build a plain text node.
            metadata = row.get(self.metadata_field) or {}
            node = TextNode(id_=row["_id"], text=content, metadata=dict(metadata))
        return node

    def _get_query(
        self,
        node_ids: Optional[List[str]],
        filters: Optional[MetadataFilters],
        limit: Optional[int],
    ) -> Query:
        expr = to_topk_filter(filters, self.metadata_field, self.ref_doc_id_field)
        q = select(self.text_field, self.node_content_field, self.node_type_field)
        if node_ids is not None:
            q = q.filter(field("_id").in_([str(i) for i in node_ids]))
        if expr is not None:
            q = q.filter(expr)
        k = limit if limit is not None else MAX_QUERY_RESULTS
        if node_ids is not None:
            k = min(k, len(node_ids))
        _check_k(k)
        return q.limit(k)

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Get nodes by ID and/or metadata filter.

        Without `node_ids`, returns at most `limit` nodes (default and
        maximum: 10,000, TopK's per-query cap). Any number of IDs can be
        requested; they are fetched 10,000 at a time and returned in the
        order given.
        """
        if node_ids is not None and not node_ids:
            return []
        partition = self._partition(kwargs)
        rows: List[Dict[str, Any]] = []
        for ids in _id_chunks(node_ids):
            q = self._get_query(ids, filters, _remaining(limit, rows))
            rows.extend(
                self._run(q, partition, kwargs.get("lsn"), kwargs.get("consistency"))
            )
            if limit is not None and len(rows) >= limit:
                break
        return self._order_nodes(rows, node_ids)

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Asynchronously get nodes by ID and/or metadata filter."""
        if node_ids is not None and not node_ids:
            return []
        partition = self._partition(kwargs)
        rows: List[Dict[str, Any]] = []
        for ids in _id_chunks(node_ids):
            q = self._get_query(ids, filters, _remaining(limit, rows))
            rows.extend(
                await self._arun(
                    q, partition, kwargs.get("lsn"), kwargs.get("consistency")
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return self._order_nodes(rows, node_ids)

    def _order_nodes(
        self, rows: List[Dict[str, Any]], node_ids: Optional[List[str]]
    ) -> List[BaseNode]:
        nodes = [self._row_to_node(r) for r in rows]
        if node_ids is None:
            return nodes
        by_id = {n.node_id: n for n in nodes}
        return [by_id[str(i)] for i in node_ids if str(i) in by_id]

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _validate_weights(self, weights: Mapping[str, float]) -> None:
        allowed = set(_SCORE_ALIAS)
        unknown = set(weights) - allowed
        if unknown:
            raise ValueError(
                f"Unknown hybrid weight keys {sorted(unknown)}. "
                f"Use any of {sorted(allowed)}."
            )

    def _available_signals(self) -> List[str]:
        available = []
        if self.text_index == TopKTextIndex.SEMANTIC:
            available.append("semantic")
        if self.text_index != TopKTextIndex.NONE:
            available.append("keyword")
        if self.embed_dim is not None:
            available.append("dense")
        if self._sparse_model is not None:
            available.append("sparse")
        if self._multi_model is not None:
            available.append("multi_vector")
        return available

    def _check_signal(self, name: str) -> None:
        if name in self._available_signals():
            return
        reasons = {
            "semantic": "a collection whose text field has a semantic index "
            "(`text_index='semantic'`)",
            "keyword": "a collection whose text field has a semantic or keyword index",
            "dense": "dense vectors: create the store with `embed_dim=`",
            "sparse": "a `sparse_embedding_model`",
            "multi_vector": "a `multi_vector_embedding_model`",
        }
        raise ValueError(f"{name.replace('_', '-')} search needs {reasons[name]}.")

    def _resolve_mode(
        self, query: VectorStoreQuery, search_mode: Optional[Union[TopKSearchMode, str]]
    ) -> Tuple[TopKSearchMode, Optional[List[str]]]:
        """Map a LlamaIndex query mode onto a TopK search mode."""
        if search_mode is not None:
            return TopKSearchMode(search_mode), None
        mode = VectorStoreQueryMode(query.mode)
        if mode == VectorStoreQueryMode.DEFAULT:
            if query.query_embedding is not None and self.embed_dim is not None:
                return TopKSearchMode.DENSE, None
            if self.text_index == TopKTextIndex.SEMANTIC:
                return TopKSearchMode.SEMANTIC, None
            if self.embed_dim is not None:
                raise ValueError(
                    "Dense search needs a query embedding. Query through a "
                    "VectorStoreIndex with an embed_model, or set "
                    "`query_embedding`."
                )
            raise ValueError(
                "This store has neither a semantic index nor dense vectors; "
                "use TEXT_SEARCH or another mode."
            )
        if mode == VectorStoreQueryMode.TEXT_SEARCH:
            return TopKSearchMode.KEYWORD, None
        if mode == VectorStoreQueryMode.SPARSE:
            if self._sparse_model is None:
                raise ValueError(
                    "SPARSE mode needs a `sparse_embedding_model` (for example "
                    "SPLADE via FastEmbedSparseEmbedding). For BM25 keyword "
                    "search, use TEXT_SEARCH."
                )
            return TopKSearchMode.SPARSE, None
        if mode == VectorStoreQueryMode.HYBRID:
            return TopKSearchMode.HYBRID, None
        if mode == VectorStoreQueryMode.SEMANTIC_HYBRID:
            return TopKSearchMode.HYBRID, ["semantic", "keyword"]
        raise ValueError(
            f"Query mode {mode.value!r} is not supported by TopKVectorStore. Use "
            "DEFAULT, TEXT_SEARCH, SPARSE, HYBRID or SEMANTIC_HYBRID."
        )

    def _signals(
        self,
        mode: TopKSearchMode,
        weights: Optional[Mapping[str, float]],
        alpha: Optional[float],
        restrict: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Return the scoring signals for `mode` with their weights."""
        if mode != TopKSearchMode.HYBRID:
            self._check_signal(mode.value)
            return {mode.value: 1.0}
        available = self._available_signals()
        if restrict is not None:
            available = [s for s in available if s in restrict]
        # Every available signal starts at 1.0; the store's `hybrid_weights`
        # and the query's `weights` override it. A weight of 0 drops a signal.
        overrides = dict(self.hybrid_weights)
        if restrict is not None:
            overrides = {k: v for k, v in overrides.items() if k in restrict}
        if weights:
            self._validate_weights(weights)
            overrides.update(weights)
        unavailable = sorted(
            k for k, w in overrides.items() if w and k not in available
        )
        if unavailable:
            raise ValueError(
                f"Hybrid weights {unavailable} refer to signals this store is not "
                f"configured for. Available signals: {available}."
            )
        merged = {name: 1.0 for name in available}
        merged.update({k: v for k, v in overrides.items() if k in available})
        if alpha is not None:
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"`alpha` must be between 0 and 1, got {alpha}.")
            merged = {
                k: v * (alpha if k in _VECTOR_SIGNALS else 1.0 - alpha)
                for k, v in merged.items()
            }
        signals = {k: float(v) for k, v in merged.items() if v}
        if not signals:
            raise ValueError("Hybrid search needs at least one non-zero weight.")
        return signals

    def _query_vectors(
        self, query: VectorStoreQuery, signals: Mapping[str, float]
    ) -> Dict[str, Any]:
        vectors: Dict[str, Any] = {}
        if "sparse" in signals and self._sparse_model is not None:
            vectors["sparse"] = self._sparse_model.get_query_embedding(
                _require_text(query, "Sparse")
            )
        if "multi_vector" in signals and self._multi_model is not None:
            vectors["multi_vector"] = self._multi_model.get_query_embedding(
                _require_text(query, "Multi-vector")
            )
        return vectors

    async def _aquery_vectors(
        self, query: VectorStoreQuery, signals: Mapping[str, float]
    ) -> Dict[str, Any]:
        vectors: Dict[str, Any] = {}
        if "sparse" in signals and self._sparse_model is not None:
            vectors["sparse"] = await self._sparse_model.aget_query_embedding(
                _require_text(query, "Sparse")
            )
        if "multi_vector" in signals and self._multi_model is not None:
            vectors["multi_vector"] = await self._multi_model.aget_query_embedding(
                _require_text(query, "Multi-vector")
            )
        return vectors

    def _build_query(
        self,
        query: VectorStoreQuery,
        signals: Mapping[str, float],
        vectors: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> Query:
        k = query.similarity_top_k
        _check_k(k)
        scores: Dict[str, Any] = {}
        text_filter = None
        keyword_text = options.get("keyword_query") or query.query_str

        if "semantic" in signals:
            scores[_SCORE_ALIAS["semantic"]] = fn.semantic_similarity(
                self.text_field, _require_text(query, "Semantic")
            )
        if "keyword" in signals:
            if not keyword_text:
                raise ValueError("Keyword search needs a text query (`query_str`).")
            scores[_SCORE_ALIAS["keyword"]] = fn.bm25_score()
            if len(signals) == 1 or options.get("keyword_required"):
                text_filter = match(
                    keyword_text,
                    field=self.text_field,
                    all=bool(options.get("keyword_match_all")),
                )
            else:
                # Boost documents that contain query terms without excluding
                # the ones that only match through another signal.
                text_filter = should_match(keyword_text, field=self.text_field)
        if "dense" in signals:
            if query.query_embedding is None:
                raise ValueError("Dense search needs a query embedding.")
            if len(query.query_embedding) != self.embed_dim:
                raise ValueError(
                    f"The query embedding has {len(query.query_embedding)} "
                    f"dimensions, but the store expects embed_dim={self.embed_dim}."
                )
            scores[_SCORE_ALIAS["dense"]] = fn.vector_distance(
                self.vector_field, [float(x) for x in query.query_embedding]
            )
        if "sparse" in signals:
            scores[_SCORE_ALIAS["sparse"]] = fn.vector_distance(
                self.sparse_vector_field,
                sparse_data(normalize_sparse(vectors["sparse"])),
            )
        if "multi_vector" in signals:
            scores[_SCORE_ALIAS["multi_vector"]] = fn.multi_vector_distance(
                self.multi_vector_field,
                matrix_data(
                    normalize_multi_vector(vectors["multi_vector"]), value_type="f32"
                ),
            )

        q = select(
            self.text_field, self.node_content_field, self.node_type_field, **scores
        )
        if text_filter is not None:
            q = q.filter(text_filter)
        for expr in (
            to_topk_filter(query.filters, self.metadata_field, self.ref_doc_id_field),
            to_topk_filter(
                options.get("topk_filter"), self.metadata_field, self.ref_doc_id_field
            ),
        ):
            if expr is not None:
                q = q.filter(expr)
        # An empty list means "no restriction": `VectorStoreIndex.as_retriever()`
        # passes `node_ids=[]` for stores that keep the text themselves.
        if query.doc_ids:
            q = q.filter(field(self.ref_doc_id_field).in_(list(query.doc_ids)))
        if query.node_ids:
            q = q.filter(field("_id").in_(list(query.node_ids)))

        ranking = options.get("ranking")
        if ranking is not None:
            sort_expr = ranking(TopKScores()) if callable(ranking) else ranking
            if not isinstance(sort_expr, (LogicalExpr, FunctionExpr)):
                raise ValueError(
                    "`ranking` must be a TopK scoring expression, or a callable "
                    "returning one."
                )
            q = q.sort(sort_expr, asc=bool(options.get("ranking_ascending")))
        else:
            q = q.sort(self._combined_score(signals), asc=self._ascending(signals))
        return q.limit(k)

    def _signal_sign(self, name: str) -> float:
        # Euclidean distance is lower-is-better; every other score is
        # higher-is-better. Negate it inside hybrid sums.
        if name == "dense" and self.distance_metric == "euclidean":
            return -1.0
        return 1.0

    def _ascending(self, signals: Mapping[str, float]) -> bool:
        return list(signals) == ["dense"] and self.distance_metric == "euclidean"

    def _combined_score(self, signals: Mapping[str, float]) -> Any:
        if len(signals) == 1:
            (name,) = signals
            return field(_SCORE_ALIAS[name])
        expr: Any = None
        for name, weight in signals.items():
            term = field(_SCORE_ALIAS[name]).coalesce(0.0) * (
                weight * self._signal_sign(name)
            )
            expr = term if expr is None else expr + term
        return expr

    def _combined_value(
        self, row: Mapping[str, Any], signals: Mapping[str, float]
    ) -> float:
        if len(signals) == 1:
            (name,) = signals
            value = row.get(_SCORE_ALIAS[name])
            if value is None:
                return 0.0
            if name == "dense" and self.distance_metric == "euclidean":
                # Report a higher-is-better similarity in (0, 1].
                return 1.0 / (1.0 + float(value))
            return float(value)
        total = 0.0
        for name, weight in signals.items():
            value = row.get(_SCORE_ALIAS[name])
            if value is not None:
                total += float(value) * weight * self._signal_sign(name)
        return total

    # ------------------------------------------------------------------
    # Running queries
    # ------------------------------------------------------------------

    def _missing_content(self, rows: List[Dict[str, Any]]) -> List[str]:
        return [r["_id"] for r in rows if not r.get(self.node_content_field)]

    def _merge_metadata(
        self, rows: List[Dict[str, Any]], found: Mapping[str, Mapping[str, Any]]
    ) -> None:
        for row in rows:
            metadata = found.get(row["_id"], {}).get(self.metadata_field)
            if metadata is not None:
                row[self.metadata_field] = metadata

    def _run(
        self,
        q: Query,
        partition: Optional[str],
        lsn: Optional[str] = None,
        consistency: Optional[Consistency] = None,
    ) -> List[Dict[str, Any]]:
        options = self._read_options(partition, lsn, consistency)
        collection = self._collection(partition)
        try:
            rows = collection.query(q, **options)
            missing = self._missing_content(rows)
            if missing:
                # Documents written outside LlamaIndex have no serialized node;
                # queries cannot return a whole struct, so fetch metadata by ID.
                found = collection.get(missing, fields=[self.metadata_field], **options)
                self._merge_metadata(rows, found)
            return rows
        except PartitionNotFoundError:
            return []

    async def _arun(
        self,
        q: Query,
        partition: Optional[str],
        lsn: Optional[str] = None,
        consistency: Optional[Consistency] = None,
    ) -> List[Dict[str, Any]]:
        options = self._read_options(partition, lsn, consistency)
        collection = self._acollection(partition)
        try:
            rows = await collection.query(q, **options)
            missing = self._missing_content(rows)
            if missing:
                found = await collection.get(
                    missing, fields=[self.metadata_field], **options
                )
                self._merge_metadata(rows, found)
            return rows
        except PartitionNotFoundError:
            return []

    def _result(
        self, rows: List[Dict[str, Any]], signals: Mapping[str, float]
    ) -> VectorStoreQueryResult:
        nodes = [self._row_to_node(r) for r in rows]
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[self._combined_value(r, signals) for r in rows],
            ids=[n.node_id for n in nodes],
        )

    @staticmethod
    def _split_options(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = set(kwargs) - _QUERY_OPTIONS
        if unknown:
            raise ValueError(
                f"Unknown query options {sorted(unknown)}. Supported options: "
                f"{sorted(_QUERY_OPTIONS)}."
            )
        return dict(kwargs)

    def _prepare(
        self, query: VectorStoreQuery, kwargs: Mapping[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        options = self._split_options(kwargs)
        mode, restrict = self._resolve_mode(query, options.get("search_mode"))
        signals = self._signals(mode, options.get("weights"), query.alpha, restrict)
        return options, signals

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Search the collection.

        The LlamaIndex query mode picks the TopK search: `DEFAULT` (dense
        vectors if the store has them and a query embedding is given,
        otherwise server-side semantic search), `TEXT_SEARCH` (BM25),
        `SPARSE`, `HYBRID` (every configured signal in one query, weighted by
        `hybrid_weights` and `alpha`) and `SEMANTIC_HYBRID` (semantic plus
        BM25). `filters`, `doc_ids` and `node_ids` are applied in the same
        query.

        Args:
            query: The LlamaIndex query.
            **kwargs: TopK options, also accepted through a retriever's
                `vector_store_kwargs`:

                * `search_mode`: a `TopKSearchMode` (or its value) to run
                  instead of the query mode, for example `"multi_vector"`.
                * `weights`: hybrid weights for this query, for example
                  `{"semantic": 1.0, "keyword": 0.2}`.
                * `keyword_query`: BM25 query text, if different from
                  `query_str`.
                * `keyword_match_all`: in keyword mode, require every term.
                * `keyword_required`: in hybrid mode, only return nodes that
                  contain at least one query term.
                * `ranking`: custom scoring expression to sort by. A TopK
                  expression, or a callable that receives `TopKScores` and
                  returns one. Similarities stay the signal scores.
                * `ranking_ascending`: sort by `ranking` ascending.
                * `topk_filter`: an extra TopK filter expression, for example
                  `field("metadata.year") >= 2020` or `match("rust")`.
                * `partition`: partition (tenant) to search.
                * `lsn` / `consistency`: read consistency for this query.

        Returns:
            Nodes with their similarities, most relevant first.

        """
        options, signals = self._prepare(query, kwargs)
        vectors = self._query_vectors(query, signals)
        q = self._build_query(query, signals, vectors, options)
        rows = self._run(
            q,
            self._partition(options),
            options.get("lsn"),
            options.get("consistency"),
        )
        return self._result(rows, signals)

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Asynchronously search the collection. See `query`."""
        options, signals = self._prepare(query, kwargs)
        vectors = await self._aquery_vectors(query, signals)
        q = self._build_query(query, signals, vectors, options)
        rows = await self._arun(
            q,
            self._partition(options),
            options.get("lsn"),
            options.get("consistency"),
        )
        return self._result(rows, signals)

    def query_raw(
        self,
        topk_query: Query,
        *,
        partition: Optional[str] = None,
        lsn: Optional[str] = None,
        consistency: Optional[Consistency] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run any TopK query against this store's collection and partition.

        Use this for the full TopK query language, for example `group_by` or
        `offset`. Rows are returned as TopK returns them.
        """
        part = partition if partition is not None else self.partition
        options = self._read_options(part, lsn, consistency)
        try:
            return self._collection(part).query(topk_query, **options)
        except PartitionNotFoundError:
            return []


def _id_chunks(
    node_ids: Optional[List[str]],
) -> Iterable[Optional[List[str]]]:
    if node_ids is None:
        yield None
        return
    for start in range(0, len(node_ids), MAX_QUERY_RESULTS):
        yield node_ids[start : start + MAX_QUERY_RESULTS]


def _remaining(limit: Optional[int], rows: List[Dict[str, Any]]) -> Optional[int]:
    return None if limit is None else limit - len(rows)


def _check_k(k: int) -> None:
    if k < 1 or k > MAX_QUERY_RESULTS:
        raise ValueError(
            f"`similarity_top_k` must be between 1 and {MAX_QUERY_RESULTS}, got {k}."
        )


def _require_text(query: VectorStoreQuery, what: str) -> str:
    if not query.query_str:
        raise ValueError(f"{what} search needs a text query (`query_str`).")
    return query.query_str


def _dimension(spec: str) -> Optional[int]:
    import re

    m = re.search(r"dimension:\s*(\d+)", spec)
    return int(m.group(1)) if m else None
