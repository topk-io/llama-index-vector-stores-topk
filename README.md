# LlamaIndex Vector Stores Integration: TopK

[TopK](https://topk.io) is a search database for AI applications. Its built-in semantic search computes embeddings and reranks on the server. It also has BM25 keyword search, dense, sparse and multi-vector search, metadata filtering and custom scoring, and all of these can be combined in **one query**.

`TopKVectorStore` brings all of it to LlamaIndex:

| TopK feature | How to use it in LlamaIndex | TopK guide |
|---|---|---|
| Server-side embeddings: no embedding model or key | `VectorStoreIndex(..., embed_model=TopKServerSideEmbedding())` | [Semantic search](https://docs.topk.io/guides/semantic-search) |
| BM25 keyword search | `vector_store_query_mode="text_search"` | [Keyword search](https://docs.topk.io/guides/keyword-search) |
| True hybrid search in one query | `vector_store_query_mode="hybrid"`, `alpha`, `weights` | [True hybrid search](https://docs.topk.io/guides/true-hybrid-search) |
| Custom scoring expressions | `vector_store_kwargs={"ranking": ...}` | [Custom scoring](https://docs.topk.io/concepts#custom-scoring) |
| Dense vectors from any LlamaIndex embedding model | `TopKVectorStore(embed_dim=...)` | [Vector search](https://docs.topk.io/guides/vector-search) |
| Sparse vectors (SPLADE), exact with 100% recall | `sparse_embedding_model=...`, `vector_store_query_mode="sparse"` | [Sparse vector search](https://docs.topk.io/guides/sparse-vector-search) |
| Multi-vector late interaction (ColBERT, MaxSim) | `multi_vector_embedding_model=...` | [Multi-vector search](https://docs.topk.io/guides/multi-vector-search) |
| Every `MetadataFilters` operator and condition | `filters=MetadataFilters(...)` | [Filtering](https://docs.topk.io/collections/query#filtering) |
| Partitions for multi-tenancy | `partition="tenant"` | [Multi-tenancy](https://docs.topk.io/concepts#multi-tenancy) |
| Consistency: balanced, indexed, strong, LSN, read-your-writes | `consistency=...`, `lsn=...` | [Read consistency](https://docs.topk.io/concepts#read-consistency) |
| Async | `aquery`, `async_add`, `aretrieve`, ... | |

## Installation

```bash
pip install llama-index-vector-stores-topk
```

Create an API key in the [TopK console](https://console.topk.io) and set:

```bash
export TOPK_API_KEY="your-api-key"
export TOPK_REGION="aws-us-east-1-elastica"  # see https://docs.topk.io/regions
```

## Usage

### Semantic search with no embedding model

TopK embeds documents and queries on the server. Pass `TopKServerSideEmbedding()` as the `embed_model`, so that LlamaIndex doesn't fall back to its default (OpenAI) model. It makes no network calls.

```python
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.topk import TopKServerSideEmbedding, TopKVectorStore

vector_store = TopKVectorStore(collection_name="llamaindex-quickstart")
index = VectorStoreIndex.from_documents(
    [
        Document(text="Rust guarantees memory safety without a garbage collector.", metadata={"lang": "rust", "year": 2015}),
        Document(text="Go has a garbage collector and goroutines.", metadata={"lang": "go", "year": 2012}),
    ],
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=TopKServerSideEmbedding(),
    transformations=[SentenceSplitter(chunk_size=512)],  # passage-sized chunks retrieve best
)

retriever = index.as_retriever(similarity_top_k=2)
retriever.retrieve("memory safe systems programming")
```

### Keyword (BM25) and hybrid search

```python
index.as_retriever(vector_store_query_mode="text_search").retrieve("garbage collector")

# Semantic + BM25 (+ dense, sparse, multi-vector if configured) in ONE TopK query.
index.as_retriever(vector_store_query_mode="hybrid", alpha=0.7).retrieve("garbage collector")

# Per-signal weights and a custom scoring expression that uses metadata.
from topk_sdk.query import field

index.as_retriever(
    vector_store_query_mode="hybrid",
    vector_store_kwargs={
        "weights": {"semantic": 1.0, "keyword": 0.2},
        "ranking": lambda s: s.semantic + s.keyword.coalesce(0.0) * 0.2 + field("metadata.year") * 0.001,
    },
).retrieve("garbage collector")
```

LlamaIndex query modes map onto TopK like this:

| `vector_store_query_mode` | TopK search |
|---|---|
| `default` | Dense vectors if the store has them (`embed_dim`), otherwise server-side semantic search |
| `text_search` | BM25 keyword search |
| `sparse` | Sparse vectors (needs `sparse_embedding_model`) |
| `hybrid` | Every configured signal plus BM25, combined in one query as a weighted sum. `alpha` scales vector signals (semantic, dense, multi-vector) by `alpha` and lexical ones (BM25, sparse) by `1 - alpha` |
| `semantic_hybrid` | Server-side semantic search plus BM25 |

Pass `vector_store_kwargs={"search_mode": "multi_vector"}` (or any `TopKSearchMode`) to choose a single signal directly.

Hybrid weights apply to TopK's raw scores, which are not normalized: semantic similarity is typically about 1–4, BM25 about 0–15 and cosine similarity -1–1, depending on your data. Print a few real scores (the similarities on each result) before choosing `alpha` or `weights`, or use a `ranking` expression to rescale them. Every configured signal starts with weight 1.0; a weight of 0 removes it.

### Metadata filters, partitions and consistency

```python
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

filters = MetadataFilters(filters=[MetadataFilter(key="year", value=2012, operator=FilterOperator.GTE)])
index.as_retriever(filters=filters).retrieve("garbage collection")

tenant = TopKVectorStore(collection_name="support", partition="acme")  # isolated per tenant
strong = TopKVectorStore(collection_name="support", consistency="strong")
```

Every `FilterOperator` (`==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin`, `any`, `all`, `contains`, `text_match`, `text_match_insensitive`, `is_empty`) and every `FilterCondition` (`and`, `or`, `not`, nested) is translated into a TopK filter expression that runs in the same query as the search. As in LlamaIndex's in-memory store, `!=`, `nin` and `not` match nodes that don't have the key, and `is_empty` matches missing keys, `None`, `""` and `[]`. The keys `doc_id`, `ref_doc_id` and `document_id` filter on the source document.

Metadata values are stored for filtering as TopK values: dates become ISO 8601 strings, and values TopK can't filter on (for example lists that mix types) stay in the node but are left out of filters, with a warning.

See the [example notebook](docs/TopKDemo.ipynb) for every feature, with runnable examples.

## Notes

- A value in a `semantic_index()` text field can be at most 32,768 characters. Longer nodes are rejected before anything is written; chunk documents (for example with `SentenceSplitter(chunk_size=512)`, which also gives more precise retrieval) or create the store with `text_index="keyword"`.
- A query returns at most 10,000 results.
- Don't delete and immediately recreate a collection under the same name. For about a minute, writes to it can be lost.

## Development

```bash
export UV_LINK_MODE=copy  # nltk, used by llama-index-core, refuses hardlinked data files
uv sync --group dev
uv run make lint
uv run make test
TOPK_API_KEY=... uv run make integration_test  # live tests; they delete only collections they create
```

## Links

- TopK: https://topk.io
- Console (API keys): https://console.topk.io
- Docs: https://docs.topk.io
