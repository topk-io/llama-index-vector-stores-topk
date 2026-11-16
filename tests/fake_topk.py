"""In-memory stand-in for `topk_sdk.Client` that records every call."""

from typing import Any, Dict, List, Optional, Tuple

from topk_sdk.error import CollectionNotFoundError, PartitionNotFoundError


class FakeCollection:
    def __init__(self, owner: "FakeClient", name: str, partition: Optional[str]):
        self.owner = owner
        self.name = name
        self.partition = partition

    def _key(self) -> Tuple[str, Optional[str]]:
        return (self.name, self.partition)

    def _store(self) -> Dict[str, Dict[str, Any]]:
        if self._key() not in self.owner.data:
            raise PartitionNotFoundError("partition not found")
        return self.owner.data[self._key()]

    def upsert(self, docs: List[Dict[str, Any]]) -> str:
        self.owner.calls.append(("upsert", self._key(), docs))
        store = self.owner.data.setdefault(self._key(), {})
        for d in docs:
            store[d["_id"]] = dict(d)
        self.owner.lsn += 1
        return str(self.owner.lsn)

    def delete(self, target: Any) -> str:
        self.owner.calls.append(("delete", self._key(), target))
        self._store()
        self.owner.lsn += 1
        return str(self.owner.lsn)

    def get(self, ids: List[str], fields: Optional[List[str]] = None, **opts: Any):
        self.owner.calls.append(("get", self._key(), ids, fields, opts))
        store = self._store()
        return {
            i: {k: v for k, v in store[i].items() if fields is None or k in fields}
            for i in ids
            if i in store
        }

    def count(self, **opts: Any) -> int:
        self.owner.calls.append(("count", self._key(), opts))
        return len(self._store())

    def query(self, q: Any, **opts: Any) -> List[Dict[str, Any]]:
        self.owner.calls.append(("query", self._key(), q, opts))
        store = self._store()
        if self.owner.query_results is not None:
            return [dict(r) for r in self.owner.query_results]
        keep = ("_id", "text", "node_content", "node_type")
        return [
            {**{k: v for k, v in d.items() if k in keep}, "semantic_score": 1.0}
            for d in store.values()
        ]

    def delete_partition(self, name: str) -> None:
        self.owner.calls.append(("delete_partition", self.name, name))
        self.owner.data.pop((self.name, name), None)

    def list_partitions(self) -> List[Any]:
        class P:
            def __init__(self, name: str) -> None:
                self.name = name

        return [P(p) for (c, p) in self.owner.data if c == self.name and p]


class FakeAsyncCollection:
    def __init__(self, sync: FakeCollection) -> None:
        self.sync = sync

    async def upsert(self, docs: List[Dict[str, Any]]) -> str:
        return self.sync.upsert(docs)

    async def delete(self, target: Any) -> str:
        return self.sync.delete(target)

    async def get(self, ids: List[str], fields: Optional[List[str]] = None, **o: Any):
        return self.sync.get(ids, fields, **o)

    async def query(self, q: Any, **opts: Any) -> List[Dict[str, Any]]:
        return self.sync.query(q, **opts)


class FakeCollectionInfo:
    def __init__(self, name: str, schema: Dict[str, Any]) -> None:
        self.name = name
        self.schema = schema


class FakeCollections:
    def __init__(self, owner: "FakeClient") -> None:
        self.owner = owner

    def get(self, name: str) -> FakeCollectionInfo:
        if name not in self.owner.schemas:
            raise CollectionNotFoundError("collection not found")
        return FakeCollectionInfo(name, self.owner.schemas[name])

    def create(self, name: str, schema: Dict[str, Any]) -> FakeCollectionInfo:
        self.owner.calls.append(("create", name, schema))
        self.owner.schemas[name] = schema
        self.owner.data.setdefault((name, None), {})
        return FakeCollectionInfo(name, schema)

    def delete(self, name: str) -> None:
        self.owner.calls.append(("delete_collection", name))
        self.owner.schemas.pop(name, None)


class FakeClient:
    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.schemas: Dict[str, Any] = {}
        self.data: Dict[Tuple[str, Optional[str]], Dict[str, Dict[str, Any]]] = {}
        self.query_results: Optional[List[Dict[str, Any]]] = None
        self.lsn = 0

    def collection(self, name: str, partition: Optional[str] = None) -> FakeCollection:
        return FakeCollection(self, name, partition)

    def collections(self) -> FakeCollections:
        return FakeCollections(self)

    def last(self, kind: str) -> tuple:
        return next(c for c in reversed(self.calls) if c[0] == kind)

    def all(self, kind: str) -> List[tuple]:
        return [c for c in self.calls if c[0] == kind]


class FakeAsyncClient:
    def __init__(self, sync: FakeClient) -> None:
        self.sync = sync

    def collection(
        self, name: str, partition: Optional[str] = None
    ) -> FakeAsyncCollection:
        return FakeAsyncCollection(self.sync.collection(name, partition))
