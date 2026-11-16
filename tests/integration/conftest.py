"""Fixtures for live tests against TopK.

Requires `TOPK_API_KEY` (and optionally `TOPK_REGION`, default
`aws-us-east-1-elastica`). Every collection gets a unique
`llamaindex-topk-it-` name, and only collections created by this test run are
deleted afterwards.
"""

import os
import uuid
from typing import Callable, Iterator, List

import pytest
from topk_sdk import Client
from topk_sdk.error import CollectionNotFoundError

PREFIX = "llamaindex-topk-it-"


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    if os.environ.get("TOPK_API_KEY"):
        return
    skip = pytest.mark.skip(reason="TOPK_API_KEY is not set")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def region() -> str:
    return os.environ.get("TOPK_REGION", "aws-us-east-1-elastica")


@pytest.fixture(scope="session")
def topk_client(region: str) -> Client:
    return Client(api_key=os.environ["TOPK_API_KEY"], region=region)


@pytest.fixture(scope="session")
def new_collection_name(topk_client: Client) -> Iterator[Callable[[str], str]]:
    """Return a factory for unique collection names; delete them at the end."""
    run_id = uuid.uuid4().hex[:8]
    created: List[str] = []

    def factory(label: str) -> str:
        name = f"{PREFIX}{run_id}-{label}-{len(created)}"
        created.append(name)
        return name

    yield factory

    for name in created:
        try:
            topk_client.collections().delete(name)
        except CollectionNotFoundError:
            pass
