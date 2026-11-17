# uv hardlinks packages by default; nltk (used by llama-index-core) refuses
# multiply-linked data files, so install with copies.
export UV_LINK_MODE ?= copy

help:	## Show all Makefile targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format:	## Run code autoformatters (black, ruff).
	black llama_index tests
	ruff check --fix llama_index tests

lint:	## Run linters: black, ruff, codespell and mypy.
	black --check llama_index tests
	ruff check llama_index tests
	codespell llama_index tests README.md
	mypy llama_index

test:	## Run unit tests via pytest.
	pytest tests --ignore=tests/integration

integration_test:	## Run live tests against TopK (needs TOPK_API_KEY).
	pytest tests/integration -v

docs_test:	## Run every cell of the example notebook against TopK (needs TOPK_API_KEY).
	jupyter nbconvert --to notebook --execute --output-dir=/tmp docs/TopKDemo.ipynb
