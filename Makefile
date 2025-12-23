.PHONY: install corpus index compare test

install:
	python -m venv .venv && .venv/bin/pip install -e '.[all]'

corpus:
	./scripts/fetch_corpus.sh

index:
	python -m stack.retrieval

compare:
	python -m stack.compare

test:
	pytest -q
