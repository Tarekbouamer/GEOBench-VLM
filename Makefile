.PHONY: dev clean ruff

dev:
	pip install -e .

clean:
	pip uninstall -y geobench-vlm

ruff:
	ruff format .
	ruff check --select I --fix .
	ruff check --fix .
