.PHONY: dev clean ruff

dev:
	pip install -e .

clean:
	pip uninstall -y geobench-vlm
	rm -rf dist build geobench_vlm.egg-info

ruff:
	ruff format .
	ruff check --select I --fix .
	ruff check --fix .
