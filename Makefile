.PHONY: install test bench run lint clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short --cov=src/gpu_memory_profiler --cov-report=term-missing

bench:
	python benchmarks/bench_core.py

run:
	gpu-profiler demo --iterations 20

run-leak:
	gpu-profiler demo --iterations 30 --leak --output output/

estimate:
	gpu-profiler estimate --params 7B --dtype fp16 --optimizer adam

lint:
	ruff check .
	mypy src/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf output/
