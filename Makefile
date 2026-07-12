PYTHON ?= python3
PIP ?= pip
CONFIG_DIR := configs

.PHONY: help setup download-data preprocess partition validate train evaluate export-tflite fl-simulate docker-build docker-up docker-down docker-logs docker-bench figures test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "%-16s %s\n", $$1, $$2}'

setup: ## Install dependencies and show Python version
	$(PYTHON) --version
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

download-data: ## Placeholder dataset acquisition entrypoint
	$(PYTHON) -m src.data.download --config $(CONFIG_DIR)/paths.yaml

preprocess: ## Placeholder preprocessing entrypoint
	$(PYTHON) -m src.data.preprocessor --config $(CONFIG_DIR)/model.yaml --paths $(CONFIG_DIR)/paths.yaml

partition: ## Placeholder FL partitioning entrypoint
	$(PYTHON) -m src.data.partitioner --config $(CONFIG_DIR)/fl.yaml --paths $(CONFIG_DIR)/paths.yaml

validate: ## Placeholder data validation entrypoint
	$(PYTHON) -m src.data.validator --config $(CONFIG_DIR)/paths.yaml

train: ## Placeholder centralized training entrypoint
	$(PYTHON) -m src.models.train --config $(CONFIG_DIR)/model.yaml --paths $(CONFIG_DIR)/paths.yaml

evaluate: ## Placeholder evaluation entrypoint
	$(PYTHON) -m src.models.evaluate --config $(CONFIG_DIR)/model.yaml --paths $(CONFIG_DIR)/paths.yaml

export-tflite: ## Placeholder model export entrypoint
	$(PYTHON) -m src.models.export --config $(CONFIG_DIR)/model.yaml --paths $(CONFIG_DIR)/paths.yaml

fl-simulate: ## Placeholder Flower simulation entrypoint
	$(PYTHON) -m src.fl.simulate --config $(CONFIG_DIR)/fl.yaml --paths $(CONFIG_DIR)/paths.yaml

docker-build: ## Build server and client images
	docker compose -f docker/compose.yaml build

docker-up: ## Start FL server and edge clients
	docker compose -f docker/compose.yaml up --build

docker-down: ## Stop Docker services
	docker compose -f docker/compose.yaml down

docker-logs: ## Tail compose logs
	docker compose -f docker/compose.yaml logs -f

docker-bench: ## Placeholder Docker benchmark entrypoint
	$(PYTHON) -m src.evaluation.latency --config $(CONFIG_DIR)/docker.yaml --paths $(CONFIG_DIR)/paths.yaml

figures: ## Placeholder thesis figure generation entrypoint
	$(PYTHON) -m src.evaluation.plots --config $(CONFIG_DIR)/paths.yaml

test: ## Run the test suite
	pytest

clean: ## Remove Python cache files
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
