.PHONY: clean build docs help clean-logs gen-docs format test test-full sync-reqs-files
.DEFAULT_GOAL := help
PLATFORM ?= local

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


clean:  ## clean all build, python, and testing files
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

clean-logs: ## Clean logs
	rm -rf logs/**

gen-docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/cyto_dl*.rst
	rm -f docs/modules.rst
	sphinx-apidoc -Me -o docs/ cyto_dl **/tests/ cyto_dl/cli cyto_dl/config
	touch docs/*.rst
	$(MAKE) -C docs html

docs: ## generate Sphinx HTML documentation, including API docs, and serve to browser
	make gen-docs
	$(BROWSER) docs/_build/html/index.html

format: ## Run pre-commit hooks
	pre-commit run -a

test: ## Run not slow tests
	pytest -k "not slow"

test-full: ## Run all tests
	pytest

# `uv lock` should respect versions in an existing uv.lock file if they do not conflict
# with the pyproject.toml
uv.lock: pyproject.toml
	uv lock

# --no-emit-project is required here because the requirements.txt files have hashes
# and the cyto-dl source is a directory, which pip cannot hash.
requirements/requirements.txt: uv.lock
	mkdir -p requirements/
	uv export --no-emit-project -o $@

requirements/all-requirements.txt: uv.lock
	mkdir -p requirements/
	uv export --no-emit-project --all-extras -o $@

requirements/%-requirements.txt: uv.lock
	mkdir -p requirements/
	uv export --no-emit-project --extra $(subst -requirements.txt,,$(notdir $@)) -o $@

sync-reqs-files: requirements/requirements.txt \
                 requirements/torchserve-requirements.txt \
                 requirements/equiv-requirements.txt \
                 requirements/spharm-requirements.txt \
                 requirements/all-requirements.txt \
                 requirements/test-requirements.txt \
                 requirements/docs-requirements.txt
