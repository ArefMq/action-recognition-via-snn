export PIPENV_VERBOSITY := -1
export PIPENV_QUIET := 1
export PYTHONPATH := `pwd`


PIPENV_RUN = PYTHONPATH=`pwd` PIPENV_VERBOSITY=-1 pipenv run
PYTEST_RUN = $(PIPENV_RUN) python -m pytest


.PHONY: check-lint
check-lint:  ## Checks linting issues
	@ echo "Checking linting issues"
	@ $(PIPENV_RUN) ruff check

.PHONY: lint
lint:
	@ echo "Fixing linting issues"
	@ $(PIPENV_RUN) ruff check --fix

.PHONY: format-check
format-check:
	@ echo "Checking formatting issues"
	@ $(PIPENV_RUN) ruff format --check

.PHONY: format
format:
	@ echo "Fixing formatting issues"
	@ $(PIPENV_RUN) ruff format

.PHONY: types
types:
	@ echo "Checking types"
	@ $(PIPENV_RUN) mypy --strict . --check

.PHONY: check
check: check-lint check-format types
	@ echo "Checked for formatting, linting and typing errors"

.PHONY: fix
fix: fix-lint fix-format types  ## Fixes all fixable linting, formatting, and typing issues
	@ echo "Fixed all fixable formatting and linting errors"

.PHONY: types
types:
	pipenv run mypy spikenet

.PHONY: fix
fix: format lint test types

.PHONY: test
test:
	pipenv run pytest tests

.PHONY: test-coverage
test-coverage:
	pipenv run pytest --cov=drivers tests

.PHONY: clean
clean:
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf .coverage
	@rm -rf .ruff_cache
	@rm -rf .ruff
	@rm -rf .ruff.lock
	@rm -rf .ruff.toml
	@rm -rf .ipynb_checkpoints
	@rm -rf ~data
