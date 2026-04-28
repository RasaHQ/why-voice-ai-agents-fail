# ── Terminal colours ───────────────────────────────────────────────────────────
GREEN  := $(shell tput -Txterm setaf 2 2>/dev/null)
YELLOW := $(shell tput -Txterm setaf 3 2>/dev/null)
WHITE  := $(shell tput -Txterm setaf 7 2>/dev/null)
RESET  := $(shell tput -Txterm sgr0 2>/dev/null)
BLUE   := $(shell tput -Txterm setaf 4 2>/dev/null)
RED    := $(shell tput -Txterm setaf 1 2>/dev/null)

# ── Project settings ───────────────────────────────────────────────────────────
PYTHON_VERSION := 3.13
VENV_NAME      := .venv
PROJECT_NAME   := why-voice-ai-agents-fail
REPO_ROOT      := $(shell pwd)
PYTHON         := $(REPO_ROOT)/$(VENV_NAME)/bin/python
SCRIPTS_DIR    := scripts

# Resolve a python binary even before the venv exists (used in `structure` etc.)
PYTHON_BIN     := $(shell \
  if [ -x "$(PYTHON)" ]; then echo "$(PYTHON)"; \
  elif command -v python3 >/dev/null 2>&1; then echo python3; \
  elif command -v python  >/dev/null 2>&1; then echo python; \
  else echo ""; fi)

.DEFAULT_GOAL := help

# ── Help ───────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@echo ''
	@echo '${YELLOW}Why (Voice) AI Agents Fail — companion code${RESET}'
	@echo '${WHITE}Four runnable scripts demonstrating production voice-agent failures.${RESET}'
	@echo ''
	@echo '${YELLOW}Setup:${RESET}'
	@echo '  ${GREEN}make env${RESET}                Create virtual environment (uv)'
	@echo '  ${GREEN}make install${RESET}            Install runtime + dev dependencies'
	@echo '  ${GREEN}make check-env${RESET}          Verify Python + uv + API keys'
	@echo ''
	@echo '${YELLOW}Run the failure-mode scripts:${RESET}'
	@echo '  ${GREEN}make script-01${RESET}          Failure 1: Turn-taking is harder than you think'
	@echo '  ${GREEN}make script-02${RESET}          Failure 2: Backchannels vs interrupts'
	@echo '  ${GREEN}make script-03${RESET}          Failure 3: The system disagrees with itself'
	@echo '  ${GREEN}make script-04${RESET}          Failure 4: The bot says goodbye too soon'
	@echo '  ${GREEN}make scripts${RESET}            Run all four scripts in sequence'
	@echo ''
	@echo '${YELLOW}API key checks:${RESET}'
	@echo '  ${GREEN}make deepgram-check${RESET}     Verify DEEPGRAM_API_KEY works'
	@echo '  ${GREEN}make rime-check${RESET}         Verify RIME_API_KEY works'
	@echo '  ${GREEN}make openai-check${RESET}       Verify OPENAI_API_KEY works'
	@echo '  ${GREEN}make keys-check${RESET}         Verify all three credentials'
	@echo ''
	@echo '${YELLOW}Quality:${RESET}'
	@echo '  ${GREEN}make format${RESET}             Auto-format with ruff'
	@echo '  ${GREEN}make lint${RESET}               Lint with ruff + mypy'
	@echo '  ${GREEN}make test${RESET}               Run tests'
	@echo '  ${GREEN}make check${RESET}              format + lint + test'
	@echo ''
	@echo '${YELLOW}Available targets:${RESET}'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*?## / \
	  {printf "  ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Environment ────────────────────────────────────────────────────────────────
.PHONY: env
env: ## Create virtual environment with uv
	@command -v uv >/dev/null 2>&1 || { \
	  echo "${RED}✗ uv is not installed.${RESET}"; \
	  echo "  Install it: ${BLUE}curl -LsSf https://astral.sh/uv/install.sh | sh${RESET}"; \
	  exit 1; \
	}
	@echo "${BLUE}→ Creating venv with Python $(PYTHON_VERSION)...${RESET}"
	@uv venv --python $(PYTHON_VERSION) $(VENV_NAME)
	@echo "${GREEN}✓ Virtual env ready at $(VENV_NAME)/${RESET}"
	@echo "  Activate manually with: ${BLUE}source $(VENV_NAME)/bin/activate${RESET}"

.PHONY: install
install: ## Install runtime + dev dependencies via uv
	@if [ ! -x "$(PYTHON)" ]; then $(MAKE) --no-print-directory env; fi
	@echo "${BLUE}→ Installing dependencies with uv...${RESET}"
	@uv pip install -e ".[dev]" --python $(PYTHON)
	@echo "${GREEN}✓ Dependencies installed${RESET}"

.PHONY: lock
lock: ## Generate uv lockfile (uv.lock)
	@echo "${BLUE}→ Locking dependencies...${RESET}"
	@uv lock
	@echo "${GREEN}✓ uv.lock written${RESET}"

.PHONY: sync
sync: ## Sync env to match uv.lock exactly
	@if [ ! -f uv.lock ]; then \
	  echo "${YELLOW}⚠ No uv.lock — running 'make lock' first${RESET}"; \
	  $(MAKE) --no-print-directory lock; \
	fi
	@echo "${BLUE}→ Syncing env from lockfile...${RESET}"
	@uv sync --extra dev
	@echo "${GREEN}✓ Env synced${RESET}"

.PHONY: check-env
check-env: ## Verify installation + tools
	@echo "${BLUE}→ Environment check${RESET}"
	@echo ""
	@printf "  Python: "
	@$(PYTHON) --version 2>/dev/null && echo "${GREEN}✓${RESET}" || echo "${RED}✗ venv not created — run: make install${RESET}"
	@printf "  uv:     "
	@uv --version 2>/dev/null && echo "${GREEN}✓${RESET}" || echo "${RED}✗ not installed${RESET}"
	@printf "  deepgram-sdk: "
	@$(PYTHON) -c "import deepgram; print(deepgram.__version__)" 2>/dev/null \
	  || echo "${RED}✗ not installed — run: make install${RESET}"
	@printf "  openai: "
	@$(PYTHON) -c "import openai; print(openai.__version__)" 2>/dev/null \
	  || echo "${RED}✗ not installed${RESET}"
	@printf "  requests: "
	@$(PYTHON) -c "import requests; print(requests.__version__)" 2>/dev/null \
	  || echo "${RED}✗ not installed${RESET}"
	@echo ""
	@$(MAKE) --no-print-directory keys-check

# ── API key verification ──────────────────────────────────────────────────────
.PHONY: deepgram-check
deepgram-check: ## Verify DEEPGRAM_API_KEY is valid
	@$(PYTHON) -c "from scripts._utils import check_deepgram; check_deepgram()" \
	  || (echo "${RED}✗ Deepgram check failed. Set DEEPGRAM_API_KEY in .env${RESET}" && exit 1)

.PHONY: rime-check
rime-check: ## Verify RIME_API_KEY is valid
	@$(PYTHON) -c "from scripts._utils import check_rime; check_rime()" \
	  || (echo "${RED}✗ Rime check failed. Set RIME_API_KEY in .env${RESET}" && exit 1)

.PHONY: openai-check
openai-check: ## Verify OPENAI_API_KEY is valid
	@$(PYTHON) -c "from scripts._utils import check_openai; check_openai()" \
	  || (echo "${RED}✗ OpenAI check failed. Set OPENAI_API_KEY in .env${RESET}" && exit 1)

.PHONY: keys-check
keys-check: ## Verify all three API keys
	@echo "${BLUE}→ Checking API credentials${RESET}"
	@$(PYTHON) -c "from scripts._utils import check_all; check_all()"

# ── Run the scripts ────────────────────────────────────────────────────────────
.PHONY: script-01
script-01: ## Failure 01 — Turn-taking is harder than you think
	@echo "${YELLOW}━━━ Failure 01: Turn-taking ━━━${RESET}"
	@$(PYTHON) $(SCRIPTS_DIR)/01_turn_taking.py

.PHONY: script-02
script-02: ## Failure 02 — Backchannels vs interrupts
	@echo "${YELLOW}━━━ Failure 02: Backchannels vs interrupts ━━━${RESET}"
	@$(PYTHON) $(SCRIPTS_DIR)/02_backchannels_vs_interrupts.py

.PHONY: script-03
script-03: ## Failure 03 — The system disagrees with itself
	@echo "${YELLOW}━━━ Failure 03: Split state ━━━${RESET}"
	@$(PYTHON) $(SCRIPTS_DIR)/03_split_state.py

.PHONY: script-04
script-04: ## Failure 04 — The bot says goodbye too soon
	@echo "${YELLOW}━━━ Failure 04: Premature goodbye ━━━${RESET}"
	@$(PYTHON) $(SCRIPTS_DIR)/04_premature_goodbye.py

.PHONY: scripts
scripts: keys-check script-01 script-02 script-03 script-04 ## Run all four failure-mode scripts in sequence
	@echo ""
	@echo "${GREEN}✓ All four scripts completed${RESET}"

# ── Notebooks (jupytext sync) ──────────────────────────────────────────────────
.PHONY: notebooks
notebooks: ## Convert scripts to .ipynb (jupytext)
	@echo "${BLUE}→ Converting scripts to notebooks...${RESET}"
	@mkdir -p notebooks
	@for f in $(SCRIPTS_DIR)/0*.py; do \
	  out=notebooks/$$(basename $$f .py).ipynb; \
	  $(PYTHON) -m jupytext --to notebook -o $$out $$f; \
	done
	@echo "${GREEN}✓ Notebooks in notebooks/${RESET}"

# ── Quality ────────────────────────────────────────────────────────────────────
.PHONY: format
format: ## Format with ruff
	@echo "${BLUE}→ Formatting...${RESET}"
	@$(PYTHON) -m ruff format $(SCRIPTS_DIR)/ tests/ 2>/dev/null || $(PYTHON) -m ruff format $(SCRIPTS_DIR)/
	@$(PYTHON) -m ruff check $(SCRIPTS_DIR)/ tests/ --fix 2>/dev/null || $(PYTHON) -m ruff check $(SCRIPTS_DIR)/ --fix
	@echo "${GREEN}✓ Formatted${RESET}"

.PHONY: lint
lint: ## Lint with ruff + mypy
	@echo "${BLUE}→ Linting...${RESET}"
	@$(PYTHON) -m ruff check $(SCRIPTS_DIR)/
	@$(PYTHON) -m mypy $(SCRIPTS_DIR)/ || true
	@echo "${GREEN}✓ Lint complete${RESET}"

.PHONY: test
test: ## Run pytest with coverage
	@if [ ! -d tests ]; then \
	  echo "${YELLOW}⚠ No tests/ directory yet — skipping${RESET}"; \
	  exit 0; \
	fi
	@echo "${BLUE}→ Running tests...${RESET}"
	@$(PYTHON) -m pytest tests/ -v
	@echo "${GREEN}✓ Tests passed${RESET}"

.PHONY: check
check: format lint test ## Run all quality checks (format + lint + test)
	@echo "${GREEN}✓ All quality checks passed${RESET}"

# ── Housekeeping ───────────────────────────────────────────────────────────────
.PHONY: clean
clean: ## Remove build/test artefacts and caches
	@echo "${BLUE}→ Cleaning...${RESET}"
	@rm -rf build/ dist/ *.egg-info .coverage .mypy_cache .pytest_cache .ruff_cache htmlcov
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
	@echo "${GREEN}✓ Clean${RESET}"

.PHONY: clean-output
clean-output: ## Remove generated audio output
	@rm -rf audio_output/
	@echo "${GREEN}✓ audio_output/ removed${RESET}"

.PHONY: clean-all
clean-all: clean clean-output ## Clean everything except .venv
	@echo "${GREEN}✓ Everything cleaned (venv preserved)${RESET}"

.PHONY: nuke
nuke: clean-all ## Clean everything INCLUDING .venv (rare)
	@rm -rf $(VENV_NAME)/
	@echo "${GREEN}✓ Nuked${RESET}"

# Catch-all to silence "no rule to make target X" for stray words like WEEK=foo
%:
	@:

# ── Structure ──────────────────────────────────────────────────────────────────
.PHONY: structure
structure: ## Show project structure
	@echo "${YELLOW}Project: $(PROJECT_NAME)${RESET}"
	@echo "  Python: $$( [ -n "$(PYTHON_BIN)" ] && $(PYTHON_BIN) -V 2>/dev/null || echo 'n/a')"
	@echo "  Branch: $$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
	@echo "  Files:  $$(find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './audio_output/*' | wc -l | tr -d ' ')"
	@echo ""
	@echo "${YELLOW}Tree:${RESET}"
	@echo "${BLUE}"
	@IGNORE='\.git|\.venv|__pycache__|\.DS_Store|\.idea|\.pytest_cache|\.ruff_cache|\.mypy_cache|\.coverage|htmlcov|build|dist|audio_output|.*\.egg-info|notebooks'; \
	if command -v tree > /dev/null; then \
	  tree -a -I "$$IGNORE"; \
	else \
	  find . \
	    -not -path './.git/*' -not -path './.venv/*' \
	    -not -path './audio_output/*' -not -path './notebooks/*' \
	    -not -path './__pycache__/*' \
	    -not -path './.pytest_cache/*' -not -path './.ruff_cache/*' \
	    -not -path './.mypy_cache/*' \
	    | sort; \
	fi
	@echo "${RESET}"
