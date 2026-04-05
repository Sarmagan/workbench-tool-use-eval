# WorkBench — OpenAI-only Fork

> **Forked from [olly-styles/WorkBench](https://github.com/olly-styles/WorkBench)**  
> Original paper: *WorkBench: A Benchmark Dataset for Agents in a Realistic Workplace Setting*

This fork removes all LangChain dependencies and uses the **OpenAI Chat Completions API** (with tool calling) directly.  
The default model is **`gpt-5-nano`**, but any model in `AVAILABLE_LLMS` can be used.

---

## Quick Start

### 1. Install `uv` (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the environment and install dependencies
```bash
cd WorkBench
uv sync
```

### 3. Add your OpenAI API key
```bash
echo "sk-..." > openai_key.txt
```

### 4. Run the benchmark

**Single domain (e.g. calendar):**
```bash
uv run python scripts/inference/generate_results.py \
  --model_name gpt-5-nano \
  --queries_path data/processed/queries_and_answers/calendar_queries_and_answers.csv
```

**All domains at once:**
```bash
for domain in calendar email analytics project_management customer_relationship_manager multi_domain; do
  uv run python scripts/inference/generate_results.py \
    --model_name gpt-5-nano \
    --queries_path data/processed/queries_and_answers/${domain}_queries_and_answers.csv
done
```

**Use only domain-relevant tools per query (faster & cheaper):**
```bash
uv run python scripts/inference/generate_results.py \
  --model_name gpt-5-nano \
  --queries_path data/processed/queries_and_answers/calendar_queries_and_answers.csv \
  --tool_selection domains
```

---

## Available Models

Set `--model_name` to any of:

| Name | OpenAI model string |
|------|-------------------|
| `gpt-5-nano` | gpt-5-nano |

To add a new model, just add its name to `AVAILABLE_LLMS` in `src/evals/utils.py` — no other changes needed.

---

## Results

Results are saved to `data/results/<domain>/` as timestamped CSV files.  
Metrics are printed to stdout at the end of each run.

The gpt-5-nano results are as follows:

| | Analytics | Calendar | CRM | Email | Project Management | Multi Domain |
|---|---|---|---|---|---|---|
| Number of tasks | 120 | 110 | 80 | 90 | 80 | 210 |
| Accuracy  | 45.83% | 86.36% | 68.75% | 82.22% | 71.25% | 45.24% |

---

## Changes from upstream

| File | What changed |
|------|-------------|
| `src/tools/*.py` | Removed `@tool` LangChain decorator; added `.openai_schema`, `.name`, `.func` attributes to each function |
| `src/tools/toolkits.py` | Removed LangChain imports; toolkits are now plain Python lists |
| `src/evals/utils.py` | Replaced `initialize_agent` + LangChain LLMs with `openai.OpenAI` + a hand-rolled tool-calling loop |
| `scripts/inference/generate_results.py` | Removed LangChain warning suppression |
| `pyproject.toml` | New file — minimal deps for `uv` |

## Tool Selection Pipeline

Giving the agent only the tools relevant to a task improves performance by reducing noise in the prompt and limiting the action space. Two scripts support this:

**1. Analyze tool dependencies** — runs once to extract which tools are prerequisites for others (e.g. `calendar__search_events` must run before `calendar__delete_event` to obtain the `event_id`). Output is saved to `data/tool_dependencies.json`.
```bash
uv run python scripts/analyze_tool_dependencies.py
```

**2. Plan tools for a task** — given a task string, asks an LLM to select the relevant tools, then checks the selection against the dependency graph and adds any missing prerequisites.
```bash
uv run python scripts/plan_tools.py --task "Delete my first meeting on December 13"
```

Run step 1 once, then use step 2 at inference time to pass only the planned tools to the agent instead of the full toolkit.


Example:

Task: Move all of Dmitri's tasks that are in progress to in review

LLM selected (2):
  - project_management__search_tasks
  - project_management__update_task

Missing dependencies added (1):
  + company_directory__find_email_address

Final tool plan (3):
  ✓ project_management__search_tasks
  ✓ project_management__update_task
  ✓ company_directory__find_email_address
