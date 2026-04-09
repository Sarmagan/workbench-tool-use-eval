# Hugging Face WorkBench Runs

This directory contains a standalone runner for evaluating local Hugging Face chat models on the WorkBench benchmark.

## Script

Run:

```bash
python run_workbench_llama32_3b.py \
  --queries_path WorkBench/data/processed/queries_and_answers/calendar_queries_and_answers.csv
```

Useful options:

- `--tool_selection all`
- `--tool_selection domains`
- `--agent_baseline tool_calling`
- `--agent_baseline react`
- `--num_queries 5`
- `--print_first_prompt`
- `--load_in_4bit`
- `--load_in_8bit`

Results are written under `WorkBench/data/results/<domain>/`.

## Results Table

Supported tool selection modes are `all` and `domains`.

Agent baselines:

- `tool_calling`: uses the chat template tool-calling path directly.
- `react`: uses a ReAct-style `Thought -> Action -> Observation` loop and executes one parsed JSON action per turn.

`Qwen2.5-7B` and `Qwen3-8B` (all tools) use `--tool_selection all`. The `Qwen3-8B` (domains) column uses `--tool_selection domains` (domain tools only). The `Qwen3-8B` (domains, 8-bit) column uses `--tool_selection domains --load_in_8bit`. The `Qwen3-8B` (domains, 8-bit, ReAct) column uses `--tool_selection domains --load_in_8bit --agent_baseline react`.


| Domain                          | Queries | `Qwen2.5-7B` | `Qwen3-8B` | `Qwen3-8B` (domains) | `Qwen3-8B` (domains, 8-bit) | `Qwen3-8B` (domains, 8-bit, ReAct) |
| ------------------------------- | ------- | ------------ | ---------- | -------------------- | --------------------------- | ---------------------------------- |
| `calendar`                      | 110     | 24.55%       | 37.27%     | 47.27%               | 50.0%                       | 70.91%                             |
| `email`                         | 90      | 8.89%        | 23.33%     | 26.67%               | 24.44%                      | 56.67%                             |
| `analytics`                     | 120     | 12.5%        | 16.67%     | 16.67%               | 18.33%                      | 14.17%                             |
| `project_management`            | 80      | 32.5%        | 41.25%     | 37.5%                | 38.75%                      | 43.75%                             |
| `customer_relationship_manager` | 80      | 12.5%        | 6.25%      | 10.0%                | 8.75%                       | 36.25%                             |

Note: the ReAct setup improved performance in most domains relative to the `Qwen3-8B` (domains, 8-bit) baseline, including `calendar` (50.0% -> 70.91%), `email` (24.44% -> 56.67%), `project_management` (38.75% -> 43.75%), and `customer_relationship_manager` (8.75% -> 36.25%). The lower `analytics` ReAct score appears to be partly caused by run failures rather than task performance alone: 16 out of 120 analytics tasks ended with `### Error: Agent stopped due to iteration limit.`, which likely explains the decrease.


## Example Commands

All tools:

```bash
python run_workbench_llama32_3b.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection all
```

Domain tools only:

```bash
python run_workbench_llama32_3b.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains
```

ReAct baseline with domain tools:

```bash
python run_workbench_llama32_3b.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains \
  --agent_baseline react
```

Qwen 2.5 14B with 4-bit quantization:

```bash
python run_workbench_llama32_3b.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \
  --load_in_4bit \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains
```

