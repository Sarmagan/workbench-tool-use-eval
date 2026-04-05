# Hugging Face WorkBench Runs

This directory contains a standalone runner for evaluating local Hugging Face chat models on the WorkBench benchmark.

## Script

Run:

```bash
python run_workbench_hf.py \
  --queries_path WorkBench/data/processed/queries_and_answers/calendar_queries_and_answers.csv
```

Useful options:

- `--tool_selection all`
- `--tool_selection domains`
- `--num_queries 5`
- `--print_first_prompt`
- `--load_in_4bit`
- `--load_in_8bit`

Results are written under `WorkBench/data/results/<domain>/`.

## Results Table

`Qwen2.5-7B` and `Qwen3-8B` (all tools) use `--tool_selection all`. The `Qwen3-8B` (domains) column uses `--tool_selection domains` (domain tools only).


| Domain                          | Queries | `Qwen2.5-7B` | `Qwen3-8B` | `Qwen3-8B` (domains) |
| ------------------------------- | ------- | ------------ | ---------- | -------------------- |
| `calendar`                      | 110     | 24.55%       | 37.27%     | 47.27%               |
| `email`                         | 90      | 8.89%        | 23.33%     | 26.67%               |
| `analytics`                     | 120     | 12.5%        | 16.67%     | 16.67%               |
| `project_management`            | 80      | 32.5%        | 41.25%     | 37.5%                |
| `customer_relationship_manager` | 80      | 12.5%        | 6.25%      | 10.0%                |


## Example Commands

All tools:

```bash
python run_workbench_hf.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection all
```

Domain tools only:

```bash
python run_workbench_hf.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains
```

Qwen 2.5 14B with 4-bit quantization:

```bash
python run_workbench_hf.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \
  --load_in_4bit \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains
```

