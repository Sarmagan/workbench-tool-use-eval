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
- `--num_queries 5`
- `--print_first_prompt`
- `--load_in_4bit`
- `--load_in_8bit`

Results are written under `WorkBench/data/results/<domain>/`.

## Results Table

Supported tool selection modes are `all` and `domains`.

`Qwen2.5-7B` and `Qwen3-8B` (all tools) use `--tool_selection all`. The `Qwen3-8B` (domains) column uses `--tool_selection domains` (domain tools only). The `Qwen3-8B` (domains, 8-bit) column uses `--tool_selection domains --load_in_8bit`.


| Domain                          | Queries | `Qwen2.5-7B` | `Qwen3-8B` | `Qwen3-8B` (domains) | `Qwen3-8B` (domains, 8-bit) |
| ------------------------------- | ------- | ------------ | ---------- | -------------------- | --------------------------- |
| `calendar`                      | 110     | 24.55%       | 37.27%     | 47.27%               | 50.0%                       |
| `email`                         | 90      | 8.89%        | 23.33%     | 26.67%               | 24.44%                      |
| `analytics`                     | 120     | 12.5%        | 16.67%     | 16.67%               | 18.33%                      |
| `project_management`            | 80      | 32.5%        | 41.25%     | 37.5%                | 38.75%                      |
| `customer_relationship_manager` | 80      | 12.5%        | 6.25%      | 10.0%                | 8.75%                       |


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

Qwen 2.5 14B with 4-bit quantization:

```bash
python run_workbench_llama32_3b.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \
  --load_in_4bit \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains
```

