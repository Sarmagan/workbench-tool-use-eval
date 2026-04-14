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
- `--agent_baseline route_then_act`
- `--agent_baseline plan_then_act`
- `--agent_baseline self_reflection`
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
- `route_then_act`: first asks the model to choose a smaller subset of tools, prints the selected tools for debugging, then runs native tool calling with only that subset.
- `plan_then_act`: runs a separate planning pass with plain-text tool descriptions and no callable tools, then starts a fresh native tool-calling run using that plan as context.
- `self_reflection`: before each tool-calling turn, runs a short reflection pass over the current conversation and tool results, then uses that critique as context for the next action.
- `react`: uses a ReAct-style `Thought -> Action -> Observation` loop and executes one parsed JSON action per turn.

`Qwen2.5-7B` and `Qwen3-8B` (all tools) use `--tool_selection all`. The `Qwen3-8B` (domains) column uses `--tool_selection domains` (domain tools only). The `Qwen3-8B` (domains, 8-bit) column uses `--tool_selection domains --load_in_8bit`. The `Qwen3-8B` (domains, 8-bit, route_then_act) column uses `--tool_selection domains --load_in_8bit --agent_baseline route_then_act`. The `Qwen3-8B` (domains, 8-bit, plan_then_act) column uses `--tool_selection domains --load_in_8bit --agent_baseline plan_then_act`. The `Qwen3-8B` (domains, 8-bit, self_reflection) column uses `--tool_selection domains --load_in_8bit --agent_baseline self_reflection`. The `Qwen3-8B` (domains, 8-bit, ReAct) column uses `--tool_selection domains --load_in_8bit --agent_baseline react`.


| Domain                          | Queries | `Qwen2.5-7B` | `Qwen3-8B` | `Qwen3-8B` (domains) | `Qwen3-8B` (domains, 8-bit) | `Qwen3-8B` (domains, 8-bit, route_then_act) | `Qwen3-8B` (domains, 8-bit, plan_then_act) | `Qwen3-8B` (domains, 8-bit, self_reflection) | `Qwen3-8B` (domains, 8-bit, ReAct) |
| ------------------------------- | ------- | ------------ | ---------- | -------------------- | --------------------------- | ------------------------------------------- | ------------------------------------------ | -------------------------------------------- | ---------------------------------- |
| `calendar`                      | 110     | 24.55%       | 37.27%     | 47.27%               | 50.0%                       | 59.09%                                      | 62.73%                                     | 43.64%                                       | 70.91%                             |
| `email`                         | 90      | 8.89%        | 23.33%     | 26.67%               | 24.44%                      | 32.22%                                      | 44.44%                                     | 47.78%                                       | 56.67%                             |
| `analytics`                     | 120     | 12.5%        | 16.67%     | 16.67%               | 18.33%                      | 22.5%                                       | 23.33%                                     | 21.67%                                       | 14.17%                             |
| `project_management`            | 80      | 32.5%        | 41.25%     | 37.5%                | 38.75%                      | 53.75%                                      | 48.75%                                     | 50.0%                                        | 43.75%                             |
| `customer_relationship_manager` | 80      | 12.5%        | 6.25%      | 10.0%                | 8.75%                       | 10.0%                                       | 43.75%                                     | 41.25%                                       | 36.25%                             |


Note: the `plan_then_act` setup improved performance in every domain relative to the `Qwen3-8B` (domains, 8-bit) baseline: `calendar` (50.0% -> 62.73%), `email` (24.44% -> 44.44%), `analytics` (18.33% -> 23.33%), `project_management` (38.75% -> 48.75%), and `customer_relationship_manager` (8.75% -> 43.75%). The likely reason is that it separates planning from execution: the model first reasons over plain-text tool descriptions without having to emit calls immediately, then uses that plan during a fresh native tool-calling pass. That seems to reduce premature or unnecessary actions and helps more multi-step domains such as CRM and project management, where choosing the right sequence matters as much as choosing the right tool.

Note: the `route_then_act` setup improved performance in every domain relative to the `Qwen3-8B` (domains, 8-bit) baseline: `calendar` (50.0% -> 59.09%), `email` (24.44% -> 32.22%), `analytics` (18.33% -> 22.5%), `project_management` (38.75% -> 53.75%), and `customer_relationship_manager` (8.75% -> 10.0%). The likely mechanism is simpler tool routing: by first shrinking the candidate set, the acting model has fewer schemas to confuse and a smaller action space to search over. That appears especially helpful for `project_management`, where route-then-act beat both `plan_then_act` (48.75%) and ReAct (43.75%), but the lighter routing step seems less robust than full planning in domains that require more careful long-horizon reasoning.

Note: the `self_reflection` setup is now complete across all domains and produced a mixed but mostly positive result relative to the `Qwen3-8B` (domains, 8-bit) baseline: it improved `email` (24.44% -> 47.78%), `analytics` (18.33% -> 21.67%), `project_management` (38.75% -> 50.0%), and `customer_relationship_manager` (8.75% -> 41.25%), but regressed on `calendar` (50.0% -> 43.64%). The likely pattern is that the reflection pass helps when the model benefits from pausing to check whether it has enough evidence before acting, especially in multi-step domains like email, project management, and CRM. But the same extra caution can become over-conservative on lookup-first tasks: in weaker runs, the model often over-indexed on "missing information" and "write safety", asked for an event ID or end date instead of first using an available search tool, and then terminated early because a natural-language reply counts as the final action in this loop.

Note: the ReAct setup improved performance in most domains relative to the `Qwen3-8B` (domains, 8-bit) baseline, including `calendar` (50.0% -> 70.91%), `email` (24.44% -> 56.67%), `project_management` (38.75% -> 43.75%), and `customer_relationship_manager` (8.75% -> 36.25%). The likely reason is that ReAct exposes an explicit `Thought -> Action -> Observation` loop, which encourages the model to inspect intermediate results before committing to the next step. That seems particularly helpful for domains like `calendar` and `email`, where one lookup often resolves the key ambiguity for the next action. The lower `analytics` ReAct score appears to be partly caused by run failures rather than task performance alone: 16 out of 120 analytics tasks ended with `### Error: Agent stopped due to iteration limit.`, which likely explains the decrease.

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

Route-then-act baseline with domain tools:

```bash
python run_workbench_llama32_3b.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains \
  --agent_baseline route_then_act
```

Self-reflection baseline with domain tools:

```bash
python run_workbench_llama32_3b.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains \
  --agent_baseline self_reflection
```

Plan-then-act baseline with domain tools:

```bash
python run_workbench_llama32_3b.py \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains \
  --agent_baseline plan_then_act
```

Qwen 2.5 14B with 4-bit quantization:

```bash
python run_workbench_llama32_3b.py \
  --model_id Qwen/Qwen2.5-14B-Instruct \
  --load_in_4bit \
  --queries_path WorkBench/data/processed/queries_and_answers/email_queries_and_answers.csv \
  --tool_selection domains
```

