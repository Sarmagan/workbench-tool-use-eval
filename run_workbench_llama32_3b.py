#!/usr/bin/env python3
from __future__ import annotations
import argparse
import ast
import csv
import json
import os
import re
import sys
from typing import Any
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from few_shot_utils import (
    build_few_shot_messages_for_query,
    format_few_shot_messages,
    load_few_shot_examples_by_domain,
    normalize_domain_name,
)
from memory_utils import (
    EpisodicMemory,
    build_memory_messages_for_query,
    format_memory_messages,
)


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKBENCH_ROOT = os.path.join(PROJECT_ROOT, "WorkBench")
sys.path.append(WORKBENCH_ROOT)
os.chdir(WORKBENCH_ROOT)

from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
from src.tools import (  # noqa: E402
    analytics,
    calendar,
    company_directory,
    customer_relationship_manager,
    email,
    project_management,
)
from src.tools.toolkits import (  # noqa: E402
    analytics_toolkit,
    calendar_toolkit,
    company_directory_toolkit,
    customer_relationship_manager_toolkit,
    email_toolkit,
    project_management_toolkit,
    tools_with_side_effects,
)


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_FEW_SHOT_K = 3
DEFAULT_TOT_NUM_THOUGHTS = 3
DEFAULT_MEMORY_K = 3
DEFAULT_MEMORY_MIN_EPISODES = 1
# Reflexion (Shinn et al., 2023) repeats the task across several trials and
# keeps a small sliding window of verbal self-reflections written between
# trials. The paper uses 1-3 reflections in the buffer and up to a small
# number of trials per task; we mirror that with cost-aware defaults.
DEFAULT_REFLEXION_MAX_TRIALS = 3
DEFAULT_REFLEXION_MEMORY_SIZE = 3
# The ReWOO Planner has to emit the entire multi-step plan in a single
# generation call. The default per-turn token budget is enough for one tool
# call but typically truncates a 3-4 step plan with brief reasoning lines.
REWOO_PLANNER_MAX_NEW_TOKENS = 512
DEFAULT_QUERIES_PATH = os.path.join(
    WORKBENCH_ROOT,
    "data",
    "processed",
    "queries_and_answers",
    "calendar_queries_and_answers.csv",
)
AGENT_BASELINE_CHOICES = (
    "tool_calling",
    "chain_of_thought",
    "few_shot",
    "memory",
    "tree_of_thoughts",
    "react",
    "plan_then_act",
    "route_then_act",
    "self_reflection",
    "rewoo",
    "reflexion",
)

DOMAINS = [calendar, email, analytics, project_management, customer_relationship_manager]

_TOOL_REGISTRY: dict[str, Any] = {
    fn.openai_schema["function"]["name"]: fn
    for toolkit in [
        calendar_toolkit,
        email_toolkit,
        analytics_toolkit,
        project_management_toolkit,
        customer_relationship_manager_toolkit,
        company_directory_toolkit,
    ]
    for fn in toolkit
}


def _openai_name_to_internal(openai_name: str) -> str:
    return openai_name.replace("__", ".", 1)


def convert_tool_call_to_function_call(tool_name: str, tool_args: dict[str, Any]) -> str:
    internal_name = _openai_name_to_internal(tool_name)
    args = ", ".join(f'{key}="{value}"' for key, value in tool_args.items())
    return f"{internal_name}.func({args})"


def normalize_domain_name(domain: str) -> str:
    return {
        "crm": "customer_relationship_manager",
    }.get(domain, domain)


def parse_domains_field(domains_value: Any) -> list[str]:
    if not isinstance(domains_value, str):
        return []
    cleaned = domains_value.strip("][")
    if not cleaned:
        return []
    return [
        normalize_domain_name(domain)
        for domain in cleaned.replace("'", "").split(", ")
        if domain
    ]


def get_toolkits(toolkits: list[str]) -> list[Any]:
    tools: list[Any] = []
    if "email" in toolkits:
        tools += email_toolkit
    if "calendar" in toolkits:
        tools += calendar_toolkit
    if "analytics" in toolkits:
        tools += analytics_toolkit
    if "project_management" in toolkits:
        tools += project_management_toolkit
    if "customer_relationship_manager" in toolkits:
        tools += customer_relationship_manager_toolkit
    tools += company_directory_toolkit
    return tools


def execute_actions_and_reset_state(actions: list[str]) -> tuple[bool, ...]:
    for domain in DOMAINS:
        domain.reset_state()

    for action in actions:
        try:
            eval(action)
        except Exception:
            continue

    new_calendar_state = calendar.CALENDAR_EVENTS.copy()
    new_email_state = email.EMAILS.copy()
    new_analytics_state = analytics.PLOTS_DATA.copy()
    new_project_management_state = project_management.PROJECT_TASKS.copy()
    new_customer_relationship_manager_state = customer_relationship_manager.CRM_DATA.copy()

    for domain in DOMAINS:
        domain.reset_state()

    return (
        True,
        new_calendar_state,
        new_email_state,
        new_analytics_state,
        new_project_management_state,
        new_customer_relationship_manager_state,
    )


def end_date_minor_error(ground_truth: list[str], prediction: list[str]) -> bool:
    matches = 0
    for func in ground_truth:
        if "2023-11-29" in func and func.replace("2023-11-29", "2023-11-30") in prediction:
            matches += 1
    if len(ground_truth) == 0:
        return False
    return matches == len(ground_truth)


def meeting_start_time_error(ground_truth: list[str], prediction: list[str]) -> bool:
    matches = 0
    next_free_time_ground_truth = "13:00:00"
    common_error_times = ["09:00:00", "11:00:00", "15:00:00", "15:30:00"]
    for func in ground_truth:
        if next_free_time_ground_truth in func:
            for time in common_error_times:
                if func.replace(next_free_time_ground_truth, time) in prediction:
                    matches += 1
                    break
    if len(ground_truth) == 0:
        return False
    return matches == len(ground_truth)


def get_function_name(action: str) -> str:
    return ".".join(action.split("(")[0].split(".")[0:2])


def is_exact_match(predicted_actions: list[str], ground_truth_actions: list[str]) -> bool:
    tools_with_side_effects_names = [fn.name for fn in tools_with_side_effects]
    predicted_actions_with_side_effects = [
        action for action in predicted_actions if get_function_name(action) in tools_with_side_effects_names
    ]
    predicted_actions_with_side_effects = sorted(action.lower() for action in predicted_actions_with_side_effects)
    ground_truth_actions = sorted(action.lower() for action in ground_truth_actions)
    return predicted_actions_with_side_effects == ground_truth_actions


def is_correct(predicted_actions: list[str], ground_truth_actions: list[str], error: str) -> bool:
    if error:
        return False

    (
        successful_execution,
        predicted_calendar_state,
        predicted_email_state,
        predicted_analytics_state,
        predicted_project_management_state,
        predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)
    (
        _,
        ground_truth_calendar_state,
        ground_truth_email_state,
        ground_truth_analytics_state,
        ground_truth_project_management_state,
        ground_truth_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(ground_truth_actions)

    def convert_strs_to_lowercase(df: pd.DataFrame) -> pd.DataFrame:
        fields_not_to_convert = ["status", "list_name", "board"]
        for col in df.columns:
            if col not in fields_not_to_convert:
                df[col] = df[col].str.lower()
        return df

    predicted_calendar_state = convert_strs_to_lowercase(predicted_calendar_state)
    predicted_email_state = convert_strs_to_lowercase(predicted_email_state)
    predicted_analytics_state = convert_strs_to_lowercase(predicted_analytics_state)
    predicted_project_management_state = convert_strs_to_lowercase(predicted_project_management_state)
    predicted_customer_relationship_manager_state = convert_strs_to_lowercase(
        predicted_customer_relationship_manager_state
    )
    ground_truth_calendar_state = convert_strs_to_lowercase(ground_truth_calendar_state)
    ground_truth_email_state = convert_strs_to_lowercase(ground_truth_email_state)
    ground_truth_analytics_state = convert_strs_to_lowercase(ground_truth_analytics_state)
    ground_truth_project_management_state = convert_strs_to_lowercase(ground_truth_project_management_state)
    ground_truth_customer_relationship_manager_state = convert_strs_to_lowercase(
        ground_truth_customer_relationship_manager_state
    )

    return (
        successful_execution
        and predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(
            ground_truth_customer_relationship_manager_state
        )
    )


def has_side_effects(predicted_actions: list[str], ground_truth_actions: list[str]) -> bool:
    for domain in DOMAINS:
        domain.reset_state()
    original_state = {
        "calendar": calendar.CALENDAR_EVENTS.copy(),
        "email": email.EMAILS.copy(),
        "analytics": analytics.PLOTS_DATA.copy(),
        "project_management": project_management.PROJECT_TASKS.copy(),
        "customer_relationship_manager": customer_relationship_manager.CRM_DATA.copy(),
    }
    (
        _,
        predicted_calendar_state,
        predicted_email_state,
        predicted_analytics_state,
        predicted_project_management_state,
        predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)

    state_changed = not predicted_calendar_state.equals(original_state["calendar"])
    state_changed |= not predicted_email_state.equals(original_state["email"])
    state_changed |= not predicted_analytics_state.equals(original_state["analytics"])
    state_changed |= not predicted_project_management_state.equals(original_state["project_management"])
    state_changed |= not predicted_customer_relationship_manager_state.equals(
        original_state["customer_relationship_manager"]
    )

    correct = is_correct(predicted_actions, ground_truth_actions, "")
    return state_changed and not correct


def calculate_metrics(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    print_errors: bool = True,
) -> pd.DataFrame:
    predictions = predictions_df.rename(columns={"function_calls": "prediction"})
    predictions = predictions.fillna("")
    ground_truth = ground_truth_df.rename(columns={"answer": "ground_truth"})
    df = predictions.merge(ground_truth, on="query")
    assert len(predictions) == len(ground_truth) == len(df), (
        f"{len(predictions)} predictions does not match {len(ground_truth_df)} ground truth answers."
    )

    df["prediction"] = df["prediction"].apply(lambda actions: [action.replace("\n", "\\n") for action in actions])
    df["ground_truth"] = df["ground_truth"].apply(lambda actions: [action.replace("\n", "\\n") for action in actions])

    df["exact_match"] = [is_exact_match(pred, gt) for pred, gt in zip(df["prediction"], df["ground_truth"])]
    df["correct"] = [is_correct(pred, gt, error) for pred, gt, error in zip(df["prediction"], df["ground_truth"], df["error"])]
    df["unwanted_side_effects"] = [has_side_effects(pred, gt) for pred, gt in zip(df["prediction"], df["ground_truth"])]
    df["no_actions"] = [not len(pred) for pred in df["prediction"]]
    df["wrong_email"] = [("@example" in str(pred)) and ("@atlas" not in str(pred)) for pred in df["prediction"]]
    df["wrong_email"] = df["wrong_email"] & ~df["correct"]
    df["end_date_minor_error"] = [end_date_minor_error(gt, pred) for gt, pred in zip(df["ground_truth"], df["prediction"])]
    df["end_date_minor_error"] = df["end_date_minor_error"] & ~df["correct"]
    df["meeting_start_time_error"] = [
        meeting_start_time_error(gt, pred) for gt, pred in zip(df["ground_truth"], df["prediction"])
    ]
    df["meeting_start_time_error"] = df["meeting_start_time_error"] & ~df["correct"]

    if print_errors:
        print("--------------------------------------------")
        print("ERRORS without unwanted side effects:")
        print("--------------------------------------------")
        for _, row in df[~df["correct"] & ~df["unwanted_side_effects"]].iterrows():
            if not row["wrong_email"] and not row["no_actions"] and not row["end_date_minor_error"] and not row["meeting_start_time_error"]:
                print("--------------------------------------------")
                print(f"Query:\n    {row['query']}")
                print("\nPrediction:")
                for action in row["prediction"]:
                    print(f"    {action}")
                print("\nGround truth:")
                for action in row["ground_truth"]:
                    print(f"    {action}")
                print(f"\nError: {row['error']}\n")

        print("--------------------------------------------")
        print("ERRORS with unwanted side effects:")
        print("--------------------------------------------")
        for _, row in df[~df["correct"] & df["unwanted_side_effects"]].iterrows():
            if not row["wrong_email"] and not row["no_actions"] and not row["end_date_minor_error"] and not row["meeting_start_time_error"]:
                print("--------------------------------------------")
                print(f"Query:\n    {row['query']}")
                print("\nPrediction:")
                for action in row["prediction"]:
                    print(f"    {action}")
                print("\nGround truth:")
                for action in row["ground_truth"]:
                    print(f"    {action}")
                print(f"\nError: {row['error']}\n")

    num_errors_without_side_effects = len(df[(~df["correct"]) & ~df["unwanted_side_effects"]])
    num_errors_with_side_effects = len(df[(~df["correct"]) & df["unwanted_side_effects"]])
    print(f"Accuracy: {round(df['correct'].mean() * 100, 2)}% ({df['correct'].sum()} out of {len(df)})")
    print(
        "Errors without unwanted side effects: "
        f"{round(num_errors_without_side_effects / len(df) * 100, 2)}% "
        f"({num_errors_without_side_effects} out of {len(df)})"
    )
    print(
        "Errors with unwanted side effects: "
        f"{round(num_errors_with_side_effects / len(df) * 100, 2)}% "
        f"({num_errors_with_side_effects} out of {len(df)})"
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WorkBench with a local Hugging Face chat model."
    )
    parser.add_argument(
        "--model_id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--model_label",
        default=None,
        help="Optional short label used in the output filename. Defaults to a sanitized model ID.",
    )
    parser.add_argument(
        "--queries_path",
        default=DEFAULT_QUERIES_PATH,
        help="Path to a WorkBench queries_and_answers CSV file.",
    )
    parser.add_argument(
        "--tool_selection",
        choices=["all", "domains"],
        default="all",
        help="Use all tools or only domain-relevant tools from the CSV.",
    )
    parser.add_argument(
        "--agent_baseline",
        choices=list(AGENT_BASELINE_CHOICES),
        default="tool_calling",
        help="Agent loop to use: native tool calling, a chain-of-thought variant, a few-shot prompting variant, an episodic-memory retrieval variant, a Tree-of-Thoughts variant, a route-then-act variant, a plan-then-act variant, a self-reflection variant, ReAct-style text actions, ReWOO (Reasoning WithOut Observation) with a Planner/Worker/Solver split, or Reflexion (verbal reinforcement learning across trials).",
    )
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=DEFAULT_FEW_SHOT_K,
        help="Number of fixed domain examples to prepend when --agent_baseline few_shot.",
    )
    parser.add_argument(
        "--memory_k",
        type=int,
        default=DEFAULT_MEMORY_K,
        help="Number of past episodes to retrieve as demonstrations when --agent_baseline memory.",
    )
    parser.add_argument(
        "--memory_path",
        default=None,
        help=(
            "Optional JSON file used to persist episodic memory across runs when "
            "--agent_baseline memory. If unset, memory is kept in-process for the "
            "current run only."
        ),
    )
    parser.add_argument(
        "--memory_min_episodes",
        type=int,
        default=DEFAULT_MEMORY_MIN_EPISODES,
        help=(
            "Cold-start guard for --agent_baseline memory: do not retrieve any "
            "past episodes until the verified-correct store contains at least this "
            "many entries. Set to 0 to retrieve from the very first query. "
            f"Default: {DEFAULT_MEMORY_MIN_EPISODES}."
        ),
    )
    parser.add_argument(
        "--tot_num_thoughts",
        type=int,
        default=DEFAULT_TOT_NUM_THOUGHTS,
        help="Number of candidate next steps to generate per Tree-of-Thoughts expansion.",
    )
    parser.add_argument(
        "--reflexion_max_trials",
        type=int,
        default=DEFAULT_REFLEXION_MAX_TRIALS,
        help=(
            "Maximum number of Actor trials per query when --agent_baseline reflexion. "
            "The first trial runs without reflections; subsequent trials see the verbal "
            "reflections written by the Self-Reflection module after each failed trial. "
            f"Default: {DEFAULT_REFLEXION_MAX_TRIALS}."
        ),
    )
    parser.add_argument(
        "--reflexion_memory_size",
        type=int,
        default=DEFAULT_REFLEXION_MEMORY_SIZE,
        help=(
            "Maximum number of past reflections kept in the Reflexion memory buffer "
            "and shown to the Actor at the start of each new trial (sliding window). "
            f"Default: {DEFAULT_REFLEXION_MEMORY_SIZE}."
        ),
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=20,
        help="Maximum number of assistant/tool turns per query.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate at each turn.",
    )
    parser.add_argument(
        "--num_retries",
        type=int,
        default=0,
        help="Retry queries that produce no tool calls.",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help="Optional cap on how many queries to run from the input CSV.",
    )
    parser.add_argument(
        "--print_first_prompt",
        action="store_true",
        help="Print the system prompt and first user query before running inference.",
    )
    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model with bitsandbytes 4-bit quantization.",
    )
    quant_group.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the model with bitsandbytes 8-bit quantization.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def default_model_label(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")


def is_qwen3_model(model_id: str) -> bool:
    return "qwen3" in model_id.lower()


def choose_torch_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_available_system_memory_gib() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return max(1, (page_size * available_pages) // (1024**3))


def get_max_memory_map() -> dict[int | str, str] | None:
    if not torch.cuda.is_available():
        return None

    max_memory: dict[int | str, str] = {}
    gib = 1024**3
    for gpu_idx in range(torch.cuda.device_count()):
        total_memory_gib = torch.cuda.get_device_properties(gpu_idx).total_memory // gib
        # Reserve extra headroom for activations, CUDA runtime state, and model-specific kernels.
        reserved_memory_gib = max(2, int(total_memory_gib * 0.15))
        usable_memory_gib = max(1, total_memory_gib - reserved_memory_gib)
        max_memory[gpu_idx] = f"{usable_memory_gib}GiB"

    available_system_memory_gib = get_available_system_memory_gib()
    if available_system_memory_gib is not None:
        # Let accelerate offload layers to CPU instead of packing the GPUs to the limit.
        cpu_memory_gib = max(8, int(available_system_memory_gib * 0.75))
        max_memory["cpu"] = f"{cpu_memory_gib}GiB"

    return max_memory


def build_quantization_config(load_in_4bit: bool, load_in_8bit: bool):
    if not load_in_4bit and not load_in_8bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "Quantized loading requires bitsandbytes. Install it in the runtime environment first."
        ) from exc

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=choose_torch_dtype(),
        )

    return BitsAndBytesConfig(load_in_8bit=True)


def build_base_system_prompt() -> str:
    return (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, "
        f"{HARDCODED_CURRENT_TIME.date()} and the current time is "
        f"{HARDCODED_CURRENT_TIME.time()}. "
        "Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm. "
        "Use the available tools to complete the user's request. "
    )


def build_tool_calling_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "When a tool is needed, return a tool call. "
        "When you have finished all necessary tool calls, provide a brief final answer."
    )


def build_plan_then_act_planning_system_prompt(tools: list[Any]) -> str:
    return (
        build_base_system_prompt()
        + "You are in a planning-only phase. Do not call tools. "
        + "Return a short plan in 1-3 brief bullet points that references the available tools when useful. "
        + "Do not include JSON, tool-call syntax, or a final answer.\n\n"
        + "Available tools:\n"
        + format_tools_for_text_prompt(tools)
    )


def build_chain_of_thought_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "Think step by step before calling any tool. "
        + "When a tool is needed, return a tool call. "
        + "When you have finished all necessary tool calls, provide a brief final answer."
    )


def build_few_shot_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "Before the real user request, you will be given worked examples for this domain. "
        + "Treat those examples as demonstrations of the expected tool-use pattern. "
        + "Prefer lookup and search tools before write tools whenever information must be resolved first. "
        + "Follow the same argument formatting and tool-call style, but adapt the actions to the current request. "
        + "When a tool is needed, return a tool call. "
        + "After completing the necessary tool calls, provide a brief final answer."
    )


def build_memory_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "Before the real user request, you may be given a small set of retrieved past episodes that solved similar requests. "
        + "Treat each retrieved episode as a demonstration of the expected tool-use pattern, but adapt the actions to the current request rather than copying values verbatim. "
        + "If no past episodes are shown, proceed using the available tools alone. "
        + "Prefer lookup and search tools before write tools whenever information must be resolved first. "
        + "When a tool is needed, return a tool call. "
        + "After completing the necessary tool calls, provide a brief final answer."
    )

# Shared ToT policy text is reused across expansion and judging so both phases
# optimize for the same execution constraints.
def build_tot_policy_prompt() -> str:
    return (
        build_base_system_prompt()
        + "Before each step, consider multiple plausible next actions and choose the single best next step before acting. "
        + "Prefer prerequisite lookup, search, and read tools before write tools whenever identifiers or current state must be resolved first. "
        + "Execute only one chosen next step at a time."
    )


# Expansion is the "propose branches" phase: generate structured candidates,
# but do not execute anything yet.
def build_tot_expand_prompt(
    tools: list[Any],
    num_thoughts: int,
    policy_prompt: str,
) -> str:
    candidate_count = max(1, num_thoughts)
    return (
        policy_prompt
        + f" You are in a Tree-of-Thoughts expansion phase. Return exactly {candidate_count} distinct candidate next steps with ids "
        + f'"T1" through "T{candidate_count}". '
        + "Do not call tools directly and do not answer the user request yet. "
        + "Each candidate must be exactly one of these:\n"
        + '1. a single tool call with an exact available tool name and JSON arguments, or\n'
        + "2. a brief final answer, but only if the task is already complete.\n"
        + "Prefer candidates that resolve missing IDs, names, emails, dates, or existing records before any write action. "
        + "Keep reasoning brief and action-oriented. "
        + 'Return exactly one JSON object with a top-level "thoughts" array and nothing else. '
        + 'Each thought must include "id", "reasoning", and exactly one of "action" or "final_answer". '
        + "Use this action shape when a tool is needed:\n"
        + '{"id": "T1", "reasoning": "brief why this is promising", "action": {"name": "exact_tool_name", "arguments": {"arg": "value"}}}\n'
        + "Use this final-answer shape only when the task is already complete:\n"
        + '{"id": "T2", "reasoning": "brief why finishing is safe", "final_answer": "brief answer"}\n\n'
        + "Available tools:\n"
        + format_tools_for_text_prompt(tools)
    )


# Judging is a separate "select one branch" phase so execution only advances
# along one candidate path per iteration.
def build_tot_judge_prompt(
    tools: list[Any],
    policy_prompt: str,
) -> str:
    return (
        policy_prompt
        + " You are evaluating candidate thoughts in a Tree-of-Thoughts search. "
        + "Do not call tools and do not answer the user request yet. "
        + "Select the single best next step. "
        + "Prefer the candidate that is most likely to complete the task safely and efficiently, uses exact available tool names, resolves missing information before write actions, and avoids premature final answers. "
        + "Return exactly one JSON object with this schema and nothing else:\n"
        + '{"best_thought_id": "T1", "reason": "brief justification"}\n\n'
        + "Available tools:\n"
        + format_tools_for_text_prompt(tools)
    )


def build_route_then_act_routing_system_prompt(tools: list[Any]) -> str:
    return (
        build_base_system_prompt()
        + "You are in a routing-only phase. Do not answer the user request and do not call tools. "
        + "Choose the smallest executable subset of available tools that is likely sufficient to solve the request end to end. "
        + "Select at least two tools. "
        + "Include any prerequisite lookup, search, list, or read tools needed to discover required IDs, emails, names, or existing records before a write action. "
        + "For update, delete, move, reassign, or status-change requests, include both a search/read tool and the corresponding write tool. "
        + "If a person's email may need to be resolved, include the company directory tool. "
        + "Prefer a slightly larger executable subset over a too-small subset that would block execution. "
        + "Return exactly one JSON object with this schema and nothing else:\n"
        + '{"selected_tools": ["exact_tool_name"]}\n\n'
        + "Available tools:\n"
        + format_tools_for_text_prompt(tools)
    )


def build_self_reflection_system_prompt(tools: list[Any]) -> str:
    return (
        build_base_system_prompt()
        + "You are in a self-reflection phase. Do not call tools and do not answer the user request yet. "
        + "Review the conversation so far, including any tool results, and write 2-4 short bullet points. "
        + "Focus on missing information, risky assumptions, whether a write action is safe yet, and the single best next step. "
        + "Mention exact tool names when useful. "
        + "If the task is already complete, say that no more tools are needed. "
        + "Do not include JSON or tool-call syntax.\n\n"
        + "Available tools:\n"
        + format_tools_for_text_prompt(tools)
    )


def build_rewoo_planner_system_prompt(tools: list[Any]) -> str:
    return (
        build_base_system_prompt()
        + "You are the Planner in a ReWOO loop (Reasoning WithOut Observation). "
        + "Decompose the task into a complete plan that fully solves it before any tool is executed. "
        + "Do not call tools directly, do not return JSON tool calls, and do not include observations or final answers. "
        + "Each step must be exactly two lines and use this format:\n"
        + "Plan: <brief reasoning for this step>\n"
        + "#E<n> = <exact_tool_name>[<JSON object with the tool arguments>]\n"
        + "where <n> is the 1-based step index. "
        + "You may reference earlier evidence by writing the bare token #E1, #E2, ... anywhere inside the JSON arguments; "
        + "the worker will textually replace that exact token with the prior tool's output before execution. "
        + "Strict reference rules: "
        + "(1) Use only the bare token form (#E1, #E2, ...). "
        + "(2) Do NOT use Python-style indexing or field access such as #E1[0], #E1.id, #E1['email_id'], or #E1.rows[0].field; "
        + "the worker cannot resolve those. "
        + "(3) If you need a single value (an id, an email, a date) that lives inside a structured prior observation, "
        + "do not try to extract it inside the JSON arguments. Instead, plan a dedicated lookup step that returns just that single value "
        + "(for example a get_*_by_id, find_*, or company directory lookup tool from the list below) and reference its bare #E<k> in the next step. "
        + "Use exact tool names from the list below. "
        + "Plan ALL the tool calls needed to fully complete the task, including any prerequisite lookup, search, or read steps required before write actions. "
        + "Return only the sequence of (Plan, #E<n>) pairs and nothing else.\n\n"
        + "Available tools:\n"
        + format_tools_for_text_prompt(tools)
    )


def build_rewoo_solver_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "You are the Solver in a ReWOO loop. The Worker has already executed every tool call in the plan. "
        + "Use the original task, the plan, and the collected evidence to write a brief final answer for the user. "
        + "Do not call any tools and do not propose any further actions."
    )


def build_rewoo_arg_repair_system_prompt(
    tool: Any,
    args_template: str,
    evidences: dict[str, str],
) -> str:
    schema = tool.openai_schema["function"]
    properties = schema.get("parameters", {}).get("properties", {})
    evidence_text = (
        "\n".join(f"#{var} = {value}" for var, value in evidences.items())
        if evidences
        else "(no prior evidence)"
    )
    return (
        build_base_system_prompt()
        + "You are repairing arguments for a single tool call inside a ReWOO worker. "
        + "The Planner emitted an argument template that contains #E references (and possibly Python-style accessors like #E1[0].field) "
        + "that the worker could not resolve mechanically. "
        + "Your job: read the resolved evidence below, figure out the single concrete value the Planner intended for each argument "
        + "(typically one id, email address, date, or string copied from a row of a prior observation), "
        + "and output exactly one JSON object containing those concrete arguments for the tool.\n\n"
        + "Hard rules: "
        + "(1) Output exactly one JSON object and nothing else - no prose, no markdown fences, no #E references in the output. "
        + "(2) Every value in the output must be a concrete literal. Do NOT echo the template. "
        + "Do NOT include any string of the form #E<n>, #E<n>[..], or #E<n>.field in the output. "
        + "If you find yourself about to write any of those, stop and look up the actual value in the evidence below. "
        + "(3) Use exact field names from the tool schema. "
        + "(4) If a required value cannot be found in the evidence, omit that key rather than inventing one or echoing the placeholder.\n\n"
        + f"Tool: {schema['name']}\n"
        + f"Tool description: {schema.get('description', '')}\n"
        + f"Parameters: {json.dumps(properties)}\n"
        + f"Argument template from the Planner: {args_template}\n"
        + "Evidence so far (each #E<n> = the full output of the n-th planned tool call):\n"
        + evidence_text
    )


# Reflexion (Shinn et al., 2023) wires three LLM "modules" around a normal
# tool-calling Actor: the Actor itself runs one full trial, an Evaluator judges
# whether that trial succeeded, and a Self-Reflection module turns a failed
# trial into a short verbal reflection that is appended to a bounded memory
# buffer. The Actor sees the buffer at the start of every subsequent trial, so
# memory is "verbal reinforcement" rather than weight updates.
def build_reflexion_actor_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "Before the real user request, you may be given a short list of reflections written by the same agent after past failed attempts at this exact task. "
        + "Treat each reflection as concrete guidance about what to do differently this attempt: which tool to call before another, which lookup to add before a write, which argument was wrong and what value to use instead. "
        + "Do not repeat the past attempt's mistakes. "
        + "If no reflections are shown, this is the first attempt; proceed using the available tools alone. "
        + "When a tool is needed, return a tool call. "
        + "When you have finished all necessary tool calls, provide a brief final answer."
    )


# The Evaluator is a separate LLM call that sees only the task and a textual
# transcript of the Actor's trial. The paper allows the Evaluator to be ground
# truth, an exact-match check, or an LM judge depending on the environment;
# WorkBench has no automatic per-task success signal that could be used at
# inference without leaking ground truth, so we use the LM-judge variant.
def build_reflexion_evaluator_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "You are the Evaluator in a Reflexion loop. Read the user task and the transcript of one agent attempt (its tool calls, observations, and final answer). "
        + "Decide whether the attempt successfully completed the task. "
        + "Be strict: an attempt only succeeds if every requested side effect (write, update, delete, send, schedule, move, reassign, status change) was performed correctly with arguments that are consistent with the observations, and any requested information was actually retrieved and reported. "
        + "Common failure modes to flag as failure: missing required write actions, fabricated identifiers/emails not verified by a prior search, wrong recipient or wrong values, premature termination before all needed steps, calling a non-existent tool, repeating the same failing call, or making no tool calls at all when the task clearly needs them. "
        + "Return exactly one JSON object with this schema and nothing else:\n"
        + '{"success": true, "reason": "brief justification"}\n'
        + "or\n"
        + '{"success": false, "reason": "brief justification of what is missing or wrong"}'
    )


# Self-Reflection turns (failed trial, evaluator verdict) into a short verbal
# critique. The output goes into the Actor's prompt for the next trial, so it
# must be concrete (tool names, argument fixes, missing prerequisite steps)
# rather than generic advice.
def build_reflexion_self_reflection_system_prompt() -> str:
    return (
        build_base_system_prompt()
        + "You are the Self-Reflection module in a Reflexion loop. The previous attempt at this task did not succeed according to the Evaluator. "
        + "Read the original user request, the transcript of the failed attempt, and the Evaluator's reason. "
        + "Write a short reflection (at most 3 sentences) diagnosing the concrete failure and stating exactly what the next attempt should do differently: which tool to call instead, which lookup to add before a write, which argument was wrong and what value to use instead. "
        + "Mention exact tool names when useful. "
        + "Do not include JSON, do not include tool-call syntax, and do not include the next plan or any executable actions. Output only the reflection text."
    )


def format_reflexion_trajectory(
    actions: list[dict[str, Any]],
    final_answer: str,
) -> str:
    lines: list[str] = []
    for index, action in enumerate(actions, start=1):
        name = action.get("name", "?")
        try:
            arguments_text = json.dumps(action.get("arguments", {}), sort_keys=True)
        except (TypeError, ValueError):
            arguments_text = str(action.get("arguments", {}))
        observation = action.get("observation", "")
        lines.append(f"Step {index}: {name}({arguments_text})")
        lines.append(f"  Observation: {observation}")
    if final_answer:
        lines.append(f"Final answer: {final_answer}")
    if not lines:
        return "(no actions taken; no final answer)"
    return "\n".join(lines)


def format_reflexion_memory(reflections: list[str]) -> str:
    if not reflections:
        return "(none)"
    return "\n".join(f"- {reflection}" for reflection in reflections)


# Inline the reflection buffer into the user query rather than into the system
# message or as a separate chat turn. That keeps the Actor's chat-template path
# identical to the plain tool-calling baseline (no extra assistant turns to
# confuse Qwen3 tool-call decoding) and ensures reflections are always grouped
# with the task they critique.
def build_reflexion_actor_user_query(query: str, reflections: list[str]) -> str:
    if not reflections:
        return query
    return (
        "Reflections written by the agent after previous failed attempts at this exact task:\n"
        + format_reflexion_memory(reflections)
        + "\n\nUser request:\n"
        + query
    )


def parse_reflexion_evaluator_decision(text: str) -> tuple[bool, str]:
    for payload in extract_json_candidates(text):
        if "success" not in payload:
            continue
        success_value = payload["success"]
        if isinstance(success_value, bool):
            success = success_value
        elif isinstance(success_value, str):
            success = success_value.strip().lower() in {"true", "yes", "success", "succeeded"}
        else:
            success = bool(success_value)
        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            reason = json.dumps(reason)
        return success, reason.strip()

    # Fallback heuristics for weaker checkpoints that drift off the JSON schema.
    lowered = text.lower()
    if re.search(r"\bsuccess(ful|fully|ed)?\b", lowered) and not re.search(
        r"\b(not|no|did not|failed|failure)\b", lowered
    ):
        return True, text.strip()
    return False, text.strip()


def format_tools_for_text_prompt(tools: list[Any]) -> str:
    formatted_tools: list[str] = []
    for fn in tools:
        schema = fn.openai_schema["function"]
        properties = schema.get("parameters", {}).get("properties", {})
        parameter_parts = []
        for name, metadata in properties.items():
            param_type = metadata.get("type", "any")
            description = metadata.get("description", "")
            parameter_parts.append(f"{name} ({param_type}): {description}")

        params_text = "; ".join(parameter_parts) if parameter_parts else "No arguments."
        formatted_tools.append(
            f"- {schema['name']}: {schema.get('description', '')}\n"
            f"  Arguments: {params_text}"
        )
    return "\n".join(formatted_tools)


def build_react_system_prompt(tools: list[Any]) -> str:
    return (
        build_base_system_prompt()
        + "You are using a ReAct-style loop. Think step by step, but do not rely on hidden tools. "
        + "If you need a tool, respond with exactly one action using this format:\n"
        + "Thought: <brief reasoning>\n"
        + "Action:\n"
        + '{"name": "tool_name", "arguments": {"arg": "value"}}\n'
        + "After you receive an observation, continue reasoning and either call one more tool or finish with:\n"
        + "Thought: <brief reasoning>\n"
        + "Final Answer: <brief answer>\n\n"
        + "Available tools:\n"
        + format_tools_for_text_prompt(tools)
    )


def build_plan_then_act_execution_user_prompt(query: str, plan_text: str) -> str:
    return (
        "User request:\n"
        + query
        + "\n\nDraft plan:\n"
        + plan_text
        + "\n\nUse the plan if it helps, but update it if tool results suggest a better approach."
    )


def build_system_prompt(agent_baseline: str, tools: list[Any]) -> str:
    if agent_baseline == "tool_calling":
        return build_tool_calling_system_prompt()
    if agent_baseline == "chain_of_thought":
        return build_chain_of_thought_system_prompt()
    if agent_baseline == "few_shot":
        return build_few_shot_system_prompt()
    if agent_baseline == "memory":
        return build_memory_system_prompt()
    if agent_baseline == "tree_of_thoughts":
        return build_tot_policy_prompt()
    if agent_baseline == "plan_then_act":
        return build_tool_calling_system_prompt()
    if agent_baseline == "route_then_act":
        return build_tool_calling_system_prompt()
    if agent_baseline == "self_reflection":
        return build_tool_calling_system_prompt()
    if agent_baseline == "react":
        return build_react_system_prompt(tools)
    if agent_baseline == "rewoo":
        return build_rewoo_planner_system_prompt(tools)
    if agent_baseline == "reflexion":
        return build_reflexion_actor_system_prompt()
    raise ValueError(f"Unknown agent baseline: {agent_baseline}")


def decode_new_tokens(
    tokenizer: AutoTokenizer,
    outputs: torch.Tensor,
    prompt_length: int,
) -> str:
    completion_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(completion_tokens, skip_special_tokens=False).strip()


def extract_json_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates


def normalize_tool_call(payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if "function" in payload and isinstance(payload["function"], dict):
        payload = payload["function"]

    name = payload.get("name")
    parameters = payload.get("parameters")
    if parameters is None:
        parameters = payload.get("arguments")

    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            return None

    if not isinstance(name, str) or not isinstance(parameters, dict):
        return None

    return name, parameters


def parse_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    parsed_tool_calls: list[tuple[str, dict[str, Any]]] = []
    tool_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    candidate_texts = tool_blocks if tool_blocks else [text]

    for candidate_text in candidate_texts:
        for candidate in extract_json_candidates(candidate_text):
            normalized = normalize_tool_call(candidate)
            if normalized is not None:
                parsed_tool_calls.append(normalized)
                break

    if parsed_tool_calls:
        return parsed_tool_calls

    for candidate in extract_json_candidates(text):
        normalized = normalize_tool_call(candidate)
        if normalized is not None:
            parsed_tool_calls.append(normalized)

    return parsed_tool_calls


def get_tool_name(tool: Any) -> str:
    return tool.openai_schema["function"]["name"]


def get_tool_debug_name(tool: Any) -> str:
    return _openai_name_to_internal(get_tool_name(tool))


def parse_selected_tool_names(text: str, tools: list[Any]) -> list[str]:
    available_names = {get_tool_name(tool) for tool in tools}

    def normalize_selected_names(candidate: Any) -> list[str]:
        if isinstance(candidate, str):
            candidate = [part.strip() for part in candidate.split(",")]
        if not isinstance(candidate, list):
            return []

        selected_names: list[str] = []
        for name in candidate:
            if isinstance(name, str) and name in available_names and name not in selected_names:
                selected_names.append(name)
        return selected_names

    for payload in extract_json_candidates(text):
        selected_names = normalize_selected_names(payload.get("selected_tools"))
        if selected_names:
            return selected_names

        selected_names = normalize_selected_names(payload.get("tools"))
        if selected_names:
            return selected_names

    selected_names = []
    for tool in tools:
        tool_name = get_tool_name(tool)
        if tool_name in text and tool_name not in selected_names:
            selected_names.append(tool_name)
    return selected_names


# Normalize prompt-level schema variants into a single internal shape so the
# execution loop does not depend on the exact JSON field names the model chose.
def normalize_tree_of_thought_candidate(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    candidate_id = payload.get("id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return None

    reasoning = payload.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = ""

    action_payload = payload.get("action")
    if action_payload is None:
        action_payload = payload.get("tool_call")
    if action_payload is None:
        action_payload = payload.get("tool")

    normalized_action = normalize_tool_call(action_payload) if action_payload is not None else None
    if normalized_action is not None:
        tool_name, tool_args = normalized_action
        return {
            "id": candidate_id.strip(),
            "reasoning": reasoning.strip(),
            "kind": "tool_call",
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

    final_answer = payload.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return {
            "id": candidate_id.strip(),
            "reasoning": reasoning.strip(),
            "kind": "final_answer",
            "final_answer": final_answer.strip(),
        }

    return None


# First parse the intended structured output. If the model drifts from the
# schema, degrade gracefully by treating raw tool calls or plain text as a
# best-effort candidate instead of failing the entire turn.
def parse_tree_of_thought_candidates(text: str) -> list[dict[str, Any]]:
    for payload in extract_json_candidates(text):
        thoughts = payload.get("thoughts")
        if not isinstance(thoughts, list):
            continue

        candidates: list[dict[str, Any]] = []
        for candidate_payload in thoughts:
            normalized = normalize_tree_of_thought_candidate(candidate_payload)
            if normalized is not None:
                candidates.append(normalized)
        if candidates:
            return candidates

    fallback_candidates: list[dict[str, Any]] = []
    for index, (tool_name, tool_args) in enumerate(parse_tool_calls(text), start=1):
        fallback_candidates.append(
            {
                "id": f"T{index}",
                "reasoning": "",
                "kind": "tool_call",
                "tool_name": tool_name,
                "tool_args": tool_args,
            }
        )
    if fallback_candidates:
        return fallback_candidates

    stripped_text = text.strip()
    if not stripped_text:
        return []

    return [
        {
            "id": "T1",
            "reasoning": "",
            "kind": "final_answer",
            "final_answer": stripped_text,
        }
    ]


def format_tree_of_thought_candidates_for_judge(candidates: list[dict[str, Any]]) -> str:
    formatted_candidates: list[str] = []
    for candidate in candidates:
        reasoning = candidate.get("reasoning") or "No reasoning provided."
        if candidate["kind"] == "tool_call":
            action_payload = {
                "name": candidate["tool_name"],
                "arguments": candidate["tool_args"],
            }
            formatted_candidates.append(
                f'{candidate["id"]}\n'
                f"Reasoning: {reasoning}\n"
                f"Action: {json.dumps(action_payload, sort_keys=True)}"
            )
            continue

        formatted_candidates.append(
            f'{candidate["id"]}\n'
            f"Reasoning: {reasoning}\n"
            f'Final answer: {candidate["final_answer"]}'
        )
    return "\n\n".join(formatted_candidates)


# The judge is supposed to return a single thought id, but weaker checkpoints
# may drift slightly, so accept a few compatible JSON field names and finally a
# textual mention of a valid id.
def parse_selected_tree_of_thought_id(text: str, candidates: list[dict[str, Any]]) -> str | None:
    available_ids = {candidate["id"] for candidate in candidates}

    for payload in extract_json_candidates(text):
        selected_id = payload.get("best_thought_id")
        if selected_id is None:
            selected_id = payload.get("selected_thought_id")
        if selected_id is None:
            selected_id = payload.get("id")
        if isinstance(selected_id, str) and selected_id in available_ids:
            return selected_id

    for candidate in candidates:
        candidate_id = candidate["id"]
        if re.search(rf"\b{re.escape(candidate_id)}\b", text):
            return candidate_id
    return None


def execute_tool_call(tool_name: str, tool_args: dict[str, Any]) -> str:
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return f"Unknown tool: {tool_name}"

    try:
        return str(fn(**tool_args))
    except Exception as exc:
        return f"Error executing tool: {exc}"


def select_route_then_act_tools(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    max_new_tokens: int,
) -> tuple[list[Any], str]:
    routing_messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_route_then_act_routing_system_prompt(tools)},
        {"role": "user", "content": query},
    ]
    routing_text = generate_assistant_turn(
        model=model,
        tokenizer=tokenizer,
        messages=routing_messages,
        tools=None,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
    )

    selected_tool_names = parse_selected_tool_names(routing_text, tools)
    if not selected_tool_names:
        return tools, routing_text

    tool_by_name = {get_tool_name(tool): tool for tool in tools}
    return [tool_by_name[name] for name in selected_tool_names], routing_text


def generate_assistant_turn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_new_tokens: int,
    model_id: str,
) -> str:
    chat_template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if tools is not None:
        chat_template_kwargs["tools"] = tools
    if is_qwen3_model(model_id):
        # Qwen3 tool use is more reliable with thinking disabled.
        chat_template_kwargs["enable_thinking"] = False

    inputs = tokenizer.apply_chat_template(
        messages,
        **chat_template_kwargs,
    )
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return decode_new_tokens(tokenizer, outputs, prompt_length)


def run_hf_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
    prefix_messages: list[dict[str, Any]] | None = None,
    recorded_outcome: dict[str, Any] | None = None,
) -> tuple[list[str], str, str]:
    tool_schemas = [fn.openai_schema for fn in tools]
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if prefix_messages:
        messages.extend(prefix_messages)
    messages.append({"role": "user", "content": query})
    function_calls: list[str] = []
    full_response_parts: list[str] = []
    if recorded_outcome is not None:
        recorded_outcome.setdefault("actions", [])
        recorded_outcome["final_answer"] = ""

    for _ in range(max_iterations):
        assistant_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            tools=tool_schemas,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
        full_response_parts.append(assistant_text)

        parsed_tool_calls = parse_tool_calls(assistant_text)
        if not parsed_tool_calls:
            if recorded_outcome is not None:
                recorded_outcome["final_answer"] = assistant_text.strip()
            return function_calls, "\n".join(full_response_parts), ""

        assistant_tool_calls = []
        tool_result_messages = []

        for tool_name, tool_args in parsed_tool_calls:
            function_calls.append(convert_tool_call_to_function_call(tool_name, tool_args))
            recorded_action: dict[str, Any] | None = None
            if recorded_outcome is not None:
                recorded_action = {"name": tool_name, "arguments": tool_args}
                recorded_outcome["actions"].append(recorded_action)
            assistant_tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args,
                    },
                }
            )

            fn = _TOOL_REGISTRY.get(tool_name)
            if fn is None:
                tool_result = f"Unknown tool: {tool_name}"
            else:
                try:
                    tool_result = str(fn(**tool_args))
                except Exception as exc:
                    tool_result = f"Error executing tool: {exc}"

            # Reflexion needs (action, observation) pairs to build a transcript
            # for the Evaluator and Self-Reflection LLM calls. Memory's
            # _compress_actions rebuilds canonical {name, arguments} dicts so
            # the extra "observation" key here is harmless for that baseline.
            if recorded_action is not None:
                recorded_action["observation"] = tool_result

            tool_result_messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result,
                }
            )

        messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            }
        )
        messages.extend(tool_result_messages)

    return (
        function_calls,
        "\n".join(full_response_parts),
        "Agent stopped due to iteration limit.",
    )


def run_react_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
) -> tuple[list[str], str, str]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    function_calls: list[str] = []
    full_response_parts: list[str] = []

    for _ in range(max_iterations):
        assistant_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            tools=None,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
        full_response_parts.append(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})

        parsed_tool_calls = parse_tool_calls(assistant_text)
        if not parsed_tool_calls:
            return function_calls, "\n".join(full_response_parts), ""

        tool_name, tool_args = parsed_tool_calls[0]
        function_calls.append(convert_tool_call_to_function_call(tool_name, tool_args))

        fn = _TOOL_REGISTRY.get(tool_name)
        if fn is None:
            tool_result = f"Unknown tool: {tool_name}"
        else:
            try:
                tool_result = str(fn(**tool_args))
            except Exception as exc:
                tool_result = f"Error executing tool: {exc}"

        observation = f"Observation:\n{tool_result}"
        full_response_parts.append(observation)
        messages.append({"role": "user", "content": observation})

    return (
        function_calls,
        "\n".join(full_response_parts),
        "Agent stopped due to iteration limit.",
    )


def run_plan_then_act_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
) -> tuple[list[str], str, str]:
    tool_schemas = [fn.openai_schema for fn in tools]
    planning_messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_plan_then_act_planning_system_prompt(tools)},
        {"role": "user", "content": query},
    ]
    function_calls: list[str] = []
    full_response_parts: list[str] = []

    plan_text = generate_assistant_turn(
        model=model,
        tokenizer=tokenizer,
        messages=planning_messages,
        tools=None,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
    )
    full_response_parts.append(f"Plan:\n{plan_text}")

    execution_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": build_plan_then_act_execution_user_prompt(query, plan_text),
        }
    ]

    for _ in range(max_iterations):
        assistant_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=execution_messages,
            tools=tool_schemas,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
        full_response_parts.append(assistant_text)

        parsed_tool_calls = parse_tool_calls(assistant_text)
        if not parsed_tool_calls:
            return function_calls, "\n".join(full_response_parts), ""

        assistant_tool_calls = []
        tool_result_messages = []

        for tool_name, tool_args in parsed_tool_calls:
            function_calls.append(convert_tool_call_to_function_call(tool_name, tool_args))
            assistant_tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args,
                    },
                }
            )

            fn = _TOOL_REGISTRY.get(tool_name)
            if fn is None:
                tool_result = f"Unknown tool: {tool_name}"
            else:
                try:
                    tool_result = str(fn(**tool_args))
                except Exception as exc:
                    tool_result = f"Error executing tool: {exc}"

            tool_result_messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result,
                }
            )

        execution_messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            }
        )
        execution_messages.extend(tool_result_messages)

    return (
        function_calls,
        "\n".join(full_response_parts),
        "Agent stopped due to iteration limit.",
    )


def run_route_then_act_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
) -> tuple[list[str], str, str]:
    selected_tools, routing_text = select_route_then_act_tools(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        query=query,
        tools=tools,
        max_new_tokens=max_new_tokens,
    )
    selected_tool_debug_names = [get_tool_debug_name(tool) for tool in selected_tools]
    print(f"### Selected tools: {selected_tool_debug_names}")

    function_calls, execution_response, error = run_hf_agent(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        query=query,
        tools=selected_tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        max_new_tokens=max_new_tokens,
    )
    full_response_parts = [
        f"Routing response:\n{routing_text}",
        f"Selected tools: {selected_tool_debug_names}",
        execution_response,
    ]
    return function_calls, "\n".join(full_response_parts), error


def run_self_reflection_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
) -> tuple[list[str], str, str]:
    tool_schemas = [fn.openai_schema for fn in tools]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    function_calls: list[str] = []
    full_response_parts: list[str] = []

    for _ in range(max_iterations):
        reflection_messages = [
            {"role": "system", "content": build_self_reflection_system_prompt(tools)},
            *messages[1:],
        ]
        reflection_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=reflection_messages,
            tools=None,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
        full_response_parts.append(f"Reflection:\n{reflection_text}")

        execution_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Self-reflection for the next step:\n"
                    + reflection_text
                    + "\n\nUse this reflection to decide the next tool call(s), or finish with a brief final answer if the task is complete."
                ),
            }
        ]
        assistant_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=execution_messages,
            tools=tool_schemas,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
        full_response_parts.append(assistant_text)

        parsed_tool_calls = parse_tool_calls(assistant_text)
        if not parsed_tool_calls:
            return function_calls, "\n".join(full_response_parts), ""

        assistant_tool_calls = []
        tool_result_messages = []

        for tool_name, tool_args in parsed_tool_calls:
            function_calls.append(convert_tool_call_to_function_call(tool_name, tool_args))
            assistant_tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args,
                    },
                }
            )

            fn = _TOOL_REGISTRY.get(tool_name)
            if fn is None:
                tool_result = f"Unknown tool: {tool_name}"
            else:
                try:
                    tool_result = str(fn(**tool_args))
                except Exception as exc:
                    tool_result = f"Error executing tool: {exc}"

            tool_result_messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": tool_result,
                }
            )

        messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            }
        )
        messages.extend(tool_result_messages)

    return (
        function_calls,
        "\n".join(full_response_parts),
        "Agent stopped due to iteration limit.",
    )


def run_tree_of_thoughts_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
    num_thoughts: int,
) -> tuple[list[str], str, str]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    function_calls: list[str] = []
    full_response_parts: list[str] = []

    for _ in range(max_iterations):
        # Phase 1: expand the current state into a small set of candidate next
        # steps. This stays tool-free so we can compare alternatives first.
        thought_messages = [
            {
                "role": "system",
                "content": build_tot_expand_prompt(
                    tools=tools,
                    num_thoughts=num_thoughts,
                    policy_prompt=system_prompt,
                ),
            },
            *messages,
        ]
        thoughts_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=thought_messages,
            tools=None,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
        full_response_parts.append(f"Thought candidates:\n{thoughts_text}")

        candidates = parse_tree_of_thought_candidates(thoughts_text)
        if not candidates:
            return (
                function_calls,
                "\n".join(full_response_parts),
                "Tree-of-thoughts expansion produced no valid candidates.",
            )

        selected_candidate = candidates[0]
        if len(candidates) > 1:
            # Phase 2: judge the proposed branches and pick the single best
            # next step before mutating benchmark state.
            judge_messages = [
                {
                    "role": "system",
                    "content": build_tot_judge_prompt(
                        tools=tools,
                        policy_prompt=system_prompt,
                    ),
                },
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Candidate next steps:\n"
                        + format_tree_of_thought_candidates_for_judge(candidates)
                        + "\n\nChoose the single best candidate for the next step."
                    ),
                },
            ]
            judge_text = generate_assistant_turn(
                model=model,
                tokenizer=tokenizer,
                messages=judge_messages,
                tools=None,
                max_new_tokens=max_new_tokens,
                model_id=model_id,
            )
            full_response_parts.append(f"Thought judge:\n{judge_text}")

            selected_id = parse_selected_tree_of_thought_id(judge_text, candidates)
            if selected_id is not None:
                selected_candidate = next(
                    candidate for candidate in candidates if candidate["id"] == selected_id
                )

        # Keep a compact trace of the chosen branch for debugging and result inspection.
        full_response_parts.append(
            "Selected thought:\n" + format_tree_of_thought_candidates_for_judge([selected_candidate])
        )

        if selected_candidate["kind"] == "final_answer":
            full_response_parts.append(f'Final answer:\n{selected_candidate["final_answer"]}')
            return function_calls, "\n".join(full_response_parts), ""

        tool_name = selected_candidate["tool_name"]
        tool_args = selected_candidate["tool_args"]
        function_calls.append(convert_tool_call_to_function_call(tool_name, tool_args))
        tool_result = execute_tool_call(tool_name, tool_args)
        full_response_parts.append(f"Observation:\n{tool_result}")

        # Persist only the executed branch in conversation state. Rejected
        # branches remain in the debug trace but should not influence future
        # turns as if they had actually happened.
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": tool_result,
            }
        )

    return (
        function_calls,
        "\n".join(full_response_parts),
        "Agent stopped due to iteration limit.",
    )


# Parse a ReWOO plan emitted by the Planner. The expected shape is a sequence
# of (Plan, #E<n> = tool[args]) pairs; tool argument blocks may span multiple
# lines and contain nested brackets, so the parser tracks bracket depth instead
# of relying on a single-line regex.
def parse_rewoo_plan(text: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    parts = re.split(r"(?:^|\n)\s*Plan\s*:\s*", text)
    for part in parts[1:]:
        action_match = re.search(r"#E(\d+)\s*=\s*([A-Za-z_][\w]*)\s*\[", part)
        if not action_match:
            continue
        bracket_start = action_match.end()
        depth = 1
        cursor = bracket_start
        while cursor < len(part) and depth > 0:
            if part[cursor] == "[":
                depth += 1
            elif part[cursor] == "]":
                depth -= 1
            cursor += 1
        if depth != 0:
            continue
        steps.append(
            {
                "thought": part[: action_match.start()].strip(),
                "var": "E" + action_match.group(1),
                "tool_name": action_match.group(2),
                "args_template": part[bracket_start : cursor - 1].strip(),
            }
        )
    return steps


# Substitute earlier evidence into an argument template. Higher-numbered
# variables are replaced first so that "#E10" does not collide with "#E1" when
# both are present.
def substitute_rewoo_evidences(template: str, evidences: dict[str, str]) -> str:
    if not template:
        return template
    sorted_vars = sorted(evidences.keys(), key=lambda var: -int(var[1:]))
    result = template
    for var in sorted_vars:
        result = result.replace(f"#{var}", evidences[var])
    return result


# The Worker tries pure textual substitution first, matching the paper. If the
# evidence broke JSON validity (often the case when an observation is a long
# DataFrame string from which a single id must be extracted), fall back to a
# single small LLM repair call so the planned step can still execute.
def resolve_rewoo_step_arguments(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    tool: Any,
    args_template: str,
    evidences: dict[str, str],
    max_new_tokens: int,
) -> tuple[dict[str, Any], str]:
    substituted = substitute_rewoo_evidences(args_template, evidences)
    try:
        parsed = json.loads(substituted)
        if isinstance(parsed, dict):
            return parsed, substituted
    except json.JSONDecodeError:
        pass

    repair_messages = [
        {
            "role": "system",
            "content": build_rewoo_arg_repair_system_prompt(
                tool=tool,
                args_template=args_template,
                evidences=evidences,
            ),
        },
        {"role": "user", "content": "Produce the JSON arguments now."},
    ]
    repair_text = generate_assistant_turn(
        model=model,
        tokenizer=tokenizer,
        messages=repair_messages,
        tools=None,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
    )
    for candidate in extract_json_candidates(repair_text):
        if isinstance(candidate, dict):
            return candidate, repair_text
    return {}, repair_text


def run_rewoo_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
) -> tuple[list[str], str, str]:
    tool_by_name = {get_tool_name(tool): tool for tool in tools}
    full_response_parts: list[str] = []

    # Phase 1: Planner. The Planner produces the entire plan up-front,
    # without seeing any tool observations. This is what makes ReWOO
    # "reasoning without observation" - reasoning tokens are emitted once,
    # not re-emitted with every observation as in ReAct.
    planner_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    # The Planner has to emit the full multi-step plan in a single generation.
    # Use a larger token budget than the per-turn default so 3-4 step plans
    # do not get truncated mid-bracket (which would silently drop steps).
    planner_max_new_tokens = max(max_new_tokens, REWOO_PLANNER_MAX_NEW_TOKENS)
    plan_text = generate_assistant_turn(
        model=model,
        tokenizer=tokenizer,
        messages=planner_messages,
        tools=None,
        max_new_tokens=planner_max_new_tokens,
        model_id=model_id,
    )
    full_response_parts.append(f"Plan:\n{plan_text}")

    steps = parse_rewoo_plan(plan_text)
    if not steps:
        return [], "\n".join(full_response_parts), ""

    # Phase 2: Worker. Execute steps in order, substituting #E<n> placeholders
    # with the corresponding observation before calling the tool.
    function_calls: list[str] = []
    evidences: dict[str, str] = {}
    for step in steps[:max_iterations]:
        tool = tool_by_name.get(step["tool_name"])
        if tool is None:
            evidences[step["var"]] = f"Unknown tool: {step['tool_name']}"
            full_response_parts.append(
                f"Step #{step['var']}\n"
                f"Thought: {step['thought']}\n"
                f"Action: {step['tool_name']}[{step['args_template']}]\n"
                f"Observation: unknown tool"
            )
            continue

        args, resolved_text = resolve_rewoo_step_arguments(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            tool=tool,
            args_template=step["args_template"],
            evidences=evidences,
            max_new_tokens=max_new_tokens,
        )
        function_calls.append(convert_tool_call_to_function_call(step["tool_name"], args))
        observation = execute_tool_call(step["tool_name"], args)
        evidences[step["var"]] = observation
        full_response_parts.append(
            f"Step #{step['var']}\n"
            f"Thought: {step['thought']}\n"
            f"Action: {step['tool_name']}({args})\n"
            f"Resolved arguments: {resolved_text}\n"
            f"Observation: {observation}"
        )

    iteration_limit_error = (
        "Agent stopped due to iteration limit." if len(steps) > max_iterations else ""
    )

    # Phase 3: Solver. Synthesize a brief final answer from the original task,
    # the plan, and the observations. The Solver does not emit tool calls; the
    # Worker has already executed them all.
    evidence_text = (
        "\n".join(f"#{var} = {value}" for var, value in evidences.items())
        if evidences
        else "(no evidence collected)"
    )
    solver_messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_rewoo_solver_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Task:\n{query}\n\n"
                f"Plan:\n{plan_text}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                "Provide a brief final answer."
            ),
        },
    ]
    final_answer = generate_assistant_turn(
        model=model,
        tokenizer=tokenizer,
        messages=solver_messages,
        tools=None,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
    )
    full_response_parts.append(f"Final answer:\n{final_answer}")

    return function_calls, "\n".join(full_response_parts), iteration_limit_error


def run_reflexion_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
    max_trials: int,
    memory_size: int,
) -> tuple[list[str], str, str]:
    reflections: list[str] = []
    full_response_parts: list[str] = []
    last_function_calls: list[str] = []
    last_error = ""
    effective_max_trials = max(1, max_trials)
    effective_memory_size = max(0, memory_size)

    for trial_index in range(effective_max_trials):
        # Reset benchmark state between trials. Reflexion treats every trial as
        # a fresh attempt at the same task; without reset, side effects from
        # prior trials would leak into the current trial's tool observations
        # and corrupt the Evaluator's view of what the trial actually did.
        for domain in DOMAINS:
            domain.reset_state()

        recorded: dict[str, Any] = {"actions": [], "final_answer": ""}
        actor_query = build_reflexion_actor_user_query(query, reflections)
        function_calls, trial_response, trial_error = run_hf_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=actor_query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
            recorded_outcome=recorded,
        )
        last_function_calls = function_calls
        last_error = trial_error

        full_response_parts.append(f"=== Trial {trial_index + 1} of {effective_max_trials} ===")
        full_response_parts.append(
            f"Reflections in buffer:\n{format_reflexion_memory(reflections)}"
        )
        full_response_parts.append(trial_response)

        trajectory_text = format_reflexion_trajectory(
            actions=recorded["actions"],
            final_answer=recorded.get("final_answer", ""),
        )

        # Phase 2: Evaluator. LLM-as-judge over the textual trajectory.
        evaluator_messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_reflexion_evaluator_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"User task:\n{query}\n\n"
                    f"Agent attempt transcript:\n{trajectory_text}\n\n"
                    "Did the attempt successfully complete the task? Return the JSON object now."
                ),
            },
        ]
        evaluator_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=evaluator_messages,
            tools=None,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        )
        success, evaluator_reason = parse_reflexion_evaluator_decision(evaluator_text)
        full_response_parts.append(
            f"Evaluator (raw):\n{evaluator_text}\n"
            f"Evaluator decision: success={success}, reason={evaluator_reason}"
        )

        if success:
            return last_function_calls, "\n".join(full_response_parts), last_error

        # No reflection needed after the final trial; the next attempt will
        # never see it, so skip the extra LLM call.
        if trial_index == effective_max_trials - 1:
            break

        # Phase 3: Self-Reflection. Turn the failed trial into one short verbal
        # reflection and append it to the bounded memory buffer.
        reflection_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_reflexion_self_reflection_system_prompt(),
            },
            {
                "role": "user",
                "content": (
                    f"User task:\n{query}\n\n"
                    f"Agent attempt transcript:\n{trajectory_text}\n\n"
                    f"Evaluator's verdict: failure. Reason: {evaluator_reason}\n\n"
                    "Write the reflection now."
                ),
            },
        ]
        reflection_text = generate_assistant_turn(
            model=model,
            tokenizer=tokenizer,
            messages=reflection_messages,
            tools=None,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
        ).strip()
        if reflection_text and effective_memory_size > 0:
            reflections.append(reflection_text)
            reflections = reflections[-effective_memory_size:]
        full_response_parts.append(f"Reflection appended:\n{reflection_text}")

    return last_function_calls, "\n".join(full_response_parts), last_error


def run_agent(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    query: str,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int,
    max_new_tokens: int,
    agent_baseline: str,
    few_shot_messages: list[dict[str, Any]] | None = None,
    memory_messages: list[dict[str, Any]] | None = None,
    recorded_outcome: dict[str, Any] | None = None,
    tot_num_thoughts: int = DEFAULT_TOT_NUM_THOUGHTS,
    reflexion_max_trials: int = DEFAULT_REFLEXION_MAX_TRIALS,
    reflexion_memory_size: int = DEFAULT_REFLEXION_MEMORY_SIZE,
) -> tuple[list[str], str, str]:
    if agent_baseline == "tool_calling":
        return run_hf_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
        )
    if agent_baseline == "chain_of_thought":
        return run_hf_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
        )
    if agent_baseline == "few_shot":
        return run_hf_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
            prefix_messages=few_shot_messages,
        )
    if agent_baseline == "memory":
        return run_hf_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
            prefix_messages=memory_messages,
            recorded_outcome=recorded_outcome,
        )
    if agent_baseline == "tree_of_thoughts":
        return run_tree_of_thoughts_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
            num_thoughts=tot_num_thoughts,
        )
    if agent_baseline == "plan_then_act":
        return run_plan_then_act_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
        )
    if agent_baseline == "route_then_act":
        return run_route_then_act_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
        )
    if agent_baseline == "self_reflection":
        return run_self_reflection_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
        )
    if agent_baseline == "react":
        return run_react_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
        )
    if agent_baseline == "rewoo":
        return run_rewoo_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
        )
    if agent_baseline == "reflexion":
        return run_reflexion_agent(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            query=query,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            max_new_tokens=max_new_tokens,
            max_trials=reflexion_max_trials,
            memory_size=reflexion_memory_size,
        )
    raise ValueError(f"Unknown agent baseline: {agent_baseline}")


def build_results_mode_label(tool_selection: str, agent_baseline: str) -> str:
    if agent_baseline == "tool_calling":
        return tool_selection
    return f"{tool_selection}_{agent_baseline}"


def load_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_quantization_config(load_in_4bit, load_in_8bit)
    model_load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    max_memory = get_max_memory_map()
    if max_memory is not None:
        model_load_kwargs["max_memory"] = max_memory

    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
    else:
        model_load_kwargs["dtype"] = choose_torch_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_load_kwargs,
    )
    if getattr(model, "generation_config", None) is not None:
        # Some checkpoints ship sampling defaults that conflict with greedy decoding.
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    model.eval()
    return model, tokenizer


def get_query_tools(
    queries_df: pd.DataFrame,
    index: int,
    tool_selection: str,
) -> list[Any]:
    if tool_selection == "all":
        return get_toolkits(
            [
                "email",
                "calendar",
                "analytics",
                "project_management",
                "customer_relationship_manager",
            ]
        )

    if tool_selection == "domains":
        query_toolkits = parse_domains_field(queries_df["domains"].iloc[index])
        return get_toolkits(query_toolkits)

    raise ValueError(f"Unknown tool selection mode: {tool_selection}")


def generate_results_with_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    queries_path: str,
    model_label: str,
    tool_selection: str,
    agent_baseline: str,
    max_iterations: int,
    max_new_tokens: int,
    num_retries: int,
    num_queries: int | None,
    print_first_prompt: bool,
    few_shot_k: int,
    memory_k: int,
    memory_path: str | None,
    memory_min_episodes: int,
    tot_num_thoughts: int,
    reflexion_max_trials: int,
    reflexion_memory_size: int,
) -> pd.DataFrame:
    few_shot_examples_by_domain: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    if agent_baseline == "few_shot":
        few_shot_examples_by_domain = load_few_shot_examples_by_domain()

    episodic_memory = EpisodicMemory()
    if agent_baseline == "memory":
        episodic_memory = EpisodicMemory.load(memory_path)
        if memory_path:
            print(
                f"### Loaded {len(episodic_memory)} memory episode(s) from {memory_path}"
            )

    queries_df = pd.read_csv(queries_path)
    if num_queries is not None:
        queries_df = queries_df.head(num_queries).copy()
    queries = queries_df["query"].tolist()

    # Memory only stores episodes whose action trace would score as correct
    # against ground truth. Without this gate, a query that runs without a
    # Python exception but solves the wrong problem (e.g. fabricated emails,
    # missing write actions) gets stored as a positive demonstration and
    # poisons every subsequent retrieval in the same run.
    ground_truth_actions_per_query: list[list[str]] = []
    if agent_baseline == "memory":
        ground_truth_actions_per_query = [
            ast.literal_eval(answer) for answer in queries_df["answer"].tolist()
        ]
    if print_first_prompt and queries:
        first_tools = get_query_tools(
            queries_df,
            0,
            tool_selection,
        )
        if agent_baseline == "plan_then_act":
            print("### Planning system prompt:")
            print(build_plan_then_act_planning_system_prompt(first_tools))
            print("\n### Execution system prompt:")
            print(build_system_prompt(agent_baseline, first_tools))
        elif agent_baseline == "route_then_act":
            print("### Routing system prompt:")
            print(build_route_then_act_routing_system_prompt(first_tools))
            print("\n### Execution system prompt:")
            print(build_system_prompt(agent_baseline, first_tools))
        elif agent_baseline == "self_reflection":
            print("### Reflection system prompt:")
            print(build_self_reflection_system_prompt(first_tools))
            print("\n### Execution system prompt:")
            print(build_system_prompt(agent_baseline, first_tools))
        elif agent_baseline == "few_shot":
            first_examples = build_few_shot_messages_for_query(
                examples_by_domain=few_shot_examples_by_domain,
                current_domains=parse_domains_field(queries_df["domains"].iloc[0]),
                available_tool_names={get_tool_name(tool) for tool in first_tools},
                few_shot_k=few_shot_k,
                tool_registry=_TOOL_REGISTRY,
                domains=DOMAINS,
            )
            print("### System prompt:")
            print(build_system_prompt(agent_baseline, first_tools))
            print("\n### Few-shot examples:")
            print(format_few_shot_messages(first_examples))
        elif agent_baseline == "memory":
            if len(episodic_memory) >= memory_min_episodes:
                first_memory_messages = build_memory_messages_for_query(
                    memory=episodic_memory,
                    query=queries[0],
                    memory_k=memory_k,
                    available_tool_names={get_tool_name(tool) for tool in first_tools},
                    tool_registry=_TOOL_REGISTRY,
                    domains=DOMAINS,
                )
            else:
                first_memory_messages = []
            print("### System prompt:")
            print(build_system_prompt(agent_baseline, first_tools))
            print(
                f"\n### Retrieved memory episodes for first query (k={memory_k}, store size={len(episodic_memory)}, min_episodes={memory_min_episodes}):"
            )
            print(format_memory_messages(first_memory_messages))
        elif agent_baseline == "rewoo":
            print("### Planner system prompt:")
            print(build_rewoo_planner_system_prompt(first_tools))
            print("\n### Solver system prompt:")
            print(build_rewoo_solver_system_prompt())
        elif agent_baseline == "reflexion":
            print("### Actor system prompt:")
            print(build_reflexion_actor_system_prompt())
            print("\n### Evaluator system prompt:")
            print(build_reflexion_evaluator_system_prompt())
            print("\n### Self-Reflection system prompt:")
            print(build_reflexion_self_reflection_system_prompt())
        elif agent_baseline == "tree_of_thoughts":
            first_system_prompt = build_system_prompt(agent_baseline, first_tools)
            print("### Baseline system prompt:")
            print(first_system_prompt)
            print("\n### Thought generation system prompt:")
            print(
                build_tot_expand_prompt(
                    tools=first_tools,
                    num_thoughts=tot_num_thoughts,
                    policy_prompt=first_system_prompt,
                )
            )
            print("\n### Thought selection system prompt:")
            print(
                build_tot_judge_prompt(
                    tools=first_tools,
                    policy_prompt=first_system_prompt,
                )
            )
        else:
            first_system_prompt = build_system_prompt(agent_baseline, first_tools)
            print("### System prompt:")
            print(first_system_prompt)
        print("\n### First user query:")
        print(queries[0])
        print()

    results = pd.DataFrame(columns=["query", "function_calls", "full_response", "error"])

    for index, query in enumerate(queries):
        tools = get_query_tools(
            queries_df,
            index,
            tool_selection,
        )
        system_prompt = build_system_prompt(agent_baseline, tools)
        few_shot_messages = None
        if agent_baseline == "few_shot":
            few_shot_messages = build_few_shot_messages_for_query(
                examples_by_domain=few_shot_examples_by_domain,
                current_domains=parse_domains_field(queries_df["domains"].iloc[index]),
                available_tool_names={get_tool_name(tool) for tool in tools},
                few_shot_k=few_shot_k,
                tool_registry=_TOOL_REGISTRY,
                domains=DOMAINS,
            )
        memory_messages = None
        recorded_outcome: dict[str, Any] | None = None
        if agent_baseline == "memory":
            # Cold-start guard: while the verified-correct store is below the
            # minimum, run memory-free instead of seeding the prompt with one
            # or two unrepresentative early traces.
            if len(episodic_memory) >= memory_min_episodes:
                memory_messages = build_memory_messages_for_query(
                    memory=episodic_memory,
                    query=query,
                    memory_k=memory_k,
                    available_tool_names={get_tool_name(tool) for tool in tools},
                    tool_registry=_TOOL_REGISTRY,
                    domains=DOMAINS,
                )
            else:
                print(
                    f"### Memory: cold start ({len(episodic_memory)}/{memory_min_episodes} verified episodes); running without retrieval"
                )
            recorded_outcome = {"actions": [], "final_answer": ""}
        function_calls: list[str] = []
        full_response = ""
        error = ""

        try:
            function_calls, full_response, error = run_agent(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                query=query,
                tools=tools,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                max_new_tokens=max_new_tokens,
                agent_baseline=agent_baseline,
                few_shot_messages=few_shot_messages,
                memory_messages=memory_messages,
                recorded_outcome=recorded_outcome,
                tot_num_thoughts=tot_num_thoughts,
                reflexion_max_trials=reflexion_max_trials,
                reflexion_memory_size=reflexion_memory_size,
            )

            if not function_calls:
                for retry_num in range(num_retries):
                    print(f"No actions taken. Retry {retry_num + 1} of {num_retries}")
                    if recorded_outcome is not None:
                        recorded_outcome = {"actions": [], "final_answer": ""}
                    function_calls, full_response, error = run_agent(
                        model=model,
                        tokenizer=tokenizer,
                        model_id=model_id,
                        query=query,
                        tools=tools,
                        system_prompt=system_prompt,
                        max_iterations=max_iterations,
                        max_new_tokens=max_new_tokens,
                        agent_baseline=agent_baseline,
                        few_shot_messages=few_shot_messages,
                        memory_messages=memory_messages,
                        recorded_outcome=recorded_outcome,
                        tot_num_thoughts=tot_num_thoughts,
                        reflexion_max_trials=reflexion_max_trials,
                        reflexion_memory_size=reflexion_memory_size,
                    )
                    if function_calls:
                        break
        except Exception as exc:
            error = str(exc)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"### Query: {query}")
        print(f"### Answer: {function_calls}")
        if error:
            print(f"### Error: {error}")

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [[query, function_calls, full_response, error]],
                    columns=["query", "function_calls", "full_response", "error"],
                ),
            ],
            ignore_index=True,
        )

        if (
            agent_baseline == "memory"
            and not error
            and recorded_outcome is not None
            and recorded_outcome.get("actions")
        ):
            ground_truth_actions = ground_truth_actions_per_query[index]
            episode_is_correct = is_correct(
                predicted_actions=function_calls,
                ground_truth_actions=ground_truth_actions,
                error="",
            )
            # is_correct compares benchmark state after replay. When ground
            # truth itself is [] (no action needed), any sequence of read-only
            # tool calls also yields the initial state and trivially passes.
            # Those traces are not useful demonstrations and are often the
            # exact failure mode we want to avoid teaching the model (e.g. a
            # fabricated "<name>@example.com" search that happened to be a
            # no-op because the target had no matching records). Require the
            # ground truth to have at least one canonical action so a stored
            # episode actually demonstrates how to act.
            if not episode_is_correct:
                print("### Memory: skipping incorrect episode")
            elif not ground_truth_actions:
                print(
                    "### Memory: skipping no-op-correct episode (ground truth has no actions)"
                )
            else:
                episodic_memory.add(
                    query=query,
                    actions=recorded_outcome["actions"],
                    final_answer=recorded_outcome.get("final_answer", ""),
                )
                episodic_memory.save(memory_path)
                print(
                    f"### Memory: stored verified-correct episode (store size now {len(episodic_memory)})"
                )

        for domain in DOMAINS:
            domain.reset_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    domain = os.path.basename(queries_path).replace("_queries_and_answers.csv", "")
    save_dir = os.path.join(WORKBENCH_ROOT, "data", "results", domain)
    os.makedirs(save_dir, exist_ok=True)
    current_datetime = str(pd.Timestamp.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    mode_label = build_results_mode_label(tool_selection, agent_baseline)
    save_path = os.path.join(save_dir, f"{model_label}_{mode_label}_{current_datetime}.csv")
    results.to_csv(save_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nSaved results to: {save_path}")
    return results


def main() -> None:
    args = parse_args()
    args.queries_path = resolve_path(args.queries_path)
    if args.model_label is None:
        args.model_label = default_model_label(args.model_id)

    model, tokenizer = load_model_and_tokenizer(
        args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    results = generate_results_with_hf(
        model=model,
        tokenizer=tokenizer,
        model_id=args.model_id,
        queries_path=args.queries_path,
        model_label=args.model_label,
        tool_selection=args.tool_selection,
        agent_baseline=args.agent_baseline,
        max_iterations=args.max_iterations,
        max_new_tokens=args.max_new_tokens,
        num_retries=args.num_retries,
        num_queries=args.num_queries,
        print_first_prompt=args.print_first_prompt,
        few_shot_k=args.few_shot_k,
        memory_k=args.memory_k,
        memory_path=args.memory_path,
        memory_min_episodes=args.memory_min_episodes,
        tot_num_thoughts=args.tot_num_thoughts,
        reflexion_max_trials=args.reflexion_max_trials,
        reflexion_memory_size=args.reflexion_memory_size,
    )

    ground_truth = pd.read_csv(args.queries_path)
    if args.num_queries is not None:
        ground_truth = ground_truth.head(args.num_queries).copy()
    ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
    calculate_metrics(ground_truth, results)


if __name__ == "__main__":
    main()
