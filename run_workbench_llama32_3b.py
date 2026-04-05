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
DEFAULT_QUERIES_PATH = os.path.join(
    WORKBENCH_ROOT,
    "data",
    "processed",
    "queries_and_answers",
    "calendar_queries_and_answers.csv",
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
        "--max_iterations",
        type=int,
        default=12,
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


def build_system_prompt() -> str:
    return (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, "
        f"{HARDCODED_CURRENT_TIME.date()} and the current time is "
        f"{HARDCODED_CURRENT_TIME.time()}. "
        "Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm. "
        "Use the available tools to complete the user's request. "
        "When a tool is needed, return a tool call. "
        "When you have finished all necessary tool calls, provide a brief final answer."
    )


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


def generate_assistant_turn(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_new_tokens: int,
    model_id: str,
) -> str:
    chat_template_kwargs: dict[str, Any] = {
        "tools": tools,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
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
) -> tuple[list[str], str, str]:
    tool_schemas = [fn.openai_schema for fn in tools]
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


def get_query_tools(queries_df: pd.DataFrame, index: int, tool_selection: str) -> list[Any]:
    if tool_selection != "domains":
        return get_toolkits(
            [
                "email",
                "calendar",
                "analytics",
                "project_management",
                "customer_relationship_manager",
            ]
        )

    query_toolkits = (
        queries_df["domains"]
        .iloc[index]
        .strip("][")
        .replace("'", "")
        .split(", ")
    )
    return get_toolkits(query_toolkits)


def generate_results_with_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_id: str,
    queries_path: str,
    model_label: str,
    tool_selection: str,
    max_iterations: int,
    max_new_tokens: int,
    num_retries: int,
    num_queries: int | None,
    print_first_prompt: bool,
) -> pd.DataFrame:
    queries_df = pd.read_csv(queries_path)
    if num_queries is not None:
        queries_df = queries_df.head(num_queries).copy()
    queries = queries_df["query"].tolist()
    system_prompt = build_system_prompt()

    if print_first_prompt and queries:
        print("### System prompt:")
        print(system_prompt)
        print("\n### First user query:")
        print(queries[0])
        print()

    results = pd.DataFrame(columns=["query", "function_calls", "full_response", "error"])

    for index, query in enumerate(queries):
        tools = get_query_tools(queries_df, index, tool_selection)
        function_calls: list[str] = []
        full_response = ""
        error = ""

        try:
            function_calls, full_response, error = run_hf_agent(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                query=query,
                tools=tools,
                system_prompt=system_prompt,
                max_iterations=max_iterations,
                max_new_tokens=max_new_tokens,
            )

            if not function_calls:
                for retry_num in range(num_retries):
                    print(f"No actions taken. Retry {retry_num + 1} of {num_retries}")
                    function_calls, full_response, error = run_hf_agent(
                        model=model,
                        tokenizer=tokenizer,
                        model_id=model_id,
                        query=query,
                        tools=tools,
                        system_prompt=system_prompt,
                        max_iterations=max_iterations,
                        max_new_tokens=max_new_tokens,
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

        for domain in DOMAINS:
            domain.reset_state()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    domain = os.path.basename(queries_path).replace("_queries_and_answers.csv", "")
    save_dir = os.path.join(WORKBENCH_ROOT, "data", "results", domain)
    os.makedirs(save_dir, exist_ok=True)
    current_datetime = str(pd.Timestamp.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    save_path = os.path.join(save_dir, f"{model_label}_{tool_selection}_{current_datetime}.csv")
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
        max_iterations=args.max_iterations,
        max_new_tokens=args.max_new_tokens,
        num_retries=args.num_retries,
        num_queries=args.num_queries,
        print_first_prompt=args.print_first_prompt,
    )

    ground_truth = pd.read_csv(args.queries_path)
    if args.num_queries is not None:
        ground_truth = ground_truth.head(args.num_queries).copy()
    ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
    calculate_metrics(ground_truth, results)


if __name__ == "__main__":
    main()
