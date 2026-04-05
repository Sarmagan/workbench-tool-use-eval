import re
import os
import json
import csv
import ast
import random
import pandas as pd
from openai import OpenAI

from src.tools import calendar, email, analytics, project_management, customer_relationship_manager, company_directory
from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME
from src.tools.toolkits import (
    calendar_toolkit,
    email_toolkit,
    analytics_toolkit,
    project_management_toolkit,
    customer_relationship_manager_toolkit,
    company_directory_toolkit,
    tools_with_side_effects,
)

DOMAINS = [calendar, email, analytics, project_management, customer_relationship_manager]

AVAILABLE_LLMS = [
    "gpt-5-nano",
]

# --------------------------------------------------------------------------- #
# OpenAI tool-name ↔ Python function mapping
# Tool names in OpenAI use __ as separator (e.g. calendar__create_event)
# which maps to the module.function convention used internally.
# --------------------------------------------------------------------------- #

_TOOL_REGISTRY: dict[str, callable] = {
    fn.openai_schema["function"]["name"]: fn
    for toolkit in [
        calendar_toolkit, email_toolkit, analytics_toolkit,
        project_management_toolkit, customer_relationship_manager_toolkit,
        company_directory_toolkit,
    ]
    for fn in toolkit
}


def _openai_name_to_internal(openai_name: str) -> str:
    """Convert calendar__create_event -> calendar.create_event"""
    return openai_name.replace("__", ".", 1)


def convert_tool_call_to_function_call(tool_name: str, tool_args: dict) -> str:
    """Convert an OpenAI tool call into the string format expected by the benchmark."""
    internal_name = _openai_name_to_internal(tool_name)
    args = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
    return f"{internal_name}.func({args})"


def execute_actions_and_reset_state(actions):
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


def end_date_minor_error(ground_truth, prediction):
    matches = 0
    for func in ground_truth:
        if "2023-11-29" in func:
            if func.replace("2023-11-29", "2023-11-30") in prediction:
                matches += 1
    if len(ground_truth) == 0:
        return False
    return matches == len(ground_truth)


def meeting_start_time_error(ground_truth, prediction):
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


def is_exact_match(predicted_actions, ground_truth_actions):
    tools_with_side_effects_names = [fn.name for fn in tools_with_side_effects]
    predicted_actions_with_side_effects = [
        action for action in predicted_actions if get_function_name(action) in tools_with_side_effects_names
    ]
    predicted_actions_with_side_effects = sorted([action.lower() for action in predicted_actions_with_side_effects])
    ground_truth_actions = sorted([action.lower() for action in ground_truth_actions])
    return predicted_actions_with_side_effects == ground_truth_actions


def get_function_name(action):
    return ".".join(action.split("(")[0].split(".")[0:2])


def is_correct(predicted_actions, ground_truth_actions, error):
    if error:
        return False
    (
        successful_execution,
        predicted_calendar_state, predicted_email_state, predicted_analytics_state,
        predicted_project_management_state, predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)
    (
        _,
        ground_truth_calendar_state, ground_truth_email_state, ground_truth_analytics_state,
        ground_truth_project_management_state, ground_truth_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(ground_truth_actions)

    def convert_strs_to_lowercase(df):
        fields_not_to_convert = ["status", "list_name", "board"]
        for col in df.columns:
            if col not in fields_not_to_convert:
                df[col] = df[col].str.lower()
        return df

    predicted_calendar_state = convert_strs_to_lowercase(predicted_calendar_state)
    predicted_email_state = convert_strs_to_lowercase(predicted_email_state)
    predicted_analytics_state = convert_strs_to_lowercase(predicted_analytics_state)
    predicted_project_management_state = convert_strs_to_lowercase(predicted_project_management_state)
    predicted_customer_relationship_manager_state = convert_strs_to_lowercase(predicted_customer_relationship_manager_state)
    ground_truth_calendar_state = convert_strs_to_lowercase(ground_truth_calendar_state)
    ground_truth_email_state = convert_strs_to_lowercase(ground_truth_email_state)
    ground_truth_analytics_state = convert_strs_to_lowercase(ground_truth_analytics_state)
    ground_truth_project_management_state = convert_strs_to_lowercase(ground_truth_project_management_state)
    ground_truth_customer_relationship_manager_state = convert_strs_to_lowercase(ground_truth_customer_relationship_manager_state)

    return (
        successful_execution
        and predicted_calendar_state.equals(ground_truth_calendar_state)
        and predicted_email_state.equals(ground_truth_email_state)
        and predicted_analytics_state.equals(ground_truth_analytics_state)
        and predicted_project_management_state.equals(ground_truth_project_management_state)
        and predicted_customer_relationship_manager_state.equals(ground_truth_customer_relationship_manager_state)
    )


def extract_function_names(s):
    return re.findall(r"(\b\w+\.\w+)\(", s)


def has_side_effects(predicted_actions, ground_truth_actions):
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
        predicted_calendar_state, predicted_email_state, predicted_analytics_state,
        predicted_project_management_state, predicted_customer_relationship_manager_state,
    ) = execute_actions_and_reset_state(predicted_actions)

    state_changed = not predicted_calendar_state.equals(original_state["calendar"])
    state_changed |= not predicted_email_state.equals(original_state["email"])
    state_changed |= not predicted_analytics_state.equals(original_state["analytics"])
    state_changed |= not predicted_project_management_state.equals(original_state["project_management"])
    state_changed |= not predicted_customer_relationship_manager_state.equals(original_state["customer_relationship_manager"])

    correct = is_correct(predicted_actions, ground_truth_actions, "")
    return state_changed and not correct


def generate_query_and_answer(template):
    logic = template["logic"]()
    if "alternative_queries" in template:
        possible_queries = [template["query"]] + template["alternative_queries"]
        query_template = random.choice(possible_queries)
        query = query_template.format(**logic)
    else:
        query_template = template["query"]
        query = query_template.format(**logic)
    answer = logic["answer"]
    domains = template.get("domains", [])
    return {
        "query": query,
        "answer": answer,
        "base_template": template["query"],
        "chosen_template": query_template,
        "domains": domains,
    }


def generate_all_queries_and_answers(templates, max_queries_per_template, verbose=False):
    generated_queries_and_answers = []
    for template in templates:
        queries_generated_for_template = 0
        while queries_generated_for_template < max_queries_per_template:
            q_and_a = generate_query_and_answer(template)
            queries = [q["query"] for q in generated_queries_and_answers]
            if q_and_a["query"] not in queries:
                generated_queries_and_answers.append(q_and_a)
                queries_generated_for_template += 1

    if verbose:
        for query_and_answer in generated_queries_and_answers:
            print(f"Base template:   {query_and_answer['base_template']}")
            print(f"Chosen template: {query_and_answer['chosen_template']}")
            print(f"Query:           {query_and_answer['query']}")
            print(f"Answer:          {query_and_answer['answer']}")
            print("--------------------------------------------")

    return generated_queries_and_answers


def calculate_metrics(ground_truth_df, predictions_df, print_errors=True):
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
    df["meeting_start_time_error"] = [meeting_start_time_error(gt, pred) for gt, pred in zip(df["ground_truth"], df["prediction"])]
    df["meeting_start_time_error"] = df["meeting_start_time_error"] & ~df["correct"]

    if print_errors:
        print("--------------------------------------------")
        print("ERRORS without unwanted side effects:")
        print("--------------------------------------------")
        for _, row in df[~df["correct"] & ~df["unwanted_side_effects"]].iterrows():
            if not row["wrong_email"] and not row["no_actions"] and not row["end_date_minor_error"] and not row["meeting_start_time_error"]:
                print("--------------------------------------------")
                print(f"Query:\n    {row['query']}")
                print(f"\nPrediction:")
                for action in row["prediction"]:
                    print(f"    {action}")
                print(f"\nGround truth:")
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
                print(f"\nPrediction:")
                for action in row["prediction"]:
                    print(f"    {action}")
                print(f"\nGround truth:")
                for action in row["ground_truth"]:
                    print(f"    {action}")
                print(f"\nError: {row['error']}\n")

    num_errors_without_side_effects = len(df[(~df["correct"]) & ~df["unwanted_side_effects"]])
    num_errors_with_side_effects = len(df[(~df["correct"]) & df["unwanted_side_effects"]])
    print(f"Accuracy: {round(df['correct'].mean() * 100, 2)}% ({df['correct'].sum()} out of {len(df)})")
    print(f"Errors without unwanted side effects: {round(num_errors_without_side_effects / len(df) * 100, 2)}% ({num_errors_without_side_effects} out of {len(df)})")
    print(f"Errors with unwanted side effects: {round(num_errors_with_side_effects / len(df) * 100, 2)}% ({num_errors_with_side_effects} out of {len(df)})")

    return df


def get_output(full_response):
    """Extract the final text output from the stored response string."""
    return full_response


def get_latest_results_path(results_root_dir, model, tool, all_tools_in_prompt=True):
    results_dir = os.path.join(results_root_dir, tool)
    results_files = os.listdir(results_dir)
    model_results_files = [os.path.join(results_dir, file) for file in results_files if model in file]
    if all_tools_in_prompt:
        model_results_files = [file for file in model_results_files if "all" in file]
    else:
        model_results_files = [file for file in model_results_files if "domains" in file]
    ground_truth_path = os.path.join("data", "processed", "queries_and_answers", f"{tool}_queries_and_answers.csv")
    if not model_results_files:
        return None
    return max(model_results_files, key=os.path.getctime), ground_truth_path


def get_latest_results_from_dir(results_root_dir, model, tool, print_errors=False, all_tools_in_prompt=True):
    results = get_latest_results_path(results_root_dir, model, tool, all_tools_in_prompt)
    if not results:
        print(f"\nNo results found for {tool} with {model}")
        return None
    model_results_path, ground_truth_path = results
    predictions = pd.read_csv(model_results_path, dtype=str)
    ground_truth = pd.read_csv(ground_truth_path, dtype=str)
    ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)
    predictions["function_calls"] = predictions["function_calls"].apply(ast.literal_eval)
    print(f"\nCalculating metrics for {tool} with {model}")
    df = calculate_metrics(ground_truth, predictions, print_errors=print_errors)
    num_correct = df["correct"].sum()
    num_incorrect = len(df) - num_correct
    num_side_effects = df["unwanted_side_effects"].sum()
    num_correct_no_actions = df[df["ground_truth"].apply(len) == 0]["correct"].sum()
    num_incorrect_no_actions = len(df[df["ground_truth"].apply(len) == 0]) - num_correct_no_actions
    num_correct_non_zero_actions = df[df["ground_truth"].apply(len) > 0]["correct"].sum()
    num_incorrect_non_zero_actions = len(df[df["ground_truth"].apply(len) > 0]) - num_correct_non_zero_actions
    num_correct_two_or_more_actions = df[df["ground_truth"].apply(len) > 1]["correct"].sum()
    num_incorrect_two_or_more_actions = len(df[df["ground_truth"].apply(len) > 1]) - num_correct_two_or_more_actions
    num_context_window_errors = len(df[df["error"] == "Context window exceeded"])
    return (
        num_correct, num_incorrect, num_side_effects,
        num_correct_no_actions, num_incorrect_no_actions,
        num_correct_non_zero_actions, num_incorrect_non_zero_actions,
        num_correct_two_or_more_actions, num_incorrect_two_or_more_actions,
        num_context_window_errors,
    )


def get_toolkits(toolkits):
    tools = []
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


def _message_to_dict(message) -> dict:
    """
    Convert an OpenAI ChatCompletionMessage object to a plain dict
    suitable for inclusion in the messages list of the next API call.
    """
    d = {"role": message.role, "content": message.content or ""}
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return d


def _run_openai_agent(client: OpenAI, model_name: str, query: str, tools: list, system_prompt: str, max_iterations: int = 20) -> tuple[list[str], str, str]:
    """
    Run a tool-calling loop with the OpenAI Chat Completions API.

    Returns (function_calls, full_response_str, error)
    """
    openai_tools = [fn.openai_schema for fn in tools]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    function_calls = []
    full_response_parts = []
    error = ""

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            # temperature=0,
        )

        message = response.choices[0].message
        full_response_parts.append(str(message))

        # Convert to plain dict before appending — the SDK object is not
        # accepted as a message on subsequent calls.
        messages.append(_message_to_dict(message))

        # If no tool calls, the model is done
        if not message.tool_calls:
            break

        # Execute each tool call and feed results back
        tool_results = []
        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            # Record the function call in benchmark format
            function_calls.append(convert_tool_call_to_function_call(fn_name, fn_args))

            # Actually execute the tool
            fn = _TOOL_REGISTRY.get(fn_name)
            if fn is not None:
                try:
                    result = fn(**fn_args)
                except Exception as e:
                    result = f"Error executing tool: {e}"
            else:
                result = f"Unknown tool: {fn_name}"

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

        messages.extend(tool_results)

    else:
        # Reached max_iterations
        error = "Agent stopped due to iteration limit or time limit."

    return function_calls, "\n".join(full_response_parts), error


def generate_results(queries_path, model_name, tool_selection="all", num_retrys=0):
    """Generates results for a given model and set of queries using the OpenAI Chat Completions API."""

    OPENAI_KEY = open("openai_key.txt", "r").read().strip()
    client = OpenAI(api_key=OPENAI_KEY)

    toolkits = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
    queries_df = pd.read_csv(queries_path)
    queries = queries_df["query"].tolist()

    system_prompt = (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, "
        f"{HARDCODED_CURRENT_TIME.date()} and the current time is {HARDCODED_CURRENT_TIME.time()}. "
        "Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm. "
        "Use the available tools to complete the user's request. "
        "When you have finished all necessary tool calls, provide a brief final answer."
    )

    results = pd.DataFrame(columns=["query", "function_calls", "full_response", "error"])

    tools = get_toolkits(toolkits)

    for i, query in enumerate(queries):
        if tool_selection == "domains":
            query_toolkits = queries_df["domains"].iloc[i].strip("][").replace("'", "").split(", ")
            tools = get_toolkits(query_toolkits)

        error = ""
        function_calls = []
        response_str = ""

        try:
            function_calls, response_str, error = _run_openai_agent(
                client=client,
                model_name=model_name,
                query=query,
                tools=tools,
                system_prompt=system_prompt,
            )

            # Retry if no actions were taken
            if len(function_calls) == 0:
                for retry_num in range(num_retrys):
                    print(f"No actions taken. Retry {retry_num + 1} of {num_retrys}")
                    function_calls, response_str, error = _run_openai_agent(
                        client=client,
                        model_name=model_name,
                        query=query,
                        tools=tools,
                        system_prompt=system_prompt,
                    )
                    if len(function_calls) > 0:
                        break

        except Exception as e:
            context_window_error_messages = [
                "maximum input length",
                "maximum context length",
                "prompt is too long",
                "Request too large",
                "context_length_exceeded",
            ]
            # Always print the full exception so nothing is silently swallowed
            print(f"ERROR on query: {query}")
            print(f"  Exception type : {type(e).__name__}")
            print(f"  Exception detail: {e}")
            if any(msg in str(e) for msg in context_window_error_messages):
                error = "Context window exceeded"
            else:
                error = str(e)

        print(f"### Query: {query}")
        print(f"### Answer: {function_calls}")

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [[query, function_calls, response_str, error]],
                    columns=["query", "function_calls", "full_response", "error"],
                ),
            ],
            ignore_index=True,
        )

        # Reset all data after each query
        for domain in DOMAINS:
            domain.reset_state()

    domain = queries_path.split("/")[-1].split(".")[0].replace("_queries_and_answers", "")
    save_dir = os.path.join("data", "results", domain)
    os.makedirs(save_dir, exist_ok=True)
    current_datetime = str(pd.Timestamp.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    save_path = os.path.join(save_dir, model_name + "_" + tool_selection + "_" + current_datetime + ".csv")
    results.to_csv(save_path, index=False, quoting=csv.QUOTE_ALL)
    return results

# ---------------------------------------------------------------------------
# Tool planner — selects relevant tools per query, then checks dependencies
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = (
    "Carefully examine the given workplace task, and return a list of names of the most "
    "appropriate and relevant tools. Ensure that returned tools only do what the given task "
    "requires and do nothing extra. To be able to use some tools, you need to use prerequisite "
    "tools. Therefore, pay attention to the parameters each tool accepts. "
    "Return your answer as a JSON array of tool name strings. Return nothing else."
)

_PLANNER_USER_TEMPLATE = (
    "You are going to select tools from the following tools:\n{tools_json}\n\nTask: {task}"
)


def _load_dependency_map(dependencies_path: str = "data/tool_dependencies.json") -> dict[str, list[str]]:
    """Load tool_dependencies.json and return a {tool: [depends_on, ...]} map."""
    if not os.path.exists(dependencies_path):
        print(f"WARNING: '{dependencies_path}' not found. Run analyze_tool_dependencies.py first. Skipping dependency check.")
        return {}
    with open(dependencies_path) as f:
        data = json.load(f)
    dep_map: dict[str, list[str]] = {}
    for d in data.get("dependencies", []):
        dep_map.setdefault(d["tool"], []).append(d["depends_on"])
    return dep_map


def _resolve_missing_dependencies(selected: list[str], dep_map: dict[str, list[str]]) -> list[str]:
    """Recursively add any prerequisite tools missing from the selected list."""
    result = list(selected)
    queue = list(selected)
    while queue:
        tool = queue.pop(0)
        for prereq in dep_map.get(tool, []):
            if prereq not in result:
                result.append(prereq)
                queue.append(prereq)
    return result


def _plan_tools_for_query(
    client: OpenAI,
    planner_model: str,
    query: str,
    dep_map: dict[str, list[str]],
) -> list:
    """
    Ask the planner LLM to select tools for a query, validate against the
    dependency map, and return the resolved list of tool functions.
    """
    import re

    all_tool_fns = {fn.openai_schema["function"]["name"]: fn for fn in _TOOL_REGISTRY.values()}
    full_schemas = [fn.openai_schema for fn in all_tool_fns.values()]

    user_prompt = _PLANNER_USER_TEMPLATE.format(
        tools_json=json.dumps(full_schemas, indent=2),
        task=query,
    )

    response = client.chat.completions.create(
        model=planner_model,
        messages=[
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Parse the JSON array from the response
    valid_names = set(all_tool_fns.keys())
    selected_names = []
    json_match = re.search(r"\[[\s\S]*?\]", raw)
    if json_match:
        try:
            candidates = json.loads(json_match.group(0))
            if isinstance(candidates, list):
                selected_names = [c for c in candidates if c in valid_names]
        except json.JSONDecodeError:
            pass
    if not selected_names:
        # Fallback: scan for known tool names in the text
        selected_names = [name for name in valid_names if name in raw]

    # Add missing prerequisites
    resolved_names = _resolve_missing_dependencies(selected_names, dep_map)

    added = [n for n in resolved_names if n not in selected_names]
    if added:
        print(f"  [planner] added missing dependencies: {added}")

    print(f"  [planner] tools selected: {resolved_names}")
    return [all_tool_fns[name] for name in resolved_names if name in all_tool_fns]


def generate_results_with_planned_tools(
    queries_path: str,
    model_name: str,
    planner_model: str = "gpt-5-nano",
    dependencies_path: str = "data/tool_dependencies.json",
    num_retrys: int = 0,
):
    """
    Like generate_results(), but selects tools per query using the planner LLM
    instead of passing the full toolkit. Requires tool_dependencies.json to be
    generated first via scripts/analyze_tool_dependencies.py.

    Parameters
    ----------
    queries_path : str
        Path to the queries-and-answers CSV.
    model_name : str
        The agent model that executes the task.
    planner_model : str
        The model used to plan which tools to use (default: gpt-4o).
    dependencies_path : str
        Path to tool_dependencies.json (default: data/tool_dependencies.json).
    num_retrys : int
        Number of retries if the agent takes no actions.
    """
    OPENAI_KEY = open("openai_key.txt", "r").read().strip()
    client = OpenAI(api_key=OPENAI_KEY)

    dep_map = _load_dependency_map(dependencies_path)

    queries_df = pd.read_csv(queries_path)
    queries = queries_df["query"].tolist()

    system_prompt = (
        f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, "
        f"{HARDCODED_CURRENT_TIME.date()} and the current time is {HARDCODED_CURRENT_TIME.time()}. "
        "Remember the current date and time when answering queries. "
        "Meetings must not start before 9am or end after 6pm. "
        "Use the available tools to complete the user's request. "
        "When you have finished all necessary tool calls, provide a brief final answer."
    )

    results = pd.DataFrame(columns=["query", "function_calls", "full_response", "error"])

    for i, query in enumerate(queries):
        print(f"\n[{i + 1}/{len(queries)}] {query}")

        # Plan tools for this specific query
        tools = _plan_tools_for_query(client, planner_model, query, dep_map)

        error = ""
        function_calls = []
        response_str = ""

        try:
            function_calls, response_str, error = _run_openai_agent(
                client=client,
                model_name=model_name,
                query=query,
                tools=tools,
                system_prompt=system_prompt,
            )

            if len(function_calls) == 0:
                for retry_num in range(num_retrys):
                    print(f"No actions taken. Retry {retry_num + 1} of {num_retrys}")
                    function_calls, response_str, error = _run_openai_agent(
                        client=client,
                        model_name=model_name,
                        query=query,
                        tools=tools,
                        system_prompt=system_prompt,
                    )
                    if len(function_calls) > 0:
                        break

        except Exception as e:
            context_window_error_messages = [
                "maximum input length",
                "maximum context length",
                "prompt is too long",
                "Request too large",
                "context_length_exceeded",
            ]
            print(f"ERROR on query: {query}")
            print(f"  Exception type : {type(e).__name__}")
            print(f"  Exception detail: {e}")
            if any(msg in str(e) for msg in context_window_error_messages):
                error = "Context window exceeded"
            else:
                error = str(e)

        print(f"### Query: {query}")
        print(f"### Answer: {function_calls}")

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    [[query, function_calls, response_str, error]],
                    columns=["query", "function_calls", "full_response", "error"],
                ),
            ],
            ignore_index=True,
        )

        for domain in DOMAINS:
            domain.reset_state()

    domain = queries_path.split("/")[-1].split(".")[0].replace("_queries_and_answers", "")
    save_dir = os.path.join("data", "results", domain)
    os.makedirs(save_dir, exist_ok=True)
    current_datetime = str(pd.Timestamp.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    save_path = os.path.join(save_dir, model_name + "_planned_" + current_datetime + ".csv")
    results.to_csv(save_path, index=False, quoting=csv.QUOTE_ALL)
    return results
