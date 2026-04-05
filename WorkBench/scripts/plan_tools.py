"""
Planner LLM: selects the tools needed to execute a workplace task,
then validates the selection against known tool dependencies and
adds any missing prerequisite tools.

Usage:
    uv run python scripts/plan_tools.py --task "Delete my first meeting on December 13"
    uv run python scripts/plan_tools.py --task "..." --dependencies data/tool_dependencies.json --model gpt-4o
"""

import json
import argparse
import sys
import os
import re

project_root = os.path.abspath(os.path.curdir)
sys.path.append(project_root)

from openai import OpenAI
from src.tools.toolkits import all_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_tool_names() -> list[str]:
    return [tool.openai_schema["function"]["name"] for tool in all_tools]


def load_dependencies(path: str) -> list[dict]:
    """Load the dependency list produced by analyze_tool_dependencies.py."""
    if not os.path.exists(path):
        print(f"WARNING: Dependencies file not found at '{path}'. "
              "Run analyze_tool_dependencies.py first. Skipping dependency check.")
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("dependencies", [])


def build_dependency_map(dependencies: list[dict]) -> dict[str, list[str]]:
    """
    Returns a dict mapping each tool to the list of tools it depends on.
    e.g. {"calendar__delete_event": ["calendar__search_events"], ...}
    """
    dep_map: dict[str, list[str]] = {}
    for d in dependencies:
        tool = d["tool"]
        dep_map.setdefault(tool, [])
        dep_map[tool].append(d["depends_on"])
    return dep_map


def resolve_missing_dependencies(
    selected: list[str],
    dep_map: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """
    Walk selected tools and recursively ensure all prerequisites are present.
    Returns (full_list, added_tools).
    """
    result = list(selected)
    added = []
    # Use a queue so we also check dependencies of newly added tools
    queue = list(selected)
    while queue:
        tool = queue.pop(0)
        for prereq in dep_map.get(tool, []):
            if prereq not in result:
                result.append(prereq)
                added.append(prereq)
                queue.append(prereq)
    return result, added


def extract_tool_names(text: str, valid_names: set[str]) -> list[str]:
    """
    Extract tool names from the LLM response.
    Tries JSON first, then falls back to scanning for known tool name strings.
    """
    # Try JSON array
    json_match = re.search(r"\[[\s\S]*?\]", text)
    if json_match:
        try:
            candidates = json.loads(json_match.group(0))
            if isinstance(candidates, list):
                found = [c for c in candidates if c in valid_names]
                if found:
                    return found
        except json.JSONDecodeError:
            pass

    # Fallback: scan for known tool name substrings in the text
    return [name for name in valid_names if name in text]


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Carefully examine the given workplace task, and return a list of names of the most appropriate and relevant tools. Ensure that returned tools only do what the given task requires and do nothing extra. To be able to use some tools, you need to use prerequisite tools. Therefore, pay attention to the parameters each tool accepts.

Return your answer as a JSON array of tool name strings, e.g. ["tool_a", "tool_b"]. Return nothing else."""

USER_PROMPT_TEMPLATE = """You are going to select tools from the following tools:
{tools_json}

Task: {task}"""


# ---------------------------------------------------------------------------
# Main planner function
# ---------------------------------------------------------------------------

def plan_tools(
    client: OpenAI,
    model: str,
    task: str,
    dep_map: dict[str, list[str]],
) -> dict:
    tool_names = get_all_tool_names()
    valid_names = set(tool_names)

    # Build a compact tool list for the prompt (name + description only)
    tools_summary = [
        {
            "name": t.openai_schema["function"]["name"],
            "description": t.openai_schema["function"]["description"],
        }
        for t in all_tools
    ]

    user_prompt = USER_PROMPT_TEMPLATE.format(
        tools_json=json.dumps(tools_summary, indent=2),
        task=task,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    selected = extract_tool_names(raw, valid_names)

    # Dependency check
    full_list, added = resolve_missing_dependencies(selected, dep_map)

    return {
        "task": task,
        "llm_selected": selected,
        "missing_dependencies_added": added,
        "final_tool_plan": full_list,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        help="The workplace task to plan tools for.")
    parser.add_argument("--dependencies", type=str, default="data/tool_dependencies.json",
                        help="Path to tool_dependencies.json (default: data/tool_dependencies.json)")
    parser.add_argument("--model", type=str, default="gpt-5-nano",
                        help="OpenAI model to use (default: gpt-5-nano)")
    args = parser.parse_args()

    OPENAI_KEY = open("openai_key.txt").read().strip()
    client = OpenAI(api_key=OPENAI_KEY)

    dependencies = load_dependencies(args.dependencies)
    dep_map = build_dependency_map(dependencies)

    result = plan_tools(client, args.model, args.task, dep_map)

    print(f"\nTask: {result['task']}")
    print(f"\nLLM selected ({len(result['llm_selected'])}):")
    for t in result["llm_selected"]:
        print(f"  - {t}")

    if result["missing_dependencies_added"]:
        print(f"\nMissing dependencies added ({len(result['missing_dependencies_added'])}):")
        for t in result["missing_dependencies_added"]:
            print(f"  + {t}")
    else:
        print("\nNo missing dependencies.")

    print(f"\nFinal tool plan ({len(result['final_tool_plan'])}):")
    for t in result["final_tool_plan"]:
        print(f"  ✓ {t}")

    print()


if __name__ == "__main__":
    main()