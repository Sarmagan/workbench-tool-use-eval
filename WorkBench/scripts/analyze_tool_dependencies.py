"""
Analyze tool dependencies using an LLM.

Usage:
    uv run python scripts/analyze_tool_dependencies.py
    uv run python scripts/analyze_tool_dependencies.py --output data/tool_dependencies.json --model gpt-4o
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


def build_enriched_schemas():
    enriched = []
    for tool in all_tools:
        schema = json.loads(json.dumps(tool.openai_schema))
        fn_name = schema["function"]["name"]
        enriched.append(schema)
    return enriched


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are given a list of tool descriptions, tool parameters, and example tasks that can be solved with these tools. Analyze the descriptions and parameters, and match parameters across tools to identify possible dependencies.

Some tasks that are solved by using these tools:
1) "lena and aisha need the last email about 'Update on Team Building Retreat'. Can you forward it?"
2) "can you add Morgan Wilson to the crm? They're a new lead and need to be assigned to Nadia"
3) "Move all of dmitri's tasks that are in progress to in review"
4) "Push back my first meeting with nia on December 12 by 1.5 hours"

Return a list of each tool with its possible dependencies as a JSON object using this exact schema:
{
  "dependencies": [
    {
      "tool": "<tool that has the dependency>",
      "depends_on": "<tool whose output is needed>",
      "reason": "<one sentence: what output from depends_on is used as input to tool, and which parameter>"
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """The tool names, descriptions and their parameters are as follows:

{tools_json}"""


# ---------------------------------------------------------------------------
# LLM call and parsing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> str:
    """Extract a JSON object from the response, with fallback to raw text."""
    json_match = re.search(r"\{[\s\S]*\"dependencies\"[\s\S]*\}", text)
    return json_match.group(0) if json_match else ""


def analyze_dependencies(client: OpenAI, model: str, enriched_schemas: list) -> dict:
    tools_json = json.dumps(enriched_schemas, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(tools_json=tools_json)

    print(f"Sending {len(enriched_schemas)} tool schemas to {model}...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    json_str = extract_json(raw)

    if not json_str:
        print("WARNING: Could not extract JSON from response. Raw output:")
        print(raw)
        return {"dependencies": [], "raw_response": raw}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse JSON: {e}")
        return {"dependencies": [], "raw_response": raw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/tool_dependencies.json",
                        help="Path to save dependency output (default: data/tool_dependencies.json)")
    parser.add_argument("--model", type=str, default="gpt-5-nano",
                        help="OpenAI model to use (default: gpt-5-nano)")
    args = parser.parse_args()

    OPENAI_KEY = open("openai_key.txt", "r").read().strip()
    client = OpenAI(api_key=OPENAI_KEY)

    enriched_schemas = build_enriched_schemas()
    result = analyze_dependencies(client, args.model, enriched_schemas)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    deps = result.get("dependencies", [])
    print(f"\nFound {len(deps)} dependencies. Saved to {args.output}\n")

    for d in deps:
        print(f"  {d['tool']}")
        print(f"    └─ needs {d['depends_on']}")
        print(f"       {d['reason']}")
        print()


if __name__ == "__main__":
    main()