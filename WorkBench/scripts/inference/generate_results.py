import pandas as pd
import argparse
import ast
import sys
import os

project_root = os.path.abspath(os.path.curdir)
sys.path.append(project_root)

from src.evals.utils import AVAILABLE_LLMS, generate_results, generate_results_with_planned_tools, calculate_metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    help="model name. Must be one of " + ", ".join(AVAILABLE_LLMS),
    required=True,
)
parser.add_argument(
    "--queries_path",
    type=str,
    help="path to queries and answers csv. By default these are stored in data/processed/queries_and_answers/",
    required=True,
)
parser.add_argument(
    "--toolkits",
    action="append",
    nargs="*",
    help="toolkits to be used for generating answers. By default all toolkits are used.",
    default=[],
)
parser.add_argument(
    "--tool_selection",
    type=str,
    help="tool selection method. One of: 'all', 'domains', 'planned'",
    default="all",
)
parser.add_argument(
    "--planner_model",
    type=str,
    help="model used to plan tools when --tool_selection planned is set (default: gpt-4o)",
    default="gpt-5-nano",
)
parser.add_argument(
    "--dependencies_path",
    type=str,
    help="path to tool_dependencies.json, used when --tool_selection planned is set (default: data/tool_dependencies.json)",
    default="data/tool_dependencies.json",
)

args = parser.parse_args()

if __name__ == "__main__":
    ground_truth = pd.read_csv(args.queries_path)
    ground_truth["answer"] = ground_truth["answer"].apply(ast.literal_eval)

    if args.tool_selection == "planned":
        results = generate_results_with_planned_tools(
            queries_path=args.queries_path,
            model_name=args.model_name,
            planner_model=args.planner_model,
            dependencies_path=args.dependencies_path,
        )
    else:
        results = generate_results(args.queries_path, args.model_name, args.tool_selection)

    calculate_metrics(ground_truth, results)