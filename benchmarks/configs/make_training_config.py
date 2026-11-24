"""Render a runner config for one reviewed candidate and selected workload IDs."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--init-seed", type=int, action="append", required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "benchmarks/backend/data_registry.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.task) != len(set(args.task)) or len(args.init_seed) != len(set(args.init_seed)):
        parser.error("task IDs and initialization seeds must each be distinct")
    registry = load(ROOT / "cal3/metadata_screening/inputs/frozen_candidate_registry.json")
    candidates = {row["candidate_id"]: row for row in registry["candidates"]}
    tasks = {row["task_id"]: row for row in load(HERE / "workloads_118.json")["tasks"]}
    if args.candidate not in candidates or not set(args.task) <= tasks.keys():
        parser.error("unknown candidate or task ID")
    candidate = candidates[args.candidate]
    algorithm = candidate["raw_algorithm"]
    device = "cpu" if candidate["device"] == "cpu" else "gpu"
    config = load(HERE / f"{device}_example.json")
    config.pop("seed", None)
    config["experiment_name"] = "configured_" + args.candidate.replace(":", "_")
    config["algorithm_options"] = {algorithm: candidate.get("options_match", {})}
    config["run_plan"] = [
        {"run_seed": seed, "init_seed": seed, "data_seed": args.data_seed}
        for seed in args.init_seed
    ]
    config["cases"] = []
    registered = load(args.registry)["datasets"]
    for task_id in args.task:
        task = tasks[task_id]
        case = {
            "id": task_id, "enabled": True, "n": task["n"], "d": task["d"],
            "k": task["k"], "max_iter": task["max_iter"],
            "dtype": candidate["compute_dtype"], "algorithms": [algorithm],
        }
        dataset = task["dataset"]
        if dataset == "synthetic":
            case["source"] = "synthetic"
        elif dataset in registered:
            case.update(source="dataset", dataset=dataset)
        else:
            parser.error(f"dataset {dataset!r} lacks a loader entry in --registry")
        config["cases"].append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
