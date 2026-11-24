from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PREPARED_STATUS = "PREPARED_NOT_AUTHORIZED"
FIXTURE_IDENTITY = "SYNTHETIC_FIXTURE"
AUTH_SCHEMA = "downstream_runtime_authorization_v1"
OUTPUT_SCHEMA = "formal_output_namespace_v1"


class ContractError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def deterministic_id(prefix: str, parts: Iterable[Any]) -> str:
    normalized = [str(part) for part in parts]
    digest = sha256_json(normalized)[:20]
    return f"{prefix}-{digest}"


def validate_prepared_manifest(manifest: dict[str, Any]) -> None:
    require(manifest.get("status") == PREPARED_STATUS, "manifest must remain PREPARED_NOT_AUTHORIZED")
    require(manifest.get("scientific_units_executed", 0) == 0, "prepared manifest reports scientific execution")
    require(not manifest.get("authorization"), "authorization cannot be embedded in prepared manifest")
    require(isinstance(manifest.get("target_stage_id"), str), "missing target_stage_id")
    require(isinstance(manifest.get("denominator"), int) and manifest["denominator"] >= 0, "invalid denominator")
    require(isinstance(manifest.get("seeds"), list), "missing seeds")


def accepted_stage_ids(state: dict[str, Any]) -> set[str]:
    return {
        row["stage_id"]
        for row in state.get("stage_statuses", [])
        if isinstance(row, dict) and row.get("status") == "accepted" and isinstance(row.get("stage_id"), str)
    }


def build_runtime_authorization(
    *,
    prepared_manifest: dict[str, Any],
    prepared_sha256: str,
    goal: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    output_root: Path,
    run_id: str,
    host_lock: str,
    input_identities: dict[str, str],
    human_gate_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_prepared_manifest(prepared_manifest)
    target = prepared_manifest["target_stage_id"]
    require(state.get("current_stage_id") == target, f"active stage is not {target}")
    require(state.get("phase_status") == "running", "active stage is not running")
    require(state.get("active_plan", {}).get("plan_id") == plan.get("plan_id"), "state/plan ID mismatch")
    require(state.get("active_plan", {}).get("version") == plan.get("version"), "state/plan version mismatch")
    require(plan.get("goal_ref", {}).get("goal_id") == goal.get("goal_id"), "goal ID mismatch")
    require(plan.get("goal_ref", {}).get("goal_version") == goal.get("version"), "goal version mismatch")
    stage = next((row for row in plan.get("stages", []) if row.get("id") == target), None)
    require(stage is not None, "target stage is absent from plan")
    missing_deps = sorted(set(stage.get("dependencies", [])) - accepted_stage_ids(state))
    require(not missing_deps, f"stage dependencies not accepted: {missing_deps}")
    require(Path(output_root).is_absolute(), "output root must be absolute")
    require(bool(run_id.strip()), "run_id is required")
    require(bool(host_lock.strip()), "host_lock is required")
    required_gates = set(prepared_manifest.get("required_human_gates", []))
    supplied_gates = {row.get("gate_id") for row in human_gate_receipts if row.get("status") == "approved"}
    require(required_gates <= supplied_gates, f"missing human gates: {sorted(required_gates - supplied_gates)}")
    require(not prepared_manifest.get("unbound_inputs"), f"unbound inputs remain: {prepared_manifest.get('unbound_inputs')}")
    payload = {
        "schema": AUTH_SCHEMA,
        "status": "AUTHORIZED",
        "goal_ref": {"goal_id": goal["goal_id"], "version": goal["version"]},
        "plan_ref": {"plan_id": plan["plan_id"], "version": plan["version"]},
        "stage_id": target,
        "prepared_manifest_sha256": prepared_sha256,
        "denominator": prepared_manifest["denominator"],
        "seeds": prepared_manifest["seeds"],
        "input_identities": dict(sorted(input_identities.items())),
        "output_root": str(output_root),
        "run_id": run_id,
        "host_lock": host_lock,
        "human_gate_receipts": human_gate_receipts,
    }
    payload["authorization_id"] = deterministic_id("AUTH", [sha256_json(payload)])
    return payload


def validate_runtime_authorization(
    *, manifest: dict[str, Any], manifest_sha256: str, authorization: dict[str, Any]
) -> None:
    validate_prepared_manifest(manifest)
    require(authorization.get("schema") == AUTH_SCHEMA, "wrong authorization schema")
    require(authorization.get("status") == "AUTHORIZED", "authorization is not active")
    require(isinstance(authorization.get("goal_ref"), dict), "authorization goal_ref missing")
    require(isinstance(authorization.get("plan_ref"), dict), "authorization plan_ref missing")
    require(authorization.get("stage_id") == manifest.get("target_stage_id"), "authorization stage mismatch")
    require(authorization.get("prepared_manifest_sha256") == manifest_sha256, "prepared manifest hash mismatch")
    require(authorization.get("denominator") == manifest.get("denominator"), "authorization denominator mismatch")
    require(authorization.get("seeds") == manifest.get("seeds"), "authorization seed mismatch")
    require(Path(str(authorization.get("output_root", ""))).is_absolute(), "authorization output root is not absolute")
    require(bool(authorization.get("run_id")), "authorization run_id missing")
    require(bool(authorization.get("host_lock")), "authorization host_lock missing")


def validate_runtime_context(
    *, authorization: dict[str, Any], goal: dict[str, Any], plan: dict[str, Any], state: dict[str, Any]
) -> None:
    stage_id = authorization.get("stage_id")
    require(state.get("phase_status") == "running", "current research phase is not running")
    require(state.get("current_stage_id") == stage_id, "authorization is stale for the current stage")
    require(authorization.get("goal_ref") == {"goal_id": goal.get("goal_id"), "version": goal.get("version")}, "authorization goal version is stale")
    require(authorization.get("plan_ref") == {"plan_id": plan.get("plan_id"), "version": plan.get("version")}, "authorization plan version is stale")
    require(state.get("active_plan", {}).get("plan_id") == plan.get("plan_id"), "state/plan ID mismatch")
    require(state.get("active_plan", {}).get("version") == plan.get("version"), "state/plan version mismatch")
    stage = next((row for row in plan.get("stages", []) if row.get("id") == stage_id), None)
    require(stage is not None, "authorization target stage is absent from current plan")
    missing_deps = sorted(set(stage.get("dependencies", [])) - accepted_stage_ids(state))
    require(not missing_deps, f"current dependencies not accepted: {missing_deps}")


def output_namespace_header(manifest: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    validate_runtime_authorization(
        manifest=manifest,
        manifest_sha256=authorization["prepared_manifest_sha256"],
        authorization=authorization,
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "protocol": manifest.get("protocol_id"),
        "stage_id": manifest["target_stage_id"],
        "tranche": manifest.get("tranche_id"),
        "seeds": manifest["seeds"],
        "run_id": authorization["run_id"],
        "authorization_id": authorization["authorization_id"],
        "prepared_manifest_sha256": authorization["prepared_manifest_sha256"],
    }
