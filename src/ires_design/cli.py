"""Command-line interface for registry, schema, and low-cost audit operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import generate_random_mutants
from .baselines import DEFAULT_REGISTRY, REPOSITORY_ROOT, load_registry
from .experiments import load_and_validate_experiment
from .io import iter_fasta, load_external_records, write_jsonl
from .metrics import summarize_sequences
from .provenance import build_run_manifest


def _print_baselines(registry_path: Path, check_assets: bool) -> int:
    specs = load_registry(registry_path)
    header = ("ID", "TASKS", "ROLE", "ADAPTER", "STATUS")
    rows: list[tuple[str, ...]] = []
    failures = 0
    for spec in specs:
        status, asset_path = spec.availability(REPOSITORY_ROOT)
        display_status = status
        if asset_path is not None:
            display_status = f"{status}: {asset_path}"
        rows.append((spec.id, ",".join(spec.tasks), spec.role, spec.adapter, display_status))
        if check_assets and status == "asset_missing":
            failures += 1
    widths = [max(len(row[index]) for row in [header, *rows]) for index in range(len(header))]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return 1 if failures else 0


def _cmd_audit_fasta(args: argparse.Namespace) -> int:
    sequences = [sequence for _, sequence in iter_fasta(args.input)]
    summary = summarize_sequences(sequences, args.expected_length)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if summary.get("n_raw", 0) > 0 else 1


def _cmd_normalize_output(args: argparse.Namespace) -> int:
    records = load_external_records(
        args.input,
        method=args.method,
        task=args.task,
        seed=args.seed,
        sequence_column=args.sequence_column,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} canonical candidates to {args.output}")
    return 0


def _cmd_random_mutate(args: argparse.Namespace) -> int:
    parents = list(iter_fasta(args.input))
    records = generate_random_mutants(
        parents,
        num_per_parent=args.num_per_parent,
        n_mutations=args.n_mutations,
        seed=args.seed,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} random-mutation candidates to {args.output}")
    return 0


def _cmd_validate_experiment(args: argparse.Namespace) -> int:
    payload = load_and_validate_experiment(args.config, args.registry)
    print(
        f"valid experiment: {payload['experiment_id']} "
        f"task={payload['task']} baselines={len(payload['baseline_ids'])}"
    )
    return 0


def _cmd_create_manifest(args: argparse.Namespace) -> int:
    load_and_validate_experiment(args.config, args.registry)
    manifest = build_run_manifest(args.config, REPOSITORY_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote run manifest to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ires-design")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baselines", help="inspect the canonical baseline registry")
    baseline_parser.add_argument("action", choices=("list", "check"))
    baseline_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)

    audit_parser = subparsers.add_parser("audit-fasta", help="run dependency-light sequence audits")
    audit_parser.add_argument("input", type=Path)
    audit_parser.add_argument("--expected-length", type=int, default=None)
    audit_parser.add_argument("--output", type=Path, default=None)
    audit_parser.set_defaults(handler=_cmd_audit_fasta)

    normalize_parser = subparsers.add_parser(
        "normalize-output", help="convert external FASTA/CSV into canonical candidate JSONL"
    )
    normalize_parser.add_argument("input", type=Path)
    normalize_parser.add_argument("output", type=Path)
    normalize_parser.add_argument("--method", required=True)
    normalize_parser.add_argument("--task", choices=("de_novo", "mutation"), required=True)
    normalize_parser.add_argument("--seed", type=int, required=True)
    normalize_parser.add_argument("--sequence-column", default=None)
    normalize_parser.set_defaults(handler=_cmd_normalize_output)

    mutation_parser = subparsers.add_parser("random-mutate", help="run the built-in random baseline")
    mutation_parser.add_argument("input", type=Path)
    mutation_parser.add_argument("output", type=Path)
    mutation_parser.add_argument("--num-per-parent", type=int, default=10)
    mutation_parser.add_argument("--n-mutations", type=int, default=3)
    mutation_parser.add_argument("--seed", type=int, default=42)
    mutation_parser.set_defaults(handler=_cmd_random_mutate)

    validate_parser = subparsers.add_parser("validate-experiment", help="validate an experiment contract")
    validate_parser.add_argument("config", type=Path)
    validate_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    validate_parser.set_defaults(handler=_cmd_validate_experiment)

    manifest_parser = subparsers.add_parser("create-manifest", help="initialize a run manifest")
    manifest_parser.add_argument("config", type=Path)
    manifest_parser.add_argument("output", type=Path)
    manifest_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    manifest_parser.set_defaults(handler=_cmd_create_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "baselines":
        return _print_baselines(args.registry, check_assets=args.action == "check")
    return args.handler(args)

