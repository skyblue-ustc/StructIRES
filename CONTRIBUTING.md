# Contributing

## Scientific requirements

- Keep assay audit, full-length seeded optimization, and legacy 174-nt de novo results separate.
- Use the canonical baseline IDs in `configs/baselines.json`.
- Do not compare released external outputs as if they used the same data or query budget.
- Record raw and surviving candidate counts; never report only filtered top candidates.
- Separate optimization scorers from locked evaluation scorers.
- Record the assay context for every function label and evaluator.
- Add a run manifest for every result intended for a paper table.

## Code requirements

1. Add or update tests for registry, schema, or metric changes.
2. Run `python -m unittest discover -s tests -v`.
3. Run `python -m compileall -q src tests`.
4. Do not commit weights, datasets, generated libraries, logs, external repositories, or secrets.
5. Update `THIRD_PARTY.md` before adding an adapter for a new external method.

## Baseline adapters

An adapter must produce canonical candidate JSONL with method, task, seed, parent information, and source provenance. A partially implemented adapter must remain marked `planned` in the registry and cannot appear in a formal result table.
