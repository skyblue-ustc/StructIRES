# Run outputs

Each run gets a unique directory named by experiment ID, date, protocol revision, and seed. A formal run should contain:

```text
run_manifest.json
config.snapshot.json
candidates.raw.jsonl
candidates.evaluated.jsonl
summary.json
logs/
```

The directory contents are ignored by Git. Only reviewed aggregate tables and provenance manifests may later be copied into a release artifact.

