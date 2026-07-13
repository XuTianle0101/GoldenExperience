# Contributing

GoldenExperience changes must preserve the fail-closed runtime contract and the separation
between development, semantic, and runtime approval evidence.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the same gates used by CI before committing:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy goldenexperience
python -m coverage run -m pytest
python -m coverage report
python -m build
```

Use focused commits with the repository's existing prefixes, such as `[fix]`, `[feat]`,
`[test]`, `[docs]`, `[data]`, and `[quality]`. Keep generated model weights, raw runtime
caches, and machine-specific payloads out of Git.

## Research Integrity

- Never use the semantic sealed split for model selection or threshold tuning.
- Keep validation candidates non-authoritative until all manifest gates pass.
- Record dataset, code, model, transport, predictor, and threshold hashes for evidence.
- Preserve raw result provenance and document any rejected or superseded experiment.
- Do not weaken runtime fallback or artifact-authority checks to make an experiment pass.

Security-sensitive runtime changes must include tests for corrupt input, partial failure,
identity mismatch, and safe native-prefill fallback.
