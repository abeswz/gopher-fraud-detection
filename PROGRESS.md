# PROGRESS

**Last updated:** 2026-05-30

## Status

Base scaffold created. No real detection logic yet — `CalculateFraudScore` returns a hardcoded stub response.

## What exists

- HTTP server on unix socket (`cmd/api/main.go`)
- Routes: `GET /ready`, `POST /fraud-score` (`internal/router`, `internal/handler`)
- DTOs (`internal/dto/fraud.go`)
- Stub service (`internal/service/fraud_detection.go`)
- Design spec at `docs/superpowers/specs/2026-05-30-gopher-fraud-detection-design.md`
- CLAUDE.md with architecture, rules, and code standards

## What is missing

- `internal/vectorizer` — Vectorize(req, norm, mcc) → [14]float32
- `internal/search` — LoadIndex + KNN over int16 binary index
- `ml/build_index.py` — references.json.gz → index/references.bin
- Wire vectorizer + search into service
- Readiness gate (block /ready until index loaded)
- Tests for vectorizer and KNN
- Makefile
- Dockerfile update for index copy

## Last test result

Not run yet.
