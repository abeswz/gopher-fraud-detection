# Security Cleanup — Public Repo Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `ml/` from git history, hide submission secrets in a gitignored script, fix `info.json`, and push a single clean commit to the recreated remote.

**Architecture:** Remote repo was deleted and recreated clean. Approach is: fix files first, then destroy `.git`, reinitialize, and push a single clean commit. The submission script lives in `references/tools/` which is gitignored — it is never committed.

**Tech Stack:** Go, bash, git, gh CLI, docker, uv

---

## File Map

| Action | Path | What changes |
|---|---|---|
| Modify | `info.json` | Fix `stack` and `source-code-repo` |
| Modify | `Makefile` | Remove secret vars, stub `submission` target |
| Create (not committed) | `references/tools/submission.sh` | Full submission logic, executable |
| Destroy + recreate | `.git/` | Fresh history, no ml/ artifacts |

---

## Task 1: Fix info.json

**Files:**
- Modify: `info.json`

- [ ] **Step 1: Verify current wrong values**

```bash
cat info.json
```

Expected output contains `"rust"` and `fraud-detection-rinha-backend-2026`.

- [ ] **Step 2: Write correct values**

Replace `info.json` with:

```json
{
    "participants": ["abesnow"],
    "social": ["https://github.com/abeswz", "https://www.linkedin.com/in/anves"],
    "source-code-repo": "https://github.com/abeswz/gopher-fraud-detection",
    "stack": ["go", "haproxy"],
    "open_to_work": false
}
```

- [ ] **Step 3: Verify**

```bash
cat info.json | jq '.stack, .["source-code-repo"]'
```

Expected:
```
["go","haproxy"]
"https://github.com/abeswz/gopher-fraud-detection"
```

---

## Task 2: Create submission script

**Files:**
- Create (not committed): `references/tools/submission.sh`

`references/` is gitignored. This file will never appear in git. Create the directory and script manually.

- [ ] **Step 1: Create directory**

```bash
mkdir -p references/tools
```

- [ ] **Step 2: Write the script**

Create `references/tools/submission.sh` with this exact content:

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE_REPO="ghcr.io/abeswz/gopher-fraud-detection"
PARTICIPANT="abeswz-gopher"
RINHA_REPO="zanfranceschi/rinha-de-backend-2026"

GIT_SHA=$(git rev-parse --short HEAD)
IMAGE="${IMAGE_REPO}:${GIT_SHA}"

# Build index
make index

# Build and push image
docker build --network=host -t "${IMAGE}" .
docker push "${IMAGE}"

# Save original branch
ORIG=$(git rev-parse --abbrev-ref HEAD)

# Return to original branch on exit (even on error)
trap 'git checkout "${ORIG}" 2>/dev/null || true' EXIT

# Prepare submission files
sed "s|build: \.|image: ${IMAGE}|" docker-compose.yml > /tmp/sub-compose.yml
cp info.json /tmp/sub-info.json
cp haproxy.cfg /tmp/sub-haproxy.cfg

# Create orphan submission branch
git checkout --orphan submission-tmp
git rm -rf . > /dev/null 2>&1

cp /tmp/sub-compose.yml docker-compose.yml
cp /tmp/sub-info.json info.json
cp /tmp/sub-haproxy.cfg haproxy.cfg

git add docker-compose.yml info.json haproxy.cfg
git commit -m "submission: ${GIT_SHA}"

git branch -D submission 2>/dev/null || true
git branch -m submission-tmp submission
git push origin submission --force

git checkout "${ORIG}"
echo "pushed submission branch: ${IMAGE}"

# Open test issue on rinha repo
gh issue create \
  --repo "${RINHA_REPO}" \
  --title "rinha/test ${PARTICIPANT}" \
  --body "rinha/test ${PARTICIPANT}"
```

- [ ] **Step 3: Make executable**

```bash
chmod +x references/tools/submission.sh
```

- [ ] **Step 4: Verify not tracked by git**

```bash
git status references/
```

Expected: nothing shown (path is gitignored). If it shows up, stop — `.gitignore` has a problem.

---

## Task 3: Update Makefile

**Files:**
- Modify: `Makefile`

Remove secret variables (`IMAGE_REPO`, `GIT_SHA`, `IMAGE`, `PARTICIPANT`, `RINHA_REPO`). Replace `submission` target with stub. `PORT` and `READY_TIMEOUT` stay — they are used by `bench`.

- [ ] **Step 1: Verify `bench` does not use IMAGE_REPO or PARTICIPANT**

```bash
grep -n 'IMAGE_REPO\|PARTICIPANT\|RINHA_REPO\|GIT_SHA\b\|IMAGE\b' Makefile
```

All matches should be only in lines 1–3, 6–7, and the `submission` target. Lines 14–22 (`bench`) must not appear in output.

- [ ] **Step 2: Write new Makefile**

Replace `Makefile` with:

```makefile
PORT           := 9999
READY_TIMEOUT  := 300

.PHONY: index bench submission

index:
	uv run ml/build_index.py

bench: index
	docker compose --compatibility down
	docker compose --compatibility up --build --force-recreate -d
	@i=0; until curl -sf http://localhost:$(PORT)/ready >/dev/null 2>&1; do \
		printf '.'; sleep 1; i=$$((i+1)); \
		[ $$i -ge $(READY_TIMEOUT) ] && echo " timeout" && exit 1; \
	done; echo " ready"
	k6 run test/test.js
	@jq -r '"p99:\(.p99) score:\(.scoring.final_score) FP:\(.scoring.breakdown.false_positive_detections) FN:\(.scoring.breakdown.false_negative_detections) ERR:\(.scoring.breakdown.http_errors)"' test/results.json

submission:
	@echo "Use: ./references/tools/submission.sh"
	@exit 1
```

**Note:** The indentation in `Makefile` must use hard tabs, not spaces. Verify with `cat -A Makefile | grep '^\^I'` — each recipe line must start with `^I`.

- [ ] **Step 3: Verify stub behavior**

```bash
make submission; echo "exit: $?"
```

Expected output:
```
Use: ./references/tools/submission.sh
make: *** [Makefile:20: submission] Error 1
exit: 2
```

- [ ] **Step 4: Verify `make index` still works (dry run)**

```bash
make -n index
```

Expected:
```
uv run ml/build_index.py
```

---

## Task 4: Reset git history and push clean state

**Files:**
- Destroy + recreate: `.git/`

Remote repo was already deleted and recreated at `git@github.com:abeswz/gopher-fraud-detection.git`. This task destroys local history, reinitializes, stages everything (gitignore will exclude `ml/`, `index/`, `references/`, `resources/`, `test/`), and pushes one clean commit.

- [ ] **Step 1: Verify ml/ is in .gitignore**

```bash
grep '^ml/' .gitignore
```

Expected: `ml/` — must be present or ml files will be staged.

- [ ] **Step 2: Destroy git history**

```bash
rm -rf .git
```

> **Warning:** This is irreversible. All local commits and branches are gone. Remote was already recreated clean — this is intentional.

- [ ] **Step 3: Reinitialize**

```bash
git init
git remote add origin git@github.com:abeswz/gopher-fraud-detection.git
```

- [ ] **Step 4: Verify ml/ will NOT be staged**

```bash
git status --short | grep '^?? ml/'
```

Expected: no output. If `ml/` appears here, stop — fix `.gitignore` before continuing.

- [ ] **Step 5: Verify no secrets will be staged**

```bash
git status --short
```

Scan the output. Must NOT contain:
- Any file under `ml/`
- Any file under `references/`
- Any file under `resources/`
- Any file under `test/`
- Any file under `index/`

Only expected staged candidates: `cmd/`, `internal/`, `docs/`, `Makefile`, `Dockerfile`, `.dockerignore`, `.gitignore`, `docker-compose.yml`, `go.mod`, `haproxy.cfg`, `info.json`, `CLAUDE.md`, `PROGRESS.md`, `README.md`

- [ ] **Step 6: Stage all non-ignored files**

```bash
git add .
```

- [ ] **Step 7: Verify ml/ is not staged**

```bash
git ls-files ml/
```

Expected: empty output. If any path appears, stop — do NOT commit.

- [ ] **Step 8: Commit**

```bash
git commit -m "feat: initial public release — fraud detection API"
```

- [ ] **Step 9: Push**

```bash
git push -u origin main
```

Expected: clean push with no errors.

- [ ] **Step 10: Verify on remote**

```bash
git log --oneline
```

Expected: exactly one commit.

```bash
gh browse --no-browser 2>/dev/null || echo "check https://github.com/abeswz/gopher-fraud-detection"
```

Open the GitHub repo and confirm:
- Only one commit in history
- No `ml/` folder visible in file tree
- No `feature/vector` or `submission` branches listed
