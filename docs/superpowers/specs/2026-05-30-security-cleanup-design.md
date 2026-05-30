# Security Cleanup — Public Repo Hardening

**Date:** 2026-05-30  
**Status:** Approved  

## Problem

Repository is going public. Three issues require resolution before that:

1. `ml/build_index.py` and `ml/pyproject.toml` are tracked in git history (added in commit `15e67fb` before `.gitignore` got `ml/`). Visible on `main` and in merged PR #1. Must be eliminated.
2. `Makefile` exposes `IMAGE_REPO`, `PARTICIPANT`, and `RINHA_REPO` inline — identifies contestant and container registry path.
3. `info.json` has wrong `stack` (`"rust"` instead of `"go"`) and wrong `source-code-repo` URL.

Remote repo was deleted and recreated clean. No rewrite tools needed.

---

## Design

### 1. Git History Reset

Destroy local `.git` and reinitialize:

```bash
rm -rf .git
git init
git remote add origin git@github.com:abeswz/gopher-fraud-detection.git
```

`ml/` is already in `.gitignore` — files will not be staged in the new history. Single clean initial commit pushed to `main`. All previous commit SHAs, PR #1 artifacts, and the `feature/vector` / `submission` remote branches are gone (remote was recreated).

### 2. Submission Script

Create `./references/tools/submission.sh` — this path falls under `references/` which is already gitignored.

The script receives all logic currently in the `make submission` target:
- `IMAGE_REPO`, `PARTICIPANT`, `RINHA_REPO` variables defined here
- `docker build` + `docker push`
- Git orphan branch creation (`submission`) with sed-substituted `docker-compose.yml`
- Force push `submission` branch to origin
- `gh issue create` on the rinha contest repo
- Returns to original branch on exit (and on error via `trap`)

Script must be executable (`chmod +x`).

Makefile `submission` target becomes a stub:

```makefile
submission:
	@echo "Use: ./references/tools/submission.sh"
	@exit 1
```

`index` and `bench` targets are unaffected.

### 3. info.json Fix

Before the initial commit, update:

| Field | Old | New |
|---|---|---|
| `stack` | `["rust", "haproxy"]` | `["go", "haproxy"]` |
| `source-code-repo` | `https://github.com/abeswz/fraud-detection-rinha-backend-2026` | `https://github.com/abeswz/gopher-fraud-detection` |

---

## Security Audit — Public Repo

| Item | Risk | Status |
|---|---|---|
| `ml/build_index.py` + `ml/pyproject.toml` | Was tracked, eliminates index-building strategy | Eliminated by git reset |
| `Makefile` IMAGE_REPO / PARTICIPANT | Exposes GHCR path + contestant identity | Moved to gitignored script |
| `info.json` stack / URL | Wrong data, misleading | Fixed before push |
| `references/`, `resources/`, `test/` | Gitignored | Safe |
| `index/` | Gitignored (87 MB binary) | Safe |
| `CLAUDE.md` | Reveals AI-assisted development — not a security risk | Acceptable, keep |
| No secrets, tokens, credentials found | — | Safe |

---

## Success Criteria

- `git ls-files ml/` returns empty on new repo
- `make submission` prints usage and exits 1
- `./references/tools/submission.sh` contains all submission logic and is executable
- `info.json` has correct stack and repo URL
- `git log --oneline` shows single clean commit
- Repo is safe to set public
