# AGENTS.md

Scratch files are stored in `.scratch/`, do not delete anything within it without permission.

## Commit conventions

Use [conventional commits](https://www.conventionalcommits.org) for commit messages and PR titles (PRs are squash-merged, so the PR title becomes the commit message): `feat:` triggers a minor version bump, `fix:` a patch bump, and a `!` suffix (e.g. `feat!:`) marks a breaking change. release-plz derives version bumps and changelogs from these.

## Pre-commit hooks

This repo uses [prek](https://github.com/j178/prek) for git hooks (`.pre-commit-config.yaml`). Install once with `prek install`; validate the whole tree with `prek run --all-files`. At pre-commit, `cargo fmt --all` and `cargo clippy --fix` mirror the CI `Rustfmt & Clippy` job and fix what they can in place: a hook that modifies files fails the commit, but the fixes are already on disk — re-stage and retry. Never bypass the hooks with `--no-verify`; CI runs the same checks.

## Agent skills

### Issue tracker

Issues and PRDs live in this repo's GitHub Issues, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
