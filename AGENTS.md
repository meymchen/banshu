# AGENTS.md

## Commit conventions

Use [conventional commits](https://www.conventionalcommits.org) for commit messages and PR titles (PRs are squash-merged, so the PR title becomes the commit message): `feat:` triggers a minor version bump, `fix:` a patch bump, and a `!` suffix (e.g. `feat!:`) marks a breaking change. release-plz derives version bumps and changelogs from these.

## Agent skills

### Issue tracker

Issues and PRDs live in this repo's GitHub Issues, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
