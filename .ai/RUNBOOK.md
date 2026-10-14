# Runbook

## Purpose
This file explains how to install, run, test, debug, and verify this project.

Keep this file practical. A new AI assistant or developer should be able to use it without asking you to repeat setup details.

---

## Environment

### Operating system
- Primary OS:
- Secondary OS / WSL / container:
- Notes:

### Required tools
- Runtime:
- Package manager:
- Database:
- External services:
- AI tools:
  - Claude:
  - Codex:
  - GPT / ChatGPT:

### Required secrets / environment variables
Do not paste secret values here. Only list names and where they should be defined.

```bash
# Example
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DATABASE_URL=
```

---

## Install

```bash
# Add installation commands here
```

Example:

```bash
# npm install
# pip install -r requirements.txt
# poetry install
# uv sync
```

---

## Run locally

```bash
# Add local run command here
```

Example:

```bash
# npm run dev
# python app.py
# docker compose up
```

---

## Test

```bash
# Add test commands here
```

Example:

```bash
# npm test
# pytest
# python -m unittest
```

---

## Lint / format

```bash
# Add lint and format commands here
```

Example:

```bash
# npm run lint
# ruff check .
# black .
# prettier --write .
```

---

## Build

```bash
# Add build command here
```

Example:

```bash
# npm run build
# docker build -t project-name .
```

---

## Useful commands

### Check git status

```bash
git status --short
```

### Show current diff

```bash
git diff
```

### Show staged diff

```bash
git diff --staged
```

### Run Codex review manually

```bash
codex exec "Read AGENTS.md and review the current git diff. Focus on correctness, regressions, security, tests, and maintainability. Do not edit files. Return prioritized findings."
```

---

## Common workflows

### Starting a new AI session

Prompt:

```txt
Read AGENTS.md, CLAUDE.md, .ai/CONTEXT_INDEX.md, .ai/CURRENT_STATE.md, .ai/TASKS.md, .ai/RUNBOOK.md, and .ai/SESSION_HANDOFF.md first.

Then summarize:
1. What this project does
2. Current state
3. Current risks
4. Next safest step

Do not edit files yet.
```

### Starting implementation

Prompt:

```txt
Task: [describe task]

Acceptance criteria:
- [criterion 1]
- [criterion 2]
- [criterion 3]

Before editing:
1. Inspect relevant files.
2. Give a short plan.
3. List files you expect to modify.

After editing:
1. Run relevant checks if available.
2. Update .ai/SESSION_HANDOFF.md.
3. Summarize changed files, commands run, and remaining risks.
```

### Ending a session

Prompt:

```txt
Before we stop, update .ai/SESSION_HANDOFF.md with:
- current goal
- what changed
- files touched
- commands/tests run
- unresolved issues
- exact next recommended prompt

Also update .ai/TASKS.md if task status changed.
```

---

## Debugging guide

### Known issue template

#### Issue
Describe the symptom.

#### How to reproduce

```bash
# commands or steps
```

#### Expected behavior
Describe expected behavior.

#### Actual behavior
Describe actual behavior.

#### Likely cause
Add current hypothesis.

#### Fix status
- [ ] Not started
- [ ] Investigating
- [ ] Fix in progress
- [ ] Fixed
- [ ] Verified

---

## Verification checklist

Before marking work as done:

- [ ] Relevant files were inspected.
- [ ] Change is minimal and targeted.
- [ ] Tests/checks were run, or reason for not running them is documented.
- [ ] Git diff was reviewed.
- [ ] Codex second-eye review was run if available.
- [ ] `.ai/SESSION_HANDOFF.md` was updated.
- [ ] `.ai/TASKS.md` was updated if task status changed.
- [ ] Remaining risks are documented.
