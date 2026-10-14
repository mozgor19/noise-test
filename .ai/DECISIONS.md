# Decisions

This file records stable project decisions so they do not get lost across AI sessions.

Use it for decisions that future assistants or developers should not re-litigate unless there is a strong reason.

---

## Decision template

```md
## YYYY-MM-DD - Decision title

### Decision
What was decided?

### Reason
Why was this chosen?

### Alternatives considered
- Alternative 1:
- Alternative 2:
- Alternative 3:

### Consequences
What does this imply for future work?

### Status
Accepted / Superseded / Rejected / Under review
```

---

## Active decisions

<!-- Add decisions below this line. Newest first. -->

## YYYY-MM-DD - AI context must live inside the repository

### Decision
Project memory should be stored in repository files such as `AGENTS.md`, `CLAUDE.md`, and `.ai/` documents, not only in chat history.

### Reason
AI sessions can expire, fill up, or move between tools. File-based context allows Claude, Codex, GPT, and future assistants to reconstruct the project state without requiring the user to explain everything again.

### Alternatives considered
- Relying on Claude Desktop chat history.
- Pasting a long project description at the start of every session.
- Keeping notes outside the repository.

### Consequences
Every meaningful task should update `.ai/SESSION_HANDOFF.md`. Stable decisions should be appended here. Setup and verification commands should stay in `.ai/RUNBOOK.md`.

### Status
Accepted

---

## YYYY-MM-DD - Codex should act as second-eye reviewer

### Decision
After meaningful code changes, Codex should review the current git diff before the work is considered complete.

### Reason
Using a second model reduces the chance of missing regressions, security issues, overengineering, unclear code, or missing tests.

### Alternatives considered
- Only trusting the implementing assistant.
- Manual review without a structured prompt.
- Running Codex only before final commit.

### Consequences
The standard completion checklist should include a Codex review step. Automation can be implemented through a VS Code task, a Git pre-commit hook, or a project script.

### Status
Accepted
