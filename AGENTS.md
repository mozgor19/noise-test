# Agent Instructions

## First action in every session
Before doing any work, read:
- .ai/PROJECT_BRIEF.md
- .ai/CONTEXT_INDEX.md
- .ai/CURRENT_STATE.md
- .ai/TASKS.md
- .ai/RUNBOOK.md
- .ai/SESSION_HANDOFF.md

Then summarize:
1. What this project does
2. Current task status
3. Relevant files
4. Risks or unknowns
5. Proposed next step

Do not implement before giving a short plan unless the user explicitly asks for direct implementation.

## Work rules
- Make minimal, targeted changes.
- Do not rewrite unrelated files.
- Do not invent project facts. Read files first.
- Do not claim tests passed unless you actually ran them.
- After meaningful work, update `.ai/SESSION_HANDOFF.md`.
- If you discover a stable project fact, add it to the appropriate `.ai/` file.
- If a decision is made, append it to `.ai/DECISIONS.md`.

## Verification
Before saying the task is complete:
- Show changed files.
- State commands/tests run.
- State remaining risks.
- Update SESSION_HANDOFF.md.