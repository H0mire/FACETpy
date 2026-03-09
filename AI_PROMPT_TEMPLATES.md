# AI Prompt Templates

Copy and adapt these templates for FACETpy tasks.

## 1) Implement Processor

```text
Task:
Implement a processor for [short goal].

Scope:
- In scope: [file paths]
- Out of scope: [file paths]

Hard Requirements:
- Follow docs/PROCESSOR_GUIDELINES.md strictly.
- Keep public APIs and naming conventions consistent.
- Add or update tests in tests/... for new behavior and edge cases.
- Update docs if user-visible behavior changes.

Edge Cases:
- [edge case 1]
- [edge case 2]

Definition of Done:
- Code compiles and tests pass locally.
- No unrelated refactors.
- Changes are small, reviewable, and explained.
```

## 2) Refactor Existing Processor

```text
Task:
Refactor [processor/class] for readability and maintainability without behavior changes.

Scope:
- In scope: [file paths]
- Out of scope: algorithm changes

Hard Requirements:
- Follow docs/PROCESSOR_GUIDELINES.md.
- Preserve behavior and API.
- Keep validate/process separation clear.
- Keep diff minimal and focused.

Verification:
- Add/adjust tests only if needed to lock current behavior.
- Explain why behavior is unchanged.
```

## 3) Review Uncommitted Changes

```text
Task:
Review this diff with focus on bugs, regressions, and missing tests.

Review Priorities:
1. Correctness and edge cases.
2. Contract with docs/PROCESSOR_GUIDELINES.md.
3. Test coverage quality.
4. Readability/maintainability.

Output Format:
- Findings first, ordered by severity.
- For each finding: file, line, issue, concrete fix.
- Then: assumptions/questions.
- Then: short change summary.
```

## 4) Docs Sync Check

```text
Task:
Check if this code change requires documentation updates.

Context:
- Changed files: [file paths]
- Relevant docs: docs/source/...

Output:
- Required doc updates (if any), with exact target files.
- Suggested text snippets for each changed section.
```
