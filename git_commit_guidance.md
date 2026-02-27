## Git Commit Message Guidelines

### Subject Line Format
```
<type>: <Short imperative description>
```

**Valid types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `style`, `build`, `ci`, `revert`

**Rules:**
- Start with a valid type followed by colon (e.g., `fix:`, `feat:`)
- Keep under ~72 characters
- No period at the end

**Examples:**
- `feat: Add support for GQA in CTE attention kernel`
- `fix: Remove unused activation function parameter in MLP`

### Body Structure

**1. Problem/Context (What & Why):**
Explain what the current behavior is and why it's problematic. Reference the commit that introduced the issue if this is a fix.

**2. Solution (How):**
Describe what this commit does to address the problem.

**3. Technical Details:**
Include relevant implementation details, algorithm explanations, or performance data when applicable.

### Body Writing Patterns

**For Bug Fixes:**
```
Commit <hash> ("<subject>") introduced/removed/changed <what>.

This resulted in <problem description>.

Fix by <solution>.

Fixes: <hash> ("<subject>")
```

**For Refactoring:**
```
<Current state description>.

<Why change is needed>.

<What the refactoring does>.

Only refactoring. No functional change intended.
```

**For New Features:**
```
<Context/motivation for the feature>.

<How the feature works - algorithm/implementation details>.

<Performance benchmarks if applicable - use tables for data>.

<Dependencies on other commits/components if any>.
```

**For Reverts:**
```
Commit <hash> ("<subject>") <what it did>.

Reverting this patch because <reason>.

<Additional context about why original shouldn't have been merged>.
```

### Key Phrases to Use

| Situation | Phrase |
|-----------|--------|
| Refactoring only | "Only refactoring. No functional change intended." |
| Preparation for future work | "This is done as a preparation for..." |
| Bug fix | "Fix by...", "Fix this issue by..." |
| Explaining rationale | "This is done because...", "The reason is..." |
| Referencing broken commit | `Fixes: <hash> ("<subject>")` (optional footer) |
| Performance context | Include benchmark tables with size/time/bandwidth columns |

### Formatting Rules

1. Wrap body at 72 characters per line
2. Blank line between subject and body
3. Use bullet points sparingly - prefer prose
4. Include benchmark data as formatted tables when relevant
5. Reference related commits with full hash and subject in quotes

### Example Complete Commit Message

```
fix: Correct mask application in CTE kernel

The softmax mask was being applied after the scale factor instead of
before, causing numerical differences with the reference implementation.

This was introduced when refactoring the attention score computation to
support variable sequence lengths.
```

### Anti-Patterns to Avoid

- Generic subjects like "Fix bug" or "Update code"
- Missing type prefix (e.g., `fix:`, `feat:`)
- Invalid type (must be one of: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `style`, `build`, `ci`, `revert`)
- Past tense in subject ("Fixed" instead of "fix")
- Body without context/rationale
- Forgetting "No functional change intended" on pure refactors
