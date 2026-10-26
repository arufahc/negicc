# Agent Instructions for negicc

This file contains mandatory rules for any agent working in this repository.

---

## Git Configuration Rules

All agents committing to this repository must configure their Git author information as follows:
- **Name**: `Agent for Alpha Lam`
- **Email**: `arufa.hc@gmail.com`

**Command to run at startup:**
```bash
git config user.name "Agent for Alpha Lam" && git config user.email "arufa.hc@gmail.com"
```

### Commit Message Style Rules

Do **NOT** use conventional commit prefixes (such as `feat:`, `feat(...)`, `fix:`, `fix(...)`, `refactor:`, `refactor(...)`, `chore:`, `chore(...)`, etc.) in commit messages, titles, or CL descriptions.
Write direct, plain English summaries (imperative mood, concise, descriptive), for example:
- "Add native C++ DINOv3 inference engine, runners, and neg_process integration"
- "Address review follow-ups for device check, mutex, input validation, and build targets"

---

## Documentation Rules

### Never Use Hardcoded Absolute Paths

Do **not** embed machine-specific absolute paths (e.g., `/home/user/Projects/...`) anywhere in documentation, markdown files, scripts, or source comments. Use relative paths instead.

---

## Code Style Rules

- Do not introduce arbitrary namespaces (e.g. `namespace dinov3`) in application and engine sources (`neg_process.cc`, `raw_info.cc`, etc.), which reside in the global namespace. The six shared pipeline files consumed by `negicc-station` (`cfa_pipeline.h`, `cfa_pipeline.cc`, `color_pipeline.h`, `flat_field.h`, `flat_field.cc`, and `flat_sample.h`) use `namespace negicc`.
