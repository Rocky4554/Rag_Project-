---
name: db-fix
description: Inspect and fix the project's Supabase Postgres database (schema, constraints, rows, RLS) using the Supabase MCP server. Use for tasks like "fix the active_interviews FK", "drop a constraint", "check a table's columns", "query interview_results", "add a column", or any read/write against this project's Supabase DB.
disable-model-invocation: true
---

## Purpose

Work on this project's Supabase Postgres database through the **Supabase MCP
server** (configured in `.mcp.json` for `project_ref=lqxgzqwxfhxjbptjdodm`).
Use this for schema inspection, fixing constraints, querying/patching rows,
checking RLS, and running migrations — instead of hand-rolling `pg` scripts.

## How to use the Supabase MCP

The Supabase MCP tools are exposed as `mcp__supabase__*`. They may be deferred —
if so, load them first with ToolSearch:

```
ToolSearch  query: "select:mcp__supabase__list_tables,mcp__supabase__execute_sql,mcp__supabase__apply_migration"
```

Common tools (names may vary slightly — discover with `ToolSearch query:"supabase"`):
- `list_tables` / `list_extensions` — inspect schema
- `execute_sql` — run a SELECT/UPDATE/INSERT/DELETE (read or write)
- `apply_migration` — run DDL (CREATE/ALTER/DROP) as a named migration
- `get_logs` / `get_advisors` — diagnostics and security/perf advisories

## Workflow (follow every time)

1. **Understand first.** Before changing anything, inspect the relevant
   table(s): list columns, constraints, and a few sample rows.
2. **Show the plan.** State exactly what SQL you will run and which rows/objects
   it affects. For any **DDL or multi-row write**, show the statement and the
   blast radius (how many rows / which constraint / which table).
3. **Confirm destructive actions.** ALWAYS get my explicit go-ahead before:
   - `DROP` / `ALTER` (constraints, columns, tables)
   - `DELETE` / `UPDATE` without a tight `WHERE`
   - anything touching `auth.*` or RLS policies
   Read-only `SELECT` and single-row, well-scoped writes can proceed directly.
4. **Run it** via the MCP tool (prefer `apply_migration` for DDL so it's named
   and tracked; `execute_sql` for queries / data fixes).
5. **Verify.** Re-query to confirm the change landed (e.g. re-check the
   constraint list, the column, or the affected rows). Report the result.

## Guardrails

- This is the **shared production database** — treat every write as production.
- Never run unbounded `DELETE`/`UPDATE`. Always include a `WHERE`.
- Prefer additive/reversible changes; note how to roll back DDL.
- Don't print secrets or full key values in output.

## Project schema notes (known tables)

- `active_interviews` — resume pointers, keyed by `session_id` (UNIQUE).
  Its `user_id` FK to `auth.users` does NOT match app user IDs; the app passes
  `null` for it. Drop `active_interviews_user_id_fkey` if you want app user IDs.
- `documents`, `chat_messages`, `interview_results`, `quiz_results`,
  `activities` — core app tables (see `lib/db.js` for the access layer).
