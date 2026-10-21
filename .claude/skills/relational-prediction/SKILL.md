---
name: relational-prediction
description: >
  Turn a raw relational database (CSV or Parquet tables) and a plain-language
  question ("which customers will stop buying next month?") into a TabPFN-Rel
  prediction task, then compare TabPFN-Rel with simple baselines. Covers
  inspecting the tables, preparing them without leaking future information,
  writing the database and task YAML files, checking the labels, and scoring
  held-out rows. TRIGGER when the user wants to predict something per entity
  over a multi-table database, to format a database for TabPFN-Rel or RPI, or
  to write a task YAML or label SQL.
---

# Predict over a relational database with TabPFN-Rel

The user brings tables and a question. The result is this layout in a working
directory of the user's choice (default `data/<name>/`; keep it out of version
control):

```
data/<name>/
  raw/                  downloaded files, never edited
  prepare.py            raw -> tables/ (only the changes listed in step 3)
  tables/               the prepared tables, one file per table
  database.yaml         schema: keys, time columns, kept columns
  <task>.yaml           label SQL, prediction window, split timestamps
```

`<skill-dir>` below is the directory that holds this file. Its scripts are in
`<skill-dir>/scripts/`; run them from the working directory with the user's
Python environment.

Read these references when the step says so:

- [references/database-checklist.md](references/database-checklist.md): what
  to check in every table and column, and the default treatment for each problem.
- [references/task-definition.md](references/task-definition.md): the YAML
  formats, the label SQL rules, split timestamps and common task patterns.

## Rules

- Keep what the source says. Change a value only when it is clearly broken or
  leaks the future, and list every change. Keep the source's table and column
  names.
- Every claim about the data carries a number computed from the full tables,
  with its scope. A column name is a hypothesis about its meaning, not proof.
- Models see the data only through `database.yaml` and the task. A leaky column
  must leave the prepared tables or the `columns` list; hiding it anywhere else
  hides it from some models only.
- Ask the user before every change the checklist marks **ask**, and before
  choosing the task. Propose a default with its evidence.
- Prefer the hosted backend (`model="client"`) unless the user has a GPU and
  local weights. Hosted fits consume API quota, so fit the default
  configuration (`n_trials=0`) unless the user asks for tuning.

## Steps

1. **Set up.** Install TabPFN-Rel and the baselines with Python 3.11 or 3.12:

   ```bash
   pip install "tabpfn-rel[api]" "relarena[lightgbm]>=0.1.4,<0.2"
   ```

   For local inference, install `tabpfn-rel[local]` instead of
   `tabpfn-rel[api]`. `relarena` holds the constant and LightGBM baselines.
   Inside a checkout of the tabpfn-rel repository, run
   `uv sync --extra api --group cpu --group baselines` instead and prefix the
   commands below with `uv run --no-sync`. On macOS, prefix every Python command
   with `OMP_NUM_THREADS=1` (torch and LightGBM ship separate OpenMP runtimes,
   and LightGBM segfaults if torch loads first). For hosted inference,
   `TABPFN_TOKEN` must be set or `tabpfn_client.init()` run once.

2. **Get and inspect the raw data.** Download into `raw/` and record the URL,
   date, licence and file sizes. If the tables come in another format (XML,
   JSON, an SQL dump), convert each one to Parquet with every value kept as a
   string and no other change. Read CSV files with `dtype=str`,
   `keep_default_na=False` and no `na_values`, so you see the markers the source
   uses. Then profile every table and column:

   ```bash
   OMP_NUM_THREADS=1 python <skill-dir>/scripts/profile_tables.py data/<name>/raw
   ```

   Then work through
   [the checklist](references/database-checklist.md) and write down for each
   table: what one row is, its primary key, its foreign keys (with match rates),
   its time column and what that date records, and whether the table is an event
   log, a snapshot or static.

3. **Agree the preparation with the user.** Show a short table of the planned
   changes, one line each: table, column, problem with its number, proposed
   treatment. Below it, list the tables and columns that stay under the
   static-row assumption (see the checklist), so the user accepts that
   approximation explicitly. Wait for the user's answer.

4. **Write `prepare.py` and `database.yaml`.** `prepare.py` reads `raw/`,
   applies only the agreed changes, and writes `tables/`. It must be rerunnable
   from scratch. Then write `database.yaml` following
   [task-definition.md](references/task-definition.md): every table that carries
   signal, with `pkey`, `fkeys`, `time_col` for event tables, and a `columns`
   list without the columns the checklist and step 3 drop.

5. **Choose the task with the user.** Propose two or three plain-language
   questions the data can answer, each with the entity table, the forward
   window, the label definition, and why someone would pay for the answer.
   Prefer questions whose label comes from future event rows (activity, churn,
   spend) and whose entities number at least a few thousand per cutoff. Let the
   user pick.

6. **Write the task YAML and check the labels.** Follow
   [task-definition.md](references/task-definition.md). Then run:

   ```bash
   OMP_NUM_THREADS=1 python <skill-dir>/scripts/check_task.py data/<name>/<task>.yaml
   ```

   Pass `--data-end <date>` when the label's source table ends earlier than the
   other tables. Fix the task when a split is empty, has fewer than about 200
   rows, or has a positive rate below 2% or above 98%. Before fitting, confirm
   that every entity filter in the label SQL uses only rows dated at or before
   the anchor.

7. **Compare TabPFN-Rel with baselines.**

   ```bash
   OMP_NUM_THREADS=1 python <skill-dir>/scripts/compare_models.py data/<name>/<task>.yaml
   ```

   It fits three baselines and TabPFN-Rel and prints their test metrics:
   `constant-global` (the share of positive labels in the training rows, or
   the median for regression), `constant-per-entity` (each entity's own share
   of positive training labels, or the global share for an entity without
   training rows) and `lightgbm` (the entity table's own columns). TabPFN-Rel is the only one that uses the other
   tables, so a gap over `lightgbm` measures what the relational context adds.
   Use `--models` to run a subset, for example the baselines only.

8. **Report.** Give the user the question, the task definition in one sentence,
   the split sizes and class balance, the metric table, and the changes from
   step 3. Say plainly when TabPFN-Rel does not beat the baselines: that means no
   signal was found for this task as posed, not that the method failed or worked.
