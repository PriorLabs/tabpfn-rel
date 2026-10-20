# Database checklist

What to check in every table and column before defining a task, and the default
treatment. TabPFN-Rel cuts the database at each prediction time `t` and keeps
the rows whose `time_col` is at or before `t`; a table without `time_col` is
visible at every `t`. Most problems below are ways in which that cut still shows
the model something from after `t`.

Measure on full tables, not the first rows. Give every number its scope (rows
examined, comparison rule). Treatments marked **ask** need the user's agreement.

## Keys

- **Primary keys** are unique and non-null. A table identified by several
  columns gets a surrogate key in `prepare.py`, or no `pkey` if nothing points
  to it.
- **Foreign keys** resolve: count non-null values without a parent row, as rows
  and as distinct ids, and give the likely cause (deleted parent, privacy,
  sampled parent table). TabPFN-Rel sets a foreign key without a parent row to
  null when it loads the database, so the link is lost. Record dangling keys as
  a known flaw and **ask** before adding placeholder parent rows that would keep
  the link. Declare only relationships whose meaning matches at both ends; equal
  numbers alone are not a relationship.
- **Key representation** is the same at both ends: watch for `123` vs `123.0`,
  leading zeros, and ids above 2^53 passing through floats. Keep codes as
  strings.
- **Codes** whose meaning lives in a lookup table: keep the lookup table and
  declare the foreign key.

## Missing values

- **String markers** (`\N`, `NULL`, `N/A`, `-`, empty strings, `unknown`) and
  **numeric markers** (0, -1, 999 where real values are positive): count each
  marker per column. Convert a marker to null only when the source documents it
  or the data proves it (a value outside the valid range). **Ask** before
  merging several markers into one null: different markers can carry different
  information.
- **Real values that look like markers** (a person named "Null", a setting
  "NONE") stay.
- **Placeholder dates** (1900-01-01, 1970-01-01, 1753-01-01, midnight standing in
  for an unknown time): keep them unknown, never fill them.
- **Imputation** of any kind: **ask**.

## Timestamps

- **What each date records**: the event, the row's creation, or when the value
  became known. A table's `time_col` must be the time its row became known. A row
  about other rows (a review of an order, a closure of a shop) becomes known no
  earlier than the rows it refers to.
- **Event tables without a time column** leak every future row into every cut.
  Derive a time column from the parent row (an order item gets its order's
  purchase time). Every derived timestamp is a decision: **ask**.
- **Child rows dated before their parent** (a review before its order, activity
  before the account): count them and find which date is wrong. Tasks that select
  entities by activity must also check the entity's own creation date.
- **Precision and time zones**: note day-only columns next to second-precision
  ones; a cut at midnight then keeps or drops a whole day of the day-only table.
- **Outlier dates** (before the domain existed, after the export, closures before
  openings): report them; null them only when the source shows they are errors.
- **Rows dated after the export** (scheduled events with no outcome yet): they
  are not observations. Note them, and keep them out of label windows.

## Snapshot versus event

**The static-row assumption.** Most exports keep only the latest value of each
row. Without a history, TabPFN-Rel shows a row from its `time_col` onward with
the values it has in the export, and shows a table without `time_col` at every
cut. Every database relies on this somewhere. The items below are the cases
where it fails badly enough to need a treatment; everything else stays under
the assumption and is listed as such.

Classify every table as **event** (each row dated when it became known),
**snapshot** (only the latest value of something that changed survives), or
**static** (truly fixed: a country list). Many tables that look static are not:
an entity table (customers, drivers, shops) usually gains rows over time and its
fields change. Where the data allows, recover the time information instead of
declaring a table static:

- **Entity tables** get an availability date as `time_col`: the creation date,
  or the first event the entity appears in. Otherwise every entity is visible
  before it exists.
- **Columns computed over the whole history** (counts, averages, loyalty
  scores, rating averages, last-activity dates, "last 9 months" windows) contain activity
  from after every past cut. Drop them; the model rebuilds such aggregates from
  the event rows at each cut.
- **Columns set only once an outcome is known** (a closure date, a final result,
  a delivery date, a status like "cancelled"): compare the outcome rate
  when the column is set with the rate when it is unset; a split near 0% or 100%
  marks a label proxy. If the source dates the change, move the value into its
  own dated table so a cut hides it until it happens; otherwise drop it. **Ask**
  either way.
- **Last-edit fields** (`updated_at`, last editor) reveal that an edit happens
  later. Drop them; keep the rows.
- **Descriptive fields that can be edited** (names, descriptions) usually have no
  history. Keep them and say that the exported value stands in for earlier ones.
- **Link tables without dates** are visible at every cut. If the links were added
  over time, date each link at the later of its two ends where possible, or
  **ask** whether to keep the table.

## Values

- **Structured values** (lists, JSON, tag strings like `<a><b>`): keep the whole
  value; never keep only its first element. Splitting into a child table is
  optional.
- **Artifact columns** (`Unnamed: 0`, a column constant over the full table,
  all-null columns, replication metadata): leave them out of `columns`.
- **Identifiers without meaning** (URLs, slugs, hashes) stay out of `columns`.
  Names of entities (product names, titles) carry meaning and stay.
- **Exact duplicate rows**: count them and keep them unless the source shows they
  are errors.
- **Columns with unclear meaning**: keep them, unless the doubt is whether they
  summarize later activity; then drop them and say why.

Drop a column only for one of the reasons above, and list every dropped column
with its reason.

## Population

- Keep every row in the prepared tables. Choosing which entities a task scores
  belongs in the label SQL, not in `prepare.py`. A row filter the preparation
  applies anyway is a decision: **ask**.
- Check whether a table mixes populations of a different nature (test accounts,
  bots, migrated records) and whether coverage changes over time (a feature
  launched mid-history, a column populated only in recent years). Report both.
