# Defining the task

TabPFN-Rel predicts one value per entity (a customer, a seller, a patient) at a
prediction time `t`, about that entity's near future: will this customer order
again in the next 30 days, how much will this customer spend next month. The label is
not a column in the data. A SQL query computes it from the database's own rows
after `t`, so any database with event timestamps and a question of that shape
becomes a supervised task. Two task types exist: `binary_classification` (0/1)
and `regression`.

A task is two YAML files. The full field reference is
[the predictive-task guide](https://github.com/PriorLabs/relarena/blob/main/docs/predictive-task.md);
[`examples/olist_database.yaml`](../../../../examples/olist_database.yaml) and
[`examples/olist_seller_churn.yaml`](../../../../examples/olist_seller_churn.yaml)
are a complete pair to adapt.

## The database file

Table name -> schema. Paths resolve against the `data_dir` passed to
`from_yaml`, or the YAML file's own directory.

```yaml
customers:
  pkey: customer_id
  time_col: signup_date         # the entity's availability date
  path: tables/customers.parquet
  columns: [customer_id, signup_date, region]
orders:
  pkey: order_id
  time_col: order_date          # event table
  path: tables/orders.parquet
  fkeys:
    customer_id: customers      # orders.customer_id references customers.<pkey>
order_items:
  time_col: order_date          # copied from the order in prepare.py
  path: tables/order_items.parquet
  fkeys:
    order_id: orders
    product_id: products
products:
  pkey: product_id              # static lookup table: no time_col
  path: tables/products.parquet
```

- `pkey` must be unique and non-null; ids are reindexed internally, so strings
  and gaps are fine. A table without `pkey` cannot be a foreign-key target.
- Every `fkeys` target must be a table in the file with a `pkey`.
- `time_col` goes on event tables and on entity tables with an availability
  date. A table without `time_col` is visible at every cut.
- `columns` restricts what is loaded. It must include the `pkey`, `time_col` and
  every foreign key column. Leaving a column out here hides it from every model.
- Table names in the label SQL must match these keys exactly, including case.
- `timestamp_df` is reserved; no table may use that name.

## The task file

```yaml
database: database.yaml        # relative to this file
entity_table: customers
entity_col: customer_id        # the entity id column the query returns
time_col: timestamp            # the anchor-time column the query returns
target_col: orders_again
task_type: binary_classification
timedelta: 30 days             # forward label window (t, t + timedelta]
num_eval_timestamps: 1
val_timestamp: '2023-04-30'    # more than one timedelta before test
test_timestamp: '2023-06-01'
query: |
  ...
```

### Label SQL rules

- Start `FROM timestamp_df`. TabPFN-Rel registers it with one column,
  `timestamp`, holding the anchor times. Your joins supply the entities.
- Return exactly three columns, aliased to `time_col`, `entity_col` and
  `target_col`. Any extra column is rejected because it would become a feature
  computed from the future.
- The label comes from the forward window:
  `x.time > timestamp AND x.time <= timestamp + INTERVAL '{timedelta}'`.
  The lower bound is strict.
- `{timedelta}` is the only placeholder. It is replaced as plain text with the
  full duration (`90 days 00:00:00`), so write `INTERVAL '{timedelta}'` with no
  unit after it.
- The query runs in DuckDB over the tables after `columns` filtering, so it can
  use only kept columns. Keys are already reindexed to integers there: join on
  them, but never compare them with literal ids from the raw files.
- Only `time_col` is parsed as a date. Write the prepared tables as Parquet so
  other dates and numbers keep their types; a CSV column may need a `CAST`.
- Each (anchor, entity) pair must appear once; aggregate with `GROUP BY` or use
  `EXISTS`.

### Choosing who gets scored

The query decides which entities are scored at each anchor. This is where most
task bugs live.

- **Use only the past to select entities.** Every condition that selects
  entities ("ordered in the last 30 days", "has at least one review") uses rows
  dated at or before the anchor, with both bounds written:
  `x.time > timestamp - INTERVAL '{timedelta}' AND x.time <= timestamp`.
- **Score only entities that exist at the anchor.** Require the entity's own
  availability date to be at or before the anchor. Earlier child activity does
  not prove existence when children can predate their parent.
- **Do not select by the outcome.** Joining the forward-window events with an
  inner join scores only entities that will be active, which makes "will be
  active" unlearnable.

### Recurrence and churn

"Will this active entity act again" seeds the population from a backward window
and labels from the forward window:

```sql
SELECT timestamp, c.customer_id,
  CAST(EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.order_date > timestamp
      AND o.order_date <= timestamp + INTERVAL '{timedelta}'
  ) AS INTEGER) AS orders_again
FROM timestamp_df, customers c
WHERE c.signup_date <= timestamp
  AND EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.customer_id
      AND o.order_date > timestamp - INTERVAL '{timedelta}'
      AND o.order_date <= timestamp
  )
```

For regression (spend, count), select from the entity table, `LEFT JOIN` the
forward-window events, and `COALESCE(SUM(...), 0)`.

## Split timestamps

Anchors for training step backward from `val_timestamp - timedelta` in
`timedelta` steps to the start of the data (at least 3 anchors). Validation and
test each get one anchor, at `val_timestamp` and `test_timestamp`; keep
`num_eval_timestamps: 1`. The model sees the database frozen at
`test_timestamp`, so a later test anchor can score only entities that existed
by then, and the others are left out of its labels.

- Set `test_timestamp` so that its label window ends on or before the last date
  with complete data: `test_timestamp + timedelta <= data end`. Check the last
  dates of the event tables; the export's final days are often incomplete. The
  library checks against the latest date in any table, so pass `--data-end` to
  the scripts when the label's source table ends earlier.
- Set `val_timestamp` slightly more than one `timedelta` before
  `test_timestamp`, for example `timedelta` plus one day. Loading fails when
  `val_timestamp + timedelta` is later than the last row dated at or before
  `test_timestamp`.
- Choose `timedelta` from the question (30 days for churn in a busy shop, 90 or
  365 days for slower activity), then check that each split has enough labelled
  rows and both classes.
- The model sees the database frozen at `test_timestamp`. A prediction for a
  later anchor uses no rows after `test_timestamp`.

## Fitting

`scripts/compare_models.py` fits and scores every model through the same
`PredictiveContext.fit` call, so all models read the same database and labels;
the [README](../../../../README.md) shows the `fit`/`predict` API. `fit`
defaults to `n_trials=10` tuning trials; pass `n_trials=0` for one fit of the
default configuration.

When a fit hangs or runs out of memory, an entity probably links to many large
tables. Drop the tables that cannot plausibly carry signal from
`database.yaml`.
