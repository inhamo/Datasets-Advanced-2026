# Monthly Subfolder Organization Guide

Both `generate_customer_data.py` and `generate_accounts.py` now support monthly subfolder organization for organizing large-scale data generation.

## Output Directory Structures

### Flat Structure (Default - No Month Parameter)
```
banking_data/
├── customers_2024.parquet          # All 2024 customers
├── accounts_2024.parquet           # All 2024 accounts (main table)
├── account_limits_history_2024.parquet
├── account_status_events_2024.parquet
├── account_product_enrollments_2024.parquet
├── account_signatories_2024.parquet
└── rejected_applications_2024.parquet
```

### Monthly Structure (With --month Parameter)
```
banking_data/
└── 2024/
    ├── 01/
    │   ├── customers_2024_01.parquet           # January 2024 customers only
    │   ├── accounts_2024_01.parquet            # January 2024 accounts only
    │   ├── account_limits_history_2024_01.parquet
    │   ├── account_status_events_2024_01.parquet
    │   ├── account_product_enrollments_2024_01.parquet
    │   ├── account_signatories_2024_01.parquet
    │   └── rejected_applications_2024_01.parquet
    ├── 02/
    │   └── [February data in same structure]
    └── ...
```

## Usage Examples

### Customer Data Generation

**Full year (flat structure):**
```bash
python generate_customer_data.py --year 2024 --target-records 20000
```

**Specific month (nested monthly structure):**
```bash
python generate_customer_data.py --year 2024 --month 1 --target-records 2000
```

**With cadence options:**
```bash
# Daily cadence (more frequent, smaller batches)
python generate_customer_data.py --year 2024 --month 6 --cadence daily --target-records 1000

# Monthly cadence (default)
python generate_customer_data.py --year 2024 --month 6 --cadence monthly --target-records 18000
```

### Account Generation

**Full year (flat structure):**
```bash
python generate_accounts.py --year 2024
```

**Specific month (nested monthly structure):**
```bash
python generate_accounts.py --year 2024 --month 1
```

**Interactive prompts (terminal will ask for month if needed):**
```bash
python generate_customer_data.py
python generate_accounts.py
```

## Key Features

### 1. **Global Account Counter Persistence**
- Maintains a single `account_counter.json` across all months
- Ensures unique account IDs across entire year and across monthly runs
- Automatically scans both flat and monthly directory structures on startup

### 2. **Account Distribution by Opening Date**
- When a month is specified, only accounts with `opening_date` in that month are output
- Allows generating a full year worth of accounts across 12 monthly runs
- Supports concurrent/sequential month generation

### 3. **Customer Distribution by Entry Date**
- When a month is specified, only customers with `date_of_entry` in that month are output
- Enables streaming customer onboarding simulation
- Can be combined with daily cadence for micro-batch ingestion patterns

### 4. **History Tables Always Grouped with Main Table**
- All 4 history tables follow the same directory/month structure as main `accounts_*` file
- Maintains referential integrity: history for month 1 is always in month 1 folder
- Supports downstream CDC processing by month/year boundaries

### 5. **Backward Compatibility**
- Omitting `--month` flag uses flat (legacy) structure
- Existing parquet/csv files remain untouched
- Global counter reads from both flat and monthly structures on initialization

## Use Cases

### Scale Testing
```bash
# Generate 500K accounts across 12 months
for m in {1..12}; do
  python generate_accounts.py --year 2024 --month $m
done
```

### Streaming Simulation
```bash
# Generate one month of customers per day
for d in {1..31}; do
  python generate_customer_data.py --year 2024 --month 1 --cadence daily --target-records 1000
  # Simulate daily ingestion...
done
```

### CDC Practice
Each month's folder contains a complete CDC-ready dataset:
- Main denormalized table with embedded JSON history (`limits_history_json`, `status_events_json`, etc.)
- Separate event-stream tables (limits, status, enrollments, signatories)
- Rejected applications for referential integrity

### Data Warehouse Loading
Process one month at a time into data warehouse:
```bash
# Load January 2024 accounts + history into warehouse
# All files are in banking_data/2024/01/
```

## Global Account Counter (`account_counter.json`)

Located at: `banking_data/account_counter.json`

Example content:
```json
{
  "counter": 4575,
  "timestamp": "2024-05-11"
}
```

- Incremented each time `generate_accounts.py` runs
- Persists across all runs (flat + monthly modes)
- Ensures globally unique account IDs: `ACC0000001`, `ACC0000002`, etc.

## Troubleshooting

### No files generated for a month?
- Check if `date_of_entry` or `opening_date` falls in the specified month
- Run with lower `--target-records` if upstream customer data is sparse

### Want to regenerate a specific month?
- Simply re-run the command with the same `--year` and `--month`
- Existing files will be overwritten
- Account counter is persistent; IDs will not reset

### Mixing flat and monthly files?
- Supported! Global counter scans both structures
- Recommended: pick one strategy (all flat OR all monthly) for a given year

## Performance Notes

- **Monthly generation**: ~15-20 seconds per month (typical for ~60 customers → ~4500 accounts)
- **Flat generation**: ~17 seconds for full year
- History tables scale with account count (typically 2-3× row count vs accounts)
- Parquet is default; falls back to CSV if parquet export fails

