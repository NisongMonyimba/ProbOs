# Audit Run Archive

This directory archives the output of `pre_week_audit.sh` runs, one
file per week, so there is a permanent record of the audit result at
the start of each week's work -- not just the pass/fail signal from
CI, but the actual detailed report.

## Naming convention
monthN_weekM_YYYY-MM-DD.txt
Example: `month2_week8_2026-07-10.txt`

## How to generate an entry

Before starting a new week's work, run the audit and save its output:

```bash
bash pre_week_audit.sh > docs/audits/monthN_weekM_$(date +%Y-%m-%d).txt 2>&1
```

Then commit the archived report alongside that week's plan document
(`docs/monthly_plans/monthN/weekM/main.tex`).

## Why this exists

`pre_week_audit.sh` and CI both report pass/fail, but neither keeps a
historical record of what the FULL report looked like at a given point
in time -- e.g. the exact coverage percentage, which specific security
findings existed and were resolved, or how test counts grew week over
week. Archiving the raw output here makes that history inspectable
without needing to dig through CI logs, which GitHub Actions does not
retain indefinitely.
