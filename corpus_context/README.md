# Corpus context (emails, news, events)

Synthetic **internal Outlook emails**, **South African news articles**, and **macro event summaries** aligned to `banking_data/` loan payment statistics. Use this layer for end-to-end data engineering and analytics practice (ingest → join → narrate).

The email corpus now includes project-style stakeholder briefs from all major departments. These messages are designed to become practical portfolio projects: dashboard requests, loan portfolio analysis, bank reconciliation, compliance monitoring, product analytics, operations SLA tracking, fraud monitoring, CX analysis, credit-risk modelling, payments settlement reconciliation, and lakehouse/data-contract work.

## Layout

```
corpus_context/
  index/
    monthly_signals.csv      # KPIs pulled from loan_payment_transactions
    corpus_manifest.csv      # catalogue of every email / article / event file
  events/
    macro_events.jsonl       # SA timeline 2019–2025
  {year}/{month}/
    emails/*.eml             # RFC 5322 source (for parsing pipelines)
    emails/*.pdf             # Outlook-style printable message
    news/*.md                # Article source with YAML front matter
    news/*.pdf               # Newspaper-style printable article
    events/month_context_{year}_{month}.json
```

## Generate

```bash
py -m pip install reportlab
py generate_corpus_context.py
py generate_corpus_context.py --start-year 2020 --end-year 2022
```

Export PDFs for an existing corpus (without regenerating text):

```bash
py export_corpus_pdfs.py
```

Requires monthly `loan_payment_transactions_{year}_{month}` files under `banking_data/`.

## How it links to banking data

| Signal | Source column / metric | Used in |
|--------|------------------------|---------|
| Failure rate | `status = Failed` | Collections emails, Fin24-style articles |
| NSF volume | `failure_reason = insufficient_funds` | Payments ops threads |
| Timeouts | `bank_timeout` | Infrastructure / load shedding mail |
| Volume growth | Row count YoY | Data engineering capacity emails |
| Data quality | `has_data_error` | Analytics ↔ DE reconciliation |

## Project email coverage

Each generated month includes department-backed project asks from:

- Executive Office
- Loan Products
- Finance Reconciliation
- Regulatory Compliance
- Products & Remittances
- Banking Operations
- Financial Crime
- Customer Experience
- Retail Credit Risk
- Payments Operations
- Data Engineering

The manifest includes `department` and `project_type` columns for email artifacts so you can filter the corpus into project briefs, stakeholder requests, and operational follow-ups.

News articles are generated in the same project-discovery pattern. They read as external sector/news bulletin items, not internal instructions to the bank, and each one carries `department` and `project_type` metadata so the article can be used as a trigger for a department project. Examples include finance reconciliation articles about bank charges and late postings, product articles about digital adoption, and operations articles about payment delays.
| Prime rate | `generate_loans.py` yearly curve | CFO / credit risk mail |

Fictional employer: **Keystone Retail Bank** (`@keystonebank.co.za`). News outlets are styled after real SA media; articles are synthetic training text.

## Suggested DE / DA pipeline exercises

1. **Ingest** — Parse `.eml` (headers + body), load `corpus_manifest.csv`, partition by `year/month`.
2. **Join** — `monthly_signals.csv` ⋈ manifest ⋈ `banking_data/{y}/{m}/loan_payment_transactions_*`.
3. **Feature** — Tag months with macro events; compute failure-rate delta vs prior month.
4. **Narrate** — Dashboard: “What did Collections know before the March 2020 spike?”
