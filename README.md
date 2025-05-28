# ai-compliance

AI-Powered Compliance Filing Platform – What You Get

Our solution turns the entire regulatory-filing workflow into a one-click experience:

Drag-and-Drop Ingestion – Upload invoices, CDRs or financial statements; REST API and SFTP feeds are also supported.

GPT-4 Data Extraction & Smart Mapping – Large-language-model agents read every document, validate figures and map them to the exact fields required by FCC, IRS and state portals.

Vector-Search Knowledge Base – A semantic index of all current regulations lets the AI cross-check every entry against the latest rules, automatically flagging discrepancies.

Real-Time Validation Layer – Custom rules (totals, cross-schedules, tax codes) run instantly to eliminate human error before submission.

Headless RPA Filing – Playwright-driven bots log in, fill the online forms, upload PDFs and capture confirmation receipts 24/7.

Audit-Ready Evidence – Every step (raw source, extracted JSON, portal response) is stored in encrypted S3 with immutable hashes for audit trails.

Dashboard & API – Track filing status, download receipts, or trigger filings programmatically. SOC 2 controls, SSO and granular role permissions are built in.

## Configuration

Copy `cfg/openai.cfg.example` to `cfg/openai.cfg` and fill in your OpenAI credentials. The application reads the following environment variables:

- `OPENAI_API_KEY` – your OpenAI key (or set in the config file)
- `OPENAI_MODEL` – optional model name (defaults to `gpt-4o-mini`)
- `FLASK_SECRET_KEY` – Flask session secret
- `FCC_API_KEY` – FCC Public API key for ECFS lookups

## License

This project is released under the [MIT License](LICENSE).

## CORS
export ALLOWED_ORIGINS='{
    "https://partner-site.io":  "PARTNER-TOKEN-42",
    "https://demo.example":     "DEMO-TOKEN-XYZ"
}'

 uv run python app.py
