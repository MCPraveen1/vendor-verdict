# Vendor Verdict

Local capstone runner for the Vendor Verdict competition. It screens evidence, derives facts, and uses the supplied deterministic rule engine to produce the required `submission.csv` format.

## Requirements

- Python 3.10 or newer
- The supplied competition files in the project directory
- For optional Groq mode only: a local `GROQ_API_KEY`

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The default offline run needs no API key. Optional Groq mode reads this variable from the ignored local `.env` file (replace the placeholder locally; never commit the file):

```text
GROQ_API_KEY=your_groq_api_key
```

The optional Groq mode also accepts `GROQ_API_KEY` from the process environment. It loads `.env` from the project directory and never prints the key.

## Generate the offline submission

From the project directory:

```powershell
python submission.py
```

By default the runner stays offline, uses the supplied keyword baseline only on exact-identity vendor or registry pages, and applies the deterministic rules. It excludes the starter baseline's fabricated certification dates and incident dates, and its placeholder subprocessor facts. This writes `submission.csv` with the exact `id,verdict` columns and test ID order from `test.csv`. Existing output files are never overwritten.

To smoke-run only the first two test cases to a new file:

```powershell
python submission.py --mode offline --limit-cases 2 --out outputs/smoke_submission.csv
```

Do not use a limited run as the competition submission.

## Evaluate on training data

Generate offline predictions for labeled training cases. The runner evaluates the result with the supplied `metric.py` after writing the CSV:

```powershell
python submission.py --mode offline --split train --out outputs/train_predictions.csv
```

Use `--limit-cases N` for a small smoke evaluation; the reported score then covers only the selected cases and is not comparable to the full competition score.

## Tests

Run the offline unit tests (no notebook execution and no API calls):

```powershell
python -m unittest -v
```

## Design and limitations

- The offline mode uses only non-archived pages from the vendor's declared official domain or one of three recognized certification registries. Official pages must explicitly identify the expected legal entity; registry pages must name the exact legal entity as certificate holder.
- In optional Groq mode, archived pages and third-party commentary are excluded from extraction candidates. Extracted facts must cite wording found verbatim in eligible pages and pass field-specific structural checks.
- Certification facts require a certificate identifier present on both an exact-identity registry page and an exact-identity official page. This intentionally prefers abstention over unsupported certification claims.
- In Groq mode, LLM/API errors fail closed to `UNKNOWN` facts for that case, preserving complete output coverage. The run summary reports a failure count; review that count before submission.
- The Groq adapter remains available with `--mode groq`; it loads `GROQ_API_KEY` from the project `.env` without printing it. That mode sends selected case/evidence text to Groq. It was not run here because this run is configured to keep competition data offline.
- The offline keyword path is deliberately conservative but still heuristic. It is not equivalent to a full, validated LLM extraction workflow and can miss facts or misread context. Its measured training score is 0.3962; the unfiltered supplied baseline scored 0.4750 but pools untrusted sources and can fabricate certification/incident dates.
- The offline unit tests and mock Groq request tests do not call the external API. Live Groq connectivity and credential validity remain unverified.
- The notebook retains the investigative history and exploratory functions. The standalone runner provides the reproducible local submission path without executing the notebook.
