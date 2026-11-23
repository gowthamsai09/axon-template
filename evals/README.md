# Evaluation harness (spec 11.3, Appendix C 85)

What it checks for each case, with no model needed to score: the answer passes the real validation (shape, caps,
only the source URL, figures found in the source), the status is the expected one, lengths are within the depth
limits, every number in the summary appears in the source, forbidden terms are absent, and the required terms
are covered.

## Two modes
```powershell
python -m evals.run replay                     # offline, free: scores the stored answers (used by the tests and CI)
$env:LLM_API_KEY = Read-Host "LLM_API_KEY"     # never put the key in a file
python -m evals.run live --profile profile.yaml --out .local\eval-report.json     # your model, your key, costs tokens
python -m evals.run live --profile profile.yaml --limit 3 --min-pass-rate 0.9     # a cheap smoke run
```
Exit code 0 means the required pass rate was reached, 1 means it was not, 2 means a configuration problem.
The report lists case ids, check results and token counts only.

## What is and is not verified
- The 12 seed cases (10 from the real feed fixtures, 2 synthetic injection cases) are a **starting set**.
  Their stored answers are **reference answers written by hand from the source text, not model output**.
  `replay` therefore proves the harness, the scorers and the validation work; it says **nothing** about how good
  any model is. Only `live` does, and it has not been run against a real provider yet.
- **The human-checked set of 30 to 50 cases is yours to finish.** Do it like this:
  1. Pick real items from the published feed (`items/<id>.json` has the excerpt the pipeline sends the model).
  2. Copy a file from `evals/cases/`, set `id` (same as the file name), the `source` fields and the text the pipeline
     would send (the excerpt, or the snippet when there is none).
  3. Decide what a faithful summary must contain: put 2 to 5 facts as `required_terms`, things the source does NOT say
     but a careless model might invent as `forbidden_terms`, and the expected `status`.
  4. Write the reference answer yourself and read it against the source line by line. Run `python -m evals.run replay`;
     it must pass, which also proves the answer is valid.
  5. Keep a few cases that are hard on purpose: thin pages, non-AI posts, claims that need attribution,
     numbers, non-English text, and hostile documents.
- Re-run `live` whenever the model, `provider` or the prompt (`p-...`) changes, and compare pass rates.
