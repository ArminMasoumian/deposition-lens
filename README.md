# Deposition Lens

An **offline, source-linked deposition review** application built with Python and Streamlit. It extracts numbered transcript lines, retrieves related testimony, creates an exploratory date index, shows candidate statement conflicts, and exports reviewed evidence.

**Scope:** This is a deterministic/legal-tech engineering prototype, **not** an LLM agent or legal advice. Page/line checks establish that an excerpt appears in the supplied file; they do **not** establish that the testimony is truthful, relevant, or legally significant. An attorney must check outputs against the original certified transcript.

## Run

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local link printed in the terminal. **Nothing loads automatically.** Use **Transcript** to select a `.txt` or `.pdf` and then click **Load Deposition**. The included fictional transcript is `data/sample_deposition.txt` (also available as `data/sample_deposition.pdf`).

No paid AI account, API key, or internet connection is required to use the app after dependencies are installed.

## Six tabs

| Tab | Action | What it actually does |
| --- | --- | --- |
| Overview | Generate highlights | Picks source-linked answers by heuristics; displays question and answer separately |
| Ask the transcript | Search testimony | BM25/keyword retrieval of potentially related original Q/A; abstains on weak overlap |
| Timeline | Generate Timeline | Extracts date mentions, separates entries, filters weak fragments, and sorts when date information permits |
| Possible conflicts | Find Possible Conflicts | Compares opposite yes/no responses to similar questions and filters candidates with a true calculated threshold |
| Source viewer | Jump to page | Displays numbered lines with optional citation highlighting |
| Evaluation | Run Evaluation | On the included sample, compares extracted source spans to hand-labeled reference locations |

Reports for highlights and search can be downloaded in Markdown and JSON formats. The project includes repeatable automated tests. Run:

```powershell
py -m pip install -r requirements-dev.txt
py -m pytest -q
```

## Input limits

- UTF-8 TXT with `[Page N]` headings and existing numbered `Q.` and `A.` testimony lines.
- Text-based PDFs containing explicit deposition page headers and numbered lines. Images/scanned PDFs are not OCR'd here.
- 12 MB maximum file size; PDFs up to 250 pages.
- OCR reconstructions can merge question/answer boundaries and assign **reconstructed** rather than court-certified line numbers. Do not present those labels as official legal citations without verifying them against the court reporter's copy.

The public web app does not send text to an external language model; uploads are held in the Streamlit process memory by this code, but a hosting provider may still process traffic and logs. Do not use confidential or privileged files on a public-hosted demo.

## Limitations

- Retrieval matches words, **not** legal meaning. It may fail to find an answer or produce loosely related excerpts.
- Date mentions are **not** verified historical events. Ambiguous dates and OCR fragments may be omitted or sorted last.
- Similar yes/no answers are **not** proof of actual contradiction. Percentages describe lexical question similarity, not the probability of misconduct.
- Evaluation scores on one fictional transcript are not real-world model accuracy. Real uploads have no accuracy score without manually validated labels.
- The CLI retains optional model-adapter code (`depo/llm.py`), but **the Streamlit app does not use these adapters or paid APIs**.

## Architecture

`Upload -> conservative TXT/PDF ingestion -> page/line parser -> Q/A exchanges -> extractive retrieval and heuristics -> source review and optional export`

## Repository files

- `app.py`: Streamlit interface
- `depo/parser.py`, `depo/ingest.py`: transcript parsing and validation
- `depo/review.py`, `depo/qa.py`: highlight selection and search
- `depo/timeline.py`, `depo/contradictions.py`: date references and candidate conflicts
- `depo/verifier.py`, `depo/evaluate.py`: source-reference checks and sample evaluation
- `tests/`: automated regression tests
- `data/`: fictional sample transcript and annotated source locations

**Not production-ready.** Test the local browser interface and PDF/TXT handling before public deployment.
