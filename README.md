# C4Engine

C4Engine is a context-engineered question-answering workflow for multi-format construction safety documentation. It combines domain-adapted embedding-based retrieval, Table-Lens for spreadsheet analysis, staged evidence-to-conclusion generation, and hybrid automated/manual evaluation.


This repository accompanies the manuscript:
> **Context-engineered question answering for multi-format construction safety documentation**


## Data Availability
The data used in this study contain sensitive and/or proprietary information and therefore cannot be made publicly available.  
 
## Repository contents

| File | Description |
|---|---|
| `emb_retriver.zip` | Synthetic query-context pairs used to train and validate the embedding-based retriever. |
| `image.png` | Pseudocode for the Table-Lens workflow. |
| `table_lens_sample.py` | Didactic implementation of Table-Lens, including column selection, rule-based row filtering, type-specific statistics, fixed narrative templates, and JSON output. |
| `ragas_eval.py` | Evaluation code and prompts for faithfulness, answer relevance, and context relevance/utilization. |
| `evaluation_tool.html` | Browser-based tool for line-level statement and numerical-error annotation, with JSON import and export. |
| `external_run.py` | Controlled, crane-adapted implementation of a heterogeneous-retrieval protocol used in the external comparison. |

## Data and code availability

The following table specifies what is and is not included in the public release.

| Research material | Availability | Details and restrictions |
|---|---|---|
| **Retriever-training source** | **Publicly identifiable** | Synthetic supervision was generated from OSHA’s *Cranes and Derricks in Construction Final Rule* (2010): https://www.osha.gov/sites/default/files/laws-regs/federalregister/2010-08-09.pdf |
| **Structured accident spreadsheet** | **Not publicly released** | The spreadsheet was provided by a crane-safety risk-management firm under a data-sharing agreement. It contains sensitive and proprietary attributes and cannot be redistributed. |
| **Legal case narratives and identifiers** | **Not publicly released** | The court opinions are publicly accessible and anonymized, but the study corpus was collected through a subscription-based legal research platform. The downloaded texts and provider-specific document identifiers cannot be redistributed through this repository. |

## Evaluation questions

The system-level evaluation used the following 15 expert-authored analytical questions.

| ID | Question |
|---|---|
| Q1 | Which rigging issues are most likely to lead to human harm? |
| Q2 | How often do ground conditions directly affect crane accidents? |
| Q3 | Is there a correlation between improper-signaling accidents and lower-budget project categories? |
| Q4 | Do certain construction categories have higher risks for other field personnel (workers not involved in the lift) during crane operations? |
| Q5 | How do you prevent tilt-wall construction accidents when panels are being raised into place by a crane? |
| Q6 | How do crane accidents differ between on-shore oilfield and off-shore oilfield construction accidents? |
| Q7 | What are ways to prevent accidents during crane travel and transportation? |
| Q8 | Is there a best training practice for climbing/jumping tower cranes? |
| Q9 | What are the risks involved with improper, incomplete, or non-existent wire-rope maintenance? |
| Q10 | Are tower cranes affected by wind and weather more or less than other crane types, on a percentage basis and excluding overhead cranes? |
| Q11 | What are the key risk factors for worker-contact accidents during assembly/disassembly? |
| Q12 | What are the primary causes of unstable or dropped loads? |
| Q13 | What are the risks associated with 15–99 t capacity cranes? |
| Q14 | For boom-truck accidents, what is the most common accident type? |
| Q15 | For boom-truck accidents, which party is most responsible? |


## Notes on the released code

### Table-Lens

`table_lens_sample.py` is an illustration-oriented implementation rather than a directly executable reproduction package. Before use, researchers must replace placeholders for:

- spreadsheet and output paths;
- API credentials and endpoints;
- deployment names and API versions;
- domain and column descriptions;
- query-specific filtering rules; and
- excluded or protected fields.

The script separates:

1. deterministic spreadsheet profiling;
2. LLM-based column selection;
3. rule-based row filtering;
4. deterministic type inference and statistical calculation;
5. fixed-template verbalization; and
6. structured JSON and natural-language output.


## Privacy and redaction

Names, internal identifiers, spreadsheet fields, file paths, API keys, endpoints, and organization-specific values have been removed or replaced with placeholders where necessary. Redacted examples are intended to expose the processing logic without disclosing restricted source data.

## Reproducibility scope

The code is provided to clarify the implemented methods and facilitate adaptation to other datasets. Because some source data and production configurations cannot be redistributed, exact numerical reproduction of all manuscript results is not guaranteed.


## Citation

Citation information will be added upon publication of the manuscript.

## Contact

Questions about the released materials, data restrictions, or potential access to additional de-identified examples may be submitted through the repository’s Issues page.
