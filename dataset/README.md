---
pretty_name: Source-Report Trace-Transfer Benchmark
license: cc-by-4.0
task_categories:
- question-answering
language:
- en
configs:
- config_name: default
  data_files:
  - split: development
    path: data/development.jsonl
  - split: evaluation
    path: data/evaluation.jsonl
---

# Source-Report Trace-Transfer Benchmark

This benchmark contains 280 paired mathematical problems for studying how reasoning models describe a supplied reasoning trace while answering a question. Each pair has a source question, $Q_s$, and a conflict question, $Q_c$, that changes one numerical value. The wording and solution method remain the same, while the verified answers differ.

The pairs support a controlled reasoning-trace transfer experiment. A model first solves $Q_s$, producing a trace in its native reasoning format. The same trace is then supplied in two reporting requests: one with $Q_s$ and one with $Q_c$. The model is asked to quote the relevant evidence from the trace and current question separately, compare them, and solve the current question. Holding the trace fixed makes its later description directly checkable.

## Contents

- `data/evaluation.jsonl` contains the 280 evaluation pairs.
- `data/development.jsonl` contains two separate pairs for checking an experimental setup.

Each JSONL line is one paired item. Both splits use the same fields.

## Task coverage

| Mathematical family | Evaluation pairs |
| --- | ---: |
| Arithmetic | 40 |
| One-variable algebra | 40 |
| Calculus and functions | 40 |
| Two-variable linear systems | 40 |
| Three-variable linear systems | 40 |
| Probability and combinatorics | 40 |
| Word problems | 40 |
| **Total** | **280** |

Each family has five question templates with eight numerical variants per template. In half of the evaluation pairs, the edited quantity increases; in the other half, it decreases. The changed value need not move the answer in the same direction.

## Data fields

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier for the paired item |
| `split` | `development` or `evaluation` |
| `family` | Mathematical problem family |
| `template_id` | Wording and problem subtype shared by related pairs |
| `source_question` | Question used to generate the source reasoning trace, $Q_s$ |
| `matched_question` | The same question used in the matched reporting request |
| `conflict_question` | Locally changed question used in the conflict request, $Q_c$ |
| `source_answer` | Verified answer to $Q_s$ |
| `conflict_answer` | Verified answer to $Q_c$ |
| `changed_field` | Type of quantity changed between the questions |
| `source_value`, `conflict_value` | Values of the changed quantity in $Q_s$ and $Q_c$ |
| `direction` | Whether that quantity increases or decreases |

Answers and changed values are stored as strings, preserving exact integer and rational forms. The question fields are the task inputs; the answer fields provide references for evaluating generated responses.

## Example

One algebra pair changes the right-hand side of an equation from 43 to 38:

```text
source_question:   Solve the equation (x + 14) / 2 = 43 for x.
conflict_question: Solve the equation (x + 14) / 2 = 38 for x.
source_answer:     x = 72
conflict_answer:   x = 62
```

A source trace solving the first equation can be placed before either question. Under the conflict question, an accurate report identifies 43 as the value in the trace and 38 as the value in the current question; the correct current answer is $x=62$.

## Using the benchmark

For a paired source-report evaluation:

1. Ask a model to solve `source_question` and record its reasoning-channel tokens.
2. Present `matched_question` with that trace, then collect the model's report and answer.
3. Present `conflict_question` with the identical trace, then collect a second report and answer.
4. Compare each report's claims about the trace with the recorded trace, and compare its answer with the verified answer to the current question.

The two reporting requests should use the same instruction, model revision, chat template, decoding settings and transferred trace. The supplied trace should come from the model being evaluated so its native reasoning format is preserved.

## Loading the data

Load the evaluation split from Hugging Face:

```python
from datasets import load_dataset

pairs = load_dataset(
    "VikramMV/source-report-trace-transfer-benchmark",
    split="evaluation",
)
print(pairs[0])
```

## Code

The [Source Report Experiments](https://github.com/SrivikramMV/source-report-experiments) repository provides a model-selectable runner for Gemma 4 E2B and GPT-OSS 20B. It records the source reasoning trace and the matched and conflict reports as JSONL.

## Limitations

The problems are synthetic and concentrated in exact-answer mathematical reasoning. Numerical variants within a template share substantial structure, so the 280 pairs represent 35 question templates across seven families. The single-value edits create a controlled disagreement between the trace and current question; behaviour may change with other domains, reporting instructions or reasoning interfaces. Results should be interpreted alongside the model and collection settings used to generate the traces and reports.
