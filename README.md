# jev-experimentation

Build a **System 1 typed decision model**, in the style of TypeSafe's Jev and Convai's
open-source Laya, from a pretrained Hugging Face encoder.

A System 1 model takes a *state* (text or JSON) plus named, typed questions and returns a
calibrated probability distribution for each one in a single encoder pass. It generates no
text, so there is nothing to parse:

| Type | Question | Answer |
| --- | --- | --- |
| `choice` | Pick one of the options defined in the request | `choice`, per-option `probabilities`, `confidence` |
| `score` | Rate on an ordered rubric | expected `score`, per-level `probabilities` |
| `noul` | Is this statement true? | `noul` = P(true) |

**Interactive explainer:** <https://manujmehrotra.github.io/jev-experimentation/>.
It covers the `[MASK]` sequence builder, an RLCD playground, a temperature-calibration
demo, and the results of this repository's own training runs.

## What is in the repository

```
src/jev_experimentation/
  serialization.py   [CLS] <type> question: … [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
  model.py           MarkerDecisionModel (Laya-style) + TypedDecisionModel (cross-encoder baseline)
  rlcd.py            proper scoring reward, Gaussian group sampling, REINFORCE + group baseline
  calibration.py     per-type temperature fitting, ECE, reliability bins, decision metrics
  data.py            LocalLLaMA/typed-decisions loader, case-level splits, collation
  training.py        length-bucketed batches, full-split logit collection
  agent.py           DecisionAgent.predict(state, questions) with the Jev/Laya request shape
notebooks/
  train_jev_style_decision_model.ipynb   end-to-end: data → model → RLCD → calibration → eval
docs/
  index.html         GitHub Pages site (interactive)
  model-study.md     sourced study of Jev and Laya: what is public, what is not
  assets/runs/       metrics exported by the notebook, read by the site
tests/               download-free unit tests (fake tokenizer and tiny encoder)
```

## The model in one paragraph

Each question becomes one sequence. The instructions come first, then one `[MASK]` marker
per option followed by the option's text, then the state. A pretrained bidirectional
encoder contextualises every token. A learned question-type embedding is added, and two
transformer layers refine the result. The hidden state at each marker is scored to one
logit, and the logits are softmaxed over that question's options after dividing by a
per-type temperature. The answer space is set by the request, so a new schema needs no new
output layer. All questions about one state are batched into a single forward pass. This
mirrors the design in [Laya's released code](https://github.com/NandhaKishorM/laya)
(Apache-2.0). Jev's internals are not public.

Training uses **RLCD** (reinforcement learning for calibrated decisions). The logits are
treated as the mean of a Gaussian policy. Each step samples a group of noisy logit
vectors and rewards each one with strictly proper scoring rules against the soft gold
distribution: log score plus spherical score, minus the ranked probability score for
ordinal questions. The update is REINFORCE with a group-mean baseline, plus soft
cross-entropy guidance. Temperatures are then fitted on held-out *cases*.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"
pytest

# Train (defaults: ModernBERT-base on CUDA, BERT-small on Apple MPS / CPU)
cd notebooks && jupyter lab train_jev_style_decision_model.ipynb
# or headless, with overrides:
JEV_ENCODER=answerdotai/ModernBERT-base JEV_EPOCHS=4 \
  jupyter nbconvert --to notebook --execute train_jev_style_decision_model.ipynb
```

You can also run it in
[Colab](https://colab.research.google.com/github/MANUJMEHROTRA/jev-experimentation/blob/main/notebooks/train_jev_style_decision_model.ipynb)
on a GPU runtime.

Other options are `JEV_OBJECTIVE=ce` (cross-entropy-only ablation), `JEV_QUICK=1`
(a smoke test on a data slice) and `JEV_BATCH_SIZE`. Each full run writes
`docs/assets/runs/<encoder>-<objective>.json`, and the site picks it up automatically.

```python
from transformers import AutoTokenizer
from jev_experimentation import DecisionAgent, MarkerDecisionModel

model = MarkerDecisionModel.load_pretrained("notebooks/outputs/ModernBERT-base/model")
tokenizer = AutoTokenizer.from_pretrained("notebooks/outputs/ModernBERT-base/model/tokenizer")
agent = DecisionAgent(model, tokenizer)
agent.predict({"body": "Billed twice, refund it today."}, {
    "refund": {"type": "noul", "instructions": "The customer asks for a refund."},
})
```

## Results

No full training run is committed yet. The notebook has been smoke-tested end to end on a
data slice. Run it on a GPU (Colab: about 4 epochs of ModernBERT-base) and commit
`docs/assets/runs/*.json`. The site's **Our runs** section then shows each run against three baselines
(majority label, training prior, untrained head) and the publisher-reported numbers on the
same test split (Jev 1.13.0 at 0.727 and fine-tuned Laya at 0.766). Those reference numbers
come from a 421M checkpoint that was already RLCD-trained on broad decision data, so treat
them as orientation rather than a controlled comparison.

## Scope and honesty

- This is a study reproduction. It is not affiliated with TypeSafe or Convai Innovations.
- Jev cannot be replicated from public information; only its interface and behaviour can be
  compared. See [docs/model-study.md](docs/model-study.md) for what each source does and does
  not say.
- Laya-derived design choices follow its Apache-2.0 code. This repository's code is MIT.
