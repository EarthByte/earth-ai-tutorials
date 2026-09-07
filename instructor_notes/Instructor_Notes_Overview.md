# Instructor Notes — Data Science and AI Applied to Understanding Earth Evolution

Working notes for running the module: timing, what to demo live vs. what students run
themselves, licensing, and setup. Session-by-session pedagogy and rationale live in the
companion planning document; this file is the practical companion for actually teaching it.

**On the documentation level:** every notebook is written assuming a weak or non-existent
programming background — new Python and library concepts are explained in plain language the
first time they appear, not just demonstrated. This is by design, not padding: the module's
premise is that students who can't code yet still need real, scaffolded coding practice, not a
zero-code substitute. The top-level `GLOSSARY.md` collects every technical/Python term used
across the whole module; it's worth mentioning to students in Session 1 so they know it exists
before they need it. If you edit a notebook via its `_build/build_sessionNN.py` script, keep new
code commented at the same level of explanation as the surrounding cells.

## Session-by-session timing (2.5 h blocks)

| # | Suggested breakdown | External material to show/demo | Student notebook |
|---|---|---|---|
| 1 | 10 min framing (why start with statistics, not AI) · 115 min `S01` hands-on (density and geochemistry, rank correlation, and directional statistics — rose diagrams, the Rayleigh test, and a Schmidt net) · 25 min wrap/discussion of why geochemists work in log space | — | `S01_Statistical_Foundations.ipynb` |
| 2 | 20 min intro (regression/classification analogies) · 90 min `S02` hands-on (both parts) · 20 min optional Teachable Machine demo · 20 min wrap | GeoSMART or CSU "Classic ML" notebook, as a second worked example if time allows | `S02_BestFit_and_Sorting_Machine.ipynb` |
| 3 | 20 min TensorFlow Playground (individually) · 90 min `S03` hands-on · 20 min GeoSMART "Deep Learning" notebook, live · 20 min wrap | TensorFlow Playground; GeoSMART Deep Learning | `S03_Neural_Networks_Hands_On.ipynb` |
| 4 | 15 min framing · 100 min `S04` hands-on, including the overfitting exercise · 35 min discussion of what a "reconstructed climate curve" actually is | — | `S04_Reading_Ancient_Climate.ipynb` |
| 5 | 15 min framing via the Farahbakhsh paper and its real prospectivity workflow · 100 min `S05` hands-on · 35 min discussion | — | `S05_Finding_Hidden_Ore_Deposits.ipynb` |
| 6 | 15 min framing (why simulations are slow) · 60 min `S06` hands-on (includes a ~25 s training wait — flag this in advance so it reads as "the point," not a hang) · 55 min discussion of the extrapolation danger, feeding into Session 8 | — | `S06_Speeding_Up_a_Slow_Earth_Model.ipynb` |
| 7 | 15 min on prompting an LLM well · 100 min `S07` debugging exercises (work in pairs) · 35 min share-out: which bug was hardest, and why Exercise 5 is different | — | `S07_GenAI_Assisted_Debugging.ipynb` (see answer key below) |
| 8 | Full session: discussion, `S08_Discussion_Guide.md` | GeoSMART, Workflows/Reproducibility & Rigor | — |
| 9 | Full session: close reading, `S09_Reading_a_Paper_Together.md` | Farahbakhsh et al. (2025); optionally browse `e-farahbakhsh/GPlates_Workflows` | — |
| 10 | Capstone work time + presentations, `S10_Capstone_Instructions.md` | GeoSMART, Communicating Your Science | Student's adapted Session 2/4/5/6 notebook |

Sessions 4-6 are the most compressible if the term runs short (see the planning document); they
introduce no new technical idea, so cutting hands-on time there costs less than cutting
Sessions 2, 3, or 7.

## Session 7 — running a "deliberately broken" notebook

`S07_GenAI_Assisted_Debugging.ipynb` is shipped with five buggy code cells that error (or, in
one case, silently return a wrong answer) on purpose — this has been verified, not just
written and assumed:

- Exercises 1, 3: `NameError` (a typo'd variable, a missing import)
- Exercise 2: `ValueError` (mismatched array lengths)
- Exercise 4: `KeyError` (a typo'd dictionary key)
- Exercise 5: **no error at all** — the logic is silently backwards. This one is the important
  teaching moment: an LLM tool asked to "find the bug" will often say the code looks fine,
  because nothing crashes.

The full corrected code for all five, plus the exact expected numbers, is in
`session07_answer_key.md` in this folder — **do not hand it to students before the session.**

## Licensing and attribution

- **GeoSMART, "Machine Learning in the Geosciences"**
  ([geo-smart.github.io/mlgeo-book](https://geo-smart.github.io/mlgeo-book/)) — text and
  figures CC-BY-4.0, code MIT. Used as a live-demo/backbone resource named in the course
  outline for Sessions 2-3, 8 and 10; not reproduced in this folder. Credit GeoSMART explicitly
  whenever its notebooks or sections are shown in class.
- **CSU, `ml_tutorial_csu`** ([github.com/eabarnes1010/ml_tutorial_csu](https://github.com/eabarnes1010/ml_tutorial_csu)) —
  MIT licensed. Alternative live-demo material for Sessions 2-3.
- **Session 1's real data** — rock density, Pb/Zn geochemistry, a two-species depth/size dataset,
  and a fault dip/dip-direction survey — are reused, with permission, from R.D. Müller's own
  historical "Statistics for Geoscientists" course material at the University of Sydney — not
  from GeoSMART, CSU, or any external published source.
- **All notebooks in this folder** (`S01`-`S07`) are original material, written for this
  module — they are inspired by the same two ideas GeoSMART and CSU teach, but are not derived
  from or copies of either project's notebooks.
- **Farahbakhsh et al. (2025), Tectonics** — cited and read as a primary source in Sessions 5
  and 9; not reproduced. Full citation and DOI in `S09_Reading_a_Paper_Together.md`.

## Setup

- **Students:** Google Colab, no installation — see the top-level `README.md` for the two ways
  to open a notebook there.
- **Instructor / local fallback:** `pip install -r requirements.txt` from the top level, then
  `jupyter notebook`. Every notebook only depends on NumPy, SciPy, Matplotlib, and
  scikit-learn — no GPU, no large downloads, nothing beyond what a standard Colab or conda
  environment already has.
- Session 6's training step runs the "slow model" 25 times (~20-30 seconds total on a typical
  laptop; can run somewhat slower on Colab's free tier) — worth testing once on whatever
  environment students will actually use, since the whole point of the session is that wait
  being felt, not skipped past.
