# Student Manual

## Data Science and AI Applied to Understanding Earth Evolution

Honours (4th-year) Module · School of Geosciences, Faculty of Science, University of Sydney

---

## About this module

*Data Science and AI Applied to Understanding Earth Evolution* is a 25-hour Honours module,
taught as ten 2.5-hour sessions. It is the second of two paired modules — the first, *Earth
Evolution — Spatio-Temporal Data and Model Analysis*, teaches you to build and interrogate
plate-tectonic reconstructions using GPlately and pyGMT. This module asks a different set of
questions: how do you describe real geological data honestly with numbers, and where does
artificial intelligence and machine learning actually enter Earth-evolution research — what is
it good at, and where does it quietly go wrong?

You will not leave this module able to design a machine-learning system from scratch, and that
is deliberate. The aim is narrower and more achievable: to compute and interpret the summary
statistics any real dataset deserves before any modelling starts, to open a real, working AI/ML
notebook, understand what each part of it is doing, deliberately change something in it,
correctly predict what should happen as a result, and recognise the difference between a
method that is working well and one that has quietly started extrapolating beyond what it
actually knows. Some notebooks in the companion module — the detrital-zircon tectonic-setting
predictor, the porphyry-copper prospectivity maps — are themselves applications of the same
ideas this module teaches from first principles.

## Who this module is for

No prior programming, scripting, or machine-learning experience is assumed or required. If you
have never written a line of Python before, you are the intended audience, not an exception to
it. What is assumed is basic computer literacy — using a web browser, managing files, and
having (or being willing to create) a Google account, since every notebook runs in Google Colab
and needs nothing installed on your own machine.

A few sessions refer back to specific notebooks and a specific paper from the companion module,
*Earth Evolution — Spatio-Temporal Data and Model Analysis* (T15, T24, and T73–T80, and
Farahbakhsh et al. 2025). Having taken that module, or being generally familiar with
plate-tectonic reconstructions, makes those references land a little more easily, but each one
is explained again in context here, so it is not a prerequisite.

If a notebook's code is genuinely mysterious after reading its explanations, that is a sign to
ask, not a sign you don't belong in the room — the module is built around the assumption that
this will happen to most students, most weeks, at least once. A companion glossary
(`GLOSSARY.md`) defines every technical, statistical, and Python term used across all ten
sessions, and is worth keeping open in a separate tab as you work.

## Learning outcomes

By the end of the module, you should be able to:

1. Compute and interpret the mean, median, standard deviation, and percentiles of a real
   geological dataset, build a histogram with a defensible number of bins, and recognise when a
   log transform is appropriate.
2. Open a provided Jupyter/Colab notebook, run it cell by cell, and explain in plain language
   what each stage is doing — for both a regression (best-fit line) and a classification
   (sorting machine) workflow.
3. Deliberately change a small number of things in a working notebook — which column, which
   threshold, which region or time window, how many layers in a small neural network — and
   correctly predict, then check, how the output should change.
4. Recognise a basic coding error (a typo, a wrong file path, a mismatched variable name) from
   its error message and fix it, or know when to ask for help.
5. Recognise the regression/classification idea at work in at least four different
   Earth-evolution contexts — plate kinematics, paleoclimate, mineral exploration, and speeding
   up a slow numerical model — and describe, in each case, what the input data actually is and
   what is being predicted.
6. Use a generative AI/LLM tool sensibly to help write or debug a short piece of code, and more
   broadly in your own research — knowing what it is good for and where it confidently gets
   geoscience facts wrong.
7. Read a real, published Earth-science AI paper and identify, in plain terms, what it claims,
   what evidence supports the claim, and one thing you would want to see before trusting it.

## How the module works

The module opens with real data, not AI: Session 1 works entirely with real measurements — rock
densities and geochemical concentrations — computing the same summary statistics a geologist
would compute in any field report, before any model of any kind gets fitted to anything. Every
notebook from Session 2 onward then follows the same pattern as the companion module's
`pygplates` workflows: a **`# === USER CONFIGURATION ===`** cell near the top marks the handful
of things you are meant to change, followed by cells that run correctly without needing to be
understood line by line on first read. You are never handed a blank cell and asked to write
code from nothing — every session asks you to run real, working code, then deliberately change
something specific in it and see what happens.

Because a weak or absent programming background is the expected starting point here, not the
exception, every notebook explains new Python and statistical concepts in plain language the
first time they come up — not just what a line of code does, but why it is written that way.
Session 1 opens with a five-minute primer on notebooks and Python basics before any real
content, and Session 7 includes a short guide to reading a Python error message, since learning
to read an error message calmly is itself one of the module's aims.

Two ideas carry almost the entire second half of the module: **regression** (fitting the best
curve through a set of points) and **classification** (sorting things into categories based on
what you can measure). Once you have those from Session 2, Sessions 4, 5, and 6 are the same
two ideas again, in a new Earth-science context each time, rather than new technical material —
which is what makes ten sessions of genuine hands-on work achievable in 25 hours for a cohort
with no assumed programming background.

## Session-by-session schedule

| # | Session | Format | What happens |
|---|---|---|---|
| 1 | Statistical foundations: describing real geological data | Notebook — real data | Using real rock density measurements and real lead/zinc geochemical concentrations, compute and interpret the mean, median, standard deviation, and percentiles of a dataset; build a histogram with a defensible bin count; and see first-hand why right-skewed geochemical data is usually analysed after a log transform. |
| 2 | The best-fit line and the sorting machine | Notebook | The two core ideas of the module's second half — regression and classification — introduced by analogy, then applied to real rock-property data, reusing Session 1's summary statistics to compare two rock types before a computer learns to sort them automatically. |
| 3 | What is a neural network, really? | Notebook (paired with TensorFlow Playground) | TensorFlow Playground first, for visual intuition — drag sliders, watch a network's decision boundary grow more complex. Then a real neural-network notebook where you change the number of layers or neurons and observe the effect. |
| 4 | Reading Earth's ancient climate | Notebook | The regression idea again, applied to a noisy climate proxy record. You fit a curve to real-shaped proxy data, then change which proxy or time window is used and read the new curve — including a hands-on look at what overfitting actually looks like. |
| 5 | Finding hidden ore deposits | Notebook | The classification idea again, applied to a simplified mineral-prospectivity model in the spirit of Farahbakhsh et al. (2025) — the paper behind the companion module's porphyry-copper notebooks. You change a threshold or input feature and watch the predicted "target zone" map respond. |
| 6 | Speeding up a slow Earth model | Notebook | A case study in why some Earth simulations are too slow to explore many scenarios directly, and how a fast, trained model can stand in for a slow one. The lightest hands-on session by design — the concept matters more than the code here. |
| 7 | Generative AI and LLMs in geoscience research | Notebook (deliberately buggy) | Practical, discussion-led: where LLM tools genuinely help with literature search, drafting, and code, and where they confidently invent wrong stratigraphy, ages, or citations. Hands-on: use an LLM to help debug five small, deliberately broken scripts from earlier sessions. |
| 8 | Can we trust it? | Discussion, no notebook | Why deep time is a genuinely hard case for AI — there is almost no way to check a 300-million-year-old prediction against reality — anchored in code and results you've already produced yourself in Sessions 2–6. |
| 9 | Reading a real AI paper, together | Reading/discussion, no notebook | A guided, section-by-section class reading of Farahbakhsh et al. (2025): what it claims, what evidence supports it, and what would make you doubt it — the paper behind Session 5's model. |
| 10 | Capstone | Independent notebook adaptation + presentation | You adapt one notebook from Session 2, 4, 5, or 6 to a new dataset, region, or parameter of your own choosing, write a short reflection, and give a five-minute presentation on what changed, what happened, and one open question. |

## Assessment

Participation across Sessions 1–9 makes up roughly 40% of the module mark; the Session 10
capstone makes up the remaining 60%, assessed on four things: whether the adapted notebook
runs and represents a genuine (if modest) extension rather than a re-run with different
numbers; whether the written reflection honestly describes what happened, including anything
unexpected; whether the open question raised is a real, specific concern rather than a generic
disclaimer; and whether the presentation is clear, to time, and honest about what worked and
what didn't. Check your specific unit-of-study outline for the exact weightings used this year.

## Getting set up

Everything runs in **Google Colab** — free, browser-based, and requires nothing to be installed
on your own machine, which removes the single biggest friction point for anyone without a
working local Python setup. To open a notebook: sign in to a Google account, go to
[colab.research.google.com](https://colab.research.google.com/), then **File → Upload
notebook** and select the `.ipynb` file for that session.

If you would rather work locally and already have a Python environment you're comfortable
with, every notebook only depends on NumPy, Matplotlib, and scikit-learn:

```
pip install -r requirements.txt
jupyter notebook
```

Colab is what's demonstrated in class and is the recommended option — it removes any risk of a
broken local environment eating into session time.

## Resources

- **`GLOSSARY.md`** — plain-language definitions of every technical, statistical, and Python
  term used across the module. Start here whenever a word in a notebook doesn't make sense.
- **GeoSMART, "Machine Learning in the Geosciences"**
  ([geo-smart.github.io/mlgeo-book](https://geo-smart.github.io/mlgeo-book/)) — the backbone
  reference for the module's regression/classification content, its section on workflow rigor
  (Session 8), and its guidance on communicating results (Session 10).
- **CSU's `ml_tutorial_csu`**
  ([github.com/eabarnes1010/ml_tutorial_csu](https://github.com/eabarnes1010/ml_tutorial_csu))
  — an alternative worked-example notebook set covering similar ground to Sessions 2–3.
- **TensorFlow Playground** ([playground.tensorflow.org](https://playground.tensorflow.org/))
  — Session 3's zero-code warm-up for building intuition about neural networks.
- **Elements of AI** ([elementsofai.com](https://www.elementsofai.com/)) — optional
  general-interest pre-reading, ideally before Session 2.
- **Google Teachable Machine**
  ([teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com/)) — an optional,
  zero-code way to explore classification before Session 2.
- **Farahbakhsh, E. et al. (2025), *Tectonics***, "Machine Learning-Based Spatio-Temporal
  Prospectivity Modeling of Porphyry Systems in the New Guinea and Solomon Islands Region"
  ([DOI: 10.1029/2024TC008362](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024TC008362))
  — the paper Session 9 reads in full, and the basis for Session 5's model.
- **Ehsan Farahbakhsh's `GPlates_Workflows`**
  ([github.com/e-farahbakhsh/GPlates_Workflows](https://github.com/e-farahbakhsh/GPlates_Workflows))
  — the real research repository behind that paper, worth a brief browse in Session 9.

Session 1's real density and geochemistry data are not drawn from any of the above — they come
from a historical University of Sydney statistics-for-geoscientists course.

## Getting help

Ask questions during class — the module assumes most students will hit a genuinely confusing
moment most weeks, and that's expected, not a sign of falling behind. Use the glossary before
anything else when a term is unfamiliar. From Session 7 onward, treating a generative AI tool
as a legitimate way to get unstuck on a code error is itself part of what the module teaches;
the same session covers how to do that well, and where those tools confidently get
geoscience-specific things wrong.
