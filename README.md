# Data Science and AI Applied to Understanding Earth Evolution

Module 2 of a two-module Honours (4th-year) sequence, paired with *Earth Evolution —
Spatio-Temporal Data and Model Analysis*. 25 contact hours, 10 sessions. Full session-by-session
rationale, learning outcomes, and assessment weighting are in the companion planning document;
this folder is the actual teaching material.

Every notebook follows the same pattern as the sibling module's `pygplates` workflows: a
**`# === USER CONFIGURATION ===`** cell near the top marks what students are meant to change,
followed by cells that run without needing to be understood line-by-line on first read. No
notebook requires writing code from a blank cell — every session asks students to run real,
working code and then deliberately change something specific in it.

**Written for students with little or no programming background.** Every notebook explains new
Python/library concepts in plain language the first time they're used — not just what a line of
code does, but why — rather than assuming familiarity. Session 1 opens with a five-minute primer
on notebooks and Python basics before any real content, and Session 7 includes a short guide to
reading a Python error message ("traceback"). **`GLOSSARY.md`**, at the top level of this folder,
collects plain-language definitions of every technical and Python term used across all ten
sessions in one place — point students to it any time unfamiliar vocabulary comes up.

## Session map

| # | Folder | Material | Format |
|---|---|---|---|
| 1 | `session01_statistical_foundations` | `S01_Statistical_Foundations.ipynb` | Notebook — real data, hands-on |
| 2 | `session02_bestfit_and_sorting` | `S02_BestFit_and_Sorting_Machine.ipynb` | Notebook |
| 3 | `session03_neural_networks` | `S03_Neural_Networks_Hands_On.ipynb` | Notebook (pair with TensorFlow Playground) |
| 4 | `session04_ancient_climate` | `S04_Reading_Ancient_Climate.ipynb` | Notebook |
| 5 | `session05_ore_deposits` | `S05_Finding_Hidden_Ore_Deposits.ipynb` | Notebook |
| 6 | `session06_speeding_up_models` | `S06_Speeding_Up_a_Slow_Earth_Model.ipynb` | Notebook |
| 7 | `session07_genai_assisted_coding` | `S07_GenAI_Assisted_Debugging.ipynb` | Notebook (deliberately buggy — see instructor notes) |
| 8 | `session08_can_we_trust_it` | `S08_Discussion_Guide.md` | Discussion, no notebook |
| 9 | `session09_reading_a_paper` | `S09_Reading_a_Paper_Together.md` | Reading/discussion, no notebook |
| 10 | `session10_capstone` | `S10_Capstone_Instructions.md` | Capstone (adapts a Session 2/4/5/6 notebook) |

`instructor_notes/` has the full session-by-session timing plan, the external
GeoSMART/CSU material each session pairs with, licensing/attribution, and the Session 7 answer
key. `_build/` holds the Python scripts that generate each notebook — useful if you want to
tweak a notebook's content later without hand-editing JSON (edit the relevant
`build_sessionNN.py`, then re-run it; comments explaining Python/library concepts live inline in
these scripts too, so edits should keep the same beginner-friendly commenting style).

## Running the notebooks

**Google Colab (recommended for students):** no installation needed.
- Fastest: open [colab.research.google.com](https://colab.research.google.com/), then
  **File → Upload notebook** and pick the `.ipynb` file.
- One-click links: push this whole folder to a GitHub repository, then a badge/link of the
  form `https://colab.research.google.com/github/<your-username>/<repo>/blob/main/<path-to-notebook>.ipynb`
  opens any notebook directly — worth setting up once if you'll reuse this module across years.

**Local (instructor machine, or students with a working Python setup):**
```
pip install -r requirements.txt
jupyter notebook
```

## What's real and what's synthetic

Session 1 uses **real measurements**: 40 rock density values and 58 paired lead/zinc
geochemical concentrations, both drawn from a University of Sydney statistics-for-geoscientists
course. Every other dataset, in every other notebook, is **synthetic**, generated in the
notebook itself so nothing needs to be downloaded and so the "true" underlying relationship is
always available for comparison. Each synthetic dataset is built to be realistic in shape and
grounded in a real geoscience relationship (seafloor spreading kinematics, porphyry alteration
zoning, Cenozoic-style cooling), and every notebook says so explicitly where it matters.
Session 9's paper (Farahbakhsh et al., 2025) and its bridge to Module 1's T73/T75 notebooks are
the other place this module points at real, published, external data and results.

## Attribution

These notebooks are original material written for this module, not copies of GeoSMART's or
CSU's notebooks — GeoSMART ("Machine Learning in the Geosciences",
[geo-smart.github.io/mlgeo-book](https://geo-smart.github.io/mlgeo-book/), CC-BY-4.0
text/figures, MIT code) and CSU's `ml_tutorial_csu`
([github.com/eabarnes1010/ml_tutorial_csu](https://github.com/eabarnes1010/ml_tutorial_csu),
MIT) are used as the live-demo/backbone resources named in the course outline, credited where
they're used, not embedded here. Session 1's real density and geochemistry data are reused,
with permission, from R.D. Müller's own historical "Statistics for Geoscientists" course
material at the University of Sydney — not from GeoSMART, CSU, or any external published
source. Full attribution detail is in `instructor_notes/`.

## License

This material is released under the [MIT License](LICENSE).
