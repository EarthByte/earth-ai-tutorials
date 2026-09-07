import sys
sys.path.insert(0, ".")
from nb_helper import md, code, save

cells = []

cells.append(md("""
# Session 7 — Generative AI and LLMs in Geoscience Research

**Data Science and AI Applied to Understanding Earth Evolution**

Two things happen in this session:

1. **Discussion:** where LLM tools (ChatGPT, Claude, Copilot, and similar) genuinely help a
   geoscience researcher, and where they confidently produce wrong stratigraphy, wrong ages,
   or citations that don't exist. Bring an example if you have one.
2. **Hands-on:** every code cell marked **BUGGY** below is broken on purpose, using the same
   patterns from Sessions 2, 4 and 5. Paste the code and its error message into an LLM tool,
   ask it to explain what's wrong *before* asking for a fix, then apply the fix in the cell
   underneath and re-run to check it actually works.

### Prompting an LLM for code help, well

- Paste the **whole** error message, not just the last line — the traceback often names the
  exact line and variable.
- Say what you *expected* to happen, not just "this doesn't work."
- Ask it to **explain** the bug before showing the fix — you're upskilling, not outsourcing.
- Always run the fix yourself and check the output makes sense. An LLM can be confidently
  wrong, especially about anything geoscience-specific — the next exercise has an example of a
  bug it might not even notice.
"""))

cells.append(md("""
### How to read a Python error message (a "traceback")

Before the exercises: when a code cell fails, Python doesn't just say "error" — it prints a
**traceback**, and reading it is a genuine skill worth having independent of any LLM tool.
Take this unrelated example (don't run it, it's just illustration — `mystery_value` is never
created anywhere, on purpose):

```
print(mystery_value)
```

This would print something like:

```
Traceback (most recent call last):
  Cell In[5], line 1
----> 1 print(mystery_value)

NameError: name 'mystery_value' is not defined
```

Three things worth knowing how to read there, from the bottom up (the bottom is usually the
most useful part, so read tracebacks bottom-to-top, not top-to-bottom):

1. **The last line** — `NameError: name 'mystery_value' is not defined` — names the *type* of
   error (`NameError`) and, after the colon, the specific problem. This line alone often tells
   you exactly what's wrong.
2. **The arrow (`---->`)** — points at the exact line that failed. In a longer cell with many
   lines, this is how you find the one that actually broke, rather than reading the whole cell
   top to bottom.
3. **"Cell In[5]"** — which cell it happened in, useful once a notebook has many cells.

Every exercise below produces a traceback shaped like this one (except Exercise 5 — read its
note carefully, that one is different on purpose). Practice reading the last line and the
arrow *yourself* before pasting anything into an LLM tool — most of these are fast to spot
once you know to look at exactly those two things.
"""))

cells.append(md("""
## Exercise 1 — BUGGY

This is meant to refit Session 2's spreading-rate regression and print the result. Running it
throws an error.
"""))

cells.append(code("""
# BUGGY -- run this cell, read the error, then fix it in the cell below.
import numpy as np
from sklearn.linear_model import LinearRegression

age = np.array([5, 12, 20, 33, 41, 50])
distance = np.array([22, 46, 81, 130, 165, 205])

reg = LinearRegression()
reg.fit(age.reshape(-1, 1), distance)

print("Fitted half-rate:", linear_model.coef_[0], "km/Myr")
"""))

cells.append(md("**Your fix here:**"))
cells.append(code("""
# paste your corrected version here and run it

"""))

cells.append(md("""
## Exercise 2 — BUGGY

This is meant to train Session 2's rock classifier. Running it throws an error.
"""))

cells.append(code("""
# BUGGY -- run this cell, read the error, then fix it in the cell below.
import numpy as np
from sklearn.linear_model import LogisticRegression

density = np.array([2.95, 2.65, 2.80, 2.70, 2.90])
silica  = np.array([50, 68, 60, 65])   # five rock samples were measured...
labels  = np.array(["mafic", "felsic", "mafic", "felsic", "mafic"])

X = np.column_stack([density, silica])
clf = LogisticRegression().fit(X, labels)
print("Trained on", len(labels), "samples")
"""))

cells.append(md("**Your fix here:**"))
cells.append(code("""
# paste your corrected version here and run it

"""))

cells.append(md("""
## Exercise 3 — BUGGY

This is meant to split a small proxy dataset into training and test sets, Session 4-style.
Running it throws an error.
"""))

cells.append(code("""
# BUGGY -- run this cell, read the error, then fix it in the cell below.
proxy = [12.1, 14.3, 9.8, 15.2, 11.0]
ages  = [10, 20, 30, 40, 50]

X_train, X_test, y_train, y_test = train_test_split(ages, proxy, test_size=0.4, random_state=0)
print(len(X_train), "training points,", len(X_test), "test points")
"""))

cells.append(md("**Your fix here:**"))
cells.append(code("""
# paste your corrected version here and run it

"""))

cells.append(md("""
## Exercise 4 — BUGGY

This mirrors Session 5's feature-lookup pattern. Running it throws an error.
"""))

cells.append(code("""
# BUGGY -- run this cell, read the error, then fix it in the cell below.
import numpy as np

feature_lookup = {"distance_to_fault": [12, 30, 8, 44], "geochem": [1.2, 0.4, 2.1, 0.9]}
FEATURES_TO_USE = ["distance_to_fault", "geochemistry"]

X = np.column_stack([feature_lookup[f] for f in FEATURES_TO_USE])
print(X.shape)
"""))

cells.append(md("**Your fix here:**"))
cells.append(code("""
# paste your corrected version here and run it

"""))

cells.append(md("""
## Exercise 5 — BUGGY, but it won't throw an error

This one is different: it runs perfectly fine and prints a normal-looking number. It's meant
to flag grid cells **at or above** the prospectivity threshold as target zones (Session 5's
logic). It doesn't do that — read it carefully rather than just running it.

This is the important caution for the whole session: **an LLM asked to "find the bug" will
often say the code looks fine**, because nothing crashes. It can only catch this kind of bug if
you tell it what output you *expected* and let it work backwards from there.
"""))

cells.append(code("""
# BUGGY (silently) -- runs without error, but the logic is wrong. What SHOULD the answer be,
# just by looking at the five probabilities below, before you run it?
import numpy as np

predicted_probability = np.array([0.1, 0.8, 0.05, 0.95, 0.3])
PROSPECTIVITY_THRESHOLD = 0.5

target_zone = predicted_probability <= PROSPECTIVITY_THRESHOLD

print("Number of target zone cells:", target_zone.sum())
"""))

cells.append(md("**Your fix here — and a one-line note on what made this bug harder to spot than Exercises 1-4:**"))
cells.append(code("""
# paste your corrected version here and run it

"""))

cells.append(md("""
## Exercise 6 — write it from a description, with an LLM

Ask an LLM tool: *"Write a short Python snippet using NumPy that computes the mean and
standard deviation of a list of numbers called `ages`, and prints both rounded to 1 decimal
place."* Paste what it gives you below, run it on the `ages` list already provided, and check
by eye that the printed numbers look right for that list.
"""))

cells.append(code("""
ages = [42.1, 38.7, 55.0, 29.3, 61.4, 47.8, 33.2]

# paste the LLM's snippet here (adapt the variable name if needed) and run it

"""))

cells.append(md("""
## Wrap-up

Every bug in this notebook was small and mechanical — a typo, a missing import, a mismatched
length, a flipped comparison. That's deliberate: these are exactly the kind of errors a weak
programming background runs into constantly, and exactly the kind an LLM tool is genuinely
good at helping with, fast. The one to remember is Exercise 5: an LLM (like a careless human
reviewer) will happily wave through code that runs and produces a plausible-looking number —
catching that requires knowing, independently, what the answer *should* look like. That's the
skill no tool can substitute for, and it's exactly what Session 8 discusses next.
"""))

save(cells, "../session07_genai_assisted_coding/S07_GenAI_Assisted_Debugging.ipynb")
