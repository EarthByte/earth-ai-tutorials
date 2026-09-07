import sys
sys.path.insert(0, ".")
from nb_helper import md, code, save

cells = []

cells.append(md("""
# Session 1 — Statistical Foundations: Describing Real Geological Data

**Module 2 · Data Science and AI Applied to Understanding Earth Evolution**

Before anything resembling "AI" appears in this module (Session 2 onward), this notebook
covers the more basic skill everything else is built on: describing a real dataset honestly
with a handful of numbers. No prior programming experience is assumed, and nothing here
depends on having taken any other module. The two datasets used below are real measurements,
drawn from a University of Sydney statistics-for-geoscientists course.

By the end of this notebook you should be able to:
- Run a Jupyter/Colab notebook cell by cell, and read a basic Python cell well enough to guess
  roughly what it does before running it.
- Compute the mean, median, standard deviation, and percentiles of a real dataset, and explain
  in plain language what each one tells you.
- Build a histogram with a sensible number of bins, and explain why the choice of bin count
  changes the visual impression the same numbers give.
- Recognise a right-skewed dataset from its histogram and summary statistics, and explain why
  geochemical concentration data is usually analysed after a log transform.
"""))

cells.append(md("""
## 0. Python and Jupyter notebooks, in five minutes

If you've never used a Jupyter or Colab notebook before, read this section first — everything
in it applies to every notebook in this module, so it only needs saying once.

**A notebook is a sequence of cells.** Each cell is either:
- a **text cell** (like this one — formatted text, no code, nothing to "run" in a meaningful
  sense), or
- a **code cell** (grey/shaded background, monospaced font — actual Python code that does
  something when you run it).

**Running a cell:** click on it, then press **Shift+Enter** (or click the ▶ "Run" button next
to it in Colab). This executes that cell's code and moves you to the next one. A code cell
that has been run shows a number next to it, like `[1]`, `[2]`, `[3]` — that number is just
"the 1st, 2nd, 3rd cell I've run so far," and it's how you can tell whether a cell has actually
been executed yet.

**Order matters, top-to-bottom, by *when you ran it*, not its position on the page.** If a
cell uses a variable that was created in an earlier cell, that earlier cell needs to have
already been *run* (not just be higher up on the page) — this is the single most common
source of a confusing error ("name is not defined") for anyone new to notebooks. If a
notebook ever starts behaving strangely, the fix is almost always: use the menu to **restart
the kernel and run all cells from the top**, in order.

**A few Python basics that will come up in every notebook:**

- **A comment** — anything after a `#` on a line — is a note for humans and is completely
  ignored when the code runs. Comments are how the code below explains itself.
- **A variable** is just a name that points to a value, created with `=`. `age = 42` means
  "create a variable called `age`, and set it to 42." After that line runs, typing `age`
  anywhere later gives you `42` back.
- **`import`** loads a *library* — pre-written code that does something useful so you don't
  have to write it from scratch. `import numpy as np` loads a library called NumPy (used for
  fast number-crunching on lists of numbers, called *arrays*) and lets you refer to it with the
  short nickname `np` for the rest of the notebook. Every notebook in this module starts with a
  cell like this — it always needs to be run first.
- **`print(...)`** displays whatever is inside the parentheses. It's the simplest way for code
  to show you something.

That's genuinely enough Python to follow every notebook in this module — anything more
specific (what a particular function does) is explained in a comment the first time it shows
up.
"""))

cells.append(code("""
import numpy as np
import matplotlib.pyplot as plt

print("Ready. NumPy version:", np.__version__)
"""))

cells.append(md("""
## 1. Describing a dataset with numbers

Before fitting any model — AI or otherwise — the first, most basic step with real
measurements is describing them with a handful of summary numbers. This is standard practice
in any field study or lab report. The 40 values below are real rock density measurements
(g/cm3), used in exercises from a University of Sydney statistics-for-geoscientists course.

**Central tendency — one number that summarises "the middle":**
- **mean** (the arithmetic average) — add every value, divide by how many there are.
- **median** — the middle value once every measurement is sorted. Unlike the mean, a handful
  of unusually extreme measurements barely move it, which matters for real field data that
  often has a few outliers (a mismeasured sample, a contaminated result, a transcription
  error).

**Spread — how scattered the measurements are around the middle:**
- **variance** and **standard deviation** (its square root, expressed in the same units as the
  original measurement) — how far, on average, each measurement lies from the mean.
- **range, minimum, maximum.**
- **percentiles and quartiles** — the 25th percentile is the value below which a quarter of
  the measurements fall; the 25th-to-75th-percentile span is the **interquartile range
  (IQR)**, a spread measure that, like the median, is not thrown off by a few extreme values.

A common rule of thumb defines an **outlier** as any value more than 1.5x the IQR beyond the
25th or 75th percentile — worth flagging for a second look, not automatically discarding.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
N_BINS = "auto"   # an integer (e.g. 4, 10, 20), or "auto" to use Sturges' rule (explained below)
"""))

cells.append(code("""
# 40 real rock density measurements (g/cm3).
density = np.array([
    3.2, 3.7, 2.9, 3.9, 3.4, 3.1, 3.1, 3.9, 3.5, 3.3, 3.6, 3.8, 3.7, 3.0, 3.5,
    3.2, 3.5, 3.7, 3.9, 3.6, 3.4, 2.9, 3.2, 3.4, 2.9, 3.6, 3.7, 3.3, 3.4, 4.0,
    3.8, 3.7, 3.3, 2.9, 3.1, 3.2, 3.6, 3.5, 3.3, 3.4,
])

# "if/else": run one block of code or the other, depending on whether the condition after
# "if" is True. Here: use Sturges' rule to pick a bin count automatically, unless a specific
# number was set above.
if N_BINS == "auto":
    # Sturges' rule: a common rule of thumb for how many histogram bins suit a dataset of a
    # given size n -- enough bins to reveal real structure, without so many that most bins end
    # up empty. np.log2 is the base-2 logarithm; np.ceil rounds up to the next whole number.
    n_bins = int(np.ceil(1 + np.log2(density.size)))
else:
    n_bins = N_BINS

print(f"Using {n_bins} bins for {density.size} measurements.")
"""))

cells.append(code("""
# NumPy has a direct function for each of these -- no manual formula ever needs writing out.
mean_d = np.mean(density)
median_d = np.median(density)
std_d = np.std(density)      # standard deviation
var_d = np.var(density)      # variance -- std_d squared, in squared units

min_d, max_d = np.min(density), np.max(density)

# np.percentile(values, p) finds the value below which p% of the data falls.
q25, q75 = np.percentile(density, [25, 75])
iqr = q75 - q25   # interquartile range: the spread of the middle 50% of the data

print(f"n = {density.size} samples")
print(f"mean    = {mean_d:.3f} g/cm3")
print(f"median  = {median_d:.3f} g/cm3")
print(f"std dev = {std_d:.3f} g/cm3   (variance = {var_d:.4f})")
print(f"range   = {min_d:.3f} to {max_d:.3f} g/cm3")
print(f"25th percentile = {q25:.3f}, 75th percentile = {q75:.3f}, IQR = {iqr:.3f}")

plt.figure(figsize=(6, 4))
plt.hist(density, bins=n_bins, color="#4D6480", edgecolor="white")
plt.axvline(mean_d, color="#B06E00", linewidth=2, label=f"mean = {mean_d:.2f}")
plt.axvline(median_d, color="#3F7A32", linestyle="--", linewidth=2, label=f"median = {median_d:.2f}")
plt.xlabel("Density (g/cm3)")
plt.ylabel("Number of samples")
plt.legend()
plt.title(f"40 real rock density measurements ({n_bins} bins)")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Try it

1. Set `N_BINS = 3`, re-run both cells below the USER CONFIGURATION cell, then try
   `N_BINS = 20`. The 40 numbers haven't changed at all — has the *impression* the histogram
   gives changed? Which version feels like the more honest summary of this dataset?
2. Set `N_BINS` back to `"auto"`. Sturges' rule picked a specific number of bins for you —
   print `n_bins` (add a line `print(n_bins)` to the cell) and check it's what you'd expect
   for 40 samples.
3. The mean and median above are close but not identical. What would a real geologist read
   into that small difference, versus a much larger one?
"""))

cells.append(md("""
## 2. Real geochemical data and the log transform

Not every real measurement is well described the way the density data above was. The cell
below loads real **lead (Pb)** and **zinc (Zn)** concentrations, in parts per million (ppm),
from a regional geochemical survey used in the same statistics-for-geoscientists course — 58
samples of each element.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
ELEMENT_TO_EXAMINE = "Pb"   # "Pb" or "Zn"
"""))

cells.append(code("""
# 58 real lead (Pb) concentrations, ppm.
pb = np.array([
    1026.0, 1389.0, 1219.0, 4870.0, 1750.0, 1384.0, 1224.0, 4233.0, 924.4, 339.5,
    793.2, 637.0, 2082.0, 709.0, 7168.0, 4307.0, 4317.0, 6634.0, 25260.0, 3509.0,
    522.2, 1715.0, 610.2, 4107.0, 3767.0, 3975.2, 1274.0, 1663.0, 2346.0, 1203.0,
    636.6, 525.9, 863.3, 1574.0, 3704.0, 2253.0, 18370.0, 2621.0, 928.3, 14240.0,
    266.8, 3150.0, 3522.0, 450.6, 1277.0, 1040.0, 30510.0, 4762.0, 1024.0, 7417.0,
    739.0, 1053.0, 422.0, 481.4, 248.4, 446.5, 919.1, 2209.0,
])
# 58 real zinc (Zn) concentrations, ppm, from the same 58 sample locations.
zn = np.array([
    564.2, 697.3, 470.6, 3317.0, 522.4, 1136.0, 920.1, 947.7, 394.9, 303.0,
    904.4, 202.3, 813.0, 2318.0, 2688.0, 789.2, 1189.0, 1306.0, 4840.0, 886.0,
    291.3, 1018.0, 382.2, 1253.0, 913.0, 2416.6, 357.3, 527.2, 289.9, 524.8,
    624.0, 3355.3, 3515.9, 657.7, 870.3, 721.7, 10450.0, 781.3, 392.6, 1145.0,
    3298.8, 916.1, 3948.0, 372.2, 740.7, 453.6, 3192.0, 2629.3, 143.0, 2060.0,
    2859.5, 463.3, 362.5, 59.5, 205.6, 375.3, 233.5, 943.6,
])

# A dictionary looks values up by name -- element_lookup["Pb"] gives back the pb array. This
# lets ELEMENT_TO_EXAMINE (from the CONFIGURATION cell) pick which element the rest of this
# section examines, without an if/else for every single line below.
element_lookup = {"Pb": pb, "Zn": zn}
concentration = element_lookup[ELEMENT_TO_EXAMINE]

print(f"{ELEMENT_TO_EXAMINE}: n={concentration.size}, "
      f"mean={concentration.mean():.1f} ppm, median={np.median(concentration):.1f} ppm")
"""))

cells.append(md("""
Look at how far apart the mean and median are — nothing like the density data above. That gap
is the signature of a **right-skewed** distribution: most measurements cluster at the lower
end, with a long tail of much higher values dragging the mean upward while barely moving the
median. The histogram below shows the same thing visually.
"""))

cells.append(code("""
plt.figure(figsize=(6, 4))
plt.hist(concentration, bins=12, color="#B06E00", edgecolor="white")
plt.axvline(concentration.mean(), color="#1C5AA0", linewidth=2, label=f"mean = {concentration.mean():.0f}")
plt.axvline(np.median(concentration), color="#3F7A32", linestyle="--", linewidth=2,
            label=f"median = {np.median(concentration):.0f}")
plt.xlabel(f"{ELEMENT_TO_EXAMINE} (ppm)")
plt.ylabel("Number of samples")
plt.legend()
plt.title(f"Raw {ELEMENT_TO_EXAMINE} concentrations -- strongly right-skewed")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
Geochemical concentrations are usually the result of many small multiplicative processes
(dilution, enrichment, mixing) rather than additive ones, which tends to produce exactly this
kind of long-tailed shape — and it is common enough in geochemistry that the standard fix is
to work in **log space** instead. `np.log10` takes the base-10 logarithm of every value in an
array at once.
"""))

cells.append(code("""
log_concentration = np.log10(concentration)

print(f"log10({ELEMENT_TO_EXAMINE}): mean={log_concentration.mean():.3f}, "
      f"median={np.median(log_concentration):.3f}  (much closer together than the raw values)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(concentration, bins=12, color="#B06E00", edgecolor="white")
axes[0].set_title(f"Raw {ELEMENT_TO_EXAMINE} (ppm)")
axes[0].set_xlabel(f"{ELEMENT_TO_EXAMINE} (ppm)")
axes[1].hist(log_concentration, bins=12, color="#1C5AA0", edgecolor="white")
axes[1].set_title(f"log10({ELEMENT_TO_EXAMINE})")
axes[1].set_xlabel(f"log10({ELEMENT_TO_EXAMINE} in ppm)")
for ax in axes:
    ax.set_ylabel("Number of samples")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Try it

1. Compare the mean and median for the raw values, then for the logged values. Which pair is
   closer together? What does that tell you about which version is more symmetric?
2. Set `ELEMENT_TO_EXAMINE = "Zn"` and re-run this section. Is it similarly skewed?
3. Sort the raw concentrations (`np.sort(concentration)`) and look at the two or three highest
   values. In a real exploration survey, an unusually high concentration like this is called a
   **geochemical anomaly**, and is often the first clue pointing toward a hidden ore deposit.
   Session 5 builds a full prospectivity model around exactly this "is this measurement
   unusually high" idea.
"""))

cells.append(md("""
## 3. Before Session 2

Every number introduced here — mean, median, standard deviation, percentile, and the log
transform — reappears throughout this module. Session 2 uses exactly these tools to compare
two rock types side by side before teaching a computer to do that sorting automatically, and
Session 5 returns to this exact "is this measurement unusually high" logic to search for real
ore deposits.

One short, free, no-installation resource — read/try before next session if you can, but not
required to follow along:

- **Elements of AI** — [elementsofai.com](https://www.elementsofai.com/) — a free, widely used
  course explicitly aimed at people with no programming or maths background. Session 2
  introduces the same "learning from examples" idea its first section covers.

### Wrap-up check
In your own words (a sentence is enough, no need to write it down), try answering:

1. Why might the mean of a geochemical concentration dataset be pulled much higher than its
   median, while a density dataset's mean and median stay close together?
2. Between a very small number of histogram bins and a very large one, which felt like the
   more honest summary of the same 40 density measurements, and why?
3. If a code cell says a variable "is not defined," what's the first thing to check? (Section 0
   has the answer.)
"""))

save(cells, "../session01_statistical_foundations/S01_Statistical_Foundations.ipynb")
