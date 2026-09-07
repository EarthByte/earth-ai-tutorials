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
depends on having taken any other module. The real measurements used throughout are drawn from
a University of Sydney statistics-for-geoscientists course.

By the end of this notebook you should be able to:
- Run a Jupyter/Colab notebook cell by cell, and read a basic Python cell well enough to guess
  roughly what it does before running it.
- Compute the mean, median, standard deviation, and percentiles of a real dataset, and explain
  in plain language what each one tells you.
- Build a histogram with a sensible number of bins, and explain why the choice of bin count
  changes the visual impression the same numbers give.
- Recognise a right-skewed dataset from its histogram and summary statistics, and explain why
  geochemical concentration data is usually analysed after a log transform.
- Compute a rank-based (Spearman) correlation coefficient and explain when it's a better choice
  than an ordinary correlation.
- Explain why compass-direction data needs its own tools -- a rose diagram, a test for a real
  preferred direction, and a stereonet -- rather than an ordinary histogram and mean.
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
## 3. Rank correlation with two real species datasets

Not every real relationship between two variables is well described by a straight line — the
next real dataset shows why. The measurements below come from the same
statistics-for-geoscientists course: some measure of body size recorded against sample depth,
for two different species (Species A, n=26; Species B, n=33). The units for "size" and "depth"
weren't recorded alongside the numbers, so treat the plots below as describing the
*statistical* relationship, not a specific biological story.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
SPECIES_TO_EXAMINE = "A"   # "A" or "B"
"""))

cells.append(code("""
# Species A: 26 real (depth, size) measurements.
depth_a = np.array([202, 203, 208, 233, 251, 258, 271, 282, 283, 301, 308, 314, 327, 329,
                     330, 350, 356, 378, 385, 386, 387, 399, 411, 422, 428, 446])
size_a = np.array([0.6, 0.4, 0.8, 1.2, 0.7, 0.7, 0.5, 0.4, 0.8, 0.7, 0.6, 1.1, 1.0, 1.0,
                    0.6, 0.6, 0.7, 0.8, 0.9, 0.7, 0.5, 0.5, 0.8, 1.1, 1.1, 0.9])

# Species B: 33 real (depth, size) measurements.
depth_b = np.array([242, 253, 271, 292, 305, 332, 335, 337, 338, 350, 357, 364, 365, 371,
                     372, 385, 401, 402, 410, 412, 418, 423, 427, 429, 432, 446, 451, 454,
                     460, 470, 474, 481, 497])
size_b = np.array([1.3, 0.9, 0.7, 0.8, 0.8, 1.2, 0.9, 1.1, 1.6, 1.6, 1.0, 1.2, 1.3, 1.4,
                    1.1, 0.9, 1.3, 1.5, 1.8, 1.6, 1.2, 1.5, 1.5, 1.7, 1.9, 1.5, 1.6, 1.2,
                    1.6, 1.7, 1.8, 1.8, 1.3])

# A dictionary of (depth, size) pairs, keyed by species -- same lookup pattern as Section 2.
species_lookup = {"A": (depth_a, size_a), "B": (depth_b, size_b)}
depth, size = species_lookup[SPECIES_TO_EXAMINE]

plt.figure(figsize=(6, 4))
plt.scatter(depth, size, color="#B06E00" if SPECIES_TO_EXAMINE == "A" else "#1C5AA0")
plt.xlabel("Depth")
plt.ylabel("Size")
plt.title(f"Species {SPECIES_TO_EXAMINE}: size vs. depth (n={depth.size})")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
The scatter trends upward but not as a clean straight line — exactly the kind of relationship
**Spearman's rank correlation coefficient** was built for. Instead of using the raw numbers, it
replaces each dataset with its **ranks** (1 = smallest, 2 = next smallest, and so on) and then
measures how consistently the ranks move together. Because it only cares about *order*, not
exact values, it still works when a relationship is consistently increasing but curved, and it
isn't thrown off by a single extreme outlier the way an ordinary correlation would be.
"""))

cells.append(code("""
from scipy.stats import rankdata, spearmanr

# rankdata assigns 1 to the smallest value, 2 to the next, and so on -- tied values share the
# average of the ranks they would otherwise occupy.
depth_ranks = rankdata(depth)
size_ranks = rankdata(size)

print("depth :", depth)
print("ranks :", depth_ranks)
print()
print("size  :", size)
print("ranks :", size_ranks)

# spearmanr computes the correlation directly and also returns a p-value: the probability of
# seeing a correlation at least this strong by chance alone if there were truly no relationship.
rho, p_value = spearmanr(depth, size)
print(f"\\nSpearman's rho = {rho:.3f}, p-value = {p_value:.4f}")
"""))

cells.append(md("""
### Try it

1. Switch `SPECIES_TO_EXAMINE` to the other species and re-run. Is its rho similar, stronger, or
   weaker?
2. A rho close to +1 means the ranks move together almost perfectly; close to 0 means no
   consistent relationship; close to -1 means one variable's rank falls as the other's rises.
   Where does each species fall on that scale?
3. `spearmanr` is just an ordinary correlation computed on the ranks instead of the raw values.
   If you're curious, try `from scipy.stats import pearsonr; pearsonr(depth, size)` on the raw
   numbers and compare the two coefficients.
"""))

cells.append(md("""
## 4. Directional data needs different tools: rose diagrams, the Rayleigh test, and stereonets

Every dataset so far has been a plain number line: density, concentration, size. Structural
geology adds a genuinely different data type: **orientation** — the compass direction a fault
plane dips towards. An ordinary histogram breaks down for this kind of data, because 359
degrees and 1 degree are almost the same direction but sit at opposite ends of a normal x-axis.
The measurements below are real: the dip angle and dip direction of 126 fault planes measured
underground, reused with permission from the same statistics-for-geoscientists course as the
datasets above.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
N_DIRECTION_BINS = 24   # 24 bins of 15 degrees each is a standard rose-diagram choice
"""))

cells.append(code("""
# 126 real fault-plane dip angles, degrees (0 = horizontal, 90 = vertical).
fault_dip = np.array([
    48, 65, 83, 74, 87, 52, 56, 57, 60, 73, 79, 52, 43, 57, 69, 71, 69, 87, 70, 35,
    33, 75, 87, 72, 15, 34, 60, 63, 59, 59, 64, 60, 42, 40, 80, 97, 79, 69, 83, 60,
    79, 58, 66, 16, 78, 69, 76, 82, 84, 73, 85, 73, 74, 81, 77, 52, 69, 68, 81, 83,
    71, 83, 87, 78, 69, 63, 74, 81, 86, 87, 77, 69, 72, 74, 19, 86, 81, 74, 37, 31,
    74, 79, 84, 85, 85, 79, 86, 82, 73, 75, 76, 78, 69, 86, 72, 35, 47, 51, 55, 82,
    79, 71, 86, 81, 84, 72, 79, 76, 85, 78, 79, 85, 60, 81, 71, 75, 74, 71, 78, 50,
    84, 77, 41, 53, 74, 60,
])
# The same 126 faults' dip direction, degrees clockwise from north (0-360).
fault_dipdir = np.array([
    330, 70, 84, 350, 275, 81, 84, 324, 75, 50, 69, 1, 16, 98, 85, 95,
    108, 114, 107, 271, 339, 305, 110, 311, 123, 324, 295, 295, 55, 96, 268, 290,
    67, 105, 90, 87, 119, 92, 280, 86, 89, 88, 94, 280, 302, 47, 42, 274,
    107, 114, 314, 98, 349, 88, 255, 90, 92, 65, 78, 78, 85, 28, 45, 91,
    117, 285, 295, 104, 273, 105, 108, 34, 48, 79, 145, 300, 297, 103, 81, 87,
    79, 173, 72, 71, 283, 60, 256, 270, 259, 249, 87, 93, 105, 109, 108, 125,
    95, 103, 105, 291, 266, 250, 62, 268, 271, 284, 279, 278, 105, 295, 268, 292,
    278, 96, 293, 109, 107, 314, 160, 108, 273, 245, 118, 313, 286, 112,
])

print(f"n = {fault_dip.size} fault planes")
print(f"dip: mean={fault_dip.mean():.1f} deg, range={fault_dip.min()}-{fault_dip.max()} deg")
print("dip direction spans the full 0-360 deg compass")
"""))

cells.append(md("""
### The rose diagram

A **rose diagram** is a histogram wrapped around a circle instead of laid out on a straight
line, so directions near 0 degrees and 360 degrees end up next to each other, exactly like they
are on a real compass.
"""))

cells.append(code("""
# Bin edges around the full circle, in radians (matplotlib's polar axes work in radians).
bin_edges = np.linspace(0, 2 * np.pi, N_DIRECTION_BINS + 1)
dipdir_rad = np.deg2rad(fault_dipdir)
counts, _ = np.histogram(dipdir_rad, bins=bin_edges)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="polar")
ax.set_theta_zero_location("N")   # 0 degrees points up (north), like a compass
ax.set_theta_direction(-1)        # degrees increase clockwise, like a compass
bin_width = bin_edges[1] - bin_edges[0]
ax.bar(bin_edges[:-1], counts, width=bin_width, color="#3F7A32", edgecolor="white", align="edge")
ax.set_title(f"Rose diagram: dip direction of {fault_dip.size} fault planes")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Is there a real preferred direction, or could this be random chance?

Eyeballing a rose diagram can be misleading — some clustering happens by chance even in truly
random directions. The **Rayleigh test** answers this properly. Treat each direction as a unit
vector (length 1, pointing the measured way), add all 126 vectors together, and look at the
length of the result: if directions are scattered evenly around the compass, the vectors mostly
cancel out and the resultant is short; if there's a real preferred direction, they reinforce
each other and the resultant stays long.
"""))

cells.append(code("""
n = fault_dipdir.size
rad = np.deg2rad(fault_dipdir)

sin_sum = np.sin(rad).sum()
cos_sum = np.cos(rad).sum()

# Mean resultant length R: 0 = directions perfectly scattered, 1 = all directions identical.
R = np.sqrt(sin_sum**2 + cos_sum**2) / n

# Mean direction: where the resultant vector points. arctan2 (rather than plain arctan) handles
# all four compass quadrants correctly.
mean_direction = np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360

# Rayleigh test statistic and its large-sample approximate p-value (Mardia & Jupp, 2000).
Z = n * R**2
p_approx = np.exp(-Z)

print(f"mean resultant length R = {R:.3f}")
print(f"mean preferred direction = {mean_direction:.1f} deg")
print(f"Rayleigh's Z = {Z:.2f}, approximate p-value = {p_approx:.2e}")
print()
if p_approx < 0.05:
    print("p < 0.05: reject the 'no preferred direction' null hypothesis -- these faults really")
    print("do cluster around a preferred orientation.")
else:
    print("p >= 0.05: not enough evidence to reject random, uniformly scattered directions.")
"""))

cells.append(md("""
### The Schmidt net: combining dip and dip direction

A fault plane needs *two* numbers to describe fully — how steep it is (dip) and which way it
faces (dip direction) — and a rose diagram only shows one of them. Structural geologists solve
this with a **stereonet**: every fault plane becomes a single point on a circular plot, using an
**equal-area (Schmidt) projection** so that planes scattered evenly in 3D space produce an even
scatter of points on the page (unlike an equal-*angle* projection, which visually crowds points
toward the edge).
"""))

cells.append(code("""
# Equal-area (Schmidt) projection of each fault's dip vector (the direction of steepest descent
# on the plane): plunge = dip, trend = dip direction.
theta = np.deg2rad(90 - fault_dipdir)                     # compass azimuth -> standard math angle
rho = np.sqrt(2) * np.sin(np.deg2rad(90 - fault_dip) / 2)  # equal-area radius from plunge

x = rho * np.cos(theta)
y = rho * np.sin(theta)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
circle = plt.Circle((0, 0), 1.0, fill=False, color="#101D31", linewidth=1.5)
ax.add_patch(circle)
for label, (lx, ly) in {"N": (0, 1.08), "S": (0, -1.08), "E": (1.08, 0), "W": (-1.08, 0)}.items():
    ax.text(lx, ly, label, ha="center", va="center", fontsize=11, color="#101D31")
ax.scatter(x, y, s=18, color="#B06E00", alpha=0.75, edgecolor="none")
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"Schmidt net: {fault_dip.size} fault-plane dip vectors")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
A single, dense cluster on the net (rather than points scattered evenly all over it) is the
stereonet signature of one dominant fault set — exactly what the rose diagram and the Rayleigh
test already pointed to, now confirmed with both angle *and* direction in a single plot.

### Try it

1. Change `N_DIRECTION_BINS` to 12 (30 degree bins) and re-run the rose diagram. Does the
   preferred direction still look the same with fewer, wider bins?
2. The Rayleigh test assumes the data really is directional (angles), not linear. Try
   `np.mean([350, 10])` — is 180 a sensible "average" of two directions either side of north?
   That's why an ordinary mean doesn't work here.
3. On the Schmidt net, points near the centre correspond to steep dips (close to vertical);
   points near the rim are shallow, close-to-horizontal dips. Does the net's overall shape
   (clustered inward vs. spread toward the rim) match the dip range printed above?
"""))

cells.append(md("""
## 5. Before Session 2

Every idea introduced here — descriptive statistics, the log transform, rank correlation, and
the tools directional data needs — reappears throughout this module. Session 2 uses exactly
these tools to compare two rock types side by side before teaching a computer to do that
sorting automatically, and Session 5 returns to this exact "is this measurement unusually high"
logic to search for real ore deposits.

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
3. Why does an ordinary mean give a misleading answer for compass-direction data, and what tool
   from this notebook fixes it?
4. If a code cell says a variable "is not defined," what's the first thing to check? (Section 0
   has the answer.)
"""))

save(cells, "../session01_statistical_foundations/S01_Statistical_Foundations.ipynb")
