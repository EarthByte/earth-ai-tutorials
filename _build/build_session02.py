import sys
sys.path.insert(0, ".")
from nb_helper import md, code, save

cells = []

cells.append(md("""
# Session 2 — The Best-Fit Line and the Sorting Machine

**Data Science and AI Applied to Understanding Earth Evolution**

Session 1 described real data with numbers computed by hand — mean, median, standard
deviation, percentiles. This notebook introduces the shift that "AI" or "machine learning"
actually refers to: instead of a person computing the summarising rule, the computer finds its
own rule from examples. Two versions of that one idea carry almost the entire module:

- **Regression** — the computer learns a best-fit *line or curve* through a cloud of points,
  so it can predict a number (a spreading rate, a temperature, an age).
- **Classification** — the computer learns to *sort things into groups* based on measurements,
  so it can predict a category (mafic vs felsic, "likely ore deposit" vs "not").

Everything else in AI you'll hear about — neural networks, deep learning, large language
models — is a more elaborate way of doing one of those two things; these two are enough to
make sense of most of what shows up in the Earth-science literature.

Three parts, all of which you will run and then deliberately change:

- **Part 0 — From one dataset to two groups:** reuse Session 1's summary statistics to compare
  mafic and felsic rock samples (density and silica content) before classifying them.
- **Part A — Regression ("the best-fit line"):** recover a spreading rate from noisy seafloor
  age-vs-distance data.
- **Part B — Classification ("the sorting machine"):** sort the same rock samples from Part 0
  into mafic/felsic groups automatically, from their two measurements.

Every notebook from here on follows the same shape: a short **USER CONFIGURATION** cell you
are meant to edit, followed by cells you run without needing to understand every line —
though the comments explain each new piece of Python the first time it shows up, so by the end
of the module you will.

**New in this notebook:** this is the first one that uses **scikit-learn** (imported as
`sklearn`), the standard Python library for exactly this kind of machine learning. Every
scikit-learn tool in this module works the same three-step way, so it's worth fixing in your
head right now:

1. **Create it:** `model = LinearRegression()` — this just sets up an empty, untrained model.
2. **Train it:** `model.fit(inputs, answers)` — show it example inputs *and* the correct
   answers for those inputs; it works out the rule that connects them.
3. **Use it:** `model.predict(new_inputs)` — apply the learned rule to inputs it hasn't seen
   the answer for.

Every model in every remaining notebook follows exactly this create → fit → predict pattern.
"""))

cells.append(code("""
import numpy as np
import matplotlib.pyplot as plt
# scikit-learn (the library is called sklearn in code) -- the two tools below are both
# "regression"/"classification" models exactly as described above.
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(seed=7)
print("Ready.")
"""))

cells.append(md("""
## Part 0 — From one dataset to two groups

Session 1 introduced mean, median, standard deviation, and percentiles for describing a
single set of measurements. Geological classification almost always starts with exactly that
kind of comparison, just applied to two (or more) groups side by side — do mafic and felsic
samples actually have different densities, and by how much? The tool that makes this
comparison possible in code is the **boolean mask**: comparing an array to a condition gives
back an array of True/False values, used to pick out just the matching rows. It is the exact
tool Part B's classifier also relies on.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
N_SAMPLES_PER_CLASS = 30       # how many mafic and how many felsic samples to simulate
ADD_ROGUE_MEASUREMENT = False  # set True to see what one bad measurement does to the stats below
"""))

cells.append(md("""
**Building the dataset.** Two rock types were sampled and measured for **density** and
**silica (SiO2) content** in the lab. `np.concatenate` joins two lists end to end (all mafic
values, then all felsic values); `np.column_stack` then lines up two such lists side by side as
the two **columns** of one table — scikit-learn always wants inputs arranged this way, one row
per sample and one column per measurement. `y` records the correct rock type for each row, in
the same order.
"""))

cells.append(code("""
# Simulate known-type samples. Mafic rocks: denser, lower silica. Felsic rocks: less dense,
# higher silica. Real petrology has more overlap than this -- the noise below is deliberately
# modest so the boundary Part B fits is easy to see first.
mafic_density = rng.normal(2.95, 0.08, N_SAMPLES_PER_CLASS)
mafic_silica = rng.normal(50, 3, N_SAMPLES_PER_CLASS)
felsic_density = rng.normal(2.65, 0.08, N_SAMPLES_PER_CLASS)
felsic_silica = rng.normal(70, 3, N_SAMPLES_PER_CLASS)

if ADD_ROGUE_MEASUREMENT:
    # np.append adds one more value onto the end of an array -- here, a deliberately
    # unrealistic density reading (think: a data-entry error, or a sample contaminated with
    # something dense), to see how much a single bad measurement can distort a summary
    # statistic computed below.
    mafic_density = np.append(mafic_density, 4.20)
    mafic_silica = np.append(mafic_silica, np.mean(mafic_silica))

X = np.column_stack([
    np.concatenate([mafic_density, felsic_density]),
    np.concatenate([mafic_silica, felsic_silica]),
])
y = np.array(["mafic"] * mafic_density.size + ["felsic"] * felsic_density.size)

print(f"Dataset ready: {len(y)} samples ({int(np.sum(y == 'mafic'))} mafic, {int(np.sum(y == 'felsic'))} felsic)")
"""))

cells.append(md("""
**Comparing two populations.** The same summary statistics introduced in Session 1 — mean,
median, standard deviation — computed separately for each group using a boolean mask, answer
the question that actually matters here: do mafic and felsic samples have different densities,
and by how much?
"""))

cells.append(code("""
for label in ["mafic", "felsic"]:
    mask = y == label   # True for every row whose label matches this one
    group_density = X[mask, 0]
    group_silica = X[mask, 1]
    print(f"{label:6s} (n={mask.sum():2d}):  "
          f"density mean={group_density.mean():.3f}, std={group_density.std():.3f} g/cm3   |  "
          f"SiO2 mean={group_silica.mean():.1f}, std={group_silica.std():.1f} wt%")

plt.figure(figsize=(5, 4.5))
# A boxplot draws the median (centre line) and the IQR (the box) that Session 1 introduced,
# and marks any point beyond the 1.5xIQR fences as an individual dot -- the same outlier rule
# from Session 1, applied automatically to each group here.
plt.boxplot([X[y == "mafic", 0], X[y == "felsic", 0]], tick_labels=["mafic", "felsic"])
plt.ylabel("Density (g/cm3)")
plt.title("Density by rock type")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Try it

1. Set `ADD_ROGUE_MEASUREMENT = True` and re-run every cell from the USER CONFIGURATION cell
   down. By how much did that one bad measurement move the **mean**? The **median**? Which
   statistic was more "robust" to a single bad value, and does that match what the boxplot
   shows?
2. Set `ADD_ROGUE_MEASUREMENT` back to `False` and try `N_SAMPLES_PER_CLASS = 5`. Do the mean
   and standard deviation still look like reliable estimates of the "true" 2.95 g/cm3 and
   0.08 g/cm3 the simulation was built from, or do they wander further with so little data?
3. At `N_SAMPLES_PER_CLASS = 30` with no rogue measurement, does the IQR rule flag anything as
   an outlier? Should it, given how the data were generated?

**The idea to take away:** every model in this module — including the two you're about to
build below — is ultimately built on exactly these numbers (means, spreads, boundaries between
groups). Fitting a model is a more elaborate way of asking the same question these summary
statistics already start to answer.
"""))

cells.append(md("""
## Part A — Regression: recovering a spreading rate from seafloor age

**The real relationship this is built on:** at a mid-ocean ridge spreading at a constant
*half-rate*, seafloor age increases linearly with distance from the ridge axis —
`distance = half_rate x age`. If you measured age and distance perfectly, every point would
lie exactly on that line. Real data never lies exactly on the line — sampling gaps, dating
uncertainty, and small changes in spreading rate through time all add scatter, in exactly the
sense that `NOISE_LEVEL_KM` below controls the same kind of spread you just measured with a
standard deviation in Part 0.

**The task:** given only the noisy (age, distance) measurements — not the true half-rate — use
regression to find the best-fit line, and read the spreading half-rate back off its slope.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
# Change these, then re-run this cell and everything below it.

TRUE_HALF_RATE_KM_PER_MYR = 4.0   # the (normally unknown!) real spreading half-rate
NOISE_LEVEL_KM = 8.0              # how scattered the measurements are
N_MEASUREMENTS = 40               # how many seafloor samples you "collected"
MAX_AGE_MYR = 60                  # oldest seafloor age sampled
"""))

cells.append(code("""
# Simulate a set of noisy seafloor age/distance measurements using the configuration above.
# In a real study these would come from magnetic-anomaly picks or drill-core ages -- here
# we generate them, so you can compare the fit against the TRUE value you set.

# rng.uniform(low, high, n) draws n random numbers evenly between low and high -- here, n
# random "ages" between 0 and MAX_AGE_MYR (this is different from Session 1's rng.normal,
# which draws from a bell curve instead of an even spread).
age = rng.uniform(0, MAX_AGE_MYR, N_MEASUREMENTS)

# The real relationship (distance = half_rate x age), plus random scatter to simulate
# measurement noise. rng.normal(0, NOISE_LEVEL_KM, N_MEASUREMENTS) draws N_MEASUREMENTS random
# numbers from a bell curve centred on 0 -- adding them is what makes the data noisy rather
# than lying exactly on a line.
distance = TRUE_HALF_RATE_KM_PER_MYR * age + rng.normal(0, NOISE_LEVEL_KM, N_MEASUREMENTS)

# np.clip(values, minimum, maximum) forces every number into a range -- here, floors any
# (physically impossible) negative distance at 0. `None` as the maximum means "no upper limit."
distance = np.clip(distance, 0, None)

plt.figure(figsize=(6, 4.5))
plt.scatter(age, distance, color="#1C5AA0")   # a scatter plot: one dot per (age, distance) pair
plt.xlabel("Seafloor age (Myr)")
plt.ylabel("Distance from ridge (km)")
plt.title("Your (simulated) field measurements")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
That's the raw data — on its own, a scatter of points. The next cell fits the best-fit line:
this is `LinearRegression`, scikit-learn's regression tool, doing exactly the "draw the line
that best fits these points" job.
"""))

cells.append(code("""
# Fit the best-fit line: this IS the regression / "AI" step. Everything above was just
# generating something to fit; everything below just reads off and displays the result.

model = LinearRegression()   # step 1: create an empty, untrained model (see the intro above)

# step 2: train it. Scikit-learn always wants the inputs as a table with one row per sample and
# one column per measurement -- even with only one measurement (age), it still needs to be a
# column, not a flat list. age.reshape(-1, 1) reshapes the flat list of ages into a table with
# 1 column and as many rows as needed (the -1 means "work out this number automatically").
# distance is the list of correct answers -- what the model is trying to learn to predict.
model.fit(age.reshape(-1, 1), distance)

# .coef_ holds the slope(s) the model learned. There's one input (age), so one slope -- and
# that slope IS the spreading half-rate, because distance = half_rate x age was exactly the
# relationship it was fitting.
fitted_half_rate = model.coef_[0]

# step 3: use it. Ask the trained model to predict a distance for 100 evenly-spaced ages, just
# so we have enough points to draw a smooth line.
age_line = np.linspace(0, MAX_AGE_MYR, 100)
distance_line = model.predict(age_line.reshape(-1, 1))

plt.figure(figsize=(6, 4.5))
plt.scatter(age, distance, color="#1C5AA0", label="measurements")
plt.plot(age_line, distance_line, color="#B06E00", linewidth=2, label="best-fit line")
plt.xlabel("Seafloor age (Myr)")
plt.ylabel("Distance from ridge (km)")
plt.legend()
plt.title("Regression recovers the spreading rate from noisy data")
plt.tight_layout()
plt.show()

# An f-string (the f before the quote) lets you drop a variable straight into a piece of text
# using {curly braces} -- ":.2f" inside the braces means "show this number with 2 decimal
# places." f-strings are used for almost every printed result in this module.
print(f"True half-rate you set:      {TRUE_HALF_RATE_KM_PER_MYR:.2f} km/Myr")
print(f"Half-rate the model found:   {fitted_half_rate:.2f} km/Myr")
"""))

cells.append(md("""
### Try it

Go back to the **USER CONFIGURATION** cell and try each of these, re-running both cells below
it each time:

1. Set `NOISE_LEVEL_KM = 25.0` — much noisier data. Does the fitted half-rate get worse? By how
   much?
2. Now also set `N_MEASUREMENTS = 200` — more (noisy) data. Does more data claw back some of
   the accuracy you lost to noise?
3. Set `N_MEASUREMENTS = 5`. With very little data, how much does the fitted line change if
   you re-run the *simulation* cell again (same configuration, new random noise)?

This is the single most important intuition in the whole module: **a regression is only ever
as good as the data you feed it — more data and less noise both help, and no algorithm can
make up for having neither.**
"""))

cells.append(md("""
## Part B — Classification: sorting the same rock samples by type

**The task:** Part 0 already summarised the density and silica measurements for a set of
mafic and felsic samples of known type — the same `X` (measurements) and `y` (known type)
built there. Now imagine a handful of *unidentified* samples with the same two measurements,
and the goal of classifying them automatically rather than by eye.

**New Python in this part:** the boolean masks and table-building from Part 0 reappear
unchanged. The one genuinely new tool is `zip` — it walks through two lists side by side, one
matching pair at a time — commented where it first appears below.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
# Unidentified samples to classify. Each line is one sample, written as a "tuple"
# (density in g/cm3, SiO2 in wt%) -- add, remove or edit rows (keep the (density, silica) shape).
UNKNOWN_SAMPLES = [
    (2.95, 52.0),
    (2.65, 68.0),
    (2.80, 60.0),   # deliberately in-between -- watch what the classifier does with this one
]
"""))

cells.append(code("""
plt.figure(figsize=(6, 5))
for label, colour in [("mafic", "#3F7A32"), ("felsic", "#B06E00")]:
    # The same boolean-mask trick as Part 0: y == label gives True/False for every row, and
    # X[mask, 0] keeps only the matching rows -- this is how the two rock types get plotted in
    # different colours from one shared table.
    mask = y == label
    plt.scatter(X[mask, 0], X[mask, 1], color=colour, label=f"known {label}", alpha=0.8)
plt.xlabel("Density (g/cm3)")
plt.ylabel("SiO2 (wt%)")
plt.legend()
plt.title("Known samples used to train the sorting machine")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
# Train the classifier: this is the "sorting machine" learning where the boundary is.
# Same create -> fit -> predict pattern as Part A, just with LogisticRegression (classification)
# instead of LinearRegression (regression), and two measurements (columns) instead of one.
clf = LogisticRegression()
clf.fit(X, y)

# Draw the decision boundary the classifier learned, by asking it to classify a whole grid of
# (density, SiO2) points covering the plot, then colouring each grid cell by the answer.
xx, yy = np.meshgrid(   # a grid of points covering the plot area (see Session 1 for meshgrid)
    np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200),
    np.linspace(X[:, 1].min() - 3, X[:, 1].max() + 3, 200),
)
# .ravel() flattens the 2-D grid into a single flat list (scikit-learn wants a table of rows,
# not a 2-D grid); np.column_stack pairs the flattened x's and y's back into (density, SiO2)
# rows; clf.predict(...) then classifies all of them in one call. Comparing the result to
# "felsic" turns the list of "mafic"/"felsic" answers into True/False, then .reshape(xx.shape)
# folds that flat list of answers back into the shape of the original grid so it can be plotted.
zz = (clf.predict(np.column_stack([xx.ravel(), yy.ravel()])) == "felsic").reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#E9F4DF", "#FBEFDC"], alpha=0.8)
for label, colour in [("mafic", "#3F7A32"), ("felsic", "#B06E00")]:
    mask = y == label
    plt.scatter(X[mask, 0], X[mask, 1], color=colour, label=f"known {label}", edgecolor="white")

unknown = np.array(UNKNOWN_SAMPLES)
predictions = clf.predict(unknown)
plt.scatter(unknown[:, 0], unknown[:, 1], color="black", marker="*", s=220,
            label="unidentified sample", zorder=5)
# zip(list_a, list_b) walks through two lists together, one matching pair at a time -- here,
# each (density, silica) sample alongside the classifier's prediction for it.
for (d, s), pred in zip(UNKNOWN_SAMPLES, predictions):
    plt.annotate(pred, (d, s), textcoords="offset points", xytext=(8, 6), fontsize=10, weight="bold")

plt.xlabel("Density (g/cm3)")
plt.ylabel("SiO2 (wt%)")
plt.legend(loc="upper right")
plt.title("The sorting machine's learned boundary, applied to your unidentified samples")
plt.tight_layout()
plt.show()

for (d, s), pred in zip(UNKNOWN_SAMPLES, predictions):
    print(f"Density {d:.2f} g/cm3, SiO2 {s:.1f} wt%  ->  classified as: {pred}")
"""))

cells.append(md("""
### Try it

1. Look at the sample that lies closest to the boundary in the plot. How confident would you
   be in its classification, just by eye?
2. Add two or three more rows to `UNKNOWN_SAMPLES` in the CONFIGURATION cell — including one
   you deliberately place right on the boundary — and re-run. Does the classifier's answer for
   your boundary sample match your own intuition?
3. Go back to Part 0's USER CONFIGURATION cell, set `N_SAMPLES_PER_CLASS = 5`, and re-run
   everything from there down, including Part B. With very little training data, does the
   boundary still look reasonable, or does it start to look shaky?

### Optional live demo
If your instructor is running it: **Google Teachable Machine**
([teachablemachine.withgoogle.com](https://teachablemachine.withgoogle.com/)) does the exact
same "sorting machine" idea using photos instead of measurements — worth watching once, no
setup needed.
"""))

cells.append(md("""
## Wrap-up

Part 0 described a dataset with numbers; Parts A and B fitted rules to data of exactly that
kind. In Part A the rule was a line (a number in, a number out). In Part B the rule was a
boundary (numbers in, a category out). Both used the exact same create → fit → predict
pattern — every model in every remaining notebook uses it too, just with a different model
name and a different shape of data.

Every remaining hands-on session in this module — ancient climate (Session 4), ore deposits
(Session 5), speeding up slow models (Session 6) — is one of these two ideas again, in a new
setting. Session 3 asks a different question: what if a straight line or a straight boundary
genuinely isn't good enough?
"""))

save(cells, "../session02_bestfit_and_sorting/S02_BestFit_and_Sorting_Machine.ipynb")
