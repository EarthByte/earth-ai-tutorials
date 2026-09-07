import sys
sys.path.insert(0, ".")
from nb_helper import md, code, save

cells = []

cells.append(md("""
# Session 5 — Finding Hidden Ore Deposits

**Module 2 · Data Science and AI Applied to Understanding Earth Evolution**

This is Session 2's "sorting machine" idea (classification), applied to the same kind of
problem as Module 1's T73/T75 notebooks and the paper behind them:

> Farahbakhsh, E. et al. (2025). *"Machine Learning-Based Spatio-Temporal Prospectivity
> Modeling of Porphyry Systems in the New Guinea and Solomon Islands Region."* Tectonics.

**This notebook is a deliberately simplified illustration of the same idea**, not a
reproduction of that paper's actual method or data. Real prospectivity models use many more
input layers (reconstructed tectonic settings, geophysical grids, geochemistry, and more) and
far more careful validation than we have time for in 2.5 hours — but the core mechanic is
identical to what you'll run below.

**New tool in this notebook — a random forest.** Instead of one logistic-regression boundary
(Session 2) or one neural network (Session 3), a `RandomForestClassifier` trains a large
number of small, simple decision trees (think: a flowchart of yes/no questions like "is
distance-to-fault under 10 km?") on slightly different random subsets of the data, then has
them vote — the majority answer wins. It's still create → fit → predict, same as every other
model so far; it's just an ensemble of simple models rather than one model, which tends to
handle messy, less tidy patterns well without much tuning.
"""))

cells.append(code("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(seed=5)
print("Ready.")
"""))

cells.append(md("""
## 1. The (synthetic) study area

Two input layers, both realistic things you could actually map: **distance to the nearest
mapped fault** (deposits often cluster near structural pathways) and a **geochemical anomaly
index** from stream-sediment sampling (deposits often show up as anomalies downstream). The
"true" relationship between these and deposit likelihood is built into the synthetic data below
— in a real study you never get to see this, we're only showing it here so you can check your
model against ground truth afterwards.
"""))

cells.append(code("""
# Build a synthetic 100 x 100 km study area on a grid (see Session 1 for what meshgrid does).
grid_res = 120
gx, gy = np.meshgrid(np.linspace(0, 100, grid_res), np.linspace(0, 100, grid_res))

# A synthetic fault trace (just a curve across the area) and distance to it.
fault_y_at_x = 30 + 0.35 * gx + 8 * np.sin(gx / 12)
dist_to_fault = np.abs(gy - fault_y_at_x)   # np.abs = absolute value (always positive)

# A synthetic geochemical anomaly field -- a couple of "hot spots" plus background noise.
geochem = (
    3.0 * np.exp(-(((gx - 70) ** 2 + (gy - 55) ** 2)) / (2 * 18 ** 2))
    + 2.0 * np.exp(-(((gx - 25) ** 2 + (gy - 65) ** 2)) / (2 * 12 ** 2))
    + rng.normal(0, 0.3, size=gx.shape)
)

# The TRUE (normally unknown) deposit probability: higher near the fault AND where geochem is high.
true_logit = -3.0 - 0.09 * dist_to_fault + 1.7 * geochem
true_probability = 1 / (1 + np.exp(-true_logit))   # squashes the logit into a clean 0-1 range

# plt.subplots(1, 3, ...) makes one figure holding 3 side-by-side plots at once, and hands back
# `axes` as a list of 3 drawing areas to use one at a time. The for-loop below walks through a
# list of (which drawing area, what data, what title, what colour scheme) tuples -- one pass per
# subplot -- so all three maps can be drawn with one shared block of code instead of repeating
# the same six lines three times.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, field, title, cmap in [
    (axes[0], dist_to_fault, "Distance to fault (km)", "Blues_r"),
    (axes[1], geochem, "Geochemical anomaly index", "Oranges"),
    (axes[2], true_probability, "TRUE deposit probability (hidden from the model)", "YlOrBr"),
]:
    im = ax.contourf(gx, gy, field, levels=20, cmap=cmap)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Easting (km)")
axes[0].set_ylabel("Northing (km)")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## 2. Simulating exploration: what you'd actually have

In real exploration you don't have the true probability map — you have a scattering of drill
holes and surface samples that came back either "deposit" or "no deposit." The cell below
simulates exactly that: a sparse set of known points, sampled using the true probability above
(so it behaves like real exploration data), which is all your model gets to train on.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
N_KNOWN_SAMPLES = 150          # how many drill holes / surface samples you have
FEATURES_TO_USE = ["distance_to_fault", "geochem"]   # try removing one, e.g. ["geochem"]
PROSPECTIVITY_THRESHOLD = 0.5  # probability above this = "target zone" on the final map
"""))

cells.append(code("""
# Pick N_KNOWN_SAMPLES random grid cells to be your "drill holes" (rng.choice, as in Session 4).
sample_idx = rng.choice(gx.size, size=N_KNOWN_SAMPLES, replace=False)
sample_dist = dist_to_fault.ravel()[sample_idx]
sample_geochem = geochem.ravel()[sample_idx]
sample_prob_true = true_probability.ravel()[sample_idx]

# Simulate an actual drilling outcome at each point: draw a random number between 0 and 1, and
# call it a "deposit found" if that number lands below the TRUE probability there -- so a point
# with true_probability=0.9 finds a deposit 90% of the time, one with 0.1 only 10% of the time,
# exactly like a weighted coin flip. .astype(int) turns the resulting True/False into 1/0.
sample_label = (rng.uniform(size=N_KNOWN_SAMPLES) < sample_prob_true).astype(int)
sample_x = gx.ravel()[sample_idx]
sample_y = gy.ravel()[sample_idx]

# A "dictionary" ({key: value, ...}) looks values up by name instead of by position -- here it
# lets FEATURES_TO_USE (a list of names, from the CONFIGURATION cell) pick out the matching
# arrays by name. [feature_lookup[f] for f in FEATURES_TO_USE] is a list comprehension
# (Session 3) that looks up each requested feature in turn, so removing a name from
# FEATURES_TO_USE automatically drops that column from the table fed to the model below.
feature_lookup = {"distance_to_fault": sample_dist, "geochem": sample_geochem}
X_known = np.column_stack([feature_lookup[f] for f in FEATURES_TO_USE])

plt.figure(figsize=(6, 5))
plt.scatter(sample_x[sample_label == 0], sample_y[sample_label == 0], color="#4D6480",
            marker="x", label="no deposit found", alpha=0.7)
plt.scatter(sample_x[sample_label == 1], sample_y[sample_label == 1], color="#B06E00",
            marker="o", label="deposit found", edgecolor="black")
plt.xlabel("Easting (km)")
plt.ylabel("Northing (km)")
plt.legend()
plt.title(f"What you actually have: {N_KNOWN_SAMPLES} sampled points, {sample_label.sum()} deposits")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## 3. Train the classifier and map the whole area

Same "sorting machine" idea as Session 2, just with a `RandomForestClassifier` instead of
logistic regression (it handles curved, less tidy boundaries a bit better) — trained on those
known points, then asked to predict a prospectivity **probability** for every cell in the grid.
"""))

cells.append(code("""
# n_estimators=200: build 200 individual decision trees to vote together (see the intro note
# above). max_depth=6 caps how many yes/no questions deep each tree can go, which keeps any one
# tree from getting too fussy about individual training points.
clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=5)
clf.fit(X_known, sample_label)

# Same feature_lookup trick as above, but now over every grid cell instead of just the 150
# sampled points -- this is what lets the model produce a prediction for the WHOLE map, not
# just the points it trained on.
grid_feature_lookup = {"distance_to_fault": dist_to_fault.ravel(), "geochem": geochem.ravel()}
X_grid = np.column_stack([grid_feature_lookup[f] for f in FEATURES_TO_USE])

# .predict(...) (used in earlier sessions) would only give a hard "deposit"/"no deposit" guess.
# .predict_proba(...) instead gives the model's estimated PROBABILITY of each class, as a table
# with one column per class -- [:, 1] keeps just the "deposit" column (column 1), then
# .reshape(gx.shape) folds that flat list of probabilities back into the shape of the map.
predicted_probability = clf.predict_proba(X_grid)[:, 1].reshape(gx.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].contourf(gx, gy, predicted_probability, levels=20, cmap="YlOrBr")
fig.colorbar(im0, ax=axes[0], shrink=0.8)
axes[0].scatter(sample_x[sample_label == 1], sample_y[sample_label == 1], color="black",
                marker="o", s=15, label="known deposit")
axes[0].set_title("Model's predicted prospectivity")
axes[0].legend(fontsize=8)
axes[0].set_xlabel("Easting (km)")
axes[0].set_ylabel("Northing (km)")

# Turn the probability map into a simple yes/no "target zone" map by comparing every cell to
# the threshold -- a boolean mask again, just a 2-D one this time instead of a flat list.
target_zone = predicted_probability >= PROSPECTIVITY_THRESHOLD
axes[1].contourf(gx, gy, target_zone, levels=[-0.5, 0.5, 1.5], colors=["#F7FAFD", "#B06E00"])
axes[1].set_title(f"Target zones at threshold {PROSPECTIVITY_THRESHOLD}")
axes[1].set_xlabel("Easting (km)")
plt.tight_layout()
plt.show()

# .mean() on a True/False array treats True as 1 and False as 0, so the average IS the fraction
# of cells that are True -- a quick way to turn a big grid of True/False into one summary number.
fraction_flagged = target_zone.mean()
print(f"{fraction_flagged:.1%} of the study area is flagged as a target zone at this threshold.")
"""))

cells.append(md("""
### Try it

1. Compare the model's predicted prospectivity map (left, above) to the **TRUE** probability
   map from Section 1 — remember, in a real project you'd never get to see that comparison.
   How close did the model get, working only from 150 sparse samples?
2. Lower `PROSPECTIVITY_THRESHOLD` to `0.3` and re-run — how much more area gets flagged as a
   target zone? What's the real-world trade-off in setting the threshold low (more area to
   explore, higher cost) versus high (less area, more risk of missing something)?
3. Set `FEATURES_TO_USE = ["geochem"]` only (drop distance-to-fault) and re-run everything from
   the training cell down. Does the map get noticeably worse? What does that tell you about
   which input actually mattered most here?
4. Set `N_KNOWN_SAMPLES = 20` — a much smaller, more realistic exploration budget — and re-run.
   How much does the map degrade?
"""))

cells.append(md("""
## Wrap-up

This is the exact shape of Module 1's T73/T75 prospectivity notebooks and the Farahbakhsh et
al. (2025) paper behind them — known deposit/non-deposit points, a handful of input layers, a
classifier trained to generalise from the known points to the whole area. Real published
prospectivity models add many more input layers, much larger and more carefully validated
training sets, and (per Farahbakhsh et al.) reconstructed plate-tectonic history as an input —
but "sorting machine, trained on known points, applied to a whole map" is the same idea you
just ran yourself.
"""))

save(cells, "../session05_ore_deposits/S05_Finding_Hidden_Ore_Deposits.ipynb")
