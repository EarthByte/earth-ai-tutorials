import sys
sys.path.insert(0, ".")
from nb_helper import md, code, save

cells = []

cells.append(md("""
# Session 6 — Speeding Up a Slow Earth Model

**Module 2 · Data Science and AI Applied to Understanding Earth Evolution**

Some Earth simulations — mantle convection, landscape evolution, ice-sheet models — take
hours, days, or weeks to run one scenario. Exploring many scenarios (different parameters,
different starting conditions) can be too expensive to do directly. One increasingly common
fix: run the slow model a limited number of times, train a fast model to imitate its outputs,
then use the fast model to explore new scenarios in a fraction of a second.

This notebook is a small, honest, toy version of that idea — no neural operators, no
"physics-informed" machinery, just Session 2's regression tool, used as a stand-in for an
expensive simulator. The concept scales up; the mechanics you'll see here do not change much
when it does.
"""))

cells.append(code("""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor   # like Session 5's forest, but predicting
                                                       # a NUMBER instead of a category

rng = np.random.default_rng(seed=9)
print("Ready.")
"""))

cells.append(md("""
## 1. The "slow model"

A simplified landscape-relaxation simulation: material is added uniformly at rate `U` (think
tectonic uplift) while diffusion (rate `D`, think erosion smoothing out slopes) works against
it. Run for enough time steps and the profile settles toward an equilibrium relief. This kind
of explicit, many-tiny-steps time-stepping is exactly what makes real landscape and
mantle-flow simulations slow — nothing here is artificially slowed down for effect.
"""))

cells.append(code("""
# def name(arguments): defines a function (Session 4 introduced this) -- the new piece here is
# that some arguments have a "=default" value (n_steps=200_000, nx=200, ...), so calling
# slow_model(0.2, 0.02) without the rest still works, using those defaults. The underscore in
# 200_000 is just a thousands separator Python allows for readability -- it's the number
# 200000, written so it's easier for a human to read at a glance.
# The triple-quoted text right after def is a "docstring" -- a description of what the
# function does, for humans reading the code (it doesn't affect how the function runs).
def slow_model(diffusivity, uplift_rate, n_steps=200_000, nx=200, dx=1.0, dt=0.4):
    \"\"\"A deliberately simple, deliberately slow 1-D relaxation model.
    Returns the equilibrium relief (max elevation) after n_steps of explicit time-stepping.
    \"\"\"
    h = np.zeros(nx)             # start from a flat profile: nx zeros in a row
    # range(n_steps) counts 0, 1, 2, ... up to n_steps -- "for _ in range(n_steps):" runs the
    # indented block that many times. The underscore is a Python convention meaning "this loop
    # variable is never actually used inside the loop, only the repetition matters."
    for _ in range(n_steps):
        lap = np.zeros(nx)
        # Slicing an array with [start:stop] takes a sub-range without a manual loop:
        # h[2:] is "everything from position 2 onward," h[:-2] is "everything except the last
        # two," h[1:-1] is "everything except the first and last." Lined up like this, the three
        # slices let every interior point compare itself to its two neighbours all at once --
        # the standard trick for this kind of physics simulation.
        lap[1:-1] = h[2:] - 2 * h[1:-1] + h[:-2]
        # "+=" means "add this to h[1:-1] and store the result back" -- one time-step's worth of
        # diffusion (spreading out) plus uplift (raising) applied to the whole profile at once.
        h[1:-1] += diffusivity * dt / dx**2 * lap[1:-1] + uplift_rate * dt
        h[0] = 0.0    # the two ends are held fixed at 0 -- a simple boundary condition
        h[-1] = 0.0
    return h.max()    # the final, tallest point once time-stepping is done

# Time a single run so you can feel how slow "slow" is.
# time.perf_counter() is a precise stopwatch: call it once before, once after, and the
# difference is how long that code took to run, in seconds.
t0 = time.perf_counter()
example_relief = slow_model(diffusivity=0.2, uplift_rate=0.02)
t1 = time.perf_counter()
print(f"One run of the slow model took {t1 - t0:.2f} seconds and gave a relief of {example_relief:.1f} m.")
"""))

cells.append(md("""
## 2. Building a small training set

To train a fast surrogate, the slow model has to be run enough times to cover the range of
`diffusivity` and `uplift_rate` values you might care about. This is the one-off expensive
step — the whole point of a surrogate is that you pay this cost *once*.
"""))

cells.append(code("""
diffusivity_grid = np.linspace(0.05, 0.5, 5)   # 5 diffusivity values to try
uplift_grid = np.linspace(0.005, 0.05, 5)      # 5 uplift-rate values to try

# Three empty lists, to be filled in one entry at a time below.
training_D, training_U, training_relief = [], [], []

t_train_start = time.perf_counter()
# A "nested" loop: for every D, try every U -- 5 x 5 = 25 combinations in total, each one
# actually running the slow model once. list.append(value) adds one new item to the end of a
# list -- this is how the three empty lists above get built up, one combination at a time.
for D in diffusivity_grid:
    for U in uplift_grid:
        training_D.append(D)
        training_U.append(U)
        training_relief.append(slow_model(D, U))
t_train_end = time.perf_counter()

training_time = t_train_end - t_train_start
print(f"Ran the slow model {len(training_relief)} times to build the training set: {training_time:.1f} seconds total.")

# Package the 25 (D, U) -> relief examples into the (inputs table, answers list) shape every
# scikit-learn model expects -- exactly as in every earlier session.
X_train = np.column_stack([training_D, training_U])
y_train = np.array(training_relief)
"""))

cells.append(md("""
## 3. Train the fast surrogate

Same regression idea as Session 2, just with two inputs (`diffusivity`, `uplift_rate`) instead
of one, predicting a number (equilibrium relief) instead of a category.
"""))

cells.append(code("""
# RandomForestRegressor is Session 5's random forest, but for predicting a NUMBER (equilibrium
# relief) instead of a category (deposit / no deposit) -- same create -> fit -> predict pattern.
surrogate = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=9)
surrogate.fit(X_train, y_train)
print("Surrogate trained.")
"""))

cells.append(md("""
## 4. The payoff: a new scenario, two ways

Pick a `diffusivity` / `uplift_rate` combination you haven't tried, and run it **both ways**:
through the real slow model, and through the fast surrogate. Both should give a similar answer
— but look at the time each one takes.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
# Stay within the training ranges (0.05-0.5 for diffusivity, 0.005-0.05 for uplift_rate) --
# outside that range the surrogate is extrapolating, which Session 8 comes back to.
NEW_DIFFUSIVITY = 0.32
NEW_UPLIFT_RATE = 0.018
"""))

cells.append(code("""
t0 = time.perf_counter()
true_relief = slow_model(NEW_DIFFUSIVITY, NEW_UPLIFT_RATE)
t1 = time.perf_counter()
slow_time = t1 - t0

t0 = time.perf_counter()
# .predict(...) always expects a table of rows, even for a single prediction -- [[a, b]] is a
# list containing one row [a, b], giving the "1 row, 2 columns" shape scikit-learn wants. The
# result is a list with one answer in it, so [0] pulls that single number back out.
surrogate_relief = surrogate.predict([[NEW_DIFFUSIVITY, NEW_UPLIFT_RATE]])[0]
t1 = time.perf_counter()
fast_time = t1 - t0

print(f"Slow model:  {true_relief:7.1f} m relief, in {slow_time:.3f} s")
print(f"Surrogate:   {surrogate_relief:7.1f} m relief, in {fast_time:.5f} s")
print(f"Difference:  {abs(true_relief - surrogate_relief):.1f} m "
      f"({abs(true_relief - surrogate_relief) / true_relief:.1%} of the true value)")
# max(fast_time, 1e-9) just guards against dividing by exactly zero, in case the prediction is
# ever timed as instantaneous; ":,.0f" in the f-string adds a thousands separator (e.g. "1,234").
print(f"Speed-up:    roughly {slow_time / max(fast_time, 1e-9):,.0f}x faster")

plt.figure(figsize=(5.5, 4))
bars = plt.bar(["slow model\\n(1 new run)", "surrogate\\n(1 prediction)"],
               [slow_time, fast_time], color=["#4D6480", "#B06E00"])
plt.ylabel("Time (seconds)")
plt.title("Same scenario, two ways")
for bar, value in zip(bars, [slow_time, fast_time]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}s",
              ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Try it

1. Change `NEW_DIFFUSIVITY` and `NEW_UPLIFT_RATE` to a few different combinations within the
   training ranges and re-run. Does the surrogate stay close to the slow model each time?
2. Now try a combination **outside** the training ranges, e.g. `NEW_DIFFUSIVITY = 1.0` — the
   surrogate will still confidently return a number. Does it look as trustworthy as the
   in-range predictions? (This is the exact concern Session 8 raises about pushing AI methods
   into deep time, where there's no "training data" from direct observation at all.)
3. Look back at the training-set cell: what would it cost (in your own rough estimate of
   minutes) to build a surrogate over a 20x20 grid instead of 5x5? Is that still a one-off cost
   worth paying, or does the slow model itself start becoming the bottleneck again?
"""))

cells.append(md("""
## Wrap-up

The pattern — run an expensive model enough times to cover the cases you care about, train a
cheap model to imitate it, then use the cheap model for everything else — is the same idea
behind real surrogate/emulator approaches used for mantle convection and other expensive Earth
simulations, just without the deep-learning machinery those often use in practice. The
trade-off you just measured directly (accuracy vs. speed, and the danger of trusting a
surrogate outside the range it was trained on) is the real trade-off researchers weigh when
they decide whether a surrogate is worth building at all.
"""))

save(cells, "../session06_speeding_up_models/S06_Speeding_Up_a_Slow_Earth_Model.ipynb")
