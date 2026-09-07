import sys
sys.path.insert(0, ".")
from nb_helper import md, code, save

cells = []

cells.append(md("""
# Session 4 — Reading Earth's Ancient Climate

**Data Science and AI Applied to Understanding Earth Evolution**

Same idea as Session 2's Part A (best-fit line), used the way climate scientists actually use
it: a proxy measurement (an isotope ratio, a trace-element ratio, a pollen count) is noisy and
patchy, and a smooth trend has to be recovered from it before anyone can talk about "what the
temperature was doing."

**A note on the data:** the proxy record below is **synthetic** — a warmth index (higher =
warmer) built to have a realistic long-term cooling shape (warm in the deep past, cooler
toward the present) with a short warming excursion around 55 Ma, in the style of the real
Paleocene-Eocene Thermal Maximum, plus realistic scatter — so you can compare your fitted
curve against the "true" trend it was built from. It is not a real published dataset; the
point is the method, which is exactly what you'd apply to one.
"""))

cells.append(code("""
import numpy as np
import matplotlib.pyplot as plt
# PolynomialFeatures: explained below, where it's used -- it's how a tool that can only draw
# straight lines (LinearRegression) is made able to draw curves instead.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(seed=11)
print("Ready.")
"""))

cells.append(code("""
# Build a synthetic 66-Myr warmth-index record (higher = warmer): a long-term cooling trend
# (warm in the deep past, cooler toward the present) plus a short warming excursion around
# 55 Ma (in the style of the PETM), then add noise typical of real proxy data.
age_full = np.linspace(0, 66, 400)  # Ma, present = 0

# np.exp(...) is the mathematical function e^x -- here it's used to build a smooth "bump"
# shape (large near age=55, fading quickly away from it) to represent the warming excursion,
# added on top of the steady 8 + 0.28*age_full cooling trend.
true_trend = 8 + 0.28 * age_full + 4.5 * np.exp(-((age_full - 55) ** 2) / (2 * 1.2 ** 2))

# "def" defines a reusable function -- a named block of code you can run again with different
# inputs instead of copy-pasting it. This one takes a noise level and a random seed, and
# returns the true trend plus that much random scatter -- called twice below to make two
# differently-noisy versions of the "same" underlying climate signal.
def make_proxy(noise_level, seed):
    r = np.random.default_rng(seed)
    return true_trend + r.normal(0, noise_level, size=age_full.size)

proxy_isotope = make_proxy(noise_level=1.6, seed=21)      # e.g. an isotope-ratio-based proxy
proxy_trace_element = make_proxy(noise_level=2.4, seed=22)  # e.g. a trace-element-ratio proxy
"""))

cells.append(code("""
# === USER CONFIGURATION ===
PROXY_TO_USE = "isotope"       # "isotope" or "trace_element"
POLYNOMIAL_DEGREE = 4          # how flexible the fitted curve is allowed to be
AGE_WINDOW_MA = (0, 66)        # (young, old) -- crop to a shorter window, e.g. (40, 66)
N_MEASUREMENTS = 70            # how many (sparse, irregular) samples you "measured"
"""))

cells.append(code("""
# "value_if_true if condition else value_if_false" is a one-line if/else -- reads almost like
# English: "proxy_isotope, if PROXY_TO_USE is 'isotope', otherwise proxy_trace_element."
proxy_full = proxy_isotope if PROXY_TO_USE == "isotope" else proxy_trace_element

# Simulate sparse, irregular sampling within the chosen age window -- real cores are never
# sampled on a perfect regular grid.
young, old = AGE_WINDOW_MA   # unpack the (young, old) tuple into two separate variables

# (age_full >= young) & (age_full <= old) combines two boolean masks (Session 2) with "&"
# (meaning AND) into one: True wherever an age falls inside the chosen window.
in_window = (age_full >= young) & (age_full <= old)

# np.where(mask)[0] turns a True/False mask into the actual list of positions (indices) where
# it's True -- candidate_idx is now "every position inside the age window we could sample from."
candidate_idx = np.where(in_window)[0]

# rng.choice(list, size=n, replace=False) picks n items at random from a list with no repeats
# (like drawing lottery balls) -- this simulates picking N_MEASUREMENTS real sample locations
# out of everywhere you *could* have sampled. np.sort puts them back in age order afterwards,
# purely so the scatter plot below reads left-to-right sensibly.
sample_idx = np.sort(rng.choice(candidate_idx, size=min(N_MEASUREMENTS, candidate_idx.size), replace=False))

age_sampled = age_full[sample_idx]      # picking out just those positions ("fancy indexing")
proxy_sampled = proxy_full[sample_idx]

plt.figure(figsize=(7, 4.5))
plt.scatter(age_sampled, proxy_sampled, color="#1C5AA0", s=18, label="your proxy measurements")
plt.gca().invert_xaxis()
plt.xlabel("Age (Ma)")
plt.ylabel(f"{PROXY_TO_USE} warmth index (arbitrary units, higher = warmer)")
plt.legend()
plt.title("Raw proxy measurements -- noisy and irregularly spaced, as real cores are")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
# Fit a polynomial regression -- still "the best-fit line" idea, just allowed to curve.
# LinearRegression on its own can only ever draw a straight line, no matter what you feed it.
# PolynomialFeatures(degree=D) is a pre-processing step that turns each age into a whole set of
# powers of itself (age, age^2, age^3, ... up to age^D) before LinearRegression ever sees it --
# fitting a straight line to THOSE columns is mathematically equivalent to fitting a curved,
# degree-D polynomial to the original age. Higher degree = more terms = more flexibility to
# bend. As in Session 3, make_pipeline chains the two steps into one trainable object.
poly_model = make_pipeline(PolynomialFeatures(degree=POLYNOMIAL_DEGREE), LinearRegression())
poly_model.fit(age_sampled.reshape(-1, 1), proxy_sampled)

age_line = np.linspace(young, old, 300)
fitted_curve = poly_model.predict(age_line.reshape(-1, 1))

plt.figure(figsize=(7, 4.5))
plt.scatter(age_sampled, proxy_sampled, color="#1C5AA0", s=18, alpha=0.6, label="measurements")
plt.plot(age_line, fitted_curve, color="#B06E00", linewidth=2.5, label=f"fitted curve (degree {POLYNOMIAL_DEGREE})")
plt.plot(age_full[in_window], true_trend[in_window], color="#4D6480", linestyle="--",
         linewidth=1.5, label="true underlying trend (normally unknown!)")
plt.gca().invert_xaxis()
plt.xlabel("Age (Ma)")
plt.ylabel(f"{PROXY_TO_USE} warmth index (arbitrary units, higher = warmer)")
plt.legend()
plt.title("Recovering a climate trend from noisy proxy data")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Try it

1. Set `POLYNOMIAL_DEGREE = 1` (a straight line) and re-run. Does a straight line capture the
   warming excursion around 55 Ma at all?
2. Now try `POLYNOMIAL_DEGREE = 15`. Look closely at the fitted curve, especially between
   sparse measurements — does it start oscillating in ways that don't look like real climate
   behaviour, just to pass close to every noisy point? That is **overfitting**: the model
   bending itself around noise instead of learning the real underlying shape — the same failure
   mode Session 3 hinted at, in a context where it's easy to see once you know to look for it.
3. Switch `PROXY_TO_USE` to `"trace_element"` (noisier) and go back to a moderate degree
   (4-6). Does the fitted curve get less reliable, even though the true trend hasn't changed?
4. Set `AGE_WINDOW_MA = (40, 66)` — zoom into just the older part of the record. Does the
   excursion still get picked up with fewer data points to work with?

**The one idea to take away:** a smooth "reconstructed" climate curve you see in a published
figure is *already* a fitted model, not a direct measurement — and every choice above (which
proxy, how much smoothing, which time window) is a real choice a researcher made, that could
have been made differently.
"""))

cells.append(md("""
## Wrap-up

This was Session 2's regression idea again — same tool, harder, more realistic data. Session 5
does the same thing in reverse: not a smooth number to predict, but a category (deposit /
no deposit) — Session 2's *other* idea, the sorting machine, applied somewhere with real
stakes.
"""))

save(cells, "../session04_ancient_climate/S04_Reading_Ancient_Climate.ipynb")
