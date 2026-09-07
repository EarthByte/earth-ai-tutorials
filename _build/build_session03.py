import sys
sys.path.insert(0, ".")
from nb_helper import md, code, save

cells = []

cells.append(md("""
# Session 3 — What Is a Neural Network, Really?

**Data Science and AI Applied to Understanding Earth Evolution**

**Before this notebook:** spend 10-15 minutes in **TensorFlow Playground**
([playground.tensorflow.org](https://playground.tensorflow.org/)) — pick the "circle" dataset,
and drag the "number of hidden layers" and "neurons per layer" controls. Watch how a simple
straight boundary turns into an increasingly irregular one as you add layers/neurons. No
account, no installation, nothing to save — experiment with it freely.

This notebook does the same thing with real, working code, on a mapping problem where a
straight line genuinely cannot do the job.

**New idea in this notebook — training data vs. test data.** So far the notebooks have
checked a model against the same data it was trained on. Real practice never does that,
because a model can sometimes just memorise its training examples rather than learning the
real pattern — memorising tells you nothing about whether it'll work on a *new* sample it
hasn't seen. The fix, used from here on: split the data in two before training — a
**training set** the model learns from, and a **test set** it never sees until after training,
used purely to check how well it generalises. `train_test_split` (imported below) does exactly
that split.
"""))

cells.append(code("""
import numpy as np
import matplotlib.pyplot as plt
# make_circles: a scikit-learn function that generates ready-made synthetic data for exactly
# this kind of example (points arranged in two concentric rings) -- a quick way to get a
# realistic-shaped dataset without needing a real one.
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier   # MLP = "multi-layer perceptron", a basic neural network
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng_seed = 3
print("Ready.")
"""))

cells.append(md("""
## 1. A mapping problem a straight line can't solve

Porphyry-Cu systems (the same deposit type Session 5 returns to) classically
show **concentric alteration zoning** in map view: an altered core zone surrounded by a
halo, distinguished by geochemical measurements that shift systematically from centre to edge.
Sort sample points into "core zone" vs "halo" from two such measurements, and the true pattern
is a bullseye, not two tidy clusters. The data below is synthetic, built to have exactly that
concentric shape.
"""))

cells.append(code("""
# Synthetic "alteration core vs halo" dataset -- a concentric, bullseye-shaped pattern.
# make_circles returns X (the (Easting, Northing)-style coordinates, one row per sample) and
# y (0 or 1, which ring each point belongs to) -- the same (inputs, labels) shape Session 2's
# X and y had, just generated for us instead of built by hand.
X, y = make_circles(n_samples=300, noise=0.08, factor=0.4, random_state=rng_seed)
X[:, 0] = X[:, 0] * 20 + 40   # relabel axes as Easting/Northing in km, purely for framing
X[:, 1] = X[:, 1] * 20 + 30

# The training-vs-test split described above: test_size=0.3 keeps 30% of the points aside as
# the test set and trains on the remaining 70%. random_state=rng_seed just makes the split
# reproducible (the same "random" split every time this cell runs).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rng_seed)

plt.figure(figsize=(6, 5))
for label, colour, name in [(1, "#B06E00", "core zone"), (0, "#3F7A32", "halo")]:
    mask = y_train == label   # boolean mask, as in Session 2 -- pick out one ring at a time
    plt.scatter(X_train[mask, 0], X_train[mask, 1], color=colour, label=name, alpha=0.8)
plt.xlabel("Easting (km)")
plt.ylabel("Northing (km)")
plt.legend()
plt.title("Mapped sample points: alteration core vs surrounding halo")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## 2. First, try Session 2's tool on it

Session 2's classifier (logistic regression) can only ever draw a **straight** boundary. Let's
see what happens when we point it at data that curves.
"""))

cells.append(code("""
# make_pipeline(step1, step2) chains two scikit-learn tools into one -- data flows through
# StandardScaler() first (it rescales Easting/Northing onto a comparable numeric footing),
# then into LogisticRegression(). The combined pipeline can be trained and used exactly like a
# single model (.fit(...), .predict(...)) -- logistic regression barely needs the rescaling,
# but the neural network below is quite sensitive to input scale, so it's included now for a
# fair comparison between the two.
straight_clf = make_pipeline(StandardScaler(), LogisticRegression())
straight_clf.fit(X_train, y_train)

# .score(inputs, correct_answers) runs the model on inputs it's given and reports the fraction
# it got right -- here, on the TEST set the model never trained on, which is the honest way to
# measure how good it actually is.
straight_accuracy = straight_clf.score(X_test, y_test)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 300),
                      np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 300))
zz = straight_clf.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=["#E9F4DF", "#FBEFDC"], alpha=0.8)
# A "list comprehension": [expression for item in list] builds a new list by applying
# expression to every item -- here, it turns each 0/1 label in y_test into an actual colour,
# one at a time, so the two rings plot in the right colours.
plt.scatter(X_test[:, 0], X_test[:, 1], c=["#3F7A32" if v == 0 else "#B06E00" for v in y_test],
            edgecolor="white")
plt.xlabel("Easting (km)")
plt.ylabel("Northing (km)")
plt.title(f"Straight-line boundary (Session 2's tool) — {straight_accuracy:.0%} correct on test points")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
A straight line genuinely can't separate a concentric, bullseye-shaped pattern — no amount of
extra data fixes that, because the *shape* of the boundary is wrong, not just its position.

## 3. Same problem, a small neural network

A neural network is built from the same "sorting machine" idea as Session 2's classifier, with
one addition: instead of one straight boundary, it combines several simple boundaries — one per
"neuron" — into a single, much more flexible one. More neurons and more layers (which
TensorFlow Playground let you feel) means a more flexible boundary.
"""))

cells.append(code("""
# === USER CONFIGURATION ===
# HIDDEN_LAYER_SIZES: one number per layer = neurons in that layer. This is a Python "tuple" --
# like a list, but written with ( ) instead of [ ], used here because scikit-learn specifically
# expects this setting in that form.
# (8,)        -> one layer of 8 neurons  (the comma matters -- (8) alone is just the number 8)
# (16, 8)     -> two layers: 16 neurons, then 8
# (2,)        -> a very small network -- try this to see it struggle too
HIDDEN_LAYER_SIZES = (8,)
"""))

cells.append(code("""
nn_clf = make_pipeline(
    StandardScaler(),
    # MLPClassifier IS the neural network -- hidden_layer_sizes controls its size (see the
    # CONFIGURATION cell above); max_iter caps how many rounds of training it's allowed before
    # giving up, set generously high here so it always has enough time to finish learning.
    MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=3000, random_state=rng_seed),
)
nn_clf.fit(X_train, y_train)
nn_accuracy = nn_clf.score(X_test, y_test)

zz_nn = nn_clf.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz_nn, levels=[-0.5, 0.5, 1.5], colors=["#E9F4DF", "#FBEFDC"], alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=["#3F7A32" if v == 0 else "#B06E00" for v in y_test],
            edgecolor="white")
plt.xlabel("Easting (km)")
plt.ylabel("Northing (km)")
plt.title(f"Neural net, hidden layers {HIDDEN_LAYER_SIZES} — {nn_accuracy:.0%} correct on test points")
plt.tight_layout()
plt.show()

print(f"Straight-line classifier:  {straight_accuracy:.0%} correct")
print(f"Neural network:            {nn_accuracy:.0%} correct")
"""))

cells.append(md("""
### Try it

Change `HIDDEN_LAYER_SIZES` and re-run the last two cells each time:

1. `(2,)` — a tiny network. Does it beat the straight line? By how much?
2. `(16, 8)` — does accuracy jump sharply once the network has enough neurons to represent a
   closed, roughly circular boundary? Keep increasing size from there (`(50, 50)`, then bigger
   still) — does accuracy keep climbing, or does it plateau once the network is already big
   enough to trace the true boundary?
3. Once you've found roughly where it plateaus, try to find the *smallest* `HIDDEN_LAYER_SIZES`
   that still gets close to that ceiling. Bigger is not automatically better: a needlessly large
   network takes longer to train for no real accuracy benefit, and — on messier, real-world data
   than this clean synthetic example — an oversized network is exactly what starts memorising
   individual data points instead of the underlying shape. Session 4 shows that failure mode
   directly, once there's real noise for it to bite on.
"""))

cells.append(md("""
## Wrap-up

A neural network is not a different idea from Session 2's sorting machine — it is the *same*
idea (learn a boundary from labelled examples), made more flexible by combining many simple
pieces. That flexibility is exactly why neural networks show up wherever the pattern is too
curved or complex for a straight line or plane — including the small ones behind some real
mantle-dynamics and paleoclimate research.

Sessions 4-6 go back to simpler tools (mostly regression, Session 2's kind) applied to new
Earth-evolution problems — the point of this session was to see *why* a more flexible tool
sometimes earns its extra complexity, not to use one every time.
"""))

save(cells, "../session03_neural_networks/S03_Neural_Networks_Hands_On.ipynb")
