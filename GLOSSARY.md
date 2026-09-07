# Glossary

Plain-language definitions for every technical term used across this module's notebooks and
guides — machine learning ideas first, then the Python syntax the notebooks rely on. Look
things up here rather than searching each notebook individually; every term is explained in
context the first time it appears in a notebook too, this is just the one place to find them
all together.

## Statistics terms

**Dip / dip direction** — two numbers that together describe the orientation of a geological
plane such as a fault: **dip** is the steepest angle the plane makes with the horizontal (0 =
flat, 90 = vertical), and **dip direction** is the compass bearing (0-360 degrees) that steepest
line points toward. Session 1 uses a real set of both to introduce directional statistics.

**Histogram** — a bar chart showing how many measurements fall into each of a set of equal-
width ranges ("bins"). The number of bins chosen changes the visual impression the same data
gives — too few hides real structure, too many makes the chart mostly noise (Session 1).

**Interquartile range (IQR)** — the span between the 25th and 75th percentile (see
**percentile** below): the range covering the middle 50% of a dataset. A common measure of
spread that, like the median, is not thrown off by a few extreme values.

**Log transform** — replacing every value in a dataset with its logarithm (`np.log10` in this
module). Standard practice for right-skewed data such as geochemical concentrations, which
usually becomes far more symmetric, bell-shaped, and easier to summarise honestly once logged
(Session 1).

**Mean** — the arithmetic average: add every value, divide by how many there are. Pulled
strongly toward extreme values (unlike the **median**), which is exactly why the mean and
median can differ a lot for a **skewed** dataset.

**Median** — the middle value once a dataset is sorted. Barely moved by a handful of unusually
extreme measurements, which makes it a more robust summary than the mean for real field data.

**Outlier** — a measurement that lies unusually far from the rest of a dataset. A common rule
of thumb flags anything more than 1.5x the IQR beyond the 25th or 75th percentile — worth a
second look, not automatic removal (Session 1).

**Percentile / quartile** — the *p*th percentile is the value below which *p*% of a dataset's
measurements fall. The 25th, 50th (the median), and 75th percentiles are also called the
quartiles.

**Rayleigh test** — a statistical test for whether a set of compass directions has a genuine
preferred orientation, or is just scattered randomly around the circle. Works by adding up all
the directions as unit vectors and checking whether the combined ("resultant") vector is long
enough that it's unlikely to have arisen from pure chance (Session 1).

**Rose diagram** — a histogram wrapped around a circle instead of a straight line, used for
compass-direction data so that directions near 0 degrees and 360 degrees sit next to each other
rather than at opposite ends of the chart (Session 1).

**Skewed distribution** — a dataset whose values pile up on one side with a long tail
stretching the other way, rather than being roughly symmetric. Right-skewed data (a long tail
of unusually high values) is common in geochemistry and usually handled with a **log
transform**. A distribution that becomes symmetric after logging is called **log-normal**.

**Spearman's rank correlation** — a correlation coefficient computed on the *ranks* of two
variables (1st smallest, 2nd smallest, and so on) rather than their raw values. Measures how
consistently two variables move together even when the relationship is curved rather than a
straight line, and is not thrown off by a single extreme outlier the way an ordinary
correlation can be (Session 1).

**Standard deviation / variance** — two closely related measures of spread: the variance is
the average squared distance of each measurement from the mean, and the standard deviation is
its square root, expressed in the same units as the original measurement. A **standard
deviation** is essentially what "noise level" means in Session 2's regression example.

**Stereonet (Schmidt net)** — a circular plot used to show the orientation of planes or lines in
3-D space as single points, using an **equal-area projection** so that orientations scattered
evenly in space produce an evenly scattered set of points on the page. Session 1 plots a real
set of fault-plane orientations this way.

**Sturges' rule** — a common rule of thumb for choosing a reasonable number of histogram bins
for a dataset of *n* measurements (`1 + log2(n)`, rounded up), used as a sensible default
rather than an arbitrary guess (Session 1).

## Machine learning terms

**Accuracy / score** — the fraction of predictions a model got right, usually reported as a
percentage. In this module it's always measured on the **test set** (see below), never on data
the model trained on, because a model can look better than it really is if you only check it
against examples it's already memorised.

**Boundary (decision boundary)** — the line, curve, or surface a classification model draws
between categories. A straight-line classifier can only ever draw a *straight* boundary; a
neural network or random forest can draw curved or irregular ones (Session 3).

**Classification** — training a model to sort things into categories (e.g. "mafic" vs
"felsic," "deposit" vs "no deposit") based on measurements. One of the two core ideas the whole
module is built on — nicknamed "the sorting machine" in Sessions 1–2.

**Ensemble** — a model built from many smaller models voting or averaging together, rather
than one model alone. A random forest (below) is an ensemble of decision trees.

**Extrapolation vs. interpolation** — a prediction is an *interpolation* if it falls within
the range of data the model was trained on, and an *extrapolation* if it falls outside that
range. Models are generally reliable interpolating and unreliable extrapolating — Session 6
demonstrates this directly, and Session 8 connects it to why deep-time AI predictions are
especially hard to trust.

**Feature** — one measured input a model uses to make a prediction (e.g. "density," "distance
to fault"). A model with two features is looking at two numbers per sample; a "feature table"
has one row per sample and one column per feature.

**Fit / train** — the step where a model looks at example inputs *and* their correct answers,
and works out the rule connecting them (`model.fit(inputs, answers)` in every notebook here).
Fitting is the actual "learning" part of machine learning.

**Hyperparameter** — a setting you choose *before* training that controls how a model learns,
as opposed to something the model learns on its own. `HIDDEN_LAYER_SIZES` (Session 3) and
`POLYNOMIAL_DEGREE` (Session 4) are both hyperparameters — every `USER CONFIGURATION` cell in
this module is really a set of hyperparameters to experiment with.

**Label** — the correct answer for one training example (e.g. "mafic," or "deposit found").
Also called a **target**.

**Model** — the trained thing that makes predictions — a straight line, a decision boundary, a
neural network, a random forest. "Model," "classifier," and "regressor" are all used somewhat
interchangeably in this module depending on what the model predicts.

**Neural network** — a model built from many simple units ("neurons") arranged in **layers**,
each combining its inputs in a slightly flexible way; stacking many of them lets the whole
network represent much more curved or complex boundaries than a single straight-line model
can (Session 3). A "hidden layer" is any layer between the inputs and the final answer.

**Overfitting** — when a model bends itself around the noise in its specific training examples
instead of learning the real underlying pattern — it looks great on data it's already seen and
noticeably worse on anything new. Session 4's high-degree polynomial fit is a direct,
visible example.

**Pipeline** — a way of chaining several processing steps (e.g. "rescale the data, *then* feed
it to a model") into one object that can be trained and used just like a single model. Used
from Session 3 onward, always via `make_pipeline(...)`.

**Prediction** — a model's output for a given input, produced with `model.predict(...)` (a
best-guess answer) or `model.predict_proba(...)` (a probability for each possible answer,
introduced in Session 5).

**Prospectivity** — in exploration geoscience, an estimated likelihood that a given location
hosts a mineral deposit, usually shown as a map. Session 5 builds a simplified one.

**Proxy** — a measurable stand-in for something you can't measure directly — e.g. an isotope
ratio used as a stand-in for ancient temperature, since nobody can put a thermometer 55 million
years into the past (Session 4).

**Random forest** — an ensemble model that trains many individual decision trees (simple
flowcharts of yes/no questions) on slightly different random subsets of the data, then has
them vote or average together. Tends to handle messy, non-tidy patterns well without much
tuning (Sessions 5–6).

**Regression** — training a model to predict a *number* (e.g. a spreading rate, a temperature)
from measurements, typically by fitting a best-fit line or curve. The other of the two core
ideas the module is built on — nicknamed "the best-fit line" in Sessions 1–2.

**Scaling (feature scaling)** — rescaling different measurements onto a comparable numeric
footing before training, usually so every feature has a similar typical size. Some models
(like neural networks) are quite sensitive to input scale and need this; others (like random
forests) don't. `StandardScaler`, used from Session 3 onward, is the specific tool for this.

**Surrogate model (emulator)** — a fast, cheap model trained to imitate the output of a slow,
expensive simulation, so that new scenarios can be explored quickly without re-running the
slow original every time (Session 6).

**Threshold** — a cutoff value used to turn a probability into a yes/no decision — e.g.
"anywhere the model predicts above 50% deposit probability counts as a target zone" (Session
5). Lowering a threshold flags more area (and more false alarms); raising it flags less (and
risks missing real ones).

**Training set / test set** — before training, data is split into a **training set** the model
learns from and a **test set** it never sees until evaluation, so that accuracy is measured
honestly on genuinely new examples rather than ones the model has memorised (Session 3
onward). `train_test_split` performs this split.

## Python terms used in these notebooks

**Array** — a NumPy list of numbers that supports fast, whole-list maths (`age * 2` doubles
every value at once, no manual loop needed). Created with functions like `np.array(...)` or
`np.linspace(...)`.

**Boolean mask** — an array of `True`/`False` values, usually created by comparing an array to
something (`y == "mafic"`), then used to pick out just the matching rows (`X[mask]`).

**Cell** — one block in a notebook: either a **text cell** (formatted writing) or a **code
cell** (Python that runs when you press Shift+Enter).

**Comment** — anything after a `#` on a line of code — a note for humans, ignored when the
code runs.

**Dictionary** — a collection that looks values up by name rather than by position, written
`{"name": value, ...}`. Used from Session 5 onward to look up a feature array by its name.

**f-string** — a text string written as `f"..."` that can drop a variable's value straight
into the text using `{curly braces}`, e.g. `f"{value:.2f}"` shows `value` rounded to 2 decimal
places. Used for almost every printed result in this module.

**Function** — a named, reusable block of code, created with `def name(inputs):` and reused by
calling `name(some_input)`. First appears in Session 4 (`make_proxy`) and Session 6
(`slow_model`).

**Import** — the line at the top of every notebook (`import numpy as np`, etc.) that loads a
library and gives it a short nickname to use for the rest of the notebook.

**Library / package** — pre-written code that does something useful (maths, plotting, machine
learning) so you don't have to write it from scratch. NumPy, SciPy, Matplotlib, and scikit-learn
are the ones used throughout this module.

**List** — an ordered collection of items written `[item1, item2, ...]`. A **list
comprehension** — `[expression for item in list]` — builds a new list by applying the same
operation to every item, without writing out a full loop (first used in Session 3).

**Meshgrid** — `np.meshgrid(...)` takes two lists of coordinates and builds every combination
of them as a grid, the standard way to cover a 2-D map or plot area one cell at a time
(introduced in Session 1, used throughout).

**Traceback** — the multi-line message Python prints when a code cell fails, ending in the
specific error type and message. Session 7 explains how to read one.

**Tuple** — like a list, but written with `( )` instead of `[ ]` and normally used for a fixed,
small group of values, such as `HIDDEN_LAYER_SIZES = (8,)` (Session 3) or a single
`(density, silica)` sample (Session 2).

**Variable** — a name that points to a value, created with `=` (e.g. `age = 42`). Read top to
bottom: a variable has to be created (the line that defines it has to have already *run*)
before it can be used anywhere later.
