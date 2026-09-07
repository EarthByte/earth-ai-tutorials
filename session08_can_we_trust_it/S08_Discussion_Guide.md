# Session 8 — Can We Trust It? (Discussion Guide)

**Module 2 · Data Science and AI Applied to Understanding Earth Evolution**

No notebook this session — this is a conceptual discussion, anchored in code you've already
run in Sessions 2–6 rather than anything new.

## Why deep time is a genuinely hard case for AI

Every method in this module learns from examples. That works well when the examples cover the
situation you want a prediction for. Deep time breaks that assumption in a specific way: for
almost anything more than a few hundred thousand years old, there is no direct observation to
check a prediction against — only other proxies, other models, and other assumptions. A
method trained mostly on recent, well-observed data is being asked to extrapolate the further
back it's pushed, and extrapolation is exactly where every method in this module is weakest.

## Discussion questions

1. Session 6 explicitly told you not to trust the surrogate outside its training range, and
   showed it would still confidently return a number if you did anyway. Where else in the
   module did a method quietly extrapolate without flagging it?
2. Session 5's ore-deposit map was trained on synthetic data built from a known "true"
   relationship. A real exploration dataset has no such guarantee — the known deposits you
   train on are themselves a biased sample (you only find what you've looked for, where you've
   looked). What does that do to a real prospectivity model's blind spots?
3. Session 4 showed that "the reconstructed climate curve" is already a fitted model, not a
   direct measurement, and that the fit depends on choices (which proxy, how much smoothing).
   If two research groups made different smoothing choices and got different curves, how would
   you decide which one to trust?
4. "The model is only as good as its training data" is a cliché — restate it using one of your
   own group's Session 2–6 notebooks as the concrete example, not in the abstract.
5. What would it actually take to build justified confidence in an AI prediction about
   something 300 million years old, given that direct verification isn't possible?

## Framing reading (optional, 10 minutes)

GeoSMART's **"Workflows, Reproducibility & Rigor"** section
([geo-smart.github.io/mlgeo-book](https://geo-smart.github.io/mlgeo-book/)) is a short,
non-technical read on why documenting a workflow and its limits — what data went in, what
choices were made, what the method can't tell you — matters as much as the result itself.

## One thing to carry into Session 9

Farahbakhsh et al. (2025), which Session 9 reads together, makes real predictions about
ore-deposit likelihood using exactly this kind of method. Read it with today's discussion in
mind: what does the paper do to earn trust in its predictions, and where do you think its
Discussion/Limitations section (if it has one) should say more than it does?
