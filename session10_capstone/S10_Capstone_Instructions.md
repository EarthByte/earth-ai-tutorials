# Session 10 — Capstone

**Module 2 · Data Science and AI Applied to Understanding Earth Evolution**

This mirrors Module 1's own capstone exactly: adapt a working notebook's configuration to
something new, rather than building anything from a blank cell.

## What to do

1. **Pick one notebook** from Session 2, 4, 5, or 6.
2. **Change something real about it** — not just a `USER CONFIGURATION` value (though start
   there), but a genuine small extension of your own choosing. Examples, roughly in order of
   ambition:
   - Session 2: a different pair of rock properties for the classifier (e.g. magnetic
     susceptibility instead of SiO2), or a different physical scenario for the regression part
     (e.g. a cooling-age vs distance relationship instead of spreading).
   - Session 4: a different synthetic proxy shape (e.g. two excursions instead of one, or a
     step change instead of a smooth trend) and see whether your polynomial fit still recovers
     it sensibly.
   - Session 5: a third input layer of your own design (e.g. a synthetic "structural
     complexity index"), added alongside distance-to-fault and geochem.
   - Session 6: a different physical process for the "slow model" (e.g. add a second, opposing
     process) and see whether the surrogate still tracks it well.
3. **Run it and see what actually happens** — not what you expected to happen.
4. **Write a short reflection** (half a page is enough) covering:
   - What you changed, and why.
   - What happened — including anything that surprised you.
   - One question you'd want answered before trusting the result, in the spirit of Sessions
     8–9.
5. **Present for about 5 minutes**: what you changed, what happened, your one open question.
   Presentation guidance: GeoSMART's **"Communicating Your Science"** section
   ([geo-smart.github.io/mlgeo-book](https://geo-smart.github.io/mlgeo-book/)).

## What "good" looks like

The bar is the same as everything else in this module: a small, honest change you understood,
not an ambitious one you didn't. A student who takes Session 2's regression, swaps in a
different rock property, and can clearly explain why the fit got better or worse has done
exactly what this capstone is for. A student who tries to bolt on a technique from outside the
module without understanding it has not.

## Assessment (≈60% of the module, per the course outline)

| Component | What it's checking |
|---|---|
| The adapted notebook itself | Does it run? Is the change a genuine (if modest) extension, not just re-running the original with different numbers? |
| Written reflection | Does it accurately describe what changed and what happened — including anything unexpected — rather than only reporting a clean success? |
| The open question | Is it a real, specific concern about trusting the result (tied to Sessions 8–9), not a generic disclaimer? |
| Presentation | Clear, to time, and honest about what worked and what didn't |

The remaining ≈40% is participation across Sessions 1–9, per the course outline.
