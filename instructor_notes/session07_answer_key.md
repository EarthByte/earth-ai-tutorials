---
Instructor-only. Do not share with students before Session 7 — the point of the exercise is
finding these by working with an LLM, not being handed the answer.
---

# Session 7 answer key — S07_GenAI_Assisted_Debugging.ipynb

Each fix has been run and confirmed to work.

**Exercise 1 — `NameError`**
The model was fit as `reg`, but the print line refers to `linear_model`, which was never
defined.
```python
print("Fitted half-rate:", reg.coef_[0], "km/Myr")
```

**Exercise 2 — `ValueError` (mismatched lengths)**
`silica` has 4 values but `density` and `labels` have 5 — `np.column_stack` requires equal
lengths. The missing value needs to be added back:
```python
silica = np.array([50, 68, 60, 65, 61])
```

**Exercise 3 — `NameError`**
`train_test_split` is used but never imported.
```python
from sklearn.model_selection import train_test_split
```
(needs to go before the call — otherwise everything else in the cell is unchanged.)

**Exercise 4 — `KeyError: 'geochemistry'`**
The dictionary key is `"geochem"`, but `FEATURES_TO_USE` asks for `"geochemistry"` — a typo,
not a missing feature.
```python
FEATURES_TO_USE = ["distance_to_fault", "geochem"]
```

**Exercise 5 — silent logic bug, no traceback**
The comparison is backwards: `<=` flags the *low*-probability cells as targets instead of the
high-probability ones.
```python
target_zone = predicted_probability >= PROSPECTIVITY_THRESHOLD
```
Correct count for the five values given (`0.1, 0.8, 0.05, 0.95, 0.3`) at threshold `0.5` is
**2** (the 0.8 and 0.95 cells) — the buggy version returns 3.

If a group's LLM tool doesn't catch this one without being told the expected output, that's
the intended outcome, not a tooling failure — it's the discussion hook for the wrap-up and for
Session 8.

**Exercise 6** has no fixed answer — check the LLM's snippet computes `np.mean(ages)` and
`np.std(ages)` (or equivalent) and that the printed numbers match a rough by-eye estimate for
the list `[42.1, 38.7, 55.0, 29.3, 61.4, 47.8, 33.2]` (mean ≈ 43.9, population std ≈ 10.7).
