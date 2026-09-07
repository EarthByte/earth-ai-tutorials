# Session 9 — Reading a Real AI Paper, Together

**Data Science and AI Applied to Understanding Earth Evolution**

No notebook this session — this is a guided close reading, done as a class, of the paper
Sessions 1 and 5 have been building toward:

> Farahbakhsh, E., Betts, P. G., Ailleres, L., & Armit, R. J. (2025). *Machine
> Learning-Based Spatio-Temporal Prospectivity Modeling of Porphyry Systems in the New Guinea
> and Solomon Islands Region.* **Tectonics**.
> [https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024TC008362](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024TC008362)

This is the direct real-world version of the exercise you ran yourself, simplified, in
Session 5.

## How to read it (the reusable skill, not just this paper)

For each section, work out three things before moving to the next section:

1. **What is being claimed?** — in one sentence, no jargon.
2. **What evidence supports it?** — is it a number, a map, a comparison against something
   independent, or mostly a plausibility argument?
3. **What would make you doubt it?** — a missing comparison, an untested assumption, a result
   that could have another explanation.

## Section-by-section prompts

- **Abstract/Introduction:** What gap in existing porphyry-exploration methods is the paper
  trying to fill? Why does the New Guinea/Solomon Islands region need this specifically?
- **Data & input features:** What goes into the model — and which inputs (if any) come from
  plate reconstruction? How were "known deposit" and
  "no deposit" locations decided?
- **Method:** Which of Session 2/3/5's ideas (regression, classification, or something more
  elaborate) is actually being used? Is it closer to Session 5's random forest, or something
  you haven't seen in this module?
- **Validation:** How does the paper check that its predictions are trustworthy, rather than
  just plausible-looking? Does it test the model on deposits it wasn't trained on?
- **Results/prospectivity maps:** Compare the shape of these maps to the one you produced in
  Session 5. What's genuinely more sophisticated here, beyond just "more inputs"?
- **Discussion/Limitations (if present):** Does the paper raise any of Session 8's concerns
  itself — extrapolation, biased training samples, deep-time uncertainty — or does that need to
  come from you?

## Wrap-up

As a class, agree on one sentence each for: what the paper claims, the strongest piece of
evidence for it, and the one thing you'd most want to see before fully trusting its
predictions for a region with no known deposits yet. That three-part habit — claim, evidence,
what would change your mind — is the actual takeaway, transferable to any AI paper you read
after this module ends, in geoscience or otherwise.

## Optional: browsing real code (not running it)

Farahbakhsh's own workflow repository —
[github.com/e-farahbakhsh/GPlates_Workflows](https://github.com/e-farahbakhsh/GPlates_Workflows)
— is worth a five-minute browse as a class: not to run, just to see what a real research
codebase behind a published AI paper actually looks like, next to the small notebooks you've
been running yourselves.
