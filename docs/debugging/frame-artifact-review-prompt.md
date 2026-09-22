# Frame artifact review prompt (reusable debugging tool)

Use this prompt to get an unbiased description of rollout frame behavior,
before forming any hypothesis about the cause. Regenerate the diagnostic
image first with:

```
PYTHONPATH=. python3 scripts/render_diagnostic_grid.py \
  --checkpoints <name>=<path> [<name>=<path> ...] \
  --steps 0 1 2 3 5 8 12 20 \
  --out /tmp/artifact_grid.png
```

Then dispatch a fresh agent (no prior context about the model or known
issues) with:

---

Read the image at `<path to artifact_grid.png>`. It's a grid of frames from
a rollout/prediction model: each row is a different model checkpoint, each
column is a different rollout step (labeled at the top), row labels on the
left name the checkpoint. Each individual frame is auto-scaled to its own
min/max, not a shared scale, so you're looking at relative structure, not
absolute magnitude.

Describe, in plain terms, what you observe:
- What does the content of each row look like at step 0 vs later steps?
- Does structure change gradually or suddenly? At around which step(s)?
- Is there any regular/periodic/repeating spatial pattern (e.g. stripes,
  checkerboards, grid lines) in any row, and if so, where does it first
  appear and how does it evolve?
- Do different checkpoints/rows behave differently from each other, or the
  same way?
- Anything else visually notable.

Do not assume a cause (e.g. don't assume it's a known bug in windowed
attention, or a specific layer) - just describe what is visible. Report
findings in a few sentences to a paragraph, plainly.

---
