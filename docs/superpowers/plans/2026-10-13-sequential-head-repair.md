# Sequential Head Repair

## Observed Development Evidence

The completed phase7 run compared seven trials and 35 validation checkpoints.
Its highest global rhythm macro-F1 checkpoint, `full_balanced_asl_e40`, scored
1,257/1,476 (85.16% accuracy, 84.67% macro-F1), but its native PTB-XL SB recall
was 27/30 versus reference 30/30. Its pathology macro-F1 was 47.42% versus
reference 50.71%. It was not promoted. Shared first-batch training-gradient
cosines were negative in 24/40 full-backbone epochs. These are development
observations, not proof that interference uniquely explains the errors.

## Fixed Follow-Up Design

Continue that checkpoint as an intermediate research initialization, selected
using validation only, and freeze every backbone parameter and buffer.
Fit only the existing rhythm/pathology heads (38,899 parameters), without
resetting them. Preserve the original total 1,143,751 parameters, four heads,
six frontal leads, 125 Hz, all source tensors and reference vectors.
Quality/biometric parameters remain frozen. The phase6 model remains the
reference and fallback, not the ineligible phase7 intermediate.

Five predeclared 32-epoch controls compare pathology-only natural masked BCE,
joint head fitting with SB weights 1, 2 and 3, and a lower-rate SB3 control.
Other class-focus weights match the earlier phase6 method. Standard natural
record traversal replaces the unsuccessful balanced sampler. Pathology has
training-only sqrt negative/positive weights and all ten existing output
columns. No evaluation labels, records or class definitions are changed.

Use the network's native `forward_features` to cache its frozen embeddings
on the CEDIA GPU; verify cached-head outputs against the full network before
fitting. All neural-network gradients remain within CEDIA Slurm CUDA.
Retain epochs 4, 8, 16, 24 and 32, select with the same validation preservation
policy, refit deterministically, then and only then open follow-up targets.

## Checklist

- [x] Test head-only scopes, unchanged buffers, no layer resets and fallback.
- [x] Review and execute the isolated GPU trainer (29384, fallback).
- [x] Recalculate validation selection and replay the original signals.
- [ ] Publish both experiments, including rejected variants, without claiming
  any requested target was attained unless the completed evaluation proves it.
