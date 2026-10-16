# Signal and Data Contract

## Offline Primary-9 Inference

The runner reads physical WFDB signals, takes the first ten seconds from leads I, II, III, aVR, aVL, and aVF, replaces non-finite samples with zero, and applies the fixed preprocessing below:

```text
Input window: 10 seconds; six frontal leads
Butterworth band-pass: 0.5-45 Hz, order 3
Filtering: causal scipy.signal.lfilter with two seconds of leading zero context
Resampling: scipy.signal.resample_poly to 125 Hz
Normalization: per-lead z-score over the 10-second window
Model tensor: [batch, 6 leads, 1250 samples]
Classes: AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, 1AVB
```

Source sample rates differ across records; the runner reads them from the WFDB headers and applies the same resampling function. Required leads are checked by name; a record with missing required leads fails rather than substituting channels.

## Labels and Evaluation

The label builder maps WFDB `Dx` SNOMED codes to the fixed Primary-9 classes, then selects the first non-NSR mapped class in header order. Prolonged-PR code `164947007` maps to 1AVB only when there is no other non-normal Primary-9 rhythm. NSR is selected when no non-normal class or prolonged-PR code is present.

The project-local `split=test` marks the frozen evaluation cohort only. Its 672 records resolve to PhysioNet Challenge 2021 v1.0.3 `training/` paths. This field is not an official hidden test designation and does not demonstrate independence from model training. The inference run performs no training or threshold tuning.

## ESP32-S3 Path

The firmware captures three physical channels at 200 Hz and derives leads using the Einthoven relations below. This is a separate signal path; its numerical equivalence to the offline SciPy preprocessing has not been established.

```text
III = II - I
aVR = -(I + II) / 2
aVL = I - II / 2
aVF = II - I / 2
```

The edge recorder stores raw and derived data as CSV, MAT, HEA, and JSON metadata. Firmware compilation and physical-device behavior require separate verification.
