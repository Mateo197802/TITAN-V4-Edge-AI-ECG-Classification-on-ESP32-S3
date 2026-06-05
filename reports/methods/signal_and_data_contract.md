# Signal And Data Contract

## ECG Signal

The model input is a 10-second ECG window represented as six frontal leads at 125 Hz:

```text
shape = [1250 samples, 6 leads]
duration = 10 seconds
sample_rate = 125 Hz
leads = I, II, III, aVR, aVL, aVF
```

The ESP32 edge firmware captures three physical channels at 200 Hz and derives the frontal lead set using Einthoven relations:

```text
III = II - I
aVR = -(I + II) / 2
aVL = I - II / 2
aVF = II - I / 2
```

The edge recorder stores raw and derived data as CSV, MAT, HEA, and JSON metadata.

## External Validation

The public label term is:

```text
final external validation label set
```

External validation records are not used for model training or threshold tuning.

