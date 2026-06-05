# Stable-Posture Hardware Diagnostic

## Acquisition Result

This 10-second recording was captured after the firmware update that separates:

- `window_accepted`: inference eligibility from physical Leads I and II.
- `full_frontal_quality`: redundant frontal acquisition quality after comparing
  physical Lead III against the Einthoven-derived value `III = II - I`.

The recording was rejected before inference because the primary acquisition is
not electrically usable.

| Channel | Minimum | Maximum | Mean | Clipped samples | Clipped fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lead I ADC | 668 | 3373 | 1951.0 | 0 | 0.00% |
| Lead II ADC | 0 | 4095 | 1854.2 | 957 | 47.85% |
| Physical Lead III ADC | 2 | 413 | 64.6 | 63 | 3.15% |

Lead II reaches both ADC rails repeatedly. The longest contiguous clipped
interval is 380 ms. The physical Lead III stage remains close to the ADC floor.

## Interpretation

The periodic rail-to-rail waveform on Lead II is incompatible with a clean ECG
window. It is not corrected by applying Einthoven's law. Einthoven's law is used
to derive `III = II - I` and to audit consistency between redundant frontal
measurements; it does not recover information lost to analog clipping.

The frontal model input remains defined from Leads I and II:

\[
III = II - I
\]

\[
aVR = -\frac{I + II}{2}
\]

\[
aVL = I - \frac{II}{2}
\]

\[
aVF = II - \frac{I}{2}
\]

## Required Hardware Checks

Perform these checks with electrodes removed from the body:

1. Measure the voltage reaching ESP32 `GPIO1`, `GPIO2`, and `GPIO3` relative to
   ESP32 ground.
2. Inspect the Lead II analog path from the second AD8232 output through its
   LM358/Sallen-Key stage and the final `10 kOhm` series resistor to `GPIO2`.
3. Confirm a shared ground between the analog board and ESP32.
4. Confirm that no LM358 output applied to an ESP32 ADC pin exceeds the ESP32-S3
   ADC input range.
5. Inspect the third analog path to `GPIO3`; its DC baseline is close to zero and
   is not centered like the Lead I and Lead II stages.
6. Repeat a short isolated-power acquisition only after the analog paths remain
   inside range.

## Long Recording Decision

The 20-minute recording is not authorized with this hardware state. A long
recording would preserve clipped data and would not be suitable for model
evaluation or dataset generation.
