# ESP32 Edge Deployment

This module contains ESP32-S3 firmware, model conversion artifacts, the Wi-Fi ECG recorder, and exported engineering recordings. The hardware scripts validate acquisition and file export; they do not establish clinical safety or numerical equivalence with the Python preprocessing pipeline.

## Build

From the repository root, install the pinned PlatformIO CLI and build a profile:

```powershell
python -m pip install -r requirements-firmware.txt
python -m platformio run -d hardware/esp32/firmware -e esp32s3
```

The build uses the project-local `esp32-s3-devkitc-1-n8r8` board profile: 8 MB flash and 8 MB octal PSRAM. The 4 MB tensor arena requires a PSRAM-equipped target; the default PlatformIO `esp32-s3-devkitc-1` profile identifies an N8 board without PSRAM and is not the intended hardware profile. This declares the firmware target, not proof that a physical N8R8 board was connected or tested. See Espressif's [DevKitC-1 variant table](https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32s3/esp32-s3-devkitc-1/user_guide_v1.1.html).

The `esp32s3` and `campus` profiles block `/api/result` and `/api/window.csv`. The `private_hotspot` profiles explicitly enable these unauthenticated HTTP endpoints for controlled recording and validation. Those endpoints expose diagnostic results and raw ECG windows to any reachable network client. Use those profiles only on an isolated private network with no untrusted peers or Internet port forwarding; do not use them on campus, public, or shared Wi-Fi.

The firmware pins TensorFlowLite_ESP32 to commit `e88e0ebee0430ed716ff5b49854795db90066e59` because this source uses its legacy Arduino API. The upstream maintainer says the library is outdated and no longer recommends it; migration to Espressif's maintained TFLite Micro integration is future work. The `esp32s3` compile is verified, but it does not verify flashing, runtime inference, clinical behavior, or equivalence with Python preprocessing. See [build evidence](../../reports/evidence/firmware-build.md).

## Recording

Disconnect electrodes before flashing or bench testing. For body-acquisition tests, use the documented isolated power arrangement and only a private hotspot profile. Follow the assembly and signal-quality procedures in `assembly_manual.md` and `signal_quality_protocol.md`.

```powershell
python hardware/esp32/recorder/record_wifi_ecg.py --base-url http://DEVICE_IP --duration-seconds 180 --max-windows 1
```

The Edge currently downsamples by linear interpolation while the training preprocessing uses `scipy.signal.resample_poly`. Numerical equivalence has not been established; see the pending measurement in `signal_quality_protocol.md`. Do not interpret an Edge recording as proof that the released model's offline metrics reproduce on-device.
