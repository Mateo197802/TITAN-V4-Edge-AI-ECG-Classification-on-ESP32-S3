# ESP32 Edge Deployment

This module contains ESP32-S3 firmware, model conversion artifacts, a WiFi ECG recorder, and exported ECG recording evidence.

Safety note: upload firmware only when electrodes are disconnected. For body acquisition tests, use isolated power and WiFi recording.

Useful commands:

```powershell
cd hardware\esp32\firmware
python -m platformio run -e esp32s3
python -m platformio run -e private_hotspot_quality
```

Recorder:

```powershell
python hardware\esp32\recorder\record_wifi_ecg.py --base-url http://DEVICE_IP --duration-seconds 180 --max-windows 1
```

