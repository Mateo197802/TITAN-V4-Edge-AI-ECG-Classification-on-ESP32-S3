from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ESP32_ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_MAIN = ESP32_ROOT / "firmware" / "src" / "main.cpp"
INFERENCE_ENGINE = ESP32_ROOT / "firmware" / "src" / "ecg_inference.cpp"
FIRMWARE_CONFIG = ESP32_ROOT / "firmware" / "src" / "config.h"
PLATFORMIO_CONFIG = ESP32_ROOT / "firmware" / "platformio.ini"
HEADER_GENERATOR = ESP32_ROOT / "03_generate_model_header.py"


def load_header_generator():
    spec = importlib.util.spec_from_file_location("generate_model_header", HEADER_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {HEADER_GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EdgeDeploymentContractTests(unittest.TestCase):
    def test_header_generator_selects_micro_compatible_float32_model(self):
        module = load_header_generator()
        model_path = Path(module.select_tflite_path())
        self.assertEqual(model_path.name, "titan_v4_edge_float32.tflite")
        self.assertEqual(model_path.parent.name, "models")

    def test_firmware_writes_nhwc_sample_major_input_layout(self):
        source = INFERENCE_ENGINE.read_text(encoding="utf-8")
        self.assertIn("input_data[s * ECG_NUM_LEADS + lead]", source)

    def test_firmware_reports_adc_diagnostics(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("printAdcDiagnostics();", source)
        self.assertIn("[ADC]", source)

    def test_firmware_samples_adc_outside_timer_isr_with_mux_settling(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertNotIn("void IRAM_ATTR onSampleTimer()", source)
        self.assertIn("void captureSampleIfDue()", source)
        self.assertIn("delayMicroseconds(ADC_MUX_SETTLE_US);", source)
        self.assertIn("analogSetPinAttenuation(AD8232_LEAD_I_PIN, ADC_11db);", source)
        self.assertIn('json += ",\\"sampling_drops\\":"', source)

    def test_firmware_rejects_flat_or_saturated_adc_windows(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("bool signalUsable = printAdcDiagnostics();", source)
        self.assertIn("lastWindowAccepted = !leadsOff && lastPrimarySignalUsable;", source)
        self.assertIn("if (lastWindowAccepted)", source)
        self.assertIn("[ECG] Signal quality invalid", source)

    def test_firmware_audits_physical_lead_iii_with_einthoven_law(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("bool checkEinthovenConsistency()", source)
        self.assertIn("float expectedIII = leadII - leadI;", source)
        self.assertIn("lastEinthovenCorrelation", source)
        self.assertIn("lastFullFrontalQuality", source)
        self.assertIn('json += ",\\"primary_signal_usable\\":"', source)
        self.assertIn('json += ",\\"third_sensor_usable\\":"', source)
        self.assertIn('json += ",\\"einthoven_consistent\\":"', source)
        self.assertIn('json += ",\\"full_frontal_quality\\":"', source)

    def test_firmware_matches_training_bandpass_with_left_context(self):
        config = FIRMWARE_CONFIG.read_text(encoding="utf-8")
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("#define ECG_FILTER_CONTEXT_S      2", config)
        self.assertIn("#define ECG_FILTER_CONTEXT_SAMPLES", config)
        self.assertIn("#define ECG_TOTAL_CAPTURE_SAMPLES", config)
        self.assertIn("constexpr float CLINICAL_FILTER_LOW_HZ = 0.5f;", source)
        self.assertIn("constexpr float CLINICAL_FILTER_HIGH_HZ = 45.0f;", source)
        self.assertIn("void applyClinicalBandpass(", source)
        self.assertIn("ecg_filtered_phys", source)
        self.assertIn("&ecg_capture[lead][ECG_FILTER_CONTEXT_SAMPLES]", source)

    def test_firmware_reports_individual_raw_lead_quality(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("uint16_t clippedSamples;", source)
        self.assertIn("float clippedFraction;", source)
        self.assertIn("float stddev;", source)
        self.assertIn("float mains50Fraction;", source)
        self.assertIn("float mains60Fraction;", source)
        self.assertIn("float estimateToneFraction(", source)
        self.assertIn('json += ",\\"clipped_samples\\":"', source)
        self.assertIn('json += ",\\"clipped_fraction\\":"', source)
        self.assertIn('json += ",\\"stddev\\":"', source)
        self.assertIn('json += ",\\"mains_50_fraction\\":"', source)
        self.assertIn('json += ",\\"mains_60_fraction\\":"', source)

    def test_firmware_settles_each_adc_channel_before_storing_sample(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("constexpr uint32_t ADC_MUX_SETTLE_US = 50;", source)
        self.assertIn("uint16_t readSettledAdc(uint8_t pin)", source)
        self.assertIn("analogRead(pin);", source)
        self.assertIn("delayMicroseconds(ADC_MUX_SETTLE_US);", source)
        self.assertIn("ecg_capture[0][sampleIndex] = readSettledAdc(AD8232_LEAD_I_PIN);", source)
        self.assertIn("ecg_capture[1][sampleIndex] = readSettledAdc(AD8232_LEAD_II_PIN);", source)
        self.assertIn("ecg_capture[2][sampleIndex] = readSettledAdc(AD8232_LEAD_III_PIN);", source)
        self.assertIn("analogSetPinAttenuation(AD8232_LEAD_I_PIN, ADC_11db);", source)
        self.assertIn("analogSetPinAttenuation(AD8232_LEAD_II_PIN, ADC_11db);", source)
        self.assertIn("analogSetPinAttenuation(AD8232_LEAD_III_PIN, ADC_11db);", source)

    def test_firmware_exposes_quality_and_raw_window_endpoints(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn('server.on("/api/quality", handleQuality);', source)
        self.assertIn('server.on("/api/window.csv", handleWindowCsv);', source)
        self.assertIn("ecg_last_window", source)
        self.assertIn("snapshotCompletedWindow();", source)
        self.assertIn('json += ",\\"leads_off\\":"', source)
        self.assertIn('json += ",\\"window_accepted\\":"', source)

    def test_firmware_quality_endpoint_reports_live_capture_state(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn('json += ",\\"state\\":\\""', source)
        self.assertIn('json += ",\\"sample_index\\":"', source)
        self.assertIn('json += ",\\"total_capture_samples\\":"', source)
        self.assertIn('json += ",\\"capture_progress\\":"', source)
        self.assertIn('json += ",\\"uptime_seconds\\":"', source)
        self.assertIn('json += ",\\"free_heap_kb\\":"', source)
        self.assertIn('json += ",\\"free_psram_kb\\":"', source)

    def test_firmware_supports_signal_quality_profile_without_inference(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        config = FIRMWARE_CONFIG.read_text(encoding="utf-8")
        platformio = PLATFORMIO_CONFIG.read_text(encoding="utf-8")
        inference = INFERENCE_ENGINE.read_text(encoding="utf-8")
        self.assertIn("#define ENABLE_TFLITE_INFERENCE 1", config)
        self.assertIn("#define INFERENCE_TIMEOUT_MS 120000", config)
        self.assertIn("#if ENABLE_TFLITE_INFERENCE", source)
        self.assertIn("Inferencia desactivada", source)
        self.assertIn('json += ",\\"inference_enabled\\":"', source)
        self.assertIn("[env:private_hotspot_quality]", platformio)
        self.assertIn("-DENABLE_TFLITE_INFERENCE=0", platformio)
        self.assertIn("pdMS_TO_TICKS(INFERENCE_TIMEOUT_MS)", inference)
        self.assertIn("vTaskDelete(task_handle);", inference)

    def test_firmware_supports_open_wifi_without_exposing_sensitive_http(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        config = FIRMWARE_CONFIG.read_text(encoding="utf-8")
        platformio = PLATFORMIO_CONFIG.read_text(encoding="utf-8")
        self.assertIn("#define WIFI_OPEN_NETWORK 0", config)
        self.assertIn("#define ALLOW_SENSITIVE_HTTP 0", config)
        self.assertIn("#if WIFI_OPEN_NETWORK", source)
        self.assertIn("WiFi.begin(WIFI_SSID);", source)
        self.assertIn("void handleSensitiveHttpBlocked()", source)
        self.assertIn("#if ALLOW_SENSITIVE_HTTP", source)
        profiles = {}
        for name in ("esp32s3", "campus", "private_hotspot", "private_hotspot_quality"):
            profile = platformio.split(f"[env:{name}]", 1)[1].split("[env:", 1)[0]
            profiles[name] = profile
        self.assertIn("-DALLOW_SENSITIVE_HTTP=0", profiles["esp32s3"])
        self.assertIn("-DALLOW_SENSITIVE_HTTP=0", profiles["campus"])
        self.assertIn("-DALLOW_SENSITIVE_HTTP=1", profiles["private_hotspot"])
        self.assertIn("-DALLOW_SENSITIVE_HTTP=1", profiles["private_hotspot_quality"])
        self.assertIn('-DWIFI_SSID=\\"YACHAYTECH\\"', platformio)
        self.assertIn("-DWIFI_OPEN_NETWORK=1", platformio)
        self.assertIn("-DALLOW_SENSITIVE_HTTP=0", platformio)

    def test_firmware_reports_boot_and_wifi_ip_on_uart0(self):
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("Serial0.begin(SERIAL_BAUD);", source)
        self.assertIn('Serial0.println("[UART0] TITAN V4 Edge boot");', source)
        self.assertIn('Serial0.printf("[UART0] WiFi IP: %s\\n"', source)

    def test_firmware_pin_mapping_matches_ecg_assembly_manual(self):
        config = FIRMWARE_CONFIG.read_text(encoding="utf-8")
        source = FIRMWARE_MAIN.read_text(encoding="utf-8")
        self.assertIn("#define AD8232_LEAD_I_PIN    1", config)
        self.assertIn("#define AD8232_LEAD_II_PIN   2", config)
        self.assertIn("#define AD8232_LEAD_III_PIN  3", config)
        self.assertIn("#define AD8232_SHARED_SDN_PIN 4", config)
        self.assertIn("#define AD8232_LO_PLUS_PIN    5", config)
        self.assertIn("#define AD8232_LO_MINUS_PIN   6", config)
        self.assertIn("digitalWrite(AD8232_SHARED_SDN_PIN, HIGH);", source)

    def test_inference_engine_preserves_psram_for_tensor_arena(self):
        source = INFERENCE_ENGINE.read_text(encoding="utf-8")
        self.assertIn("required_psram", source)
        self.assertIn("TENSOR_ARENA_SIZE", source)


if __name__ == "__main__":
    unittest.main()
