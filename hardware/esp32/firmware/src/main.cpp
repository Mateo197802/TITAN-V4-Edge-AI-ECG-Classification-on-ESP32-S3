/**
 * TITAN V4 Edge — Main Firmware (State Machine)
 * ================================================
 * ESP32-S3 + 3x AD8232 + TFLite-Micro
 * 
 * CICLO:
 *   ┌─────────────────────────────────────────────┐
 *   │  RECORDING (2s context + 10s @ 200Hz)       │
 *   │      ↓                                      │
 *   │  PROCESSING                                 │
 *   │    • Butterworth 0.5-45Hz causal            │
 *   │    • Downsample 200Hz → 125Hz (2000→1250)   │
 *   │    • Derivar 6 leads (Einthoven)            │
 *   │    • Z-score normalize                      │
 *   │    • TFLite Inference                       │
 *   │      ↓                                      │
 *   │  OUTPUT (Serial + WiFi JSON)                │
 *   │      ↓                                      │
 *   │  RECORDING ...  (loop infinito)             │
 *   └─────────────────────────────────────────────┘
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <math.h>
#include "config.h"
#include "ecg_inference.h"

// ─── Objetos Globales ────────────────────────────────────────────────────────
ECGInference inference;
WebServer server(HTTP_PORT);

// Estado del sistema
volatile SystemState currentState = STATE_RECORDING;
volatile uint32_t sampleIndex = 0;

// Buffers ECG: 2s de contexto causal + 10s útiles, a 200Hz.
uint16_t ecg_capture[ECG_NUM_LEADS_PHYS][ECG_TOTAL_CAPTURE_SAMPLES];

// Snapshot estable de la última ventana terminada para descarga WiFi
uint16_t ecg_last_window[ECG_NUM_LEADS_PHYS][ECG_CAPTURE_SAMPLES];
float ecg_filtered_phys[ECG_NUM_LEADS_PHYS][ECG_TOTAL_CAPTURE_SAMPLES];
bool windowReady = false;
uint32_t lastWindowCycle = 0;

struct AdcLeadStats {
    uint16_t minValue;
    uint16_t maxValue;
    uint16_t span;
    uint16_t clippedSamples;
    float mean;
    float stddev;
    float clippedFraction;
    float mains50Fraction;
    float mains60Fraction;
};

AdcLeadStats lastAdcStats[ECG_NUM_LEADS_PHYS];
bool lastLeadSignalUsable[ECG_NUM_LEADS_PHYS] = {false, false, false};
bool lastSignalUsable = false;
bool lastPrimarySignalUsable = false;
bool lastThirdSensorUsable = false;
bool lastLeadsOff = true;
bool lastWindowAccepted = false;
bool lastEinthovenConsistent = false;
bool lastFullFrontalQuality = false;
float lastEinthovenCorrelation = 0.0f;
float lastEinthovenScale = 0.0f;
float lastEinthovenResidualRatio = 1.0f;

constexpr uint16_t ADC_BASELINE_MARGIN = 256;
constexpr uint32_t ADC_MUX_SETTLE_US = 50;
constexpr uint16_t ADC_CLIP_LOW = 5;
constexpr uint16_t ADC_CLIP_HIGH = 4090;
constexpr float ADC_MAX_CLIPPED_FRACTION = 0.01f;
constexpr float EINTHOVEN_MIN_CORRELATION = 0.70f;

// Butterworth bandpass order 3, 0.5-45Hz, fs=200Hz.
// Coefficients match scipy.signal.butter(..., btype="band") + lfilter.
constexpr float CLINICAL_FILTER_LOW_HZ = 0.5f;
constexpr float CLINICAL_FILTER_HIGH_HZ = 45.0f;
constexpr int CLINICAL_FILTER_ORDER = 6;
constexpr double CLINICAL_FILTER_B[CLINICAL_FILTER_ORDER + 1] = {
    0.126687518735, 0.0, -0.380062556204, 0.0,
    0.380062556204, 0.0, -0.126687518735
};
constexpr double CLINICAL_FILTER_A[CLINICAL_FILTER_ORDER + 1] = {
    1.0, -3.27278699294, 4.20517017846, -2.97487533544,
    1.45535151734, -0.442301170922, 0.0294458422518
};

// Buffers procesados — modelo a 125Hz (6 leads derivadas)
float ecg_model[ECG_NUM_LEADS][ECG_MODEL_SAMPLES];           // [6][1250]

// Resultado
InferenceResult lastResult;
uint32_t cycleCount = 0;
uint32_t lastProcessingMs = 0;
bool lastInferenceAttempted = false;

// Calendario de muestreo cooperativo fuera de ISR
uint32_t nextSampleMicros = 0;
uint32_t samplingDrops = 0;
uint32_t lastWindowSamplingDrops = 0;

// ─── ADC: descartar la primera conversión después de cambiar de canal ───────
uint16_t readSettledAdc(uint8_t pin) {
    analogRead(pin);
    delayMicroseconds(ADC_MUX_SETTLE_US);
    return analogRead(pin);
}

// ─── Muestreo a 200Hz fuera de ISR ──────────────────────────────────────────
void captureSampleIfDue() {
    if (currentState != STATE_RECORDING) return;

    uint32_t nowMicros = micros();
    if ((int32_t)(nowMicros - nextSampleMicros) < 0) return;

    uint32_t lateMicros = nowMicros - nextSampleMicros;
    uint32_t missedIntervals = lateMicros / ECG_CAPTURE_PERIOD_US;
    samplingDrops += missedIntervals;
    nextSampleMicros += (missedIntervals + 1) * ECG_CAPTURE_PERIOD_US;

    // La primera conversión se descarta para permitir el asentamiento del SAR.
    ecg_capture[0][sampleIndex] = readSettledAdc(AD8232_LEAD_I_PIN);
    ecg_capture[1][sampleIndex] = readSettledAdc(AD8232_LEAD_II_PIN);
    ecg_capture[2][sampleIndex] = readSettledAdc(AD8232_LEAD_III_PIN);
    sampleIndex++;

    if (sampleIndex >= ECG_TOTAL_CAPTURE_SAMPLES) {
        currentState = STATE_PROCESSING;
    }
}

// ─── Butterworth clínico causal 0.5-45Hz ────────────────────────────────────
void applyClinicalBandpass(const uint16_t src[], float dst[], int sampleCount) {
    double state[CLINICAL_FILTER_ORDER] = {0.0};
    for (int sample = 0; sample < sampleCount; sample++) {
        const double x = static_cast<double>(src[sample]);
        const double y = CLINICAL_FILTER_B[0] * x + state[0];
        for (int index = 0; index < CLINICAL_FILTER_ORDER - 1; index++) {
            state[index] = CLINICAL_FILTER_B[index + 1] * x
                         + state[index + 1]
                         - CLINICAL_FILTER_A[index + 1] * y;
        }
        state[CLINICAL_FILTER_ORDER - 1] =
            CLINICAL_FILTER_B[CLINICAL_FILTER_ORDER] * x
            - CLINICAL_FILTER_A[CLINICAL_FILTER_ORDER] * y;
        dst[sample] = static_cast<float>(y);
    }
}

// ─── Downsample 200Hz → 125Hz ───────────────────────────────────────────────
/**
 * Downsample por interpolación lineal.
 * 200Hz (2000 muestras) → 125Hz (1250 muestras)
 * Ratio = 200/125 = 1.6 muestras originales por muestra destino
 */
void downsample200to125(const float src[], float dst[], int src_len, int dst_len) {
    float ratio = (float)(src_len - 1) / (float)(dst_len - 1);
    for (int i = 0; i < dst_len; i++) {
        float idx = i * ratio;
        int lo = (int)idx;
        int hi = lo + 1;
        if (hi >= src_len) hi = src_len - 1;
        float frac = idx - lo;
        dst[i] = src[lo] * (1.0f - frac) + src[hi] * frac;
    }
}

// ─── Snapshot y diagnóstico ADC por lead física ─────────────────────────────
void snapshotCompletedWindow() {
    for (int lead = 0; lead < ECG_NUM_LEADS_PHYS; lead++) {
        memcpy(
            ecg_last_window[lead],
            &ecg_capture[lead][ECG_FILTER_CONTEXT_SAMPLES],
            sizeof(ecg_last_window[lead])
        );
    }
    lastWindowCycle = cycleCount;
    lastWindowSamplingDrops = samplingDrops;
    windowReady = true;
}

// Fracción de energía explicada por un tono específico sobre la señal centrada.
// Se usa como proxy diagnóstico; no modifica el tensor enviado al modelo.
float estimateToneFraction(
    const uint16_t samples[],
    int sampleCount,
    float mean,
    float frequencyHz
) {
    const float omega = 2.0f * PI * frequencyHz / ECG_CAPTURE_RATE;
    const float coefficient = 2.0f * cosf(omega);
    float previous = 0.0f;
    float previous2 = 0.0f;
    float totalSq = 0.0f;
    for (int sample = 0; sample < sampleCount; sample++) {
        const float centered = (float)samples[sample] - mean;
        const float current = centered + coefficient * previous - previous2;
        previous2 = previous;
        previous = current;
        totalSq += centered * centered;
    }
    if (totalSq <= 1e-8f) return 0.0f;
    const float tonePower = previous * previous + previous2 * previous2
                          - coefficient * previous * previous2;
    return min(1.0f, max(0.0f, 2.0f * tonePower / (sampleCount * totalSq)));
}

bool printAdcDiagnostics() {
    static const char* leadNames[ECG_NUM_LEADS_PHYS] = {"I", "II", "III"};
    bool signalUsable = true;
    for (int lead = 0; lead < ECG_NUM_LEADS_PHYS; lead++) {
        uint16_t minValue = 4095;
        uint16_t maxValue = 0;
        uint32_t sum = 0;
        double sumSq = 0.0;
        uint16_t clippedSamples = 0;
        for (int s = 0; s < ECG_CAPTURE_SAMPLES; s++) {
            uint16_t value = ecg_last_window[lead][s];
            minValue = min(minValue, value);
            maxValue = max(maxValue, value);
            sum += value;
            sumSq += static_cast<double>(value) * value;
            if (value <= ADC_CLIP_LOW || value >= ADC_CLIP_HIGH) {
                clippedSamples++;
            }
        }

        uint16_t span = maxValue - minValue;
        float mean = (float)sum / ECG_CAPTURE_SAMPLES;
        float variance = max(0.0, sumSq / ECG_CAPTURE_SAMPLES - mean * mean);
        float stddev = sqrtf(variance);
        float clippedFraction = (float)clippedSamples / ECG_CAPTURE_SAMPLES;
        float mains50Fraction = estimateToneFraction(
            ecg_last_window[lead], ECG_CAPTURE_SAMPLES, mean, 50.0f
        );
        float mains60Fraction = estimateToneFraction(
            ecg_last_window[lead], ECG_CAPTURE_SAMPLES, mean, 60.0f
        );
        lastAdcStats[lead] = {
            minValue, maxValue, span, clippedSamples, mean, stddev, clippedFraction,
            mains50Fraction, mains60Fraction
        };
        Serial.printf(
            "[ADC] Lead %s min=%u max=%u span=%u mean=%.1f std=%.1f clipped=%u (%.2f%%) mains50=%.3f mains60=%.3f\n",
            leadNames[lead], minValue, maxValue, span, mean, stddev,
            clippedSamples, clippedFraction * 100.0f, mains50Fraction, mains60Fraction
        );
        bool leadUsable = true;
        if (span < 8) {
            Serial.printf("[ADC] WARN Lead %s flat signal\n", leadNames[lead]);
            leadUsable = false;
        }
        if (clippedSamples > 0) {
            Serial.printf("[ADC] WARN Lead %s reached ADC rail\n", leadNames[lead]);
        }
        if (clippedFraction > ADC_MAX_CLIPPED_FRACTION) {
            Serial.printf("[ADC] WARN Lead %s excessive clipping\n", leadNames[lead]);
            leadUsable = false;
        }
        if (mean <= ADC_BASELINE_MARGIN || mean >= 4095 - ADC_BASELINE_MARGIN) {
            Serial.printf("[ADC] WARN Lead %s baseline near ADC edge\n", leadNames[lead]);
            leadUsable = false;
        }
        lastLeadSignalUsable[lead] = leadUsable;
        signalUsable = signalUsable && leadUsable;
    }
    lastPrimarySignalUsable = lastLeadSignalUsable[0] && lastLeadSignalUsable[1];
    lastThirdSensorUsable = lastLeadSignalUsable[2];
    lastSignalUsable = signalUsable;
    return signalUsable;
}

// ─── Auditoría redundante de Einthoven ─────────────────────────────────────
/**
 * Lead III is measured physically for audit, while the model receives the
 * algebraic derivation III = II - I. Centering removes analog DC offsets and a
 * fitted scale exposes gain differences between the three analog front ends.
 */
bool checkEinthovenConsistency() {
    const float meanI = lastAdcStats[0].mean;
    const float meanII = lastAdcStats[1].mean;
    const float meanIII = lastAdcStats[2].mean;
    float expectedSq = 0.0f;
    float measuredSq = 0.0f;
    float cross = 0.0f;

    for (int sample = 0; sample < ECG_CAPTURE_SAMPLES; sample++) {
        float leadI = (float)ecg_last_window[0][sample] - meanI;
        float leadII = (float)ecg_last_window[1][sample] - meanII;
        float measuredIII = (float)ecg_last_window[2][sample] - meanIII;
        float expectedIII = leadII - leadI;
        expectedSq += expectedIII * expectedIII;
        measuredSq += measuredIII * measuredIII;
        cross += expectedIII * measuredIII;
    }

    if (expectedSq <= 1e-8f || measuredSq <= 1e-8f) {
        lastEinthovenCorrelation = 0.0f;
        lastEinthovenScale = 0.0f;
        lastEinthovenResidualRatio = 1.0f;
        lastEinthovenConsistent = false;
        return false;
    }

    lastEinthovenCorrelation = cross / sqrtf(expectedSq * measuredSq);
    lastEinthovenScale = cross / expectedSq;
    float residualSq = measuredSq - 2.0f * lastEinthovenScale * cross
                     + lastEinthovenScale * lastEinthovenScale * expectedSq;
    lastEinthovenResidualRatio = sqrtf(max(0.0f, residualSq) / measuredSq);
    lastEinthovenConsistent = lastThirdSensorUsable
                           && lastEinthovenCorrelation >= EINTHOVEN_MIN_CORRELATION
                           && lastEinthovenScale > 0.0f;
    Serial.printf("[EINTHOVEN] corr=%.3f scale=%.3f residual=%.3f consistent=%s\n",
                  lastEinthovenCorrelation,
                  lastEinthovenScale,
                  lastEinthovenResidualRatio,
                  lastEinthovenConsistent ? "true" : "false");
    return lastEinthovenConsistent;
}

// ─── Derivación de Einthoven (3 → 6 leads) ──────────────────────────────────
void deriveEinthoven() {
    // Las 3 primeras ya están downsampled
    for (int s = 0; s < ECG_MODEL_SAMPLES; s++) {
        float I   = ecg_model[0][s];
        float II  = ecg_model[1][s];
        // Lead III = II - I (ya calculada abajo, pero la guardamos explícitamente)
        ecg_model[2][s] = II - I;                    // Lead III
        ecg_model[3][s] = -(I + II) / 2.0f;         // aVR
        ecg_model[4][s] = I - II / 2.0f;            // aVL
        ecg_model[5][s] = II - I / 2.0f;            // aVF
    }
}

// ─── Normalización Z-score por lead ──────────────────────────────────────────
void normalizeZScore() {
    for (int lead = 0; lead < ECG_NUM_LEADS; lead++) {
        // Media
        float sum = 0;
        for (int s = 0; s < ECG_MODEL_SAMPLES; s++) sum += ecg_model[lead][s];
        float mean = sum / ECG_MODEL_SAMPLES;
        
        // Std dev
        float sq_sum = 0;
        for (int s = 0; s < ECG_MODEL_SAMPLES; s++) {
            float d = ecg_model[lead][s] - mean;
            sq_sum += d * d;
        }
        float std_dev = sqrtf(sq_sum / ECG_MODEL_SAMPLES);
        if (std_dev < 1e-8f) std_dev = 1e-8f;
        
        // Normalizar
        for (int s = 0; s < ECG_MODEL_SAMPLES; s++) {
            ecg_model[lead][s] = (ecg_model[lead][s] - mean) / std_dev;
        }
    }
}

// ─── Procesamiento completo ──────────────────────────────────────────────────
void processECG() {
    uint32_t t0 = millis();
    lastInferenceAttempted = false;
    
    // 1. Filtrar cada vía física con el mismo bandpass usado en entrenamiento.
    Serial.println("[PROC] Butterworth causal 0.5-45Hz...");
    for (int lead = 0; lead < ECG_NUM_LEADS_PHYS; lead++) {
        applyClinicalBandpass(
            ecg_capture[lead],
            ecg_filtered_phys[lead],
            ECG_TOTAL_CAPTURE_SAMPLES
        );
    }

    // 2. Downsample de la ventana útil, excluyendo el contexto causal.
    Serial.println("[PROC] Downsample 200Hz → 125Hz...");
    for (int lead = 0; lead < ECG_NUM_LEADS_PHYS; lead++) {
        downsample200to125(
            &ecg_filtered_phys[lead][ECG_FILTER_CONTEXT_SAMPLES],
            ecg_model[lead],
            ECG_CAPTURE_SAMPLES,
            ECG_MODEL_SAMPLES
        );
    }
    
    // 3. Derivar 6 leads con Einthoven
    Serial.println("[PROC] Derivando 6 leads (Einthoven)...");
    deriveEinthoven();
    
    // 4. Normalizar Z-score
    Serial.println("[PROC] Normalizando Z-score...");
    normalizeZScore();
    
    // 5. Inferencia TFLite
#if ENABLE_TFLITE_INFERENCE
    Serial.println("[PROC] Ejecutando inferencia TITAN V4...");
    if (inference.isReady()) {
        lastInferenceAttempted = true;
        inference.infer(ecg_model, lastResult);
    } else {
        lastResult.success = false;
        Serial.println("[PROC] ⚠ Modelo no cargado, saltando inferencia");
    }
#else
    lastResult.success = false;
    Serial.println("[PROC] Inferencia desactivada: modo quality/recording");
#endif
    
    lastProcessingMs = millis() - t0;
    Serial.printf("[PROC] Pipeline completo en %d ms\n", lastProcessingMs);
}

// ─── WiFi ────────────────────────────────────────────────────────────────────
void connectWiFi() {
    Serial.printf("[WiFi] Conectando a %s (2.4GHz)...\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
#if WIFI_OPEN_NETWORK
    WiFi.begin(WIFI_SSID);
#else
    WiFi.begin(WIFI_SSID, WIFI_PASS);
#endif
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] ✓ IP: %s\n", WiFi.localIP().toString().c_str());
        Serial0.printf("[UART0] WiFi IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WiFi] ✗ Sin conexión. Continuando offline.");
        Serial0.println("[UART0] WiFi unavailable");
    }
}

// ─── HTTP Handlers ───────────────────────────────────────────────────────────
void handleSensitiveHttpBlocked() {
    server.send(
        403,
        "application/json",
        "{\"error\":\"sensitive_http_disabled_on_shared_network\"}"
    );
}

void handleRoot() {
    String html = "<!DOCTYPE html><html><head><title>TITAN V4 Edge</title>";
    html += "<meta charset='utf-8'><meta http-equiv='refresh' content='11'>";
    html += "<style>body{font-family:'Courier New',monospace;background:#0a0a1a;color:#e0e0e0;padding:20px;max-width:700px;margin:0 auto}";
    html += "h1{color:#00d4ff;text-align:center}h2{color:#00ff88}";
    html += ".state{padding:10px;border-radius:8px;margin:10px 0;font-size:1.2em;text-align:center}";
    html += ".rec{background:#442200;color:#ffaa00;border:2px solid #ff6600}";
    html += ".proc{background:#002244;color:#00aaff;border:2px solid #0066ff}";
    html += ".ok{color:#00ff88}.warn{color:#ffaa00}.danger{color:#ff4444}";
    html += "table{border-collapse:collapse;width:100%;margin:10px 0}td,th{border:1px solid #333;padding:6px;text-align:left}";
    html += "th{background:#1a1a3a}</style></head><body>";
    html += "<h1>&#x2764; TITAN V4 Edge</h1>";
    
    // Estado actual
    const char* stateNames[] = {"GRABANDO", "PROCESANDO", "RESULTADO", "ESPERA"};
    const char* stateClass[] = {"rec", "proc", "ok", "rec"};
    int si = (int)currentState;
    html += "<div class='state " + String(stateClass[si]) + "'>Estado: " + stateNames[si] + " | Ciclo #" + String(cycleCount) + "</div>";
    
#if ALLOW_SENSITIVE_HTTP
    if (lastResult.success) {
        html += "<h2>Ritmo: <span class='ok'>" + String(lastResult.rhythm_label) + " (" + String(lastResult.rhythm_top_confidence * 100, 1) + "%)</span></h2>";
        html += "<table><tr><th>Clase</th><th>Prob</th></tr>";
        for (int i = 0; i < NUM_RHYTHM_CLASSES; i++) {
            html += "<tr><td>" + String(RHYTHM_LABELS[i]) + "</td><td>" + String(lastResult.rhythm_probs[i] * 100, 1) + "%</td></tr>";
        }
        html += "</table>";
        
        if (lastResult.pathology_count > 0) {
            html += "<h2 class='warn'>Patologías: " + String(lastResult.pathology_count) + "</h2><table><tr><th>Patología</th><th>Prob</th></tr>";
            for (int i = 0; i < NUM_PATHOLOGY_CLASSES; i++) {
                if (lastResult.pathology_detected[i]) {
                    html += "<tr><td class='danger'>" + String(PATHOLOGY_LABELS[i]) + "</td><td>" + String(lastResult.pathology_probs[i] * 100, 1) + "%</td></tr>";
                }
            }
            html += "</table>";
        } else {
            html += "<p class='ok'>&#10003; Sin patologías detectadas</p>";
        }
        html += "<p>Inferencia: " + String(lastResult.inference_time_ms) + " ms</p>";
    } else {
        html += "<p>Esperando primera inferencia...</p>";
    }
#else
    html += "<p class='warn'>Perfil campus-safe: resultados y ventanas ECG bloqueados.</p>";
    html += "<p>Disponible: <code>/api/quality</code></p>";
#endif
    
    html += "<hr><p style='font-size:0.8em'>Heap: " + String(ESP.getFreeHeap()/1024) + "KB | PSRAM: " + String(ESP.getFreePsram()/1024) + "KB | Uptime: " + String(millis()/1000) + "s</p>";
    html += "</body></html>";
    server.send(200, "text/html", html);
}

void handleApi() {
    if (lastResult.success) {
        String json = inference.resultToJson(lastResult);
        // Agregar metadata del ciclo
        json.remove(json.length() - 1); // quitar }
        json += ",\"cycle\":" + String(cycleCount);
        json += ",\"state\":\"" + String(currentState == STATE_RECORDING ? "recording" : "idle") + "\"";
        json += "}";
        server.send(200, "application/json", json);
    } else {
        server.send(200, "application/json", "{\"status\":\"waiting\",\"cycle\":" + String(cycleCount) + "}");
    }
}

void handleQuality() {
    String json = "{";
    json += "\"cycle\":" + String(lastWindowCycle);
    const char* stateNames[] = {"recording", "processing", "output", "idle"};
    int stateIndex = (int)currentState;
    if (stateIndex < 0 || stateIndex > 3) stateIndex = 3;
    json += ",\"state\":\"" + String(stateNames[stateIndex]) + "\"";
    json += ",\"sample_index\":" + String((uint32_t)sampleIndex);
    json += ",\"total_capture_samples\":" + String(ECG_TOTAL_CAPTURE_SAMPLES);
    json += ",\"capture_progress\":" + String((float)sampleIndex / ECG_TOTAL_CAPTURE_SAMPLES, 4);
    json += ",\"window_ready\":" + String(windowReady ? "true" : "false");
    json += ",\"signal_usable\":" + String(lastSignalUsable ? "true" : "false");
    json += ",\"primary_signal_usable\":" + String(lastPrimarySignalUsable ? "true" : "false");
    json += ",\"third_sensor_usable\":" + String(lastThirdSensorUsable ? "true" : "false");
    json += ",\"leads_off\":" + String(lastLeadsOff ? "true" : "false");
    json += ",\"window_accepted\":" + String(lastWindowAccepted ? "true" : "false");
    json += ",\"einthoven_consistent\":" + String(lastEinthovenConsistent ? "true" : "false");
    json += ",\"einthoven_correlation\":" + String(lastEinthovenCorrelation, 4);
    json += ",\"einthoven_scale\":" + String(lastEinthovenScale, 4);
    json += ",\"einthoven_residual_ratio\":" + String(lastEinthovenResidualRatio, 4);
    json += ",\"full_frontal_quality\":" + String(lastFullFrontalQuality ? "true" : "false");
    json += ",\"sampling_drops\":" + String(lastWindowSamplingDrops);
    json += ",\"last_processing_ms\":" + String(lastProcessingMs);
    json += ",\"last_inference_attempted\":" + String(lastInferenceAttempted ? "true" : "false");
    json += ",\"inference_enabled\":" + String(ENABLE_TFLITE_INFERENCE ? "true" : "false");
    json += ",\"inference_timeout_ms\":" + String(INFERENCE_TIMEOUT_MS);
    json += ",\"uptime_seconds\":" + String(millis() / 1000);
    json += ",\"free_heap_kb\":" + String(ESP.getFreeHeap() / 1024);
    json += ",\"free_psram_kb\":" + String(ESP.getFreePsram() / 1024);
    json += ",\"capture_rate_hz\":" + String(ECG_CAPTURE_RATE);
    json += ",\"samples_per_window\":" + String(ECG_CAPTURE_SAMPLES);
    json += ",\"filter_context_samples\":" + String(ECG_FILTER_CONTEXT_SAMPLES);
    json += ",\"model_bandpass_hz\":\"0.5-45\"";
    json += ",\"physical_leads\":[";
    static const char* leadNames[ECG_NUM_LEADS_PHYS] = {"I", "II", "III_SENSOR"};
    for (int lead = 0; lead < ECG_NUM_LEADS_PHYS; lead++) {
        if (lead > 0) json += ",";
        json += "{\"lead\":\"" + String(leadNames[lead]) + "\"";
        json += ",\"min\":" + String(lastAdcStats[lead].minValue);
        json += ",\"max\":" + String(lastAdcStats[lead].maxValue);
        json += ",\"span\":" + String(lastAdcStats[lead].span);
        json += ",\"mean\":" + String(lastAdcStats[lead].mean, 1);
        json += ",\"stddev\":" + String(lastAdcStats[lead].stddev, 1);
        json += ",\"clipped_samples\":" + String(lastAdcStats[lead].clippedSamples);
        json += ",\"clipped_fraction\":" + String(lastAdcStats[lead].clippedFraction, 4);
        json += ",\"mains_50_fraction\":" + String(lastAdcStats[lead].mains50Fraction, 4);
        json += ",\"mains_60_fraction\":" + String(lastAdcStats[lead].mains60Fraction, 4);
        json += "}";
    }
    json += "]}";
    server.send(200, "application/json", json);
}

void handleWindowCsv() {
    if (!windowReady) {
        server.send(404, "application/json", "{\"error\":\"window_not_ready\"}");
        return;
    }

    server.sendHeader("Cache-Control", "no-store");
    server.sendHeader("Connection", "close");
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "text/csv", "cycle,sample_index,lead_i_adc,lead_ii_adc,lead_iii_adc\n");

    String chunk;
    chunk.reserve(1024);
    for (int sample = 0; sample < ECG_CAPTURE_SAMPLES; sample++) {
        chunk += String(lastWindowCycle) + "," + String(sample) + ",";
        chunk += String(ecg_last_window[0][sample]) + ",";
        chunk += String(ecg_last_window[1][sample]) + ",";
        chunk += String(ecg_last_window[2][sample]) + "\n";
        if (chunk.length() >= 900) {
            server.sendContent(chunk);
            chunk = "";
        }
    }
    if (chunk.length() > 0) {
        server.sendContent(chunk);
    }
    server.sendContent("");
}

// ─── SETUP ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(SERIAL_BAUD);
    Serial0.begin(SERIAL_BAUD);
    delay(1000);
    Serial0.println("[UART0] TITAN V4 Edge boot");
    
    Serial.println("\n");
    Serial.println("╔══════════════════════════════════════════════╗");
    Serial.println("║   TITAN V4 Edge — ESP32-S3                  ║");
    Serial.println("║   Context 2s + Record 10s → Filter → Infer ║");
    Serial.println("║   Primary-9 Rhythm + Pathology-10           ║");
    Serial.println("╚══════════════════════════════════════════════╝\n");
    
    // PSRAM
    if (psramFound()) {
        Serial.printf("[MEM] PSRAM: %.1f MB\n", ESP.getPsramSize() / 1024.0 / 1024.0);
    } else {
        Serial.println("[MEM] ⚠ Sin PSRAM — puede faltar memoria");
    }
    Serial.printf("[MEM] Heap: %d KB\n", ESP.getFreeHeap() / 1024);
    
    // WiFi
    connectWiFi();
    
    // HTTP
    if (WiFi.status() == WL_CONNECTED) {
        server.on("/", handleRoot);
#if ALLOW_SENSITIVE_HTTP
        server.on("/api/result", handleApi);
        server.on("/api/window.csv", handleWindowCsv);
#else
        server.on("/api/result", handleSensitiveHttpBlocked);
        server.on("/api/window.csv", handleSensitiveHttpBlocked);
#endif
        server.on("/api/quality", handleQuality);
        server.begin();
        Serial.printf("[HTTP] http://%s/\n", WiFi.localIP().toString().c_str());
    }
    
    // Modelo TFLite
    Serial.println();
#if ENABLE_TFLITE_INFERENCE
    if (!inference.begin()) {
        Serial.println("╔══════════════════════════════════════╗");
        Serial.println("║  ⚠ Modelo no cargado                ║");
        Serial.println("║  Verifica model_data.h               ║");
        Serial.println("╚══════════════════════════════════════╝");
    }
#else
    Serial.println("[TFLite] Inferencia desactivada para validación de señal.");
#endif
    
    // ADC 12-bit
    analogReadResolution(12);
    analogSetPinAttenuation(AD8232_LEAD_I_PIN, ADC_11db);
    analogSetPinAttenuation(AD8232_LEAD_II_PIN, ADC_11db);
    analogSetPinAttenuation(AD8232_LEAD_III_PIN, ADC_11db);
    
    // Habilitar los tres AD8232 y leer la detección de leads-off compartida.
    pinMode(AD8232_SHARED_SDN_PIN, OUTPUT);
    digitalWrite(AD8232_SHARED_SDN_PIN, HIGH);
    pinMode(AD8232_LO_PLUS_PIN, INPUT);
    pinMode(AD8232_LO_MINUS_PIN, INPUT);
    
    // Iniciar primer ciclo
    sampleIndex = 0;
    samplingDrops = 0;
    nextSampleMicros = micros();
    currentState = STATE_RECORDING;
    
    Serial.println("\n[READY] ═══ Ciclo 1: Contexto 2s + ventana 10s @ 200Hz... ═══\n");
}

// ─── LOOP (Máquina de Estados) ───────────────────────────────────────────────
void loop() {
    // Atender HTTP siempre
    if (WiFi.status() == WL_CONNECTED) {
        server.handleClient();
    }
    
    switch (currentState) {
        
        case STATE_RECORDING:
            // Muestrear fuera de ISR y mostrar progreso cada 2s.
            {
                captureSampleIfDue();
                static uint32_t lastProgress = 0;
                if (millis() - lastProgress > 2000 && sampleIndex > 0) {
                    float pct = (float)sampleIndex / ECG_TOTAL_CAPTURE_SAMPLES * 100.0f;
                    float secs = (float)sampleIndex / ECG_CAPTURE_RATE;
                    Serial.printf("[REC] %.1fs / 12s (%.0f%%) — %d muestras\n", 
                                  secs, pct, (uint32_t)sampleIndex);
                    lastProgress = millis();
                }
            }
            break;
            
        case STATE_PROCESSING:
        {
            cycleCount++;
            
            Serial.printf("\n[CICLO %d] ═══ Buffer lleno (%d muestras). Procesando... ═══\n",
                          cycleCount, ECG_TOTAL_CAPTURE_SAMPLES);
            snapshotCompletedWindow();
            bool signalUsable = printAdcDiagnostics();
            bool einthovenConsistent = checkEinthovenConsistency();
            
            // Verificar leads-off
            bool leadsOff = false;
            if (digitalRead(AD8232_LO_PLUS_PIN) || digitalRead(AD8232_LO_MINUS_PIN)) {
                Serial.println("[ECG] ⚠ Leads-off detectado");
                leadsOff = true;
            }
            lastLeadsOff = leadsOff;
            lastWindowAccepted = !leadsOff && lastPrimarySignalUsable;
            lastFullFrontalQuality = lastWindowAccepted && einthovenConsistent;
            
            if (lastWindowAccepted) {
                if (!lastFullFrontalQuality) {
                    Serial.println("[ECG] WARN Physical Lead III failed Einthoven audit; using derived frontal leads");
                }
                processECG();
            } else {
                if (!lastPrimarySignalUsable) {
                    Serial.println("[ECG] Signal quality invalid - skipping inference");
                } else {
                    Serial.println("[ECG] Leads desconectadas - saltando inferencia");
                }
                lastResult.success = false;
            }
            
            currentState = STATE_OUTPUT;
            break;
        }
            
        case STATE_OUTPUT:
            // Imprimir resultado
            if (lastResult.success) {
                inference.printResult(lastResult);
                
                // Enviar por Serial como JSON para la PC
                Serial.println("[JSON] " + inference.resultToJson(lastResult));
            }
            
            Serial.printf("\n[CICLO %d] ═══ Reiniciando grabación... ═══\n\n", cycleCount);
            
            // ═══ REINICIAR: Limpiar buffer y reiniciar calendario ═══
            sampleIndex = 0;
            samplingDrops = 0;
            nextSampleMicros = micros();
            currentState = STATE_RECORDING;
            break;
            
        case STATE_IDLE:
            break;
    }
    
    // Reconexión WiFi
    static uint32_t lastWiFiCheck = 0;
    if (millis() - lastWiFiCheck > WIFI_RECONNECT_MS) {
        lastWiFiCheck = millis();
        if (WiFi.status() != WL_CONNECTED) {
            WiFi.reconnect();
        }
    }
    
    delay(1);
}
