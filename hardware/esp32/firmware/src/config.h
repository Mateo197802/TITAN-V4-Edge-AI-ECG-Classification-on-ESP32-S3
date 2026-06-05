/**
 * TITAN V4 Edge — Configuration
 * ================================
 * Hardware:  ESP32-S3 + 3x AD8232
 * Model:    TITAN V4 Primary-9 (rhythm) + Primary-10 (pathology)
 * 
 * Ciclo: CONTEXTO 2s + GRABAR 10s@200Hz → FILTRAR → DOWNSAMPLE → INFERIR
 */

#ifndef CONFIG_H
#define CONFIG_H

// ─── WiFi ────────────────────────────────────────────────────────────────────
#ifdef WIFI_PROFILE_PRIVATE
#define WIFI_SSID     "YOUR_PRIVATE_2G4_SSID"
#define WIFI_PASS     "YOUR_PRIVATE_WIFI_PASSWORD"
#endif
#ifndef WIFI_SSID
#define WIFI_SSID     "YOUR_PRIVATE_2G4_SSID"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS     "YOUR_PRIVATE_WIFI_PASSWORD"
#endif
#ifndef WIFI_OPEN_NETWORK
#define WIFI_OPEN_NETWORK 0
#endif
#ifndef ALLOW_SENSITIVE_HTTP
#define ALLOW_SENSITIVE_HTTP 1
#endif
#ifndef ENABLE_TFLITE_INFERENCE
#define ENABLE_TFLITE_INFERENCE 1
#endif
#ifndef INFERENCE_TIMEOUT_MS
#define INFERENCE_TIMEOUT_MS 120000
#endif

// ─── ECG / ADC ───────────────────────────────────────────────────────────────
// Salidas filtradas LM358 hacia ADC1 del ESP32-S3, según ecg.pdf
#define AD8232_LEAD_I_PIN    1    // Via 1: Lead I   -> GPIO1 / ADC1_CH0
#define AD8232_LEAD_II_PIN   2    // Via 2: Lead II  -> GPIO2 / ADC1_CH1
#define AD8232_LEAD_III_PIN  3    // Via 3: Lead III -> GPIO3 / ADC1_CH2

// Control compartido descrito por el manual de ensamblaje
#define AD8232_SHARED_SDN_PIN 4   // SDN de los tres módulos, activo en HIGH
#define AD8232_LO_PLUS_PIN    5   // LO+ expuesto por la vía de control
#define AD8232_LO_MINUS_PIN   6   // LO- expuesto por la vía de control

// ─── Señal ECG ───────────────────────────────────────────────────────────────
// Captura a 200 Hz (mayor resolución), downsample a 125 Hz para el modelo
#define ECG_CAPTURE_RATE    200       // Hz — frecuencia de muestreo ADC
#define ECG_MODEL_RATE      125       // Hz — frecuencia que espera el modelo
#define ECG_DURATION_S      10        // Ventana de 10 segundos
#define ECG_FILTER_CONTEXT_S      2   // Contexto causal previo para estabilizar el filtro

#define ECG_CAPTURE_SAMPLES (ECG_CAPTURE_RATE * ECG_DURATION_S)  // 2000
#define ECG_FILTER_CONTEXT_SAMPLES (ECG_CAPTURE_RATE * ECG_FILTER_CONTEXT_S) // 400
#define ECG_TOTAL_CAPTURE_SAMPLES (ECG_FILTER_CONTEXT_SAMPLES + ECG_CAPTURE_SAMPLES) // 2400
#define ECG_MODEL_SAMPLES   (ECG_MODEL_RATE * ECG_DURATION_S)    // 1250
#define ECG_NUM_LEADS_PHYS  3         // 3 leads físicas (AD8232)
#define ECG_NUM_LEADS       6         // 6 derivaciones (3 + Einthoven)

#define ECG_CAPTURE_PERIOD_US (1000000 / ECG_CAPTURE_RATE)  // 5000 us = 5 ms

// ─── Modelo ──────────────────────────────────────────────────────────────────
#define NUM_RHYTHM_CLASSES    9       // Primary-9
#define NUM_PATHOLOGY_CLASSES 10      // Pathology-10
#define MODEL_OUTPUT_SIZE     19      // 9 + 10

// Clases Primary-9 (ritmo)
static const char* RHYTHM_LABELS[NUM_RHYTHM_CLASSES] = {
    "AFIB",   // 0 - Fibrilación auricular
    "SB",     // 1 - Bradicardia sinusal
    "STACH",  // 2 - Taquicardia sinusal
    "NSR",    // 3 - Ritmo sinusal normal
    "PVC",    // 4 - Contracción ventricular prematura
    "RBBB",   // 5 - Bloqueo de rama derecha
    "LBBB",   // 6 - Bloqueo de rama izquierda
    "PAC",    // 7 - Contracción auricular prematura
    "1AVB",   // 8 - Bloqueo AV de primer grado
};

// Clases Pathology-10
static const char* PATHOLOGY_LABELS[NUM_PATHOLOGY_CLASSES] = {
    "IMI",    // 0 - Infarto inferior
    "ASMI",   // 1 - Infarto anteroseptal
    "LVH",    // 2 - Hipertrofia ventricular izquierda
    "ISC_",   // 3 - Isquemia
    "ISCAL",  // 4 - Isquemia anterolateral
    "NST_",   // 5 - Cambios inespecíficos ST-T
    "ILMI",   // 6 - Infarto inferolateral
    "AMI",    // 7 - Infarto anterior agudo
    "ALMI",   // 8 - Infarto anterolateral
    "LAE",    // 9 - Agrandamiento auricular izquierdo
};

// ─── TFLite Micro ────────────────────────────────────────────────────────────
// Arena en PSRAM — modelo ~2.2 MB + activaciones intermedias
#define TENSOR_ARENA_SIZE   (4 * 1024 * 1024)  // 4 MB arena en PSRAM

// ─── WiFi / HTTP ─────────────────────────────────────────────────────────────
#define HTTP_PORT           80
#define WIFI_RECONNECT_MS   5000
#define SERIAL_BAUD         115200

// ─── Estado del sistema ──────────────────────────────────────────────────────
enum SystemState {
    STATE_RECORDING,     // Grabando 10s de ECG @ 200Hz
    STATE_PROCESSING,    // Downsample + Einthoven + Z-score + Inferencia
    STATE_OUTPUT,        // Enviando resultados (Serial + WiFi)
    STATE_IDLE           // Pausa breve antes de reiniciar
};

#endif // CONFIG_H
