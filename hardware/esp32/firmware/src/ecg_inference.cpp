/**
 * TITAN V4 Edge — ECG Inference Engine Implementation
 * =====================================================
 * TFLite-Micro inference on ESP32-S3 with PSRAM arena.
 */

#include "ecg_inference.h"
#include "model_data.h"

// TensorFlow Lite Micro includes
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

// Pathology detection threshold
static const float PATHOLOGY_THRESHOLD = 0.5f;

// Estructura para pasar a la tarea de inferencia
struct InferTaskArgs {
    tflite::MicroInterpreter* interpreter;
    TfLiteStatus status;
    SemaphoreHandle_t sem;
};

// Tarea de FreeRTOS para ejecutar la inferencia con gran stack
static void tflite_infer_task(void* param) {
    InferTaskArgs* args = (InferTaskArgs*)param;
    args->status = args->interpreter->Invoke();
    xSemaphoreGive(args->sem);
    vTaskDelete(NULL);
}

ECGInference::ECGInference() 
    : _ready(false), _model_buffer(nullptr), _tensor_arena(nullptr), _interpreter(nullptr),
      _model(nullptr), _input(nullptr), _output(nullptr) {}

ECGInference::~ECGInference() {
    if (_tensor_arena) {
        heap_caps_free(_tensor_arena);
    }
    if (_model_buffer && _model_buffer != titan_model_data) {
        heap_caps_free(_model_buffer);
    }
}

bool ECGInference::begin() {
    Serial.println("[TFLite] Inicializando motor de inferencia...");
    
    // 0. Copiar el modelo a PSRAM solo si queda espacio suficiente para el arena.
    // El modelo FP32 cabe en Flash y la inferencia necesita preservar PSRAM.
    const size_t psram_reserve = 256 * 1024;
    const size_t required_psram = (size_t)titan_model_data_len + TENSOR_ARENA_SIZE + psram_reserve;
    if (psramFound() && ESP.getFreePsram() >= required_psram) {
        _model_buffer = (uint8_t*)heap_caps_aligned_alloc(16, titan_model_data_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (_model_buffer != nullptr) {
            memcpy(_model_buffer, titan_model_data, titan_model_data_len);
            Serial.println("[TFLite] Modelo copiado a PSRAM (16-byte align)");
        }
    } else if (psramFound()) {
        Serial.printf("[TFLite] PSRAM reservada para arena; modelo desde Flash (%u bytes libres)\n",
                      ESP.getFreePsram());
    }
    
    if (_model_buffer == nullptr) {
        _model_buffer = (uint8_t*)titan_model_data;
        Serial.println("[TFLite] ADVERTENCIA: Usando modelo directamente desde Flash");
    }
    
    // 1. Cargar modelo
    _model = tflite::GetModel(_model_buffer);
    if (_model == nullptr) {
        Serial.println("[TFLite] ERROR: No se pudo cargar el modelo");
        return false;
    }
    
    if (_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TFLite] ERROR: Versión del modelo (%d) != schema (%d)\n",
                      _model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    Serial.printf("[TFLite] Modelo cargado (%u bytes)\n", titan_model_data_len);
    
    // 2. Alocar arena en PSRAM (si disponible) o RAM
    if (psramFound()) {
        _tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, 
                                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        Serial.printf("[TFLite] Arena alocada en PSRAM (16-byte align): %d bytes\n", TENSOR_ARENA_SIZE);
    }
    
    if (_tensor_arena == nullptr) {
        _tensor_arena = (uint8_t*)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE,
                                                     MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        Serial.println("[TFLite] ADVERTENCIA: Arena en RAM interna (sin PSRAM)");
    }
    
    if (_tensor_arena == nullptr) {
        Serial.println("[TFLite] ERROR: No hay memoria suficiente para arena");
        return false;
    }
    
    // 3. Registrar operaciones
    static tflite::AllOpsResolver resolver;
    static tflite::MicroErrorReporter micro_error_reporter;
    
    // 4. Crear intérprete (esta versión requiere ErrorReporter)
    static tflite::MicroInterpreter static_interpreter(
        _model, resolver, _tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
    _interpreter = &static_interpreter;
    
    // 5. Alocar tensores
    TfLiteStatus alloc_status = _interpreter->AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        Serial.println("[TFLite] ERROR: AllocateTensors() falló");
        Serial.println("[TFLite] El modelo puede ser demasiado grande para el arena");
        return false;
    }
    
    // 6. Obtener punteros a tensores de entrada/salida
    _input = _interpreter->input(0);
    _output = _interpreter->output(0);
    
    // 7. Verificar dimensiones
    Serial.printf("[TFLite] Input:  dims=%d [", _input->dims->size);
    for (int i = 0; i < _input->dims->size; i++) {
        Serial.printf("%d%s", _input->dims->data[i], 
                      i < _input->dims->size - 1 ? "," : "");
    }
    Serial.println("]");
    
    Serial.printf("[TFLite] Output: dims=%d [", _output->dims->size);
    for (int i = 0; i < _output->dims->size; i++) {
        Serial.printf("%d%s", _output->dims->data[i], 
                      i < _output->dims->size - 1 ? "," : "");
    }
    Serial.println("]");
    
    Serial.printf("[TFLite] Input type:  %d (float32=1, int8=9)\n", _input->type);
    Serial.printf("[TFLite] Output type: %d\n", _output->type);

    if (_input->dims->size != 3 ||
        _input->dims->data[0] != 1 ||
        _input->dims->data[1] != ECG_MODEL_SAMPLES ||
        _input->dims->data[2] != ECG_NUM_LEADS) {
        Serial.println("[TFLite] ERROR: Tensor de entrada incompatible; esperado [1,1250,6]");
        return false;
    }
    
    // Reportar uso de memoria
    size_t used = _interpreter->arena_used_bytes();
    Serial.printf("[TFLite] Arena usado: %d / %d bytes (%.1f%%)\n",
                  used, TENSOR_ARENA_SIZE, 100.0f * used / TENSOR_ARENA_SIZE);
    
    _ready = true;
    Serial.println("[TFLite] ✓ Motor de inferencia listo");
    return true;
}

bool ECGInference::infer(float ecg_data[ECG_NUM_LEADS][ECG_MODEL_SAMPLES],
                          InferenceResult& result) {
    if (!_ready) {
        result.success = false;
        return false;
    }
    
    uint32_t start_ms = millis();
    
    // 1. Copiar datos de entrada al tensor
    if (_input->type == kTfLiteFloat32) {
        float* input_data = _input->data.f;
        for (int lead = 0; lead < ECG_NUM_LEADS; lead++) {
            for (int s = 0; s < ECG_MODEL_SAMPLES; s++) {
                input_data[s * ECG_NUM_LEADS + lead] = ecg_data[lead][s];
            }
        }
    } else if (_input->type == kTfLiteInt8) {
        float input_scale = _input->params.scale;
        int32_t input_zp = _input->params.zero_point;
        int8_t* input_data = _input->data.int8;
        
        for (int lead = 0; lead < ECG_NUM_LEADS; lead++) {
            for (int s = 0; s < ECG_MODEL_SAMPLES; s++) {
                float val = ecg_data[lead][s];
                int32_t quantized = (int32_t)roundf(val / input_scale) + input_zp;
                quantized = max(-128, min(127, quantized));
                input_data[s * ECG_NUM_LEADS + lead] = (int8_t)quantized;
            }
        }
    }
    
    // 2. Ejecutar inferencia en tarea con stack grande (32KB)
    InferTaskArgs task_args;
    task_args.interpreter = _interpreter;
    task_args.status = kTfLiteError;
    task_args.sem = xSemaphoreCreateBinary();
    if (task_args.sem == nullptr) {
        Serial.println("[TFLite] ERROR: No se pudo crear semáforo de inferencia");
        result.success = false;
        return false;
    }
    
    // Core 1 (app_cpu), prioridad 5
    TaskHandle_t task_handle = nullptr;
    BaseType_t task_created = xTaskCreatePinnedToCore(
        tflite_infer_task,
        "tflite_infer",
        32768,
        &task_args,
        5,
        &task_handle,
        1
    );
    if (task_created != pdPASS) {
        Serial.println("[TFLite] ERROR: No se pudo crear tarea de inferencia");
        vSemaphoreDelete(task_args.sem);
        result.success = false;
        return false;
    }
    
    // Esperar a que termine sin dejar el equipo bloqueado indefinidamente.
    if (xSemaphoreTake(task_args.sem, pdMS_TO_TICKS(INFERENCE_TIMEOUT_MS)) != pdTRUE) {
        Serial.printf("[TFLite] ERROR: Invoke() excedió timeout de %d ms\n", INFERENCE_TIMEOUT_MS);
        if (task_handle != nullptr) {
            vTaskDelete(task_handle);
        }
        vSemaphoreDelete(task_args.sem);
        result.success = false;
        _ready = false;
        return false;
    }
    vSemaphoreDelete(task_args.sem);
    
    TfLiteStatus status = task_args.status;
    if (status != kTfLiteOk) {
        Serial.println("[TFLite] ERROR: Invoke() falló");
        result.success = false;
        return false;
    }
    
    // 3. Leer resultados
    int output_size = _output->dims->data[_output->dims->size - 1];
    float output_values[MODEL_OUTPUT_SIZE];
    
    if (_output->type == kTfLiteFloat32) {
        for (int i = 0; i < output_size && i < MODEL_OUTPUT_SIZE; i++) {
            output_values[i] = _output->data.f[i];
        }
    } else if (_output->type == kTfLiteInt8) {
        // Dequantizar int8 → float
        float output_scale = _output->params.scale;
        int32_t output_zp = _output->params.zero_point;
        for (int i = 0; i < output_size && i < MODEL_OUTPUT_SIZE; i++) {
            output_values[i] = (_output->data.int8[i] - output_zp) * output_scale;
        }
    }
    
    // 4. Parsear resultados de ritmo (Primary-9)
    result.rhythm_top_class = 0;
    result.rhythm_top_confidence = 0;
    
    for (int i = 0; i < NUM_RHYTHM_CLASSES && i < output_size; i++) {
        result.rhythm_probs[i] = output_values[i];
        if (output_values[i] > result.rhythm_top_confidence) {
            result.rhythm_top_confidence = output_values[i];
            result.rhythm_top_class = i;
        }
    }
    result.rhythm_label = RHYTHM_LABELS[result.rhythm_top_class];
    
    // 5. Parsear resultados de patología (si el modelo incluye pathology head)
    result.pathology_count = 0;
    if (output_size >= MODEL_OUTPUT_SIZE) {
        for (int i = 0; i < NUM_PATHOLOGY_CLASSES; i++) {
            result.pathology_probs[i] = output_values[NUM_RHYTHM_CLASSES + i];
            result.pathology_detected[i] = (result.pathology_probs[i] >= PATHOLOGY_THRESHOLD);
            if (result.pathology_detected[i]) {
                result.pathology_count++;
            }
        }
    } else {
        // Modelo sin pathology head
        for (int i = 0; i < NUM_PATHOLOGY_CLASSES; i++) {
            result.pathology_probs[i] = 0;
            result.pathology_detected[i] = false;
        }
    }
    
    result.inference_time_ms = millis() - start_ms;
    result.success = true;
    return true;
}

void ECGInference::printResult(const InferenceResult& result) {
    if (!result.success) {
        Serial.println("[Resultado] Error en inferencia");
        return;
    }
    
    Serial.println("\n══════════════════════════════════════");
    Serial.println("  TITAN V4 Edge — Resultado ECG");
    Serial.println("══════════════════════════════════════");
    
    // Ritmo
    Serial.printf("  RITMO: %s (%.1f%%)\n", 
                  result.rhythm_label, 
                  result.rhythm_top_confidence * 100.0f);
    
    Serial.println("  ─────────────────────────────────");
    for (int i = 0; i < NUM_RHYTHM_CLASSES; i++) {
        char bar[21] = {};
        int filled = (int)(result.rhythm_probs[i] * 20);
        for (int j = 0; j < 20; j++) bar[j] = j < filled ? '#' : '.';
        Serial.printf("  %5s: [%s] %.1f%%\n", 
                      RHYTHM_LABELS[i], bar, result.rhythm_probs[i] * 100.0f);
    }
    
    // Patología
    if (result.pathology_count > 0) {
        Serial.println("  ─────────────────────────────────");
        Serial.printf("  PATOLOGÍAS DETECTADAS: %d\n", result.pathology_count);
        for (int i = 0; i < NUM_PATHOLOGY_CLASSES; i++) {
            if (result.pathology_detected[i]) {
                Serial.printf("    ⚠ %s: %.1f%%\n", 
                              PATHOLOGY_LABELS[i], 
                              result.pathology_probs[i] * 100.0f);
            }
        }
    } else {
        Serial.println("  PATOLOGÍAS: Ninguna detectada");
    }
    
    Serial.printf("  Inferencia: %d ms\n", result.inference_time_ms);
    Serial.println("══════════════════════════════════════\n");
}

String ECGInference::resultToJson(const InferenceResult& result) {
    String json = "{";
    json += "\"rhythm\":{";
    json += "\"label\":\"" + String(result.rhythm_label) + "\",";
    json += "\"confidence\":" + String(result.rhythm_top_confidence, 4) + ",";
    json += "\"probs\":[";
    for (int i = 0; i < NUM_RHYTHM_CLASSES; i++) {
        json += String(result.rhythm_probs[i], 4);
        if (i < NUM_RHYTHM_CLASSES - 1) json += ",";
    }
    json += "]},";
    
    json += "\"pathology\":{\"detected\":[";
    for (int i = 0; i < NUM_PATHOLOGY_CLASSES; i++) {
        if (result.pathology_detected[i]) {
            json += "\"" + String(PATHOLOGY_LABELS[i]) + "\",";
        }
    }
    if (result.pathology_count > 0) json.remove(json.length() - 1); // trailing comma
    json += "],\"probs\":[";
    for (int i = 0; i < NUM_PATHOLOGY_CLASSES; i++) {
        json += String(result.pathology_probs[i], 4);
        if (i < NUM_PATHOLOGY_CLASSES - 1) json += ",";
    }
    json += "]},";
    
    json += "\"inference_ms\":" + String(result.inference_time_ms);
    json += "}";
    return json;
}
