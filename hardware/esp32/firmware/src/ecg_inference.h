/**
 * TITAN V4 Edge — ECG Inference Engine
 * ======================================
 * Wrapper para TFLite-Micro en ESP32-S3.
 * Maneja la carga del modelo, allocación de tensores, e inferencia.
 */

#ifndef ECG_INFERENCE_H
#define ECG_INFERENCE_H

#include <Arduino.h>
#include "config.h"

// Forward declarations de TFLite
namespace tflite {
    class MicroInterpreter;
    class Model;
    class MicroOpResolver;
}
struct TfLiteTensor;

/**
 * Resultado de una inferencia Primary-9 + Pathology-10
 */
struct InferenceResult {
    // Ritmo (softmax — suman ~1.0)
    float rhythm_probs[NUM_RHYTHM_CLASSES];
    int   rhythm_top_class;
    float rhythm_top_confidence;
    const char* rhythm_label;
    
    // Patología (sigmoid — independientes)
    float pathology_probs[NUM_PATHOLOGY_CLASSES];
    bool  pathology_detected[NUM_PATHOLOGY_CLASSES];
    int   pathology_count;
    
    // Performance
    uint32_t inference_time_ms;
    bool     success;
};

/**
 * Motor de inferencia ECG basado en TFLite-Micro.
 */
class ECGInference {
public:
    ECGInference();
    ~ECGInference();
    
    /**
     * Inicializa el modelo TFLite.
     * Carga el modelo desde flash, aloca arena en PSRAM.
     * @return true si la inicialización fue exitosa.
     */
    bool begin();
    
    /**
     * Ejecuta inferencia sobre una ventana ECG de 6 derivaciones.
     * 
     * @param ecg_data  Buffer [6][1250] float32, normalizado Z-score.
     *                  Orden: Lead I, II, III, aVR, aVL, aVF
     *                  Ya downsampled a 125Hz (1250 muestras = 10s)
     * @param result    Estructura donde se guardan los resultados.
     * @return true si la inferencia fue exitosa.
     */
    bool infer(float ecg_data[ECG_NUM_LEADS][ECG_MODEL_SAMPLES], 
               InferenceResult& result);
    
    /**
     * Imprime el resultado por Serial.
     */
    void printResult(const InferenceResult& result);
    
    /**
     * Devuelve el resultado como JSON string.
     */
    String resultToJson(const InferenceResult& result);
    
    /**
     * @return true si el modelo está cargado y listo.
     */
    bool isReady() const { return _ready; }

private:
    bool _ready;
    uint8_t* _model_buffer;
    uint8_t* _tensor_arena;
    tflite::MicroInterpreter* _interpreter;
    const tflite::Model* _model;
    TfLiteTensor* _input;
    TfLiteTensor* _output;
};

#endif // ECG_INFERENCE_H
