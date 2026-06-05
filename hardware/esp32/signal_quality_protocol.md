# Análisis del Sistema de Acondicionamiento de Señal ECG (ECG Edge AI)

A continuación se presentan las respuestas detalladas y específicas a tus preguntas sobre el sistema ECG (basadas en el manuscrito "Design and Implementation of an Analog Signal Conditioning System with Edge AI Inference"):

### 1. How to improve better The signal Quality (Cómo mejorar la calidad de la señal)
Para mejorar la calidad de la señal, el sistema emplea una estrategia de filtrado en múltiples etapas (Analógica y Digital):
- **Filtro Pasa-Bajos Sallen-Key:** Se utiliza un filtro activo de segundo orden (con un amplificador operacional LM358) con una frecuencia de corte ($f_c$) de **33.86 Hz** (atenuación de $-40$ dB/década). Esto preserva el complejo P-QRS-T mientras atenúa severamente el ruido de alta frecuencia como los artefactos electromiográficos (EMG) y provee $-12$ dB de atenuación para la interferencia de la red eléctrica de 50/60 Hz. *(Nota: Las correcciones del manuscrito sugieren aumentar esta $f_c$ a 72-100 Hz usando capacitores de 33-47 nF para no atenuar morfológicamente el QRS).*
- **Filtro de Alimentación (Pi-topology):** Se implementa un filtro pasivo CRC tipo Pi (resistencia de 10 $\Omega$, capacitor electrolítico de 10 $\mu$F y cerámico de 100 nF) con una $f_c \approx 1575$ Hz. Este filtro aísla la circuitería analógica del ruido de conmutación digital generado por las ráfagas de transmisión WiFi del ESP32-S3, eliminando la fluctuación de la línea base (baseline wander).
- **Filtro Digital:** En el firmware, se aplica un filtro digital de promedio móvil de 4 puntos (4-point moving average) para supresión adicional de ruido en alta frecuencia.

### 2. How is The internal process of The signal (Cómo es el proceso interno de la señal)
El flujo de procesamiento de la señal se divide en cuatro capas funcionales:
1. **Interfaz Paciente-Electrodo:** Los electrodos de Ag/AgCl capturan los biopotenciales sin procesar.
2. **Analog Front-End (AFE):** La señal entra a los módulos AD8232, donde ocurre la amplificación del biopotencial y el rechazo de modo común (RLD).
3. **Acondicionamiento Analógico:** La salida de cada AD8232 pasa por el filtro Sallen-Key para eliminar el ruido fuera de banda.
4. **Procesamiento Digital e IA:** La señal analógica limpia es digitalizada por el ADC de 12 bits del ESP32-S3. Se le aplica el filtro de promedio móvil y se guarda en un *ring buffer* (búfer circular) de 2048 muestras. Cuando se acumula una ventana de 1250 muestras (6.25 segundos), la red neuronal TensorFlow Lite (cuantizada a INT8) ejecuta una inferencia para clasificar el ritmo cardíaco en 14 categorías.

### 3. How The sensors Ad86 works (Cómo funcionan los sensores AD86)
*Nota: El componente real utilizado en el diseño es el **AD8232** (es probable que "AD86" haya sido un error tipográfico).*
El AD8232 es un AFE (Analog Front-End) integrado de un solo canal para monitorización cardíaca. Funciona integrando en un solo chip:
- Un **Amplificador de Instrumentación** para proporcionar la ganancia inicial y extraer la pequeñísima señal de biopotencial de la piel.
- Un circuito **Right-Leg Drive (RLD)**, que inyecta una señal de modo común invertida de vuelta al paciente para cancelar activamente la interferencia externa (como el ruido de 50/60 Hz de la red eléctrica).
- Circuitos de **Detección de Electrodos Desconectados (Leads-off detection)** para saber si un cable se ha soltado.

### 4. How The signal is transmited internally on The circuit (Cómo se transmite la señal internamente en el circuito)
La señal se transmite a través de un ruteo físico (hardware) y de software:
- **Hardware:** El voltaje analógico sale del pin *OUT* del AD8232 y viaja por pistas del PCB hacia el filtro activo LM358. Luego de ser filtrada, la señal entra a los pines GPIO analógicos del microcontrolador ESP32-S3.
- **Software (FreeRTOS Dual-Core):** El sistema aprovecha los dos núcleos del ESP32-S3:
  - **Core 1 (Main Loop):** Se encarga de la recolección de datos críticos en el tiempo. Ejecuta el muestreo del ADC mediante interrupciones de un temporizador de hardware y llena el *lock-free ring buffer*.
  - **Core 0 (Task AI & WiFi):** Lee los datos del búfer de forma asíncrona para ejecutar la red neuronal (TensorFlow Lite Micro). Además, maneja la transmisión de las muestras y del diagnóstico a través de un socket TCP por WiFi.

### 5. How mucho voltage is processed (Cuánto voltaje es procesado)
- El microcontrolador ESP32-S3 tiene un ADC interno de 12 bits configurado con una atenuación de **11 dB**, lo que le permite procesar un rango de voltaje de entrada de **0 a 3.3 V**.
- Sin embargo, los amplificadores LM358 se alimentan con 3.3 V (single-supply) y no son verdaderamente *rail-to-rail* en la salida, lo que significa que el swing máximo de voltaje de la señal analógica procesada estará en un rango de aproximadamente **0 V a 1.8 V** ($V_{CC} - 1.5$ V).

### 6. In which frecuency the signal is sensed by the modules ADS86 (A qué frecuencia es sensada la señal)
La señal preacondicionada proveniente de los módulos AD8232 es digitalizada (muestreada) por el ADC del ESP32-S3 a una frecuencia estricta de **200 Hz** (200 muestras por segundo) por cada canal. Adicionalmente, se aplica un retraso de 50 $\mu$s entre las lecturas secuenciales de los 3 canales en el ADC para minimizar el efecto *crosstalk* (interferencia entre canales).

### 7. Tell me more about The process of sensing on The triangle distribution And how it improve The signal (Háblame sobre el proceso de sensado en la distribución de triángulo y cómo mejora la señal)
Esto hace referencia al **Triángulo de Einthoven**, el principio fundamental de la electrocardiografía.
- **El Proceso de Sensado:** El circuito emplea tres módulos físicos AD8232 conectados a electrodos en los brazos y la pierna (RA, LA, LL) para adquirir simultáneamente las tres derivaciones bipolares físicas:
  - **Lead I:** Diferencia de potencial entre Brazo Derecho (RA$^-$) y Brazo Izquierdo (LA$^+$).
  - **Lead II:** Diferencia de potencial entre Brazo Derecho (RA$^-$) y Pierna Izquierda (LL$^+$).
  - **Lead III:** Diferencia de potencial entre Brazo Izquierdo (LA$^-$) y Pierna Izquierda (LL$^+$).
- **Cómo mejora la señal (Aumentación Digital):** A partir de estas tres señales base, el microcontrolador usa las ecuaciones matemáticas de Einthoven (ej. $aVR = -(I + II)/2$) para **calcular computacionalmente tres derivaciones adicionales (aVR, aVL, aVF)**. 
- **La Mejora Real:** Esta "distribución en triángulo" mejora masivamente el sistema porque permite obtener una visión espacial completa de **6 derivaciones** de la actividad eléctrica del corazón utilizando **solo 3 sensores de hardware**. Esto ahorra espacio, reduce los costos y el consumo de energía del circuito, mientras provee a la Inteligencia Artificial de un conjunto de datos mucho más rico y robusto para detectar anomalías con precisión.

---

## 8. Estado implementado en firmware Edge y diferencias frente al manuscrito inicial

El texto anterior conserva el contexto de diseño y la justificación del circuito. Sin
embargo, varias descripciones de software corresponden a una iteración anterior. La
versión actualmente cargada en el ESP32-S3 se rige por el código fuente verificable en
`firmware/src/main.cpp` y `firmware/src/config.h`.

| Componente | Iteración inicial descrita arriba | Firmware Edge actualmente cargado |
|---|---|---|
| Frecuencia ADC | 200 Hz | 200 Hz por cada una de las 3 entradas físicas |
| Ventana útil | 1250 muestras / 6.25 s | 2000 muestras / 10 s |
| Contexto de filtro | No documentado | 400 muestras / 2 s previos |
| Adquisición ADC | ISR y ring buffer | Muestreo cooperativo fuera de ISR |
| Cambio de canal ADC | No auditado | Primera conversión descartada y espera de 50 us |
| Filtro digital | Promedio móvil de 4 puntos | Butterworth causal de orden 3, 0.5-45 Hz |
| Derivaciones de entrada | Descripción histórica | 3 entradas físicas y 6 derivaciones frontales para el modelo |
| Ritmos | 14 categorías históricas | Primary-9 |
| Patologías | No descritas en esa iteración | 10 salidas auxiliares |
| Modelo desplegado | INT8 histórico | TFLite float32 compatible con TFLite Micro |
| Evidencia descargable | No especificada | ADC crudo de 10 s en CSV, MAT, HEA y JSON |

### 8.1 Separación obligatoria entre adquisición cruda y procesamiento del modelo

El firmware conserva dos rutas:

1. **Ruta de auditoría cruda:** guarda las tres entradas ADC físicas sin filtrado
   digital: `I_ADC`, `II_ADC` y `III_SENSOR_ADC`.
2. **Ruta del modelo:** aplica filtro Butterworth causal `0.5-45 Hz`, recorta el
   contexto previo, reduce de `200 Hz` a `125 Hz`, compone las seis derivaciones
   frontales y normaliza con z-score.

Esta separación es necesaria porque un filtro digital puede reducir ruido, pero no
puede recuperar información perdida por saturación analógica o clipping del ADC.

### 8.2 Sensado individual de las tres vías con tierra común

Las tres salidas analógicas se digitalizan por separado:

| Vía física | GPIO ADC | Rol |
|---|---:|---|
| Lead I | GPIO1 | Entrada primaria para inferencia |
| Lead II | GPIO2 | Entrada primaria para inferencia |
| Lead III sensor | GPIO3 | Redundancia física y auditoría del triángulo |

Las vías comparten la referencia eléctrica del circuito. Cada entrada se evalúa antes
de componer derivaciones adicionales mediante:

- mínimo, máximo y rango ADC;
- media y desviación estándar;
- número y fracción de muestras en clipping;
- energía relativa estimada a `50 Hz` y `60 Hz`;
- estado de electrodos desconectados;
- pérdida de intervalos de muestreo.

### 8.3 Auditoría del triángulo de Einthoven

Para una adquisición frontal consistente debe cumplirse aproximadamente:

```text
III = II - I
```

Como cada frente analógico puede introducir un offset DC distinto, el firmware centra
las tres señales antes de compararlas:

```text
I_c   = I   - mean(I)
II_c  = II  - mean(II)
III_c = III - mean(III)
III_expected = II_c - I_c
```

Después calcula correlación, escala ajustada y residuo relativo:

```text
corr = dot(III_expected, III_c) /
       sqrt(dot(III_expected, III_expected) * dot(III_c, III_c))

scale = dot(III_expected, III_c) /
        dot(III_expected, III_expected)

residual_ratio = norm(III_c - scale * III_expected) / norm(III_c)
```

El criterio actual exige correlación positiva `>= 0.70`, señal física III utilizable y
escala positiva. El modelo usa `III = II - I` para mantener una composición frontal
algebraicamente consistente; la tercera vía física no se descarta: permanece en los
archivos crudos y funciona como control de calidad redundante.

### 8.4 Qué mejora realmente la calidad

La composición triangular no elimina ruido por sí sola. Su contribución es:

- detectar inversión de polaridad, ganancia incompatible o mala conexión;
- identificar qué canal físico falla;
- impedir que una derivación defectuosa quede oculta después de la composición;
- obtener `aVR`, `aVL` y `aVF` con ecuaciones reproducibles.

La reducción de ruido procede de capas distintas:

1. contacto correcto y posición estable de electrodos;
2. alimentación aislada durante pruebas corporales;
3. acondicionamiento analógico;
4. adquisición ADC sin crosstalk por multiplexación;
5. filtro digital alineado con entrenamiento;
6. rechazo de ventanas con señal inválida.

### 8.5 Estado del cascade

El **gate de calidad de señal** ya está activo y bloquea inferencias si Lead I o Lead II
presentan clipping excesivo, baseline fuera de rango o electrodos desconectados.

El **cascade de confianza diagnóstica** es una etapa diferente: no limpia el ECG.
Debe activarse únicamente después de validar ventanas corporales estables de las tres
entradas físicas. Su función será enviar a revisión las inferencias con incertidumbre
alta, no corregir fallos del circuito.

### 8.6 Pendiente técnico medible

El entrenamiento usa remuestreo polifásico a `125 Hz`. El Edge aplica actualmente
bandpass equivalente y reducción temporal por interpolación lineal. Antes de congelar
la versión final de producción se debe medir la diferencia numérica frente a
`scipy.signal.resample_poly` y, si altera logits de forma relevante, portar un
remuestreador FIR polifásico al firmware.
