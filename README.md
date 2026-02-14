# Demo ISLR-LSE (MediaPipe Holistic + TCN)

Demo reproducible de reconocimiento de gestos aislados (ISLR) para Lengua de Signos Española (LSE) con enfoque **tiempo real + estabilidad en directo**.

## 1) Estructura del repo

```text
.
├── configs/
│   └── demo.yaml
├── data/
│   ├── lse_sign/
│   │   └── videos/               # vídeos por clase (usuario)
│   ├── swl_lse/                  # keypoints .pkl (usuario)
│   └── background/               # clips idle (usuario o generados)
├── scripts/
│   ├── extract_mediapipe.py
│   ├── preprocess.py
│   ├── generate_background.py
│   ├── train.py
│   ├── eval.py
│   └── demo_webcam.py
├── src/
│   ├── data/
│   ├── preprocess/
│   ├── models/
│   ├── training/
│   ├── eval/
│   └── utils/
├── requirements.txt
└── README.md
```

## 2) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Preparación de datos

> TL;DR rápido: para entrenar **necesitas clips de signos + clips de no-signo (background)**.  
> Si no metes background, la demo tenderá a "inventar" clases cuando estás quieto.

## 3.0 ¿Qué tienes que meter exactamente para que entrene?

Piensa en 4 "bloques" de entrada:

1. **Vídeos de signos (LSE-Sign o tuyos)**
   - Qué meter: carpetas por clase (`hola/`, `gracias/`, etc.) con clips cortos (1 signo por clip).
   - Por qué: esto define el vocabulario que el modelo aprenderá.
   - Cómo complementa: aporta ejemplos reales del dominio final de la demo.

2. **Mapeo de etiquetas (`label_map.json`)**
   - Qué meter: un JSON `{"background": 0, "hola": 1, ...}`.
   - Por qué: garantiza consistencia entre extracción, entrenamiento y demo.
   - Cómo complementa: evita desalinear índices de clase entre scripts.

3. **Clips BACKGROUND (idle / sin signo)**
   - Qué meter: 2-5 minutos de reposo real (misma cámara/entorno de la demo), luego convertir a keypoints.
   - Por qué: enseña explícitamente al modelo cuándo **no** hay signo.
   - Cómo complementa: junto al activity gate, reduce falsos positivos en directo.

4. **(Opcional) SWL-LSE keypoints para pretrain**
   - Qué meter: `.pkl` de SWL-LSE convertidos al formato interno.
   - Por qué: mejora robustez de movimiento/variabilidad antes de ajustar al vocabulario social final.
   - Cómo complementa: pretrain (robustez) + fine-tune en LSE-Sign (dominio objetivo) suele estabilizar resultados.

### Mínimo viable recomendado (challenge)

- 20–40 signos.
- 40–100 clips por signo (idealmente varios signantes).
- 1 clip = 1 gesto aislado (sin frases largas).
- Duración orientativa por clip: 1–3 s.
- Background total: 2–5 min reales.
- Split por signante o sesión (si puedes) para evitar leakage.

### ¿Cómo se complementan los componentes en la práctica?

Pipeline lógico:

`videos signos` + `videos background` -> `MediaPipe keypoints` -> `preprocess normalizado` -> `TCN` -> `demo con gate + smoothing`

- **MediaPipe** unifica la entrada visual en una representación compacta (landmarks).
- **Preprocess** hace invariancia de tamaño/posición (centro pecho + escala hombros) y añade dinámica (`dx,dy`).
- **TCN** aprende patrones temporales del gesto sobre esa señal ya estabilizada.
- **Background + activity gate + smoothing** hacen que la app sea usable en vivo (menos ruido, menos parpadeo de etiquetas).

### 3.1 LSE-Sign (vídeos)

Esperado:

```text
data/lse_sign/videos/
├── hola/
│   ├── clip1.mp4
│   └── clip2.mp4
├── gracias/
└── ...
```

Si tienes un `label_map.json` (`{"hola": 1, "gracias": 2, ...}`):

```bash
python scripts/extract_mediapipe.py \
  --videos-dir data/lse_sign/videos \
  --output-dir artifacts/raw/lse_sign \
  --label-map configs/label_map.json \
  --source lse_sign
```

Ejemplo completo recomendado de mapping (incluyendo background=0):

```json
{
  "background": 0,
  "hola": 1,
  "adios": 2,
  "gracias": 3,
  "por_favor": 4
}
```

### 3.2 SWL-LSE (.pkl)

El adaptador SWL-LSE no fuerza scraping. Coloca tus `.pkl` en `data/swl_lse/` y conviértelos al formato interno (`.npz`) con campos:
- `keypoints`: `[T, V, C]`
- `label`: `int`
- `source`: `str`
- `fps`: `int`
- `meta`: `dict`

Puedes reutilizar `src/data/formats.py` para guardar en este formato.

### 3.3 Clase BACKGROUND

Opción recomendada (grabación de reposo):

```bash
python scripts/generate_background.py --output-dir data/background --seconds 180
python scripts/extract_mediapipe.py \
  --videos-dir data/background \
  --output-dir artifacts/raw/background \
  --source background
```

## 4) Preprocesado y dataset final

Incluye:
- selección de landmarks (pose superior + manos),
- normalización por pecho/hombros,
- interpolación temporal de huecos cortos,
- `features = [x, y, dx, dy]`,
- resampling a longitud fija `T`.

```bash
python scripts/preprocess.py \
  --input-dir artifacts/raw/lse_sign \
  --background-dir artifacts/raw/background \
  --background-label 0 \
  --output-dir artifacts/processed \
  --target-len 64 \
  --max-gap 5
```

Salida clave:
- `artifacts/processed/manifests/train.npz`
- `artifacts/processed/manifests/val.npz`
- `artifacts/processed/manifests/test.npz`

## 5) Entrenamiento

```bash
python scripts/train.py --config configs/demo.yaml
```

Fine-tuning opcional desde pretrain:

```bash
python scripts/train.py \
  --config configs/demo.yaml \
  --fine-tune-from artifacts/checkpoints/pretrain_best.pt
```

## 6) Evaluación

```bash
python scripts/eval.py \
  --config configs/demo.yaml \
  --checkpoint artifacts/checkpoints/best.pt
```

Métricas: top-1, top-3, F1 macro + matriz de confusión (`artifacts/eval/confusion_matrix.png`).

## 7) Demo en webcam (tiempo real)

```bash
python scripts/demo_webcam.py \
  --config configs/demo.yaml \
  --checkpoint artifacts/checkpoints/best.pt
```

Características de robustez para demo:
- **activity gate** por energía de movimiento de muñecas (evita falsos positivos cuando estás quieto),
- **smoothing temporal** de logits,
- **umbral de confianza**,
- modo controlado:
  - `c`: activar/desactivar control mode
  - `s`: start/stop captura en control mode
  - `q`: salir

## 8) Recomendaciones prácticas para el challenge

- Luz frontal uniforme y fondo simple.
- Cámara fija a altura de pecho/cara.
- Mantener manos visibles y centradas.
- Usar 20–40 signos con vocabulario diario y suficientes ejemplos por clase.
- Grabar 2–5 minutos de idle real para background.
- Revisar clases confundidas en la matriz de confusión antes de demo final.

## 9) Notas de adaptadores y rutas

- Si el dataset externo no trae split oficial, este repo genera `train/val/test` reproducibles por semilla.
- Si no hay SWL-LSE disponible, pipeline funciona solo con LSE-Sign + background.
- Si MediaPipe falla en frames puntuales, el pipeline no crashea: se registran NaNs/errores y se interpola cuando aplica.

## 10) Checklist de "si no entrena bien"

- ¿`background` está en `label_map` y también en `labels` del config?
- ¿`num_classes` coincide con el número real de clases (incluyendo background)?
- ¿Los clips contienen **un solo gesto** y no transiciones largas?
- ¿Hay clases con muy pocos ejemplos (<20)? (aumentar datos o bajar vocabulario inicial).
- ¿La luz/encuadre de train se parece a la demo real?
- ¿`input_dim` en `configs/demo.yaml` coincide con tus features reales (V * 4)?
