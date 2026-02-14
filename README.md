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
