# 🧠 NeuralPricer

> Sistema ML de radar de precios: limpieza estadística, agrupamiento NLP y predicción con redes neuronales.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📌 Descripción del Proyecto

**NeuralPricer** es un sistema de inteligencia de precios desarrollado para **Grupo Almerco**. Analiza miles de productos tecnológicos de la competencia (datos públicos), detecta anomalías en los precios y agrupa productos idénticos con nombres distintos mediante procesamiento de lenguaje natural (NLP).

### Problema que resuelve

El área de ventas de Grupo Almerco necesita ajustar sus precios basándose en la **media real del mercado**, no en suposiciones. Sin un sistema automatizado, los analistas revisan precios manualmente, perdiendo tiempo y cometiendo errores. NeuralPricer automatiza este proceso con matemáticas formales y Machine Learning.

### Impacto esperado

- Maximizar el margen de ganancia sin perder competitividad
- Detectar automáticamente precios basura en el mercado (`S/1.00` o `S/99,999`)
- Emparejar productos con nombres distintos que son físicamente el mismo artículo
- Predecir el precio óptimo de venta dado un producto nuevo

---

## 🏗️ Arquitectura del Sistema

```
CSV Kaggle (origen de datos)
        │
        ▼
┌─────────────────────────────┐
│  PR #1 — Limpieza estadística│  Pandas / NumPy
│  Elimina outliers con μ ± 3σ │  por categoría
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  PR #2 — Visualización Gauss │  Matplotlib / Seaborn
│  Boxplots + histogramas      │  línea roja en μ
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  PR #3 — Texto como álgebra  │  NLTK / SciPy
│  Matriz TF-IDF + coseno      │  stopwords limpias
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  PR #4 — Clustering          │  Scikit-learn
│  K-Means / DBSCAN            │  exporta ID de clúster
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  PR #5 — Red Neuronal        │  TensorFlow / Keras
│  Predicción de precio óptimo │  Marca + Categoría + Clúster
└─────────────────────────────┘
        │
        ▼
  💰 Precio óptimo sugerido
```

---

## 📂 Estructura del Repositorio

```
neuralpricer/
│
├── data/                        # Datasets (NO se suben a GitHub)
│   ├── raw/                     # ← Coloca aquí el electronics.csv de Kaggle
│   └── processed/               # ← Generado automáticamente por PR #1
│
├── src/                         # Código fuente de producción
│   ├── 01_outlier_cleaning.py   # PR #1 — Limpieza estadística
│   ├── 02_visualization.py      # PR #2 — Visualización Gauss
│   ├── 03_tfidf_matrix.py       # PR #3 — Matriz TF-IDF
│   ├── 04_clustering.py         # PR #4 — Clustering no supervisado
│   └── 05_neural_network.py     # PR #5 — Red neuronal predictora
│
├── outputs/                     # Resultados generados (NO se suben a GitHub)
│   ├── plots/                   # ← Gráficos generados por PR #2
│   ├── models/                  # ← Modelos entrenados por PR #5 (.h5, .pkl)
│   └── data/                    # ← Matrices y CSVs generados por PR #3 y PR #4
│
├── .env.example                 # Variables de entorno (plantilla)
├── .gitignore                   # Excluye data/, __pycache__, .env, models
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```

> **Nota:** Las carpetas `data/` y `outputs/` no se suben a GitHub pero su estructura
> está preservada con archivos `.gitkeep`. Al clonar el repositorio las carpetas
> ya existen — solo necesitas agregar el dataset CSV.

---

## 🗺️ Roadmap — Entregas Diarias (Pull Requests)

> **Regla de trabajo:** No se fusionará ningún código si no se demuestra el cálculo matemático detrás de las funciones.

| PR                    | Módulo                   | Foco                       | Estado        |
| --------------------- | ------------------------ | -------------------------- | ------------- |
| [PR #1](../../pull/1) | `01_outlier_cleaning.py` | Pandas, NumPy, Estadística | ✅ Completado |
| [PR #2](../../pull/2) | `02_visualization.py`    | Matplotlib, Seaborn        | ✅ Completado |
| [PR #3](../../pull/3) | `03_tfidf_matrix.py`     | NLTK, SciPy, Vectores      | ✅ Completado |
| [PR #4](../../pull/4) | `04_clustering.py`       | Scikit-learn               | 🔲 Pendiente  |
| [PR #5](../../pull/5) | `05_neural_network.py`   | TensorFlow, Keras          | 🔲 Pendiente  |

---

## ⚙️ Instalación y Configuración

### Prerequisitos

- Python 3.10 o superior
- pip
- Git

### 1. Clonar el repositorio

```bash
git clone https://github.com/Josias45-crypto/neuralpricer.git
cd neuralpricer
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# En Windows
venv\Scripts\activate

# En macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus rutas locales si es necesario
```

### 5. Descargar el dataset de Kaggle

Descarga el dataset oficial usado en este proyecto:

| Dataset                                | URL                                                                                    | Productos |
| -------------------------------------- | -------------------------------------------------------------------------------------- | --------- |
| Datafiniti Electronic Products Pricing | [Ver en Kaggle](https://www.kaggle.com/datasets/datafiniti/electronic-products-prices) | ~5,400    |

El archivo CSV debe contener estas columnas:

| Columna en CSV | Columna estándar | Descripción            |
| -------------- | ---------------- | ---------------------- |
| `name`         | `product_name`   | Nombre del producto    |
| `price`        | `price`          | Precio numérico        |
| `brand`        | `brand`          | Marca del fabricante   |
| `categories`   | `category`       | Categoría del producto |

**Renombrar el archivo a `electronics.csv` y colocarlo en `data/raw/electronics.csv`**

---

## 🚀 Ejecución

Ejecutar los scripts en orden — cada uno depende del anterior:

```bash
# PR #1 — Limpieza de outliers → genera data/processed/clean.csv
python src/01_outlier_cleaning.py

# PR #2 — Visualización → genera outputs/plots/*.png
python src/02_visualization.py

# PR #3 — Matriz TF-IDF → genera outputs/data/tfidf_matrix.npz
python src/03_tfidf_matrix.py

# PR #4 — Clustering (próximamente)
python src/04_clustering.py

# PR #5 — Red neuronal (próximamente)
python src/05_neural_network.py
```

---

## 📐 Fundamentos Matemáticos

Cada PR documenta en el código la fórmula matemática que respalda su lógica.

### PR #1 — Desviación estándar por categoría

$$\sigma = \sqrt{\frac{\sum_{i=1}^{N}(x_i - \mu)^2}{N}}$$

Se elimina toda fila cuyo precio se encuentre fuera del rango $[\mu - 3\sigma,\ \mu + 3\sigma]$ dentro de su categoría. Ningún `if precio < 10` — solo matemáticas.

### PR #3 — Similitud coseno entre vectores TF-IDF

$$\text{similitud}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

Dos nombres de producto son el mismo artículo físico si su similitud coseno supera el umbral definido en `.env`.

### PR #5 — Regresión con red neuronal secuencial

$$\hat{y} = f(W_n \cdot \sigma(...\sigma(W_1 \cdot x + b_1)...) + b_n)$$

Entrada: `[marca_encoded, categoria_encoded, cluster_id]`
Salida: precio óptimo de venta $\hat{y}$

---

## 🔁 Flujo de Trabajo Git

```
main              ← código aprobado y estable
  ├── feature/pr1-outlier-cleaning   ✅
  ├── feature/pr2-visualization      ✅
  ├── feature/pr3-tfidf-matrix       ✅
  ├── feature/pr4-clustering         🔲
  └── feature/pr5-neural-network     🔲
```

### Convención de commits

```
feat(pr1): descripción del cambio
fix(pr2): corrección en gráfico de distribución
docs(readme): actualizar instrucciones de instalación
chore: agregar .gitkeep para estructura de carpetas
```

---

## 📦 Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
scipy>=1.10.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
python-dotenv>=1.0.0
jupyter>=1.0.0
```

---

## 🤝 Contribución

1. Hacer fork del repositorio
2. Crear una rama: `git checkout -b feature/pr4-clustering`
3. **Documentar la fórmula matemática** en el código antes de hacer commit
4. Hacer commit: `git commit -m "feat(pr4): descripción"`
5. Hacer push: `git push origin feature/pr4-clustering`
6. Abrir un Pull Request hacia `main`

> ⚠️ **Regla:** Ningún PR será aprobado si el código no incluye la demostración matemática de cada función implementada.

---

## 🏢 Contexto Empresarial

Proyecto desarrollado para el área de ventas de **Grupo Almerco**. El objetivo final es que el equipo comercial pueda consultar el precio óptimo de cualquier producto tecnológico ingresando únicamente su marca, categoría y clúster de mercado.

---

<p align="center">
  Desarrollado con foco matemático para <strong>Grupo Almerco</strong>
</p>
