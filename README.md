# 🧠 NeuralPricer
> Sistema ML de radar de precios: limpieza estadística, agrupamiento NLP y predicción con redes neuronales.
 
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
 
---
 
## 📌 Descripción del Proyecto
 
**NeuralPricer** es un sistema de inteligencia de precios desarrollado para **Grupo Almerco**. Analiza miles de productos tecnológicos de la competencia (datos públicos), detecta anomalías en los precios y agrupa productos idénticos con nombres distintos mediante procesamiento de lenguaje natural (NLP).
 
### Problema que resuelve
El área de ventas de Grupo Almerco necesita ajustar sus precios basándose en la **media real del mercado**, no en suposiciones. Sin un sistema automatizado, los analistas revisan precios manualmente, perdiendo tiempo y cometiendo errores. NeuralPricer automatiza este proceso con matemáticas formales y Machine Learning.
 
### Impacto esperado
- Maximizar el margen de ganancia sin perder competitividad
- Detectar automáticamente precios basura en el mercado (`S/1.00` o `S/99,999`)
- Emparejar productos con nombres distintos que son físicamente el mismo artículo
- Predecir el precio óptimo de venta dado un producto nuevo ingresando solo Marca + Categoría + Clúster
 
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
│  K-Means K=10                │  exporta ID de clúster
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  PR #5 — Red Neuronal        │  TensorFlow / Keras
│  Predicción de precio óptimo │  5 features → precio
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  WEB — Flask Dashboard       │  Flask / Matplotlib
│  Interface corporativa       │  Gráficos en tiempo real
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
├── src/                         # Scripts del pipeline ML
│   ├── 01_outlier_cleaning.py   # PR #1 — Limpieza estadística μ ± 3σ
│   ├── 02_visualization.py      # PR #2 — Histogramas y boxplots
│   ├── 03_tfidf_matrix.py       # PR #3 — Matriz TF-IDF + distancia coseno
│   ├── 04_clustering.py         # PR #4 — Clustering K-Means K=10
│   └── 05_neural_network.py     # PR #5 — Red neuronal predicción de precio
│
├── web/                         # Aplicación web Flask
│   ├── app.py                   # Backend Flask con 6 endpoints
│   ├── templates/
│   │   └── index.html           # Dashboard corporativo
│   └── static/
│       ├── css/style.css        # Estilos corporativos
│       └── js/main.js           # Lógica frontend
│
├── outputs/                     # Resultados generados (NO se suben a GitHub)
│   ├── plots/                   # Gráficos generados por PR #2
│   ├── models/                  # Modelos entrenados (.keras, .h5)
│   └── data/                    # Matrices y CSVs del pipeline
│
├── .env.example                 # Variables de entorno (plantilla)
├── .gitignore                   # Excluye data/, outputs/, .env, venv/
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```
 
> **Nota:** Las carpetas `data/` y `outputs/` no se suben a GitHub pero su estructura
> está preservada con archivos `.gitkeep`. Al clonar, las carpetas ya existen —
> solo necesitas agregar el dataset CSV y ejecutar los scripts.
 
---
 
## 🗺️ Roadmap — Pull Requests
 
> **Regla de trabajo:** No se fusionará ningún código sin demostrar el cálculo matemático detrás de las funciones.
 
| PR | Módulo | Foco | Estado |
|----|--------|------|--------|
| [PR #1](../../pull/1) | `01_outlier_cleaning.py` | Pandas, NumPy, Estadística | ✅ Completado |
| [PR #2](../../pull/2) | `02_visualization.py` | Matplotlib, Seaborn | ✅ Completado |
| [PR #3](../../pull/3) | `03_tfidf_matrix.py` | NLTK, SciPy, Vectores | ✅ Completado |
| [PR #4](../../pull/4) | `04_clustering.py` | Scikit-learn K-Means | ✅ Completado |
| [PR #5](../../pull/5) | `05_neural_network.py` | TensorFlow, Keras | ✅ Completado |
| PR #5b | `05_neural_network.py` | Mejora 5 features R²=0.69 | ✅ Completado |
| Web | `web/app.py` | Flask + gráficos dinámicos | ✅ Completado |
 
---
 
## ⚙️ Instalación Paso a Paso
 
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
# El archivo .env ya tiene valores por defecto — no necesitas cambiarlo
```
 
### 5. Descargar el dataset de Kaggle
 
Descarga el dataset oficial usado en este proyecto:
 
| Dataset | URL | Productos |
|---------|-----|-----------|
| Datafiniti Electronic Products Pricing | [Ver en Kaggle](https://www.kaggle.com/datasets/datafiniti/electronic-products-prices) | ~5,400 |
 
El archivo CSV debe contener estas columnas:
 
| Columna en CSV | Descripción |
|----------------|-------------|
| `name` | Nombre del producto |
| `price` | Precio numérico |
| `brand` | Marca del fabricante |
| `categories` | Categoría del producto |
| `manufacturer` | Fabricante |
| `primaryCategories` | Categoría principal |
 
**Renombrar el archivo a `electronics.csv` y colocarlo en `data/raw/electronics.csv`**
 
---
 
## 🚀 Ejecución del Pipeline ML
 
Ejecutar los scripts **en orden** — cada uno depende del anterior:
 
```bash
# PR #1 — Limpieza de outliers → genera data/processed/clean.csv
python src/01_outlier_cleaning.py
 
# PR #2 — Visualización → genera outputs/plots/*.png
python src/02_visualization.py
 
# PR #3 — Matriz TF-IDF → genera outputs/data/tfidf_matrix.npz
python src/03_tfidf_matrix.py
 
# PR #4 — Clustering → genera outputs/data/clustered.csv
python src/04_clustering.py
 
# PR #5 — Red neuronal → genera outputs/models/neuralpricer.keras
python src/05_neural_network.py
```
 
### Resultados esperados del pipeline
 
```
data/processed/clean.csv          → 5,412 productos limpios
outputs/data/tfidf_matrix.npz     → Matriz 5,412 × 5,000
outputs/data/clustered.csv        → Productos con cluster_id
outputs/models/neuralpricer.keras → Modelo entrenado
MAE = $241  |  R² = 0.69
```
 
---
 
## 🌐 Ejecutar la Aplicación Web
 
Una vez ejecutado el pipeline completo:
 
```bash
python web/app.py
```
 
Abrir en el navegador: **http://localhost:5000**
 
### Secciones de la web
 
| Sección | Descripción |
|---------|-------------|
| Dashboard | Métricas del sistema en tiempo real |
| Predictor | Ingresa Marca + Categoría + Clúster → precio óptimo |
| Gráficos | Histogramas y boxplots generados dinámicamente |
| Clústeres | Distribución K-Means con precio y marca líder |
| Similares | Pares de productos detectados por TF-IDF |
 
### Endpoints de la API
 
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Dashboard principal |
| `/api/metrics` | GET | Métricas del sistema |
| `/api/opciones` | GET | Marcas, categorías y clústeres disponibles |
| `/api/predict` | POST | Predicción de precio |
| `/api/clusters` | GET | Distribución de clústeres |
| `/api/similares` | GET | Pares de productos similares |
| `/api/grafico/categoria/<nombre>` | GET | Gráfico dinámico por categoría |
| `/api/grafico/cluster/<id>` | GET | Gráfico dinámico por clúster |
 
---
 
## 📐 Fundamentos Matemáticos
 
### PR #1 — Desviación estándar por categoría
 
$$\sigma = \sqrt{\frac{\sum_{i=1}^{N}(x_i - \mu)^2}{N}}$$
 
Se elimina toda fila cuyo precio esté fuera de $[\mu - 3\sigma,\ \mu + 3\sigma]$ por categoría.
 
### PR #3 — Similitud coseno entre vectores TF-IDF
 
$$\text{similitud}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$
 
Detecta productos con nombres distintos que son el mismo artículo físico.
 
### PR #4 — K-Means (minimización de inercia)
 
$$J = \sum_{k} \sum_{i \in C_k} \|x_i - \mu_k\|^2$$
 
### PR #5 — Red neuronal secuencial
 
$$\hat{y} = f(W_n \cdot \sigma(...\sigma(W_1 \cdot x + b_1)...) + b_n)$$
 
Entrada: `[marca, categoría, cluster_id, fabricante, categoría_primaria]`
Salida: precio óptimo $\hat{y}$
 
---
 
## 🔁 Flujo de Trabajo Git
 
```
main ← código aprobado y estable
  ├── feature/pr1-outlier-cleaning   ✅
  ├── feature/pr2-visualization      ✅
  ├── feature/pr3-tfidf-matrix       ✅
  ├── feature/pr4-clustering         ✅
  ├── feature/pr5-neural-network     ✅
  └── feature/web-flask              ✅
```
 
### Convención de commits
 
```
feat(pr1): descripción del cambio
fix(pr2): corrección en gráfico
docs(readme): actualizar instrucciones
chore: sincronizar archivos
feat(web): nueva funcionalidad en la web
```
 
---
 
## 📦 Dependencias Principales
 
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
scipy>=1.10.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
flask>=3.0.0
flask-cors>=4.0.0
python-dotenv>=1.0.0
```
 
---
 
## 🏢 Contexto Empresarial
 
Proyecto desarrollado para el área de ventas de **Grupo Almerco**. El equipo comercial puede consultar el precio óptimo de cualquier producto tecnológico ingresando únicamente su marca, categoría y clúster de mercado a través de la interfaz web.
 
---
 
<p align="center">
  Desarrollado con foco matemático para <strong>Grupo Almerco</strong>
</p>
 