"""
================================================================================
NeuralPricer — PR #5: Predicción de Precio con Red Neuronal
================================================================================
Proyecto  : NeuralPricer — Radar de Precios de Mercado
Cliente   : Grupo Almerco
Autor     : Josias
Fecha     : Marzo 2026
PR        : #5
--------------------------------------------------------------------------------
OBJETIVO:
    Construir una Red Neuronal Secuencial que prediga el precio óptimo
    de un producto dado únicamente su Marca, Categoría y Clúster.

    Entrada : [marca_encoded, categoria_encoded, cluster_id]
    Salida  : precio_predicho (regresión)

    Ejemplo:
        Entrada: [LG=5, Monitors=3, cluster_id=1]
        Salida : $339.41  ← precio óptimo sugerido al área de ventas

¿QUÉ ES UNA RED NEURONAL?
    Es un modelo matemático inspirado en el cerebro humano.
    Está formada por capas de "neuronas" (nodos) conectadas entre sí.

    Cada neurona hace dos cosas:
        1. Suma ponderada de sus entradas:  z = Σ(wᵢ × xᵢ) + b
        2. Aplica una función de activación: a = f(z)

    Donde:
        xᵢ = valores de entrada
        wᵢ = pesos (weights) — lo que la red APRENDE
        b  = sesgo (bias) — ajuste fino
        f  = función de activación (ReLU, sigmoid, etc.)

¿QUÉ ES UNA RED SECUENCIAL?
    Es el tipo más simple de red neuronal — las capas van una detrás de otra:

        Entrada [3 features]
             ↓
        Capa Densa 1 (64 neuronas) → ReLU
             ↓
        Dropout (20%) → evita sobreajuste
             ↓
        Capa Densa 2 (32 neuronas) → ReLU
             ↓
        Dropout (20%)
             ↓
        Capa Densa 3 (16 neuronas) → ReLU
             ↓
        Salida (1 neurona) → precio predicho

¿QUÉ ES ReLU?
    ReLU = Rectified Linear Unit
    f(z) = max(0, z)

    Si z > 0 → devuelve z (pasa la señal)
    Si z ≤ 0 → devuelve 0 (bloquea la señal)

    Es la función de activación más usada en capas ocultas porque:
    - Es simple y rápida de calcular
    - No sufre del problema del gradiente que desaparece
    - Introduce no-linealidad (sin esto la red sería solo álgebra lineal)

¿QUÉ ES DROPOUT?
    Dropout es una técnica de regularización que durante el entrenamiento
    "apaga" aleatoriamente un % de neuronas en cada iteración.

    ¿Por qué? Para evitar el SOBREAJUSTE (overfitting):
        Sobreajuste = la red memoriza los datos de entrenamiento
                      pero falla con datos nuevos

        Con Dropout = la red no puede depender de neuronas específicas
                      → aprende patrones más generales y robustos

¿QUÉ ES LA FUNCIÓN DE PÉRDIDA (LOSS)?
    Mide qué tan equivocada está la red en sus predicciones.
    Usamos MSE (Mean Squared Error):

        MSE = (1/N) × Σ(yᵢ - ŷᵢ)²

    Donde:
        yᵢ  = precio real
        ŷᵢ  = precio predicho por la red
        N   = número de productos

    El entrenamiento busca MINIMIZAR el MSE ajustando los pesos wᵢ.

¿QUÉ ES BACKPROPAGATION?
    Es el algoritmo que ajusta los pesos de la red para minimizar el MSE.

    PASO FORWARD:
        Los datos entran → pasan por todas las capas → producen una predicción

    CÁLCULO DEL ERROR:
        Comparamos la predicción con el precio real → calculamos MSE

    PASO BACKWARD (backpropagation):
        Usando la regla de la cadena del cálculo diferencial, calculamos
        el gradiente del error respecto a cada peso:
        ∂MSE/∂wᵢ

    ACTUALIZACIÓN (Gradient Descent):
        wᵢ = wᵢ - α × (∂MSE/∂wᵢ)

        Donde α = learning rate (qué tan grandes son los pasos)

¿QUÉ ES ADAM?
    Adam (Adaptive Moment Estimation) es un optimizador avanzado.
    Mejora el Gradient Descent básico ajustando automáticamente
    el learning rate para cada peso individualmente.
    Es el optimizador más usado en la práctica.

ENTRADAS:
    - outputs/data/clustered.csv  (generado por PR #4)

SALIDAS:
    - outputs/models/neuralpricer.h5     (modelo entrenado)
    - outputs/models/neuralpricer.keras  (modelo en formato moderno)
    - outputs/plots/training_history.png (curvas de entrenamiento)
    - outputs/plots/prediccion_vs_real.png (gráfico de predicciones)

DEPENDENCIAS:
    pip install tensorflow keras pandas numpy matplotlib scikit-learn python-dotenv
================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------------------------
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from dotenv import load_dotenv

# Scikit-learn — para preprocesamiento y evaluación
from sklearn.model_selection import train_test_split  # Divide datos en train/test
from sklearn.preprocessing import LabelEncoder        # Convierte strings a números
from sklearn.preprocessing import StandardScaler      # Normaliza los features
from sklearn.metrics import mean_absolute_error, r2_score  # Métricas de evaluación

# TensorFlow y Keras — para construir y entrenar la red neuronal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ------------------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------------------
load_dotenv()

CLUSTERED_PATH = os.getenv("DATA_CLUSTERED_PATH", "outputs/data/clustered.csv")
MODELS_PATH    = os.getenv("MODELS_PATH",         "outputs/models/")
PLOTS_PATH     = os.getenv("PLOTS_PATH",          "outputs/plots/")

# Hiperparámetros del modelo — configurables desde .env
RANDOM_STATE   = int(os.getenv("RANDOM_STATE",   "42"))
TEST_SIZE      = float(os.getenv("TEST_SIZE",     "0.2"))   # 20% para test
EPOCHS         = int(os.getenv("EPOCHS",          "100"))   # Máximo de épocas
BATCH_SIZE     = int(os.getenv("BATCH_SIZE",      "32"))    # Productos por batch
LEARNING_RATE  = float(os.getenv("LEARNING_RATE", "0.001")) # Tasa de aprendizaje

# Fijar semilla para reproducibilidad
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ------------------------------------------------------------------------------
# FUNCIÓN 1: cargar_datos
# ------------------------------------------------------------------------------
def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset con cluster_id generado por el PR #4.

    Parámetros:
        ruta (str): Ruta al CSV con cluster_id.

    Retorna:
        pd.DataFrame: Dataset con columnas cluster_id, product_name,
                      price, brand, category.
    """

    if not os.path.exists(ruta):
        print(f"\nERROR: No se encontró: {ruta}")
        print("   Ejecuta primero: python src/04_clustering.py")
        sys.exit(1)

    print(f"\nCargando dataset desde: {ruta}")
    df = pd.read_csv(ruta, low_memory=False)
    print(f"   Productos cargados : {len(df):,}")
    print(f"   Columnas           : {', '.join(df.columns.tolist())}")
    print(f"   Clústeres únicos   : {df['cluster_id'].nunique()}")

    return df


# ------------------------------------------------------------------------------
# FUNCIÓN 2: preparar_features
# ------------------------------------------------------------------------------
def preparar_features(df: pd.DataFrame):
    """
    Convierte las columnas de texto (brand, category) en números
    y prepara los arrays X (features) e y (target) para la red neuronal.

    ¿POR QUÉ CONVERTIR TEXTO A NÚMEROS?
        Las redes neuronales solo trabajan con números.
        "LG" no significa nada matemáticamente.
        LabelEncoder convierte: "LG"=5, "Samsung"=12, "Sony"=15, etc.

    ¿QUÉ ES StandardScaler?
        Normaliza los features para que tengan media=0 y std=1:
        x_normalizado = (x - μ) / σ

        ¿Por qué normalizar?
        Si brand va de 0 a 50 y price va de 1 a 5000,
        la red le da más importancia a price solo por ser más grande.
        Con normalización todos los features tienen el mismo rango.

    FEATURES (X) — entradas de la red:
        - brand_encoded   : marca convertida a número
        - category_encoded: categoría convertida a número
        - cluster_id      : ID del clúster del PR #4

    TARGET (y) — lo que predice la red:
        - price           : precio real del producto

    Parámetros:
        df (pd.DataFrame): Dataset con columnas originales.

    Retorna:
        tuple: (X_train, X_test, y_train, y_test, scaler_X, scaler_y,
                encoder_brand, encoder_category)
    """

    print("\nPreparando features para la red neuronal...")

    df = df.copy()

    # ── Eliminar filas con valores nulos ──────────────────────────────────────
    df = df.dropna(subset=["price", "brand", "category", "cluster_id"])
    print(f"   Productos después de limpiar nulos: {len(df):,}")

    # ── Codificar variables categóricas con LabelEncoder ─────────────────────
    # LabelEncoder asigna un número entero a cada categoría única
    # Ejemplo: {"Dell":0, "HP":1, "LG":2, "Samsung":3, "Sony":4}

    encoder_brand = LabelEncoder()
    df["brand_encoded"] = encoder_brand.fit_transform(
        df["brand"].astype(str)
    )
    print(f"   Marcas únicas     : {len(encoder_brand.classes_):,}")

    encoder_category = LabelEncoder()
    df["category_encoded"] = encoder_category.fit_transform(
        df["category"].astype(str)
    )
    print(f"   Categorías únicas : {len(encoder_category.classes_):,}")

    # ── Definir features (X) y target (y) ────────────────────────────────────
    # X = lo que entra a la red (3 features)
    # y = lo que queremos predecir (1 valor: precio)
    encoder_manufacturer = LabelEncoder()
    df["manufacturer_encoded"] = encoder_manufacturer.fit_transform(
        df["manufacturer"].astype(str)
    )

    encoder_primary = LabelEncoder()
    df["primary_category_encoded"] = encoder_primary.fit_transform(
        df["primary_category"].astype(str)
    )

    print(f"   Fabricantes únicos   : {len(encoder_manufacturer.classes_):,}")
    print(f"   Cat. primarias únicas: {len(encoder_primary.classes_):,}")

    X = df[["brand_encoded", "category_encoded", "cluster_id",
        "manufacturer_encoded", "primary_category_encoded"]].values
    y = df["price"].values.reshape(-1, 1)  # reshape para que sea columna

    print(f"\n   Features (X) shape: {X.shape}  → {X.shape[0]:,} productos × {X.shape[1]} features")
    print(f"   Target  (y) shape: {y.shape}  → {y.shape[0]:,} precios")

    # ── Dividir en conjuntos de entrenamiento y prueba ────────────────────────
    # train = 80% de datos → la red APRENDE con estos
    # test  = 20% de datos → evaluamos qué tan bien GENERALIZA
    # Es CRÍTICO que la red nunca vea los datos de test durante entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE
    )

    print(f"\n   Train: {len(X_train):,} productos ({int((1-TEST_SIZE)*100)}%)")
    print(f"   Test : {len(X_test):,} productos ({int(TEST_SIZE*100)}%)")

    # ── Normalizar features con StandardScaler ────────────────────────────────
    # IMPORTANTE: fit() solo en train, transform() en train Y test
    # Si hiciéramos fit() en test, estaríamos "filtrando" información del futuro
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)  # Aprende μ y σ del train
    X_test  = scaler_X.transform(X_test)        # Aplica la misma normalización

    # Normalizar también el target (precios)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_test  = scaler_y.transform(y_test)

    print(f"\n   Features normalizados (media≈0, std≈1)")

    return (X_train, X_test, y_train, y_test,
            scaler_X, scaler_y, encoder_brand, encoder_category)


# ------------------------------------------------------------------------------
# FUNCIÓN 3: construir_modelo
# ------------------------------------------------------------------------------
def construir_modelo(learning_rate: float) -> keras.Model:
    """
    Construye la arquitectura de la Red Neuronal Secuencial.

    ARQUITECTURA:
        Input(3) → Dense(64, ReLU) → Dropout(0.2) →
        Dense(32, ReLU) → Dropout(0.2) →
        Dense(16, ReLU) →
        Dense(1)  ← precio predicho

    ¿POR QUÉ ESTAS CAPAS?
        - Empezamos con 64 neuronas y vamos reduciendo (64→32→16→1)
        - Esta forma de embudo es común en regresión
        - Las capas más grandes aprenden patrones generales
        - Las capas más pequeñas refinan la predicción final

    ¿POR QUÉ DROPOUT=0.2?
        Apaga el 20% de neuronas aleatoriamente en cada paso
        de entrenamiento. Fuerza a la red a aprender redundancia
        y evita memorizar los datos de entrenamiento.

    Parámetros:
        learning_rate (float): Tasa de aprendizaje del optimizador Adam.

    Retorna:
        keras.Model: Modelo compilado listo para entrenar.
    """

    print(f"\nConstruyendo arquitectura de la red neuronal...")

    # keras.Sequential = capas apiladas una tras otra
    modelo = keras.Sequential([

        # Capa de entrada — recibe 3 features
        # input_shape=(3,) indica que cada muestra tiene 3 valores
        layers.Input(shape=(5,)),

        # Capa oculta 1 — 64 neuronas con activación ReLU
        # Dense = todas las neuronas conectadas a todas las de la capa anterior
        # ReLU: f(z) = max(0, z) — introduce no-linealidad
        layers.Dense(64, activation="relu"),

        # Dropout 20% — apaga 20% de neuronas durante entrenamiento
        # Solo activo durante training, NO durante predicción
        layers.Dropout(0.2),

        # Capa oculta 2 — 32 neuronas con ReLU
        layers.Dense(32, activation="relu"),

        # Dropout 20%
        layers.Dropout(0.2),

        # Capa oculta 3 — 16 neuronas con ReLU
        layers.Dense(16, activation="relu"),

        # Capa de salida — 1 neurona, SIN activación
        # Sin activación = regresión lineal en la última capa
        # Puede predecir cualquier valor numérico (precio)
        layers.Dense(1)

    ], name="NeuralPricer")

    # ── Compilar el modelo ────────────────────────────────────────────────────
    # optimizer = cómo se actualizan los pesos (Adam es el más usado)
    # loss      = función a minimizar (MSE para regresión)
    # metrics   = métricas a monitorear durante entrenamiento (MAE)
    modelo.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = "mse",   # Mean Squared Error: (1/N) × Σ(y - ŷ)²
        metrics   = ["mae"]  # Mean Absolute Error: (1/N) × Σ|y - ŷ|
    )

    # Mostrar resumen de la arquitectura
    modelo.summary()

    return modelo


# ------------------------------------------------------------------------------
# FUNCIÓN 4: entrenar_modelo
# ------------------------------------------------------------------------------
def entrenar_modelo(modelo, X_train, y_train, X_test, y_test):
    """
    Entrena la red neuronal usando backpropagation y gradient descent.

    CALLBACKS USADOS:
        EarlyStopping:
            Detiene el entrenamiento si la pérdida en validación no mejora
            después de N épocas (patience). Evita sobreajuste.

        ReduceLROnPlateau:
            Reduce el learning rate si la pérdida se estanca.
            Permite pasos más finos cuando estamos cerca del mínimo.

    ¿QUÉ ES UNA ÉPOCA (EPOCH)?
        Una época = la red ve TODOS los datos de entrenamiento una vez.
        Con EPOCHS=100 y EarlyStopping, la red entrena máximo 100 épocas
        pero puede parar antes si ya convergió.

    ¿QUÉ ES UN BATCH?
        En lugar de procesar todos los datos a la vez, los divide en
        lotes (batches) de BATCH_SIZE productos.
        Esto hace el entrenamiento más rápido y estable.

    Parámetros:
        modelo          : Red neuronal compilada.
        X_train, y_train: Datos de entrenamiento.
        X_test, y_test  : Datos de validación.

    Retorna:
        History: Historial de métricas por época.
    """

    print(f"\nEntrenando red neuronal...")
    print(f"   Épocas máximas : {EPOCHS}")
    print(f"   Batch size     : {BATCH_SIZE}")
    print(f"   Learning rate  : {LEARNING_RATE}")

    # ── Callbacks — controlan el entrenamiento automáticamente ───────────────

    # EarlyStopping: para si val_loss no mejora en 15 épocas
    # restore_best_weights=True → usa los pesos de la mejor época
    early_stopping = EarlyStopping(
        monitor              = "val_loss",  # Monitorea pérdida en validación
        patience             = 15,          # Espera 15 épocas sin mejora
        restore_best_weights = True,        # Restaura los mejores pesos
        verbose              = 1
    )

    # ReduceLROnPlateau: reduce learning rate si val_loss se estanca
    reduce_lr = ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,      # Reduce el LR a la mitad
        patience = 7,        # Después de 7 épocas sin mejora
        min_lr   = 1e-6,     # Learning rate mínimo
        verbose  = 1
    )

    # ── Entrenar el modelo ────────────────────────────────────────────────────
    # validation_data = datos de test para monitorear durante entrenamiento
    # (solo para monitoreo, NO para actualizar pesos)
    historial = modelo.fit(
        X_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (X_test, y_test),
        callbacks       = [early_stopping, reduce_lr],
        verbose         = 1
    )

    epocas_reales = len(historial.history["loss"])
    print(f"\n   Entrenamiento completado en {epocas_reales} épocas")
    print(f"   Pérdida final (train): {historial.history['loss'][-1]:.4f}")
    print(f"   Pérdida final (val)  : {historial.history['val_loss'][-1]:.4f}")

    return historial


# ------------------------------------------------------------------------------
# FUNCIÓN 5: evaluar_modelo
# ------------------------------------------------------------------------------
def evaluar_modelo(modelo, X_test, y_test, scaler_y) -> dict:
    """
    Evalúa el modelo con datos que NUNCA vio durante entrenamiento.

    MÉTRICAS DE EVALUACIÓN:
        MAE (Mean Absolute Error):
            MAE = (1/N) × Σ|yᵢ - ŷᵢ|
            Promedio del error absoluto en dólares.
            Fácil de interpretar: "me equivoco en promedio $X"

        RMSE (Root Mean Squared Error):
            RMSE = √( (1/N) × Σ(yᵢ - ŷᵢ)² )
            Penaliza más los errores grandes.
            Útil para detectar predicciones muy equivocadas.

        R² (Coeficiente de Determinación):
            R² = 1 - (SS_res / SS_tot)
            Mide qué % de la varianza del precio explica el modelo.
            R²=1.0 = predicción perfecta
            R²=0.0 = el modelo no explica nada
            R²<0.0 = peor que predecir siempre la media

    Parámetros:
        modelo   : Red neuronal entrenada.
        X_test   : Features de test normalizados.
        y_test   : Precios reales normalizados.
        scaler_y : Scaler para desnormalizar precios.

    Retorna:
        dict: Métricas de evaluación con precios en USD reales.
    """

    print(f"\nEvaluando modelo con datos de test...")

    # Predicciones en escala normalizada
    y_pred_norm = modelo.predict(X_test, verbose=0)

    # Desnormalizar — convertir de vuelta a dólares reales
    y_pred_real = scaler_y.inverse_transform(y_pred_norm)
    y_test_real = scaler_y.inverse_transform(y_test)

    # Calcular métricas en dólares reales
    mae  = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
    r2   = r2_score(y_test_real, y_pred_real)

    print(f"\n   {'─'*45}")
    print(f"   MÉTRICAS DE EVALUACIÓN (en USD reales)")
    print(f"   {'─'*45}")
    print(f"   MAE  (error promedio)     : ${mae:>8.2f}")
    print(f"   RMSE (error cuadrático)   : ${rmse:>8.2f}")
    print(f"   R²   (varianza explicada) :  {r2:>8.4f}")
    print(f"   {'─'*45}")

    if r2 > 0.7:
        print(f"   Resultado: BUENO — el modelo explica el {r2*100:.1f}% de la varianza")
    elif r2 > 0.4:
        print(f"   Resultado: ACEPTABLE — explica el {r2*100:.1f}% de la varianza")
    else:
        print(f"   Resultado: MEJORABLE — explica el {r2*100:.1f}% de la varianza")
        print(f"   Sugerencia: agregar más features o más datos")

    return {
        "mae"        : mae,
        "rmse"       : rmse,
        "r2"         : r2,
        "y_pred_real": y_pred_real,
        "y_test_real": y_test_real
    }


# ------------------------------------------------------------------------------
# FUNCIÓN 6: graficar_entrenamiento
# ------------------------------------------------------------------------------
def graficar_entrenamiento(historial, ruta_salida: str) -> None:
    """
    Grafica las curvas de pérdida durante el entrenamiento.

    QUÉ BUSCAR EN ESTE GRÁFICO:
        - Curva train (azul) y val (naranja) deben bajar juntas
        - Si train baja pero val sube → SOBREAJUSTE (overfitting)
        - Si ambas bajan juntas → el modelo está aprendiendo bien
        - Si ninguna baja → el modelo no está aprendiendo (underfitting)

    Parámetros:
        historial   : Objeto History de Keras con métricas por época.
        ruta_salida : Ruta donde guardar el gráfico PNG.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epocas = range(1, len(historial.history["loss"]) + 1)

    # Gráfico 1 — Pérdida (MSE)
    ax1.plot(epocas, historial.history["loss"],     "b-", label="Train", linewidth=2)
    ax1.plot(epocas, historial.history["val_loss"], "r-", label="Validación", linewidth=2)
    ax1.set_title("Curva de Pérdida (MSE)\nTrain vs Validación", fontweight="bold")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2 — MAE
    ax2.plot(epocas, historial.history["mae"],     "b-", label="Train", linewidth=2)
    ax2.plot(epocas, historial.history["val_mae"], "r-", label="Validación", linewidth=2)
    ax2.set_title("Curva de Error Absoluto (MAE)\nTrain vs Validación", fontweight="bold")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("MAE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Curvas de entrenamiento: {ruta_salida}")


# ------------------------------------------------------------------------------
# FUNCIÓN 7: graficar_predicciones
# ------------------------------------------------------------------------------
def graficar_predicciones(metricas: dict, ruta_salida: str) -> None:
    """
    Grafica precio real vs precio predicho.

    QUÉ BUSCAR EN ESTE GRÁFICO:
        - Los puntos deberían estar cerca de la línea diagonal roja
        - La línea diagonal = predicción perfecta (predicho == real)
        - Puntos lejos de la línea = predicciones equivocadas
        - Si los puntos forman una nube alrededor de la línea → bueno

    Parámetros:
        metricas    (dict): Contiene y_pred_real e y_test_real.
        ruta_salida (str) : Ruta donde guardar el gráfico PNG.
    """

    y_real = metricas["y_test_real"].flatten()
    y_pred = metricas["y_pred_real"].flatten()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Puntos: precio real vs precio predicho
    ax.scatter(y_real, y_pred, alpha=0.4, color="#4C72B0", s=20)

    # Línea diagonal roja = predicción perfecta
    max_val = max(y_real.max(), y_pred.max())
    min_val = min(y_real.min(), y_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], "r--",
            linewidth=2, label="Predicción perfecta")

    ax.set_title(
        f"Precio Real vs Precio Predicho\n"
        f"R²={metricas['r2']:.4f}  |  MAE=${metricas['mae']:.2f}  |  RMSE=${metricas['rmse']:.2f}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Precio Real (USD)", fontsize=12)
    ax.set_ylabel("Precio Predicho (USD)", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Predicciones vs real   : {ruta_salida}")


# ------------------------------------------------------------------------------
# FUNCIÓN 8: guardar_modelo
# ------------------------------------------------------------------------------
def guardar_modelo(modelo, ruta_base: str) -> None:
    """
    Guarda el modelo entrenado en disco para uso futuro.

    Se guardan dos formatos:
        .keras : Formato moderno recomendado por TensorFlow/Keras
        .h5    : Formato legacy compatible con versiones anteriores

    El modelo guardado se puede cargar luego con:
        modelo = keras.models.load_model('outputs/models/neuralpricer.keras')
        precio = modelo.predict([[brand_id, category_id, cluster_id]])

    Parámetros:
        modelo    : Red neuronal entrenada.
        ruta_base : Ruta base sin extensión.
    """

    os.makedirs(ruta_base, exist_ok=True)

    # Formato moderno Keras
    ruta_keras = os.path.join(ruta_base, "neuralpricer.keras")
    modelo.save(ruta_keras)
    print(f"   Modelo guardado (.keras): {ruta_keras}")

    # Formato legacy .h5
    ruta_h5 = os.path.join(ruta_base, "neuralpricer.h5")
    modelo.save(ruta_h5)
    print(f"   Modelo guardado (.h5)   : {ruta_h5}")


# ------------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("  NeuralPricer -- PR #5: Red Neuronal de Predicción de Precios")
    print("  Grupo Almerco | Input: [marca, categoría, cluster] → precio")
    print("=" * 70)

    # PASO 1 — Cargar dataset con cluster_id del PR #4
    df = cargar_datos(CLUSTERED_PATH)

    # PASO 2 — Preparar features y dividir en train/test
    (X_train, X_test, y_train, y_test,
     scaler_X, scaler_y,
     encoder_brand, encoder_category) = preparar_features(df)

    # PASO 3 — Construir arquitectura de la red neuronal
    modelo = construir_modelo(learning_rate=LEARNING_RATE)

    # PASO 4 — Entrenar con backpropagation y Adam optimizer
    historial = entrenar_modelo(modelo, X_train, y_train, X_test, y_test)

    # PASO 5 — Evaluar con datos de test (nunca vistos durante entrenamiento)
    metricas = evaluar_modelo(modelo, X_test, y_test, scaler_y)

    # PASO 6 — Generar gráficos
    print(f"\nGenerando gráficos...")
    graficar_entrenamiento(
        historial,
        os.path.join(PLOTS_PATH, "training_history.png")
    )
    graficar_predicciones(
        metricas,
        os.path.join(PLOTS_PATH, "prediccion_vs_real.png")
    )

    # PASO 7 — Guardar modelo entrenado
    print(f"\nGuardando modelo...")
    guardar_modelo(modelo, MODELS_PATH)

    print("\n" + "=" * 70)
    print("  PR #5 completado exitosamente")
    print(f"  MAE  : ${metricas['mae']:.2f} (error promedio en USD)")
    print(f"  R²   :  {metricas['r2']:.4f} (varianza explicada)")
    print(f"  Modelo: {MODELS_PATH}neuralpricer.keras")
    print("=" * 70 + "\n")
