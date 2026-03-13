"""
================================================================================
NeuralPricer — PR #4: Agrupación No Supervisada (Clustering)
================================================================================
Proyecto  : NeuralPricer — Radar de Precios de Mercado
Cliente   : Grupo Almerco
Autor     : [Tu nombre]
Fecha     : Marzo 2026
PR        : #4
--------------------------------------------------------------------------------
OBJETIVO:
    Usar la matriz TF-IDF del PR #3 para agrupar automáticamente productos
    similares bajo un mismo "ID de Clúster".

    Resultado: "Monitor LG 24" y "LG 24 pulgadas IPS" quedarán en el
    mismo clúster — la máquina descubrió que son el mismo producto.

¿QUÉ ES CLUSTERING?
    Es aprendizaje NO SUPERVISADO — no le dices a la máquina cuáles
    productos son iguales. Ella misma descubre los grupos basándose
    en la similitud matemática de los vectores TF-IDF.

    Supervisado     = le dices las respuestas correctas (etiquetas)
    No supervisado  = la máquina encuentra patrones sola, sin etiquetas

¿QUÉ ES K-MEANS?
    Es el algoritmo de clustering más famoso. Funciona así:

    PASO 1 — Inicialización:
        Coloca K puntos aleatorios en el espacio (llamados "centroides")
        Cada centroide será el centro de un grupo

    PASO 2 — Asignación:
        Cada producto se asigna al centroide MÁS CERCANO
        "Cercano" = menor distancia euclidiana entre vectores

    PASO 3 — Actualización:
        Recalcula cada centroide como el PROMEDIO de todos los
        productos asignados a ese grupo

    PASO 4 — Repetir:
        Repite pasos 2 y 3 hasta que los centroides no se muevan
        (convergencia) o se alcance el máximo de iteraciones

    MATEMÁTICAS DE K-MEANS:
        Minimiza la función de inercia (Within-Cluster Sum of Squares):

        J = Σ Σ ||xᵢ - μₖ||²
            k  i∈Cₖ

        Donde:
            xᵢ  = vector TF-IDF del producto i
            μₖ  = centroide del clúster k
            Cₖ  = conjunto de productos en el clúster k
            ||·||² = distancia euclidiana al cuadrado

        El algoritmo busca minimizar J — que los productos estén
        lo más cerca posible de su centroide.

¿CÓMO ELEGIR EL NÚMERO K DE CLÚSTERES?
    Usamos el MÉTODO DEL CODO (Elbow Method):

    1. Probamos K = 2, 3, 4, ..., 20
    2. Para cada K calculamos la inercia (J)
    3. Graficamos K vs Inercia
    4. Buscamos el "codo" — el punto donde agregar más clústeres
       ya no reduce significativamente la inercia

    Por qué funciona:
        Con K=1 la inercia es máxima (todo en un grupo)
        Con K=N la inercia es 0 (cada producto en su propio grupo)
        El "codo" es el balance óptimo entre complejidad y calidad

ENTRADAS:
    - outputs/data/tfidf_matrix.npz  (generado por PR #3)
    - data/processed/clean.csv       (generado por PR #1)

SALIDAS:
    - outputs/data/clustered.csv         (dataset con columna cluster_id)
    - outputs/plots/elbow_curve.png      (gráfico del método del codo)
    - outputs/plots/cluster_prices.png   (precios por clúster)

DEPENDENCIAS:
    pip install scikit-learn pandas numpy matplotlib python-dotenv scipy
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
import seaborn as sns

from dotenv import load_dotenv

# SciPy — para cargar la matriz sparse del PR #3
from scipy.sparse import load_npz

# Scikit-learn — algoritmos de clustering
from sklearn.cluster import KMeans          # Algoritmo K-Means
from sklearn.decomposition import TruncatedSVD  # Reducción de dimensiones
from sklearn.preprocessing import normalize     # Normalización de vectores


# ------------------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------------------
load_dotenv()

DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/clean.csv")
TFIDF_MATRIX_PATH   = os.getenv("TFIDF_MATRIX_PATH",   "outputs/data/tfidf_matrix.npz")
PLOTS_PATH          = os.getenv("PLOTS_PATH",           "outputs/plots/")
OUTPUTS_DATA_PATH   = "outputs/data/"

# Semilla aleatoria fija — garantiza resultados reproducibles
# Con la misma semilla, K-Means siempre produce los mismos clústeres
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# Número de clústeres a probar en el método del codo
K_MIN = 2
K_MAX = 20

# Número final de clústeres (se ajusta después de ver el gráfico del codo)
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "10"))

# Estilo visual
sns.set_theme(style="whitegrid", palette="muted")


# ------------------------------------------------------------------------------
# FUNCIÓN 1: cargar_datos
# ------------------------------------------------------------------------------
def cargar_datos(ruta_csv: str, ruta_matriz: str):
    """
    Carga el dataset limpio y la matriz TF-IDF del PR #3.

    La matriz TF-IDF está guardada en formato .npz (sparse matrix comprimida).
    Es el resultado del trabajo del PR #3 — cada producto ya es un vector numérico.

    Parámetros:
        ruta_csv    (str): Ruta al CSV limpio del PR #1.
        ruta_matriz (str): Ruta a la matriz TF-IDF del PR #3.

    Retorna:
        tuple: (df, matriz_tfidf)
    """

    # Verificar que existan los archivos necesarios
    for ruta in [ruta_csv, ruta_matriz]:
        if not os.path.exists(ruta):
            print(f"\nERROR: No se encontró: {ruta}")
            print("   Asegúrate de haber ejecutado primero:")
            print("   python src/01_outlier_cleaning.py")
            print("   python src/03_tfidf_matrix.py")
            sys.exit(1)

    print(f"\nCargando dataset limpio desde: {ruta_csv}")
    df = pd.read_csv(ruta_csv, low_memory=False)
    print(f"   Productos cargados: {len(df):,}")

    print(f"\nCargando matriz TF-IDF desde: {ruta_matriz}")
    # load_npz carga la matriz sparse — solo los valores no-cero
    matriz_tfidf = load_npz(ruta_matriz)
    print(f"   Dimensiones matriz: {matriz_tfidf.shape[0]:,} x {matriz_tfidf.shape[1]:,}")

    # Verificar que el número de filas coincida entre CSV y matriz
    if len(df) != matriz_tfidf.shape[0]:
        print(f"\nADVERTENCIA: El CSV tiene {len(df):,} filas pero la matriz tiene")
        print(f"             {matriz_tfidf.shape[0]:,} filas.")
        print(f"             Usando el mínimo de ambos.")
        min_filas = min(len(df), matriz_tfidf.shape[0])
        df = df.iloc[:min_filas]

    return df, matriz_tfidf


# ------------------------------------------------------------------------------
# FUNCIÓN 2: reducir_dimensiones
# ------------------------------------------------------------------------------
def reducir_dimensiones(matriz_tfidf, n_componentes: int = 100):
    """
    Reduce la dimensionalidad de la matriz TF-IDF usando TruncatedSVD.

    ¿POR QUÉ REDUCIR DIMENSIONES?
        La matriz TF-IDF tiene 5,000 columnas (palabras).
        K-Means en espacios de muy alta dimensión sufre de la
        "maldición de la dimensionalidad" — las distancias euclidianas
        pierden significado cuando hay demasiadas dimensiones.

        Reducir a 100 dimensiones:
        - Mantiene el 80-90% de la información importante
        - Hace K-Means mucho más rápido y preciso
        - Elimina el "ruido" de palabras poco importantes

    ¿QUÉ ES TruncatedSVD?
        Es como PCA (Análisis de Componentes Principales) pero diseñado
        para matrices sparse. Encuentra las direcciones de mayor varianza
        en los datos y proyecta todo en ese espacio reducido.

        Matemáticamente:
            Matriz original (N x 5000) → Matriz reducida (N x 100)

    Parámetros:
        matriz_tfidf   : Matriz sparse TF-IDF.
        n_componentes  : Número de dimensiones destino. Default: 100.

    Retorna:
        np.ndarray: Matriz densa reducida de forma (N_productos x n_componentes).
    """

    print(f"\nReduciendo dimensiones: {matriz_tfidf.shape[1]:,} → {n_componentes}...")
    print(f"   (Esto elimina ruido y mejora la calidad del clustering)")

    # TruncatedSVD reduce dimensiones manteniendo la mayor varianza posible
    svd = TruncatedSVD(
        n_components  = n_componentes,
        random_state  = RANDOM_STATE,
        algorithm     = "randomized"  # Más rápido para matrices grandes
    )

    # fit_transform aprende la reducción Y aplica la transformación
    matriz_reducida = svd.fit_transform(matriz_tfidf)

    # Normalizar los vectores — importante para K-Means con texto
    # Normalizar hace que todos los vectores tengan longitud 1
    # Esto hace que K-Means mida ángulos en lugar de distancias absolutas
    matriz_normalizada = normalize(matriz_reducida)

    # Calcular cuánta varianza explicamos con estos componentes
    varianza_explicada = svd.explained_variance_ratio_.sum() * 100

    print(f"   Dimensiones reducidas a: {matriz_normalizada.shape[1]}")
    print(f"   Varianza explicada      : {varianza_explicada:.1f}%")
    print(f"   (Conservamos el {varianza_explicada:.1f}% de la información original)")

    return matriz_normalizada


# ------------------------------------------------------------------------------
# FUNCIÓN 3: metodo_del_codo
# ------------------------------------------------------------------------------
def metodo_del_codo(matriz: np.ndarray, k_min: int, k_max: int) -> int:
    """
    Aplica el Método del Codo para encontrar el número óptimo de clústeres K.

    MATEMÁTICAS DEL MÉTODO DEL CODO:
        Para cada valor de K probamos K-Means y calculamos la INERCIA:

        Inercia = Σ Σ ||xᵢ - μₖ||²
                  k  i∈Cₖ

        Donde:
            xᵢ  = vector del producto i
            μₖ  = centroide del clúster k
            ||·||² = distancia euclidiana al cuadrado

        Cuando K aumenta, la inercia siempre baja (más grupos = más específicos).
        Pero en algún punto agregar más grupos ya no ayuda mucho.
        Ese punto de inflexión es el "CODO" — el K óptimo.

    Parámetros:
        matriz (np.ndarray): Matriz reducida de productos.
        k_min  (int)       : K mínimo a probar.
        k_max  (int)       : K máximo a probar.

    Retorna:
        int: K óptimo sugerido por el método del codo.
    """

    print(f"\nAplicando Método del Codo (K = {k_min} a {k_max})...")
    print(f"   Esto puede tardar 1-2 minutos...")

    inercias = []  # Lista para guardar la inercia de cada K
    valores_k = range(k_min, k_max + 1)

    for k in valores_k:
        # Entrenar K-Means con este valor de K
        # n_init=3 = prueba 3 inicializaciones diferentes y toma la mejor
        # max_iter=100 = máximo 100 iteraciones por entrenamiento
        kmeans = KMeans(
            n_clusters   = k,
            random_state = RANDOM_STATE,
            n_init       = 3,
            max_iter     = 100
        )
        kmeans.fit(matriz)

        # inertia_ = suma de distancias al cuadrado de cada punto a su centroide
        inercias.append(kmeans.inertia_)
        print(f"   K={k:2d} → Inercia: {kmeans.inertia_:,.2f}")

    # ── Graficar la curva del codo ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(valores_k, inercias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Número de Clústeres (K)", fontsize=12)
    ax.set_ylabel("Inercia (suma de distancias²)", fontsize=12)
    ax.set_title(
        "Método del Codo — Número Óptimo de Clústeres\n"
        "El 'codo' indica el K óptimo: agregar más clústeres ya no ayuda mucho",
        fontsize=13, fontweight="bold"
    )

    # Resaltar el K configurado en .env
    ax.axvline(
        x         = N_CLUSTERS,
        color     = "red",
        linestyle = "--",
        linewidth = 2,
        label     = f"K seleccionado = {N_CLUSTERS}"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    os.makedirs(PLOTS_PATH, exist_ok=True)
    ruta_grafico = os.path.join(PLOTS_PATH, "elbow_curve.png")
    plt.tight_layout()
    plt.savefig(ruta_grafico, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n   Gráfico del codo guardado: {ruta_grafico}")
    print(f"   Revisa el gráfico y ajusta N_CLUSTERS en .env si es necesario")
    print(f"   Usando K = {N_CLUSTERS} (configurado en .env)")

    return N_CLUSTERS


# ------------------------------------------------------------------------------
# FUNCIÓN 4: entrenar_kmeans
# ------------------------------------------------------------------------------
def entrenar_kmeans(matriz: np.ndarray, n_clusters: int) -> KMeans:
    """
    Entrena el modelo K-Means final con el número de clústeres elegido.

    ALGORITMO K-MEANS PASO A PASO:
        1. Inicialización (k-means++):
           Coloca K centroides iniciales de forma inteligente —
           el primer centroide es aleatorio, los siguientes se eligen
           con probabilidad proporcional a su distancia del más cercano.
           Esto da mejores resultados que inicialización completamente aleatoria.

        2. Asignación:
           Cada producto xᵢ se asigna al clúster k* donde:
           k* = argmin_k ||xᵢ - μₖ||²

        3. Actualización:
           Cada centroide μₖ se recalcula como:
           μₖ = (1/|Cₖ|) × Σ xᵢ  para todo i en Cₖ

        4. Convergencia:
           Repite 2 y 3 hasta que ||μₖ(t) - μₖ(t-1)|| < tolerancia
           o se alcanza max_iter iteraciones.

    Parámetros:
        matriz     (np.ndarray): Matriz reducida de productos.
        n_clusters (int)       : Número de clústeres K.

    Retorna:
        KMeans: Modelo entrenado con los clústeres asignados.
    """

    print(f"\nEntrenando K-Means con K={n_clusters} clústeres...")
    print(f"   Semilla aleatoria: {RANDOM_STATE} (resultados reproducibles)")

    kmeans = KMeans(
        n_clusters   = n_clusters,
        random_state = RANDOM_STATE,
        n_init       = 10,      # 10 inicializaciones — toma la mejor
        max_iter     = 300,     # Máximo 300 iteraciones
        algorithm    = "lloyd"  # Algoritmo estándar de Lloyd
    )

    # fit() entrena el modelo — asigna cada producto a un clúster
    kmeans.fit(matriz)

    print(f"   Iteraciones realizadas : {kmeans.n_iter_}")
    print(f"   Inercia final          : {kmeans.inertia_:,.2f}")
    print(f"   Clústeres encontrados  : {n_clusters}")

    # Contar cuántos productos hay en cada clúster
    etiquetas, conteos = np.unique(kmeans.labels_, return_counts=True)
    print(f"\n   Distribución de productos por clúster:")
    for cluster_id, count in zip(etiquetas, conteos):
        barra = "█" * (count // 10)  # Barra visual proporcional
        print(f"   Clúster {cluster_id:2d}: {count:4d} productos  {barra}")

    return kmeans


# ------------------------------------------------------------------------------
# FUNCIÓN 5: graficar_precios_por_cluster
# ------------------------------------------------------------------------------
def graficar_precios_por_cluster(df: pd.DataFrame, ruta_salida: str) -> None:
    """
    Genera un boxplot mostrando la distribución de precios por clúster.

    Este gráfico permite al gerente de ventas ver:
        - Qué clústeres tienen productos baratos vs caros
        - Qué tan dispersos son los precios dentro de cada grupo
        - Si los clústeres tienen sentido desde el punto de vista de precios

    Un clúster bien formado debería tener precios similares entre sí
    (baja dispersión) porque agrupa productos parecidos.

    Parámetros:
        df          (pd.DataFrame): Dataset con columnas 'cluster_id' y 'price'.
        ruta_salida (str)         : Ruta donde guardar el gráfico PNG.
    """

    print(f"\nGenerando gráfico de precios por clúster...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Boxplot horizontal — un box por clúster
    # Muestra mediana, Q1, Q3 y outliers de precio para cada grupo
    sns.boxplot(
        data      = df,
        x         = "price",
        y         = "cluster_id",
        orient    = "h",
        palette   = "muted",
        ax        = ax
    )

    ax.set_title(
        f"Distribución de Precios por Clúster (K={df['cluster_id'].nunique()})\n"
        "Cada fila = un grupo de productos similares",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Precio (USD)", fontsize=12)
    ax.set_ylabel("ID de Clúster", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Gráfico guardado: {ruta_salida}")


# ------------------------------------------------------------------------------
# FUNCIÓN 6: guardar_resultado
# ------------------------------------------------------------------------------
def guardar_resultado(df: pd.DataFrame, kmeans: KMeans, ruta_salida: str) -> pd.DataFrame:
    """
    Agrega la columna 'cluster_id' al dataset y lo exporta como CSV.

    Este es el entregable principal del PR #4:
        Un CSV donde cada producto tiene su cluster_id asignado.

    Ejemplo del resultado:
        product_name          price   brand    category     cluster_id
        Monitor LG 24         299.99  LG       Monitors     3
        LG 24 pulgadas IPS    319.99  LG       Monitors     3   ← mismo clúster!
        Laptop Dell XPS 15   1299.99  Dell     Laptops      7

    Parámetros:
        df          (pd.DataFrame): Dataset limpio.
        kmeans      (KMeans)      : Modelo entrenado con las etiquetas.
        ruta_salida (str)         : Ruta del CSV de salida.

    Retorna:
        pd.DataFrame: Dataset con columna cluster_id agregada.
    """

    print(f"\nGuardando dataset con cluster_id...")

    # Agregar la columna cluster_id al DataFrame
    # kmeans.labels_ contiene el ID del clúster para cada producto
    df = df.copy()
    df["cluster_id"] = kmeans.labels_

    # Reordenar columnas para que cluster_id sea visible al inicio
    columnas = ["cluster_id", "product_name", "price", "brand", "category", "manufacturer", "primary_category"]
    df = df[columnas]

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    # Guardar CSV
    df.to_csv(ruta_salida, index=False, encoding="utf-8")

    print(f"   Guardado en     : {ruta_salida}")
    print(f"   Total productos : {len(df):,}")
    print(f"   Clústeres únicos: {df['cluster_id'].nunique()}")

    # Mostrar ejemplos de productos agrupados en el mismo clúster
    print(f"\n   Ejemplos de productos agrupados (mismo cluster_id):")
    print(f"   {'─'*65}")

    for cluster_id in range(min(3, df["cluster_id"].nunique())):
        grupo = df[df["cluster_id"] == cluster_id].head(3)
        print(f"\n   CLUSTER {cluster_id}:")
        for _, row in grupo.iterrows():
            nombre = str(row["product_name"])[:50]
            print(f"     · {nombre:<50} ${row['price']:>8.2f}")

    return df


# ------------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("  NeuralPricer -- PR #4: Clustering No Supervisado (K-Means)")
    print("  Grupo Almerco | Minimiza J = Σ Σ ||xᵢ - μₖ||²")
    print("=" * 70)

    # PASO 1 — Cargar dataset limpio y matriz TF-IDF del PR #3
    df, matriz_tfidf = cargar_datos(DATA_PROCESSED_PATH, TFIDF_MATRIX_PATH)

    # PASO 2 — Reducir dimensiones para mejorar K-Means
    # 5,000 dimensiones → 100 dimensiones conservando 80-90% de información
    matriz_reducida = reducir_dimensiones(matriz_tfidf, n_componentes=100)

    # PASO 3 — Método del Codo para encontrar K óptimo
    # Genera el gráfico elbow_curve.png para visualizar
    k_optimo = metodo_del_codo(matriz_reducida, K_MIN, K_MAX)

    # PASO 4 — Entrenar K-Means con el K elegido
    kmeans = entrenar_kmeans(matriz_reducida, n_clusters=k_optimo)

    # PASO 5 — Guardar dataset con cluster_id
    ruta_clustered = os.path.join(OUTPUTS_DATA_PATH, "clustered.csv")
    df_clustered = guardar_resultado(df, kmeans, ruta_clustered)

    # PASO 6 — Graficar precios por clúster
    ruta_grafico = os.path.join(PLOTS_PATH, "cluster_prices.png")
    graficar_precios_por_cluster(df_clustered, ruta_grafico)

    print("\n" + "=" * 70)
    print("  PR #4 completado exitosamente")
    print(f"  Clústeres generados: {k_optimo}")
    print(f"  Output: {ruta_clustered}")
    print("=" * 70 + "\n")
