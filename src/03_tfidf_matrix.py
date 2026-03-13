"""
================================================================================
NeuralPricer — PR #3: El Texto como Álgebra Lineal (TF-IDF)
================================================================================
Proyecto  : NeuralPricer — Radar de Precios de Mercado
Cliente   : Grupo Almerco
Autor     : Josias
Fecha     : Marzo 2026
PR        : #3
--------------------------------------------------------------------------------
OBJETIVO:
    Convertir los nombres de productos (texto) en vectores numéricos usando
    TF-IDF, y calcular la similitud entre productos usando distancia coseno.

    Esto permite detectar que "Monitor LG 24" y "LG 24 pulgadas IPS"
    son el mismo producto aunque tengan nombres distintos.

EL PROBLEMA QUE RESUELVE:
    La competencia escribe los mismos productos de formas diferentes:
        - "Monitor LG 24"
        - "LG 24 pulgadas IPS"
        - "Pantalla LG 24inch Full HD"

    Para una computadora estos son 3 strings completamente distintos.
    Para un humano es obviamente el mismo producto.

    TF-IDF convierte cada nombre en un vector matemático.
    La distancia coseno mide qué tan "parecidos" son esos vectores.

¿QUÉ ES TF-IDF?
    TF-IDF = Term Frequency × Inverse Document Frequency

    TF (Term Frequency):
        ¿Qué tan frecuente es una palabra en ESTE documento?
        TF(palabra, doc) = (veces que aparece la palabra en el doc) /
                           (total de palabras en el doc)

    IDF (Inverse Document Frequency):
        ¿Qué tan rara es esta palabra en TODOS los documentos?
        IDF(palabra) = log( N / (1 + df(palabra)) )
        Donde:
            N          = total de documentos
            df(palabra) = en cuántos documentos aparece la palabra

    La idea: una palabra que aparece en TODOS los documentos (como "de", "el")
    no ayuda a distinguir productos — su IDF es bajo.
    Una palabra rara y específica ("OLED", "NVMe") sí distingue — su IDF es alto.

    TF-IDF(palabra, doc) = TF × IDF
    Resultado: un número alto = palabra importante para este documento

¿QUÉ ES LA DISTANCIA COSENO?
    Imagina que cada producto es una flecha (vector) en el espacio.
    La distancia coseno mide el ÁNGULO entre dos flechas.

    similitud_coseno = cos(θ) = (A · B) / (||A|| × ||B||)
    distancia_coseno = 1 - similitud_coseno

    Donde:
        A · B   = producto punto entre los vectores (suma de xi*yi)
        ||A||   = magnitud del vector A (raíz de suma de xi²)
        θ       = ángulo entre los dos vectores

    Interpretación:
        distancia = 0.0 → productos idénticos (mismo ángulo, paralelos)
        distancia = 0.5 → productos algo similares
        distancia = 1.0 → productos completamente distintos (perpendiculares)

¿QUÉ SON STOPWORDS?
    Palabras tan comunes que no aportan significado:
    "el", "la", "de", "con", "para", "y", "a", "the", "of", "with"

    Las eliminamos antes de construir la matriz TF-IDF porque solo
    agregan ruido matemático sin información útil.

ENTRADAS:
    - data/processed/clean.csv  (generado por PR #1)

SALIDAS:
    - outputs/data/tfidf_matrix.npz     (matriz TF-IDF en formato sparse)
    - outputs/data/tfidf_features.csv   (nombres de las columnas/palabras)
    - outputs/data/similitud_top.csv    (pares de productos más similares)

DEPENDENCIAS:
    pip install nltk scipy scikit-learn pandas python-dotenv
================================================================================
"""

# ------------------------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------------------------
import os
import sys
import re

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# NLTK — Natural Language Toolkit
# Librería para procesar texto en lenguaje natural
import nltk
from nltk.corpus import stopwords      # Lista de palabras vacías
from nltk.tokenize import word_tokenize # Divide texto en palabras individuales
from nltk.stem import PorterStemmer    # Reduce palabras a su raíz (run/running → run)

# SciPy — Scientific Python
# Usamos sparse matrices porque la matriz TF-IDF tiene muchos ceros
from scipy.sparse import save_npz      # Guarda matriz sparse en disco
from scipy.spatial.distance import cosine  # Calcula distancia coseno entre vectores

# Scikit-learn — Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer  # Construye la matriz TF-IDF


# ------------------------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------------------------
load_dotenv()

DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/clean.csv")
TFIDF_MATRIX_PATH   = os.getenv("TFIDF_MATRIX_PATH",   "outputs/data/tfidf_matrix.npz")
OUTPUTS_DATA_PATH   = "outputs/data/"

# Umbral de similitud coseno — dos productos con distancia <= este valor
# son considerados "potencialmente el mismo producto"
# 0.25 significa que los vectores están a menos del 25% de distancia
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.25"))

# Número máximo de pares similares a guardar en el CSV de resultados
MAX_PARES_SIMILARES = 100


# ------------------------------------------------------------------------------
# FUNCIÓN 0: descargar_recursos_nltk
# ------------------------------------------------------------------------------
def descargar_recursos_nltk() -> None:
    """
    Descarga los recursos de NLTK necesarios si no están instalados.

    NLTK necesita descargar datos adicionales la primera vez:
        - stopwords : lista de palabras vacías en múltiples idiomas
        - punkt     : modelo para tokenizar (dividir texto en palabras)
        - punkt_tab : tabla de tokenización actualizada

    Estos datos se guardan localmente y no se vuelven a descargar.
    """

    print("\nVerificando recursos de NLTK...")

    recursos = [
        ("corpora/stopwords",          "stopwords"),
        ("tokenizers/punkt",           "punkt"),
        ("tokenizers/punkt_tab",       "punkt_tab"),
    ]

    for ruta, nombre in recursos:
        try:
            nltk.data.find(ruta)
            print(f"   Ya instalado: {nombre}")
        except LookupError:
            print(f"   Descargando: {nombre}...")
            nltk.download(nombre, quiet=True)
            print(f"   Descargado: {nombre}")


# ------------------------------------------------------------------------------
# FUNCIÓN 1: cargar_datos
# ------------------------------------------------------------------------------
def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset limpio del PR #1.

    Parámetros:
        ruta (str): Ruta al CSV limpio.

    Retorna:
        pd.DataFrame: Dataset con columnas product_name, price, brand, category.
    """

    if not os.path.exists(ruta):
        print(f"\nERROR: No se encontró: {ruta}")
        print("   Ejecuta primero: python src/01_outlier_cleaning.py")
        sys.exit(1)

    print(f"\nCargando dataset desde: {ruta}")
    df = pd.read_csv(ruta, low_memory=False)
    print(f"   Cargado: {df.shape[0]:,} productos")

    return df


# ------------------------------------------------------------------------------
# FUNCIÓN 2: limpiar_texto
# ------------------------------------------------------------------------------
def limpiar_texto(texto: str, stemmer: PorterStemmer, stop_words: set) -> str:
    """
    Limpia y normaliza un nombre de producto para el análisis TF-IDF.

    PASOS DEL PROCESAMIENTO:
        1. Convertir a minúsculas        → "Monitor LG" → "monitor lg"
        2. Eliminar caracteres especiales → "24-inch!" → "24 inch"
        3. Tokenizar                      → "monitor lg" → ["monitor", "lg"]
        4. Eliminar stopwords             → ["monitor", "lg", "de"] → ["monitor", "lg"]
        5. Aplicar stemming               → ["running"] → ["run"]

    ¿QUÉ ES TOKENIZAR?
        Dividir un string en una lista de palabras individuales (tokens).
        "Monitor LG 24 pulgadas" → ["Monitor", "LG", "24", "pulgadas"]

    ¿QUÉ ES STEMMING?
        Reducir una palabra a su raíz eliminando sufijos.
        "running" → "run"
        "laptops" → "laptop"
        "computing" → "comput"
        Esto ayuda a que "laptop" y "laptops" sean tratados como la misma palabra.

    Parámetros:
        texto     (str)           : Nombre del producto original.
        stemmer   (PorterStemmer) : Objeto para aplicar stemming.
        stop_words (set)          : Conjunto de palabras a eliminar.

    Retorna:
        str: Texto limpio listo para TF-IDF.
    """

    # Paso 1: Minúsculas — "Monitor" y "monitor" deben ser la misma palabra
    texto = texto.lower()

    # Paso 2: Eliminar caracteres especiales — solo letras, números y espacios
    # El regex [^a-z0-9\s] elimina todo lo que NO sea letra, número o espacio
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)

    # Paso 3: Tokenizar — dividir en palabras individuales
    tokens = word_tokenize(texto)

    # Paso 4: Eliminar stopwords y palabras muy cortas (1 caracter)
    # Las stopwords son palabras sin significado: "the", "of", "with", "de"
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    # Paso 5: Stemming — reducir cada palabra a su raíz
    tokens = [stemmer.stem(t) for t in tokens]

    # Unir los tokens de vuelta en un string limpio
    return " ".join(tokens)


# ------------------------------------------------------------------------------
# FUNCIÓN 3: construir_matriz_tfidf
# ------------------------------------------------------------------------------
def construir_matriz_tfidf(textos_limpios: pd.Series):
    """
    Construye la matriz TF-IDF a partir de los nombres de productos limpios.

    MATEMÁTICAS APLICADAS:
        Para cada producto i y cada palabra j:

        TF(j, i)  = frecuencia de la palabra j en el producto i /
                    total de palabras en el producto i

        IDF(j)    = log( (1 + N) / (1 + df(j)) ) + 1
                    Donde N = total productos, df(j) = productos con la palabra j

        TF-IDF(j, i) = TF(j, i) × IDF(j)

    RESULTADO:
        Una matriz de dimensiones (N_productos × N_palabras_únicas)
        Cada fila = un producto representado como vector numérico
        Cada columna = una palabra del vocabulario completo

        Ejemplo con 3 productos y vocabulario de 5 palabras:
                    monitor  laptop  lg   24  pulgadas
        producto_1 [  0.8,    0.0,  0.6, 0.4,  0.3  ]
        producto_2 [  0.0,    0.9,  0.5, 0.0,  0.0  ]
        producto_3 [  0.7,    0.0,  0.6, 0.4,  0.4  ]

        productos_1 y productos_3 son similares (vectores parecidos)

    ¿QUÉ ES UNA MATRIZ SPARSE?
        La mayoría de productos no contiene la mayoría de palabras del vocabulario.
        Eso significa que la matriz tiene muchísimos ceros.
        Una matriz sparse solo guarda los valores NO CERO — ahorra memoria.
        Con 5,000 productos y 10,000 palabras únicas:
            Matriz densa  = 5,000 × 10,000 = 50,000,000 números en memoria
            Matriz sparse = solo los valores distintos de cero (mucho menos)

    Parámetros:
        textos_limpios (pd.Series): Nombres de productos ya limpios.

    Retorna:
        tuple: (matriz_tfidf, vectorizador)
            - matriz_tfidf : scipy sparse matrix (N_productos × N_palabras)
            - vectorizador : TfidfVectorizer entrenado (para obtener nombres de features)
    """

    print("\nConstruyendo matriz TF-IDF...")

    # TfidfVectorizer hace todo el proceso TF-IDF automáticamente
    # max_features = máximo de palabras únicas en el vocabulario
    # min_df       = una palabra debe aparecer en al menos 2 productos para incluirse
    #                (filtra palabras rarísimas que solo aparecen en 1 producto)
    # ngram_range  = (1,2) incluye palabras solas Y pares de palabras consecutivas
    #                "lg monitor" como bigrama es más específico que "lg" solo
    vectorizador = TfidfVectorizer(
        max_features = 5000,      # Máximo 5000 palabras en el vocabulario
        min_df       = 2,         # La palabra debe aparecer en mínimo 2 productos
        ngram_range  = (1, 2),    # Unigramas y bigramas
        sublinear_tf = True,      # Aplica log(TF) en lugar de TF crudo — reduce peso de repeticiones
    )

    # fit_transform hace DOS cosas:
    # 1. fit    → aprende el vocabulario de todos los textos
    # 2. transform → convierte cada texto en su vector TF-IDF
    matriz_tfidf = vectorizador.fit_transform(textos_limpios)

    # Obtener los nombres de las palabras (features) del vocabulario
    nombres_features = vectorizador.get_feature_names_out()

    print(f"   Productos vectorizados : {matriz_tfidf.shape[0]:,}")
    print(f"   Palabras en vocabulario: {matriz_tfidf.shape[1]:,}")
    print(f"   Dimensiones de matriz  : {matriz_tfidf.shape[0]:,} × {matriz_tfidf.shape[1]:,}")
    print(f"   Valores no-cero        : {matriz_tfidf.nnz:,} de {matriz_tfidf.shape[0] * matriz_tfidf.shape[1]:,}")
    print(f"   Densidad de la matriz  : {matriz_tfidf.nnz / (matriz_tfidf.shape[0] * matriz_tfidf.shape[1]) * 100:.2f}%")

    return matriz_tfidf, vectorizador, nombres_features


# ------------------------------------------------------------------------------
# FUNCIÓN 4: calcular_pares_similares
# ------------------------------------------------------------------------------
def calcular_pares_similares(
    matriz_tfidf,
    df          : pd.DataFrame,
    threshold   : float,
    max_pares   : int
) -> pd.DataFrame:
    """
    Encuentra pares de productos con alta similitud coseno.

    MATEMÁTICAS APLICADAS — DISTANCIA COSENO:

        Para dos vectores A y B (dos productos):

        similitud_coseno(A, B) = (A · B) / (||A|| × ||B||)

        Donde:
            A · B = producto punto = Σ(aᵢ × bᵢ)
            ||A|| = norma euclidiana = √(Σ aᵢ²)

        distancia_coseno = 1 - similitud_coseno

        Valores:
            0.0 → idénticos (mismo vector, ángulo = 0°)
            0.5 → algo similares (ángulo = 60°)
            1.0 → completamente distintos (ángulo = 90°, perpendiculares)

    Por qué coseno y no distancia euclidiana:
        La distancia euclidiana se ve afectada por la LONGITUD del vector.
        Un nombre largo tendrá vectores más grandes aunque diga lo mismo.
        El coseno solo mide el ÁNGULO — independiente de la longitud.
        Eso lo hace perfecto para comparar textos de distinto largo.

    Parámetros:
        matriz_tfidf : Matriz sparse TF-IDF.
        df           : DataFrame original con nombres de productos.
        threshold    : Distancia máxima para considerar productos similares.
        max_pares    : Máximo de pares a retornar.

    Retorna:
        pd.DataFrame: Tabla con pares de productos similares y su distancia.
    """

    print(f"\nCalculando similitud coseno entre productos...")
    print(f"   Umbral de distancia: {threshold} (productos con distancia <= {threshold} son similares)")

    pares = []

    # Convertir matriz sparse a densa para calcular distancias
    # NOTA: con datasets grandes esto puede ser lento — para producción
    # se usaría una aproximación como LSH (Locality Sensitive Hashing)
    matriz_densa = matriz_tfidf.toarray()

    total_productos = len(matriz_densa)
    comparaciones   = 0

    # Comparar cada producto contra todos los siguientes (triángulo superior)
    # Si tenemos N productos, hacemos N×(N-1)/2 comparaciones
    for i in range(min(total_productos, 500)):  # Limitamos a 500 para velocidad
        for j in range(i + 1, min(total_productos, 500)):
            # Saltar si los nombres son exactamente iguales — son duplicados
            # No tiene sentido comparar un producto consigo mismo
            if df["product_name"].iloc[i] == df["product_name"].iloc[j]:
                continue

            comparaciones += 1

            # Calcular distancia coseno entre producto i y producto j
            # distancia = 1 - cos(θ) entre los dos vectores TF-IDF
            try:
                distancia = cosine(matriz_densa[i], matriz_densa[j])
            except Exception:
                continue

            # Si la distancia es menor al umbral → productos similares
            if distancia <= threshold:
                pares.append({
                    "producto_1"  : df["product_name"].iloc[i],
                    "producto_2"  : df["product_name"].iloc[j],
                    "precio_1"    : df["price"].iloc[i],
                    "precio_2"    : df["price"].iloc[j],
                    "distancia"   : round(distancia, 4),
                    "similitud"   : round(1 - distancia, 4),
                })

            # Mostrar progreso cada 10,000 comparaciones
            if comparaciones % 10000 == 0:
                print(f"   Comparaciones realizadas: {comparaciones:,} | Pares encontrados: {len(pares)}")

            if len(pares) >= max_pares:
                break
        if len(pares) >= max_pares:
            break

    print(f"   Total comparaciones  : {comparaciones:,}")
    print(f"   Pares similares encontrados: {len(pares)}")

    if not pares:
        print("   NOTA: No se encontraron pares con ese umbral.")
        print(f"   Intenta aumentar COSINE_THRESHOLD en .env (actual: {threshold})")
        return pd.DataFrame()

    # Convertir a DataFrame y ordenar por similitud descendente
    df_pares = pd.DataFrame(pares).sort_values("similitud", ascending=False)

    return df_pares


# ------------------------------------------------------------------------------
# FUNCIÓN 5: guardar_resultados
# ------------------------------------------------------------------------------
def guardar_resultados(
    matriz_tfidf,
    nombres_features : np.ndarray,
    df_pares         : pd.DataFrame,
) -> None:
    """
    Guarda la matriz TF-IDF y los pares similares en disco.

    Archivos generados:
        tfidf_matrix.npz    — matriz sparse en formato comprimido
        tfidf_features.csv  — nombres de las columnas (palabras del vocabulario)
        similitud_top.csv   — pares de productos más similares

    Parámetros:
        matriz_tfidf     : Matriz sparse TF-IDF.
        nombres_features : Array con nombres de las palabras/columnas.
        df_pares         : DataFrame con pares de productos similares.
    """

    print(f"\nGuardando resultados...")

    os.makedirs(OUTPUTS_DATA_PATH, exist_ok=True)

    # Guardar matriz TF-IDF en formato sparse comprimido (.npz)
    # Este formato ahorra espacio al no guardar los ceros
    ruta_matriz = os.path.join(OUTPUTS_DATA_PATH, "tfidf_matrix.npz")
    save_npz(ruta_matriz, matriz_tfidf)
    print(f"   Matriz TF-IDF guardada : {ruta_matriz}")

    # Guardar nombres de features (palabras del vocabulario)
    ruta_features = os.path.join(OUTPUTS_DATA_PATH, "tfidf_features.csv")
    pd.DataFrame({"feature": nombres_features}).to_csv(ruta_features, index=False)
    print(f"   Features guardadas     : {ruta_features}")

    # Guardar pares de productos similares
    if not df_pares.empty:
        ruta_pares = os.path.join(OUTPUTS_DATA_PATH, "similitud_top.csv")
        df_pares.to_csv(ruta_pares, index=False)
        print(f"   Pares similares        : {ruta_pares}")

        print(f"\n   Top 5 pares más similares:")
        print(f"   {'─'*70}")
        for _, row in df_pares.head(5).iterrows():
            p1 = str(row["producto_1"])[:35]
            p2 = str(row["producto_2"])[:35]
            print(f"   '{p1}'")
            print(f"   '{p2}'")
            print(f"   Similitud: {row['similitud']:.4f} | Distancia coseno: {row['distancia']:.4f}")
            print(f"   {'─'*70}")


# ------------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("  NeuralPricer -- PR #3: El Texto como Álgebra Lineal (TF-IDF)")
    print("  Grupo Almerco | TF-IDF + Distancia Coseno")
    print("=" * 70)

    # PASO 1 — Descargar recursos de NLTK si es la primera vez
    descargar_recursos_nltk()

    # PASO 2 — Cargar dataset limpio del PR #1
    df = cargar_datos(DATA_PROCESSED_PATH)

    # PASO 3 — Preparar herramientas de NLP
    # Stemmer: reduce palabras a su raíz (laptops → laptop)
    stemmer = PorterStemmer()

    # Stopwords: palabras vacías en inglés y español
    # El dataset es en inglés pero puede tener algunas palabras en español
    stop_words = set(stopwords.words("english")) | set(stopwords.words("spanish"))

    print(f"\n   Stopwords cargadas: {len(stop_words)} palabras a ignorar")
    print(f"   Ejemplos: {list(stop_words)[:8]}")

    # PASO 4 — Limpiar y normalizar nombres de productos
    print("\nLimpiando nombres de productos...")
    print("   Pasos: minúsculas → eliminar especiales → tokenizar → stopwords → stemming")

    df["nombre_limpio"] = df["product_name"].apply(
        lambda texto: limpiar_texto(texto, stemmer, stop_words)
    )

    # Mostrar ejemplos de transformación
    print(f"\n   Ejemplos de limpieza:")
    print(f"   {'─'*65}")
    for _, row in df.head(5).iterrows():
        print(f"   ORIGINAL : {str(row['product_name'])[:60]}")
        print(f"   LIMPIO   : {str(row['nombre_limpio'])[:60]}")
        print(f"   {'─'*65}")

    # PASO 5 — Construir matriz TF-IDF
    matriz_tfidf, vectorizador, nombres_features = construir_matriz_tfidf(
        df["nombre_limpio"]
    )

    # PASO 6 — Calcular pares de productos similares con distancia coseno
    df_pares = calcular_pares_similares(
        matriz_tfidf = matriz_tfidf,
        df           = df,
        threshold    = COSINE_THRESHOLD,
        max_pares    = MAX_PARES_SIMILARES
    )

    # PASO 7 — Guardar resultados
    guardar_resultados(matriz_tfidf, nombres_features, df_pares)

    print("\n" + "=" * 70)
    print("  PR #3 completado exitosamente")
    print(f"  Matriz TF-IDF: {matriz_tfidf.shape[0]:,} productos x {matriz_tfidf.shape[1]:,} palabras")
    print(f"  Output: {OUTPUTS_DATA_PATH}")
    print("=" * 70 + "\n")
