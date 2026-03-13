"""
================================================================================
NeuralPricer — PR #1: Limpieza Estadística de Outliers
================================================================================
Proyecto  : NeuralPricer — Radar de Precios de Mercado
Cliente   : Grupo Almerco
Autor     : [Tu nombre]
Fecha     : Marzo 2026
PR        : #1
--------------------------------------------------------------------------------
OBJETIVO:
    Cargar el dataset de productos electrónicos de Kaggle y eliminar
    automáticamente los precios atípicos (outliers) usando estadística formal.
 
FÓRMULA MATEMÁTICA APLICADA:
    ─────────────────────────────────────────────────────────────────
    Desviación estándar poblacional:
 
              σ = √( Σ(xᵢ - μ)² / N )
 
    Donde:
        xᵢ = precio individual de cada producto
        μ  = media aritmética de precios en la categoría
        N  = total de productos en la categoría
        σ  = desviación estándar (qué tanto se dispersan los precios)
 
    Rango válido de precios por categoría:
        [ μ - 3σ ,  μ + 3σ ]
 
    Todo precio fuera de ese rango es un OUTLIER y se elimina.
    ─────────────────────────────────────────────────────────────────
 
¿POR QUÉ 3 DESVIACIONES ESTÁNDAR?
    Por la Regla Empírica (Campana de Gauss):
        - μ ± 1σ cubre el 68.27% de los datos
        - μ ± 2σ cubre el 95.45% de los datos
        - μ ± 3σ cubre el 99.73% de los datos
    Cualquier precio fuera de μ ± 3σ ocurre solo el 0.27% del tiempo
    en una distribución normal — casi seguro es un error o fraude.
 
REGLA DE NEGOCIO (del documento):
    ❌ PROHIBIDO: if precio < 10  (condicional arbitrario)
    ✅ OBLIGATORIO: usar la fórmula estadística formal por categoría
 
ENTRADAS:
    - data/raw/electronics.csv
 
SALIDAS:
    - data/processed/clean.csv  (dataset sin outliers)
    - Reporte en consola con estadísticas del proceso
 
DEPENDENCIAS:
    pip install pandas numpy python-dotenv
================================================================================
"""
 
# ──────────────────────────────────────────────────────────────────────────────
# IMPORTACIONES
# Cargamos las librerías necesarias para este script
# ──────────────────────────────────────────────────────────────────────────────
 
import os           # Para manejar rutas del sistema operativo
import sys          # Para salir del programa con mensajes de error
 
import numpy as np  # Operaciones matemáticas y estadísticas
import pandas as pd # Manipulación de datos en tablas (DataFrames)
 
from dotenv import load_dotenv  # Carga variables de entorno desde .env
 
 
# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# Cargamos las variables de entorno definidas en el archivo .env
# Esto evita tener rutas hardcodeadas en el código (buena práctica)
# ──────────────────────────────────────────────────────────────────────────────
 
load_dotenv()  # Lee el archivo .env y carga las variables en el entorno
 
# Leemos las rutas desde .env — si no existen usamos valores por defecto
DATA_RAW_PATH       = os.getenv("DATA_RAW_PATH",       "data/raw/electronics.csv")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/clean.csv")
OUTLIER_THRESHOLD   = float(os.getenv("OUTLIER_THRESHOLD", "3"))  # El "3" de μ ± 3σ
 
 
# ──────────────────────────────────────────────────────────────────────────────
# COLUMNAS REQUERIDAS (RF-01 del documento de requerimientos)
# El dataset DEBE contener estas 4 columnas mínimas para que el sistema funcione
# ──────────────────────────────────────────────────────────────────────────────
 
COLUMNAS_REQUERIDAS = ["name", "price", "brand", "categories"]

MAPEO_COLUMNAS = {
    "name"       : "product_name",
    "price"      : "price",
    "brand"      : "brand",
    "categories" : "category",
}
 
# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 1: cargar_dataset
# Responsabilidad: leer el CSV y validar que tenga las columnas requeridas
# ──────────────────────────────────────────────────────────────────────────────
 
def cargar_dataset(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset CSV y valida que contenga las columnas mínimas requeridas.
 
    Esta función implementa el RF-01 del documento de requerimientos:
    mostrar un error descriptivo si falta alguna columna, sin lanzar
    una excepción genérica.
 
    Parámetros:
        ruta (str): Ruta al archivo CSV de entrada.
 
    Retorna:
        pd.DataFrame: Dataset cargado con las columnas validadas.
 
    Lanza:
        SystemExit: Si el archivo no existe o faltan columnas requeridas.
    """
 
    # ── Verificar que el archivo existe ──────────────────────────────────────
    if not os.path.exists(ruta):
        print(f"\n❌ ERROR: No se encontró el archivo en: {ruta}")
        print("   Asegúrate de haber colocado el CSV en data/raw/electronics.csv")
        sys.exit(1)  # Termina el programa con código de error
 
    print(f"\n📂 Cargando dataset desde: {ruta}")
 
    # ── Leer el CSV con pandas ────────────────────────────────────────────────
    # low_memory=False evita warnings de tipos de datos mixtos en columnas grandes
    df = pd.read_csv(ruta, low_memory=False)
 
    print(f"   ✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
 
    # ── Validar columnas requeridas (RF-01) ───────────────────────────────────
    # Verificamos que existan las columnas que necesitamos antes de continuar
    columnas_faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
 
    if columnas_faltantes:
        print(f"\n❌ ERROR: Faltan columnas requeridas en el dataset:")
        for col in columnas_faltantes:
            print(f"   - '{col}'")
        print(f"\n   Columnas disponibles en el CSV:")
        for col in df.columns.tolist():
            print(f"   · {col}")
        sys.exit(1)
 
    print(f"   ✅ Columnas requeridas verificadas correctamente")
 
    return df
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 2: normalizar_columnas
# Responsabilidad: renombrar columnas del CSV al estándar del proyecto
# ──────────────────────────────────────────────────────────────────────────────
 
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra las columnas del CSV original al estándar del proyecto y
    limpia los tipos de datos básicos.
 
    El dataset de Kaggle usa nombres como 'prices.amountMax' pero el
    proyecto trabaja con 'price'. Esta función hace esa traducción.
 
    Parámetros:
        df (pd.DataFrame): DataFrame con columnas originales del CSV.
 
    Retorna:
        pd.DataFrame: DataFrame con columnas renombradas y tipos limpios.
    """
 
    print("\n🔧 Normalizando columnas...")
 
    # ── Seleccionar solo las columnas que necesitamos ─────────────────────────
    df = df[list(MAPEO_COLUMNAS.keys())].copy()
    # .copy() evita el SettingWithCopyWarning de pandas — buena práctica
 
    # ── Renombrar columnas al estándar del proyecto ───────────────────────────
    df = df.rename(columns=MAPEO_COLUMNAS)
 
    # ── Limpiar la columna de precios ─────────────────────────────────────────
    # Los precios pueden venir como strings con símbolos ("$1,299.99")
    # Los convertimos a números eliminando caracteres no numéricos
    df["price"] = (
        df["price"]
        .astype(str)                        # Convertir a texto primero
        .str.replace(r"[^\d.]", "", regex=True)  # Eliminar todo excepto dígitos y punto
        .replace("", np.nan)                # Strings vacíos → NaN (valor nulo)
        .astype(float)                      # Convertir a número decimal
    )
 
    # ── Limpiar textos en otras columnas ─────────────────────────────────────
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["brand"]        = df["brand"].astype(str).str.strip().str.title()
    df["category"]     = df["category"].astype(str).str.strip()
 
    # ── Eliminar filas donde el precio es nulo (no se puede calcular σ) ───────
    filas_antes = len(df)
    df = df.dropna(subset=["price"])
    filas_eliminadas = filas_antes - len(df)
 
    print(f"   ✅ Columnas renombradas al estándar del proyecto")
    print(f"   ✅ Precios convertidos a formato numérico")
    print(f"   ⚠️  Filas con precio nulo eliminadas: {filas_eliminadas:,}")
 
    return df
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 3: calcular_estadisticas_por_categoria
# Responsabilidad: calcular μ y σ para cada categoría
#
# MATEMÁTICAS APLICADAS:
#
#   Media aritmética (μ):
#       μ = (Σ xᵢ) / N
#
#   Desviación estándar poblacional (σ):
#       σ = √( Σ(xᵢ - μ)² / N )
#
#   Usamos ddof=0 en NumPy/Pandas para la fórmula POBLACIONAL
#   (dividir entre N, no entre N-1 que sería la muestral)
# ──────────────────────────────────────────────────────────────────────────────
 
def calcular_estadisticas_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la media (μ) y desviación estándar (σ) de precios
    agrupados por categoría usando la fórmula estadística formal.
 
    FÓRMULA IMPLEMENTADA:
        μ = Σ(xᵢ) / N
        σ = √( Σ(xᵢ - μ)² / N )
 
    El parámetro ddof=0 indica desviación estándar POBLACIONAL
    (dividimos entre N, no entre N-1).
 
    Parámetros:
        df (pd.DataFrame): DataFrame con columnas 'price' y 'category'.
 
    Retorna:
        pd.DataFrame: DataFrame original con columnas adicionales:
                      'precio_media' (μ) y 'precio_std' (σ) por categoría.
    """
 
    print("\n📐 Calculando estadísticas por categoría (μ y σ)...")
 
    # ── Calcular μ y σ agrupado por categoría ────────────────────────────────
    # groupby("category") agrupa todos los productos de la misma categoría
    # ddof=0 → fórmula poblacional: dividir entre N (no entre N-1)
    estadisticas = df.groupby("category")["price"].agg(
        precio_media=("mean"),          # μ = media aritmética
        precio_std  =lambda x: x.std(ddof=0)  # σ = desviación estándar poblacional
    ).reset_index()
 
    # ── Unir las estadísticas al DataFrame original ───────────────────────────
    # Hacemos un merge para que cada producto tenga la μ y σ de su categoría
    df = df.merge(estadisticas, on="category", how="left")
 
    # ── Mostrar resumen de estadísticas por categoría ─────────────────────────
    print(f"\n   {'Categoría':<35} {'μ (media)':>12} {'σ (std)':>12} {'Productos':>10}")
    print(f"   {'─'*35} {'─'*12} {'─'*12} {'─'*10}")
 
    for _, row in estadisticas.iterrows():
        categoria = str(row["category"])[:34]  # Truncar nombre largo
        print(f"   {categoria:<35} {row['precio_media']:>12.2f} {row['precio_std']:>12.2f}")
 
    return df
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 4: eliminar_outliers
# Responsabilidad: aplicar el filtro μ ± 3σ por categoría
#
# MATEMÁTICAS APLICADAS:
#
#   Límite inferior:  Li = μ - (THRESHOLD × σ)
#   Límite superior:  Ls = μ + (THRESHOLD × σ)
#
#   Un precio xᵢ es VÁLIDO si:  Li ≤ xᵢ ≤ Ls
#   Un precio xᵢ es OUTLIER si: xᵢ < Li  ó  xᵢ > Ls
#
#   Con THRESHOLD = 3 (configurado en .env):
#       Se conservan el 99.73% de precios válidos
#       Se eliminan el 0.27% de precios atípicos
# ──────────────────────────────────────────────────────────────────────────────
 
def eliminar_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Elimina filas cuyo precio se aleja más de N desviaciones estándar
    de la media de su categoría.
 
    FÓRMULA IMPLEMENTADA:
        Límite inferior: Li = μ - (threshold × σ)
        Límite superior: Ls = μ + (threshold × σ)
        Precio válido si: Li ≤ precio ≤ Ls
 
    ❌ PROHIBIDO usar: if precio < 10 (condicional arbitrario)
    ✅ CORRECTO usar: la fórmula estadística formal por categoría
 
    Parámetros:
        df        (pd.DataFrame): DataFrame con columnas 'price',
                                  'precio_media' y 'precio_std'.
        threshold (float)       : Número de desviaciones estándar
                                  para definir el rango válido.
                                  Por defecto: 3.0 (μ ± 3σ)
 
    Retorna:
        pd.DataFrame: DataFrame sin outliers de precio.
    """
 
    print(f"\n🔍 Aplicando filtro de outliers con threshold = {threshold}σ...")
    print(f"   Fórmula: precio válido si μ - {threshold}σ ≤ precio ≤ μ + {threshold}σ")
 
    filas_antes = len(df)
 
    # ── Calcular límites inferior y superior por fila ─────────────────────────
    # Cada fila ya tiene su μ y σ correspondiente a su categoría (del merge anterior)
 
    # Límite inferior: μ - 3σ
    limite_inferior = df["precio_media"] - (threshold * df["precio_std"])
 
    # Límite superior: μ + 3σ
    limite_superior = df["precio_media"] + (threshold * df["precio_std"])
 
    # ── Crear máscara booleana: True = precio válido, False = outlier ─────────
    # Un precio es válido si está DENTRO del rango [Li, Ls]
    mascara_validos = (df["price"] >= limite_inferior) & (df["price"] <= limite_superior)
 
    # ── Identificar outliers para el reporte ─────────────────────────────────
    outliers = df[~mascara_validos]  # ~ invierte la máscara: True donde era False
 
    print(f"\n   Ejemplos de outliers eliminados:")
    if len(outliers) > 0:
        # Mostramos los 5 outliers más extremos para verificar visualmente
        extremos = outliers.nlargest(5, "price")[["product_name", "category", "price", "precio_media", "precio_std"]]
        for _, row in extremos.iterrows():
            li = row["precio_media"] - threshold * row["precio_std"]
            ls = row["precio_media"] + threshold * row["precio_std"]
            print(f"   · {str(row['product_name'])[:40]:<40} precio={row['price']:>10.2f}  rango=[{li:.2f}, {ls:.2f}]")
 
    # ── Aplicar el filtro — solo conservamos filas donde la máscara es True ───
    df_limpio = df[mascara_validos].copy()
 
    filas_despues   = len(df_limpio)
    outliers_eliminados = filas_antes - filas_despues
    porcentaje      = (outliers_eliminados / filas_antes) * 100
 
    print(f"\n   📊 Resultado del filtro:")
    print(f"   · Filas antes del filtro  : {filas_antes:>10,}")
    print(f"   · Outliers eliminados     : {outliers_eliminados:>10,}  ({porcentaje:.2f}%)")
    print(f"   · Filas después del filtro: {filas_despues:>10,}")
 
    return df_limpio
 
 
# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN 5: guardar_resultado
# Responsabilidad: exportar el CSV limpio al directorio de procesados
# ──────────────────────────────────────────────────────────────────────────────
 
def guardar_resultado(df: pd.DataFrame, ruta_salida: str) -> None:
    """
    Guarda el DataFrame limpio como CSV en la ruta especificada.
 
    Elimina las columnas auxiliares de estadísticas (precio_media, precio_std)
    antes de exportar, ya que son columnas de trabajo, no del dataset final.
 
    Parámetros:
        df          (pd.DataFrame): DataFrame limpio sin outliers.
        ruta_salida (str)         : Ruta donde se guardará el CSV.
    """
 
    print(f"\n💾 Guardando dataset limpio...")
 
    # ── Crear el directorio si no existe ─────────────────────────────────────
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    # exist_ok=True evita error si la carpeta ya existe
 
    # ── Eliminar columnas auxiliares de estadísticas ──────────────────────────
    # Estas columnas fueron útiles para el cálculo pero no pertenecen al dataset final
    columnas_auxiliares = ["precio_media", "precio_std"]
    df_exportar = df.drop(columns=columnas_auxiliares, errors="ignore")
 
    # ── Guardar como CSV sin el índice de pandas ──────────────────────────────
    # index=False evita que pandas agregue una columna extra de números
    df_exportar.to_csv(ruta_salida, index=False, encoding="utf-8")
 
    print(f"   ✅ Dataset guardado en: {ruta_salida}")
    print(f"   ✅ Total de productos limpios: {len(df_exportar):,}")
    print(f"   ✅ Columnas exportadas: {', '.join(df_exportar.columns.tolist())}")
 
 
# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# Este bloque solo se ejecuta cuando corres el script directamente:
#   python src/01_outlier_cleaning.py
# NO se ejecuta si otro script importa las funciones de este módulo
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
 
    print("=" * 70)
    print("  NeuralPricer — PR #1: Limpieza Estadística de Outliers")
    print("  Grupo Almerco | Fórmula: σ = √(Σ(xᵢ - μ)² / N)")
    print("=" * 70)
 
    # ── PASO 1: Cargar el dataset y validar columnas (RF-01) ──────────────────
    df = cargar_dataset(DATA_RAW_PATH)
 
    # ── PASO 2: Normalizar nombres de columnas ────────────────────────────────
    df = normalizar_columnas(df)
 
    # ── PASO 3: Calcular μ y σ por categoría (fórmula formal) ────────────────
    df = calcular_estadisticas_por_categoria(df)
 
    # ── PASO 4: Eliminar outliers con el rango μ ± 3σ (RF-02) ────────────────
    df_limpio = eliminar_outliers(df, threshold=OUTLIER_THRESHOLD)
 
    # ── PASO 5: Guardar el dataset limpio ────────────────────────────────────
    guardar_resultado(df_limpio, DATA_PROCESSED_PATH)
 
    print("\n" + "=" * 70)
    print("  ✅ PR #1 completado exitosamente")
    print(f"  📁 Output: {DATA_PROCESSED_PATH}")
    print("=" * 70 + "\n")
 