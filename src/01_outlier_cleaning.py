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
 
              sigma = sqrt( sum(xi - mu)^2 / N )
 
    Donde:
        xi = precio individual de cada producto
        mu = media aritmética de precios en la categoría
        N  = total de productos en la categoría
        sigma = desviación estándar
 
    Rango válido de precios por categoría:
        [ mu - 3*sigma ,  mu + 3*sigma ]
 
    Todo precio fuera de ese rango es un OUTLIER y se elimina.
    ─────────────────────────────────────────────────────────────────
 
POR QUÉ 3 DESVIACIONES ESTÁNDAR:
    Regla Empírica (Campana de Gauss):
        - mu +- 1sigma cubre el 68.27% de los datos
        - mu +- 2sigma cubre el 95.45% de los datos
        - mu +- 3sigma cubre el 99.73% de los datos
    Precio fuera de mu +- 3sigma ocurre solo el 0.27% del tiempo.
 
REGLA DE NEGOCIO (del documento):
    PROHIBIDO : if precio < 10  (condicional arbitrario)
    OBLIGATORIO: usar la fórmula estadística formal por categoría
 
DATASET UTILIZADO:
    Amazon Electronics Products & Pricing (Kaggle - Datafiniti)
    Columnas mapeadas:
        'Product Name'  -> product_name
        'Selling Price' -> price
        'Brand Name'    -> brand
        'Category'      -> category
 
ENTRADAS:
    - data/raw/electronics.csv
 
SALIDAS:
    - data/processed/clean.csv
 
DEPENDENCIAS:
    pip install pandas numpy python-dotenv
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
 
 
# ------------------------------------------------------------------------------
# CONFIGURACIÓN — variables desde .env
# Nunca se hardcodean rutas en el código (buena práctica)
# ------------------------------------------------------------------------------
load_dotenv()
 
DATA_RAW_PATH       = os.getenv("DATA_RAW_PATH",       "data/raw/electronics.csv")
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/clean.csv")
OUTLIER_THRESHOLD   = float(os.getenv("OUTLIER_THRESHOLD", "3"))
 
 
# ------------------------------------------------------------------------------
# COLUMNAS REQUERIDAS (RF-01 del documento de requerimientos)
# ------------------------------------------------------------------------------
COLUMNAS_REQUERIDAS = ["Product Name", "Selling Price", "Brand Name", "Category"]
 
# Traducción: nombre en el CSV original -> nombre estándar del proyecto
MAPEO_COLUMNAS = {
    "Product Name"  : "product_name",
    "Selling Price" : "price",
    "Brand Name"    : "brand",
    "Category"      : "category",
}
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 1: cargar_dataset
# ------------------------------------------------------------------------------
def cargar_dataset(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset CSV y valida que contenga las columnas mínimas requeridas.
 
    Implementa RF-01: mostrar error descriptivo si falta alguna columna,
    sin lanzar una excepción genérica de Python.
 
    Parámetros:
        ruta (str): Ruta al archivo CSV de entrada.
 
    Retorna:
        pd.DataFrame: Dataset cargado con columnas validadas.
    """
 
    # Verificar que el archivo existe antes de intentar abrirlo
    if not os.path.exists(ruta):
        print(f"\nERROR: No se encontró el archivo en: {ruta}")
        print("   Asegúrate de haber colocado el CSV en data/raw/electronics.csv")
        sys.exit(1)
 
    print(f"\nCargando dataset desde: {ruta}")
 
    # low_memory=False evita warnings de tipos mixtos en columnas grandes
    df = pd.read_csv(ruta, low_memory=False)
 
    print(f"   Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
 
    # Validar que existan las columnas requeridas (RF-01)
    columnas_faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
 
    if columnas_faltantes:
        print(f"\nERROR: Faltan columnas requeridas en el dataset:")
        for col in columnas_faltantes:
            print(f"   - '{col}'")
        print(f"\n   Columnas disponibles en el CSV:")
        for col in df.columns.tolist():
            print(f"   · {col}")
        sys.exit(1)
 
    print(f"   Columnas requeridas verificadas correctamente")
    return df
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 2: normalizar_columnas
# ------------------------------------------------------------------------------
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra las columnas del CSV original al estándar del proyecto
    y limpia los tipos de datos.
 
    El precio viene como string con simbolos (ej: "$1,299.99").
    Esta función lo convierte a float eliminando caracteres no numéricos.
 
    Parámetros:
        df (pd.DataFrame): DataFrame con columnas originales del CSV.
 
    Retorna:
        pd.DataFrame: DataFrame con columnas renombradas y tipos limpios.
    """
 
    print("\nNormalizando columnas...")
 
    # Seleccionar solo columnas necesarias — .copy() evita SettingWithCopyWarning
    df = df[list(MAPEO_COLUMNAS.keys())].copy()
 
    # Renombrar al estándar del proyecto
    df = df.rename(columns=MAPEO_COLUMNAS)
 
    # Limpiar precios con función robusta
    # El dataset tiene precios concatenados como '74.99249.99' (dos precios pegados)
    # La función extrae solo el PRIMER precio válido de cada string
    #
    # Pasos:
    #   1. Eliminar símbolo $ y comas de miles
    #   2. Extraer el primer número con hasta 2 decimales usando regex
    #   3. Convertir a float — si no hay número válido retorna None (NaN)
    #
    # Ejemplos:
    #   '$74.99'      -> 74.99   (precio normal con símbolo)
    #   '74.99249.99' -> 74.99   (dos precios pegados — tomamos el primero)
    #   '1,299.99'    -> 1299.99 (precio con coma de miles)
    #   ''            -> None    (vacío — se eliminará como NaN)
 
    def extraer_primer_precio(valor: str) -> float:
        """
        Extrae el primer precio numérico válido de un string.
        Maneja precios concatenados, símbolos de moneda y comas de miles.
 
        Parámetros:
            valor (str): String con el precio crudo del CSV.
 
        Retorna:
            float: Primer precio válido encontrado, o None si no hay ninguno.
        """
        texto = str(valor).replace("$", "").replace(" ", "").replace(",", "")
        # Regex: busca el primer número entero o decimal con hasta 2 decimales
        match = re.match(r'^(\d+(?:\.\d{1,2})?)', texto)
        if match:
            return float(match.group(1))
        return None
 
    df["price"] = df["price"].apply(extraer_primer_precio)
 
    # Limpiar textos — eliminar espacios extra y normalizar mayúsculas
    df["product_name"] = df["product_name"].astype(str).str.strip()
    df["brand"]        = df["brand"].astype(str).str.strip().str.title()
    df["category"]     = df["category"].astype(str).str.strip()
 
    # Eliminar filas sin precio — sin precio no se puede calcular sigma
    filas_antes = len(df)
    df          = df.dropna(subset=["price"])
    eliminadas  = filas_antes - len(df)
 
    print(f"   Columnas renombradas al estándar del proyecto")
    print(f"   Precios convertidos a formato numérico")
    print(f"   Filas con precio nulo eliminadas: {eliminadas:,}")
 
    return df
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 3: calcular_estadisticas_por_categoria
#
# MATEMÁTICAS APLICADAS:
#   Media aritmética:             mu = suma(xi) / N
#   Desviación estándar pobl.:    sigma = sqrt( suma(xi - mu)^2 / N )
#
#   ddof=0 -> fórmula POBLACIONAL (dividir entre N)
#   Usamos ddof=0 porque tenemos TODOS los productos de la categoría,
#   no una muestra parcial.
# ------------------------------------------------------------------------------
def calcular_estadisticas_por_categoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la media (mu) y desviación estándar (sigma) de precios
    agrupados por categoría usando la fórmula estadística formal.
 
    FÓRMULA:
        mu    = suma(xi) / N
        sigma = sqrt( suma(xi - mu)^2 / N )   [poblacional, ddof=0]
 
    Parámetros:
        df (pd.DataFrame): DataFrame con columnas 'price' y 'category'.
 
    Retorna:
        pd.DataFrame: DataFrame con columnas adicionales 'precio_media' (mu)
                      y 'precio_std' (sigma) por categoría.
    """
 
    print("\nCalculando estadísticas por categoría (mu y sigma)...")
 
    # Agrupar por categoría y calcular mu y sigma
    estadisticas = df.groupby("category")["price"].agg(
        precio_media="mean",
        precio_std=lambda x: x.std(ddof=0)  # sigma poblacional
    ).reset_index()
 
    # Merge: cada producto queda con la mu y sigma de SU categoría
    df = df.merge(estadisticas, on="category", how="left")
 
    # Mostrar resumen de las primeras 10 categorías
    print(f"\n   {'Categoría':<40} {'mu (media)':>12} {'sigma (std)':>12}")
    print(f"   {'-'*40} {'-'*12} {'-'*12}")
    for _, row in estadisticas.head(10).iterrows():
        categoria = str(row["category"])[:39]
        print(f"   {categoria:<40} {row['precio_media']:>12.2f} {row['precio_std']:>12.2f}")
 
    if len(estadisticas) > 10:
        print(f"   ... y {len(estadisticas) - 10} categorías más")
 
    return df
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 4: eliminar_outliers
#
# MATEMÁTICAS APLICADAS:
#   Límite inferior: Li = mu - (threshold * sigma)
#   Límite superior: Ls = mu + (threshold * sigma)
#   Precio VÁLIDO  : Li <= precio <= Ls
#   Precio OUTLIER : precio < Li  o  precio > Ls
#
#   Con threshold=3: se conserva el 99.73% de precios válidos
# ------------------------------------------------------------------------------
def eliminar_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Elimina filas cuyo precio se aleja más de N desviaciones estándar
    de la media de su categoría.
 
    FÓRMULA:
        Li = mu - (threshold * sigma)
        Ls = mu + (threshold * sigma)
        Precio válido si: Li <= precio <= Ls
 
    PROHIBIDO : if precio < 10
    CORRECTO  : filtro estadístico formal por categoría
 
    Parámetros:
        df        (pd.DataFrame): DataFrame con 'price', 'precio_media', 'precio_std'.
        threshold (float)       : Número de sigmas para el rango válido. Default: 3.0
 
    Retorna:
        pd.DataFrame: DataFrame sin outliers.
    """
 
    print(f"\nAplicando filtro mu +- {threshold}*sigma por categoría...")
 
    filas_antes = len(df)
 
    # Límite inferior: mu - 3*sigma
    limite_inferior = df["precio_media"] - (threshold * df["precio_std"])
 
    # Límite superior: mu + 3*sigma
    limite_superior = df["precio_media"] + (threshold * df["precio_std"])
 
    # Máscara booleana: True = precio válido, False = outlier
    mascara_validos = (
        (df["price"] >= limite_inferior) &
        (df["price"] <= limite_superior)
    )
 
    # Mostrar ejemplos de outliers para verificación visual
    outliers = df[~mascara_validos]
    if len(outliers) > 0:
        print(f"\n   Ejemplos de outliers eliminados (precios más extremos):")
        extremos = outliers.nlargest(5, "price")[
            ["product_name", "category", "price", "precio_media", "precio_std"]
        ]
        for _, row in extremos.iterrows():
            li     = row["precio_media"] - threshold * row["precio_std"]
            ls     = row["precio_media"] + threshold * row["precio_std"]
            nombre = str(row["product_name"])[:35]
            print(f"   · {nombre:<35} precio={row['price']:>10.2f}  rango=[{li:.2f}, {ls:.2f}]")
 
    # Aplicar filtro — conservar solo filas donde la máscara es True
    df_limpio = df[mascara_validos].copy()
 
    filas_despues       = len(df_limpio)
    outliers_eliminados = filas_antes - filas_despues
    porcentaje          = (outliers_eliminados / filas_antes) * 100
 
    print(f"\n   Resultado del filtro:")
    print(f"   · Filas antes del filtro  : {filas_antes:>10,}")
    print(f"   · Outliers eliminados     : {outliers_eliminados:>10,}  ({porcentaje:.2f}%)")
    print(f"   · Filas después del filtro: {filas_despues:>10,}")
 
    return df_limpio
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 5: guardar_resultado
# ------------------------------------------------------------------------------
def guardar_resultado(df: pd.DataFrame, ruta_salida: str) -> None:
    """
    Guarda el DataFrame limpio como CSV eliminando columnas auxiliares
    que fueron usadas solo para el cálculo estadístico.
 
    Parámetros:
        df          (pd.DataFrame): DataFrame limpio sin outliers.
        ruta_salida (str)         : Ruta de salida del CSV.
    """
 
    print(f"\nGuardando dataset limpio...")
 
    # Crear directorio si no existe — exist_ok=True evita error si ya existe
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
 
    # Eliminar columnas auxiliares del cálculo (no pertenecen al dataset final)
    df_exportar = df.drop(columns=["precio_media", "precio_std"], errors="ignore")
 
    # index=False evita que pandas agregue columna extra de números
    df_exportar.to_csv(ruta_salida, index=False, encoding="utf-8")
 
    print(f"   Guardado en    : {ruta_salida}")
    print(f"   Total productos: {len(df_exportar):,}")
    print(f"   Columnas       : {', '.join(df_exportar.columns.tolist())}")
 
 
# ------------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL
# Este bloque solo se ejecuta cuando corres el script directamente:
#   python src/01_outlier_cleaning.py
# NO se ejecuta si otro módulo importa las funciones de este archivo
# ------------------------------------------------------------------------------
if __name__ == "__main__":
 
    print("=" * 70)
    print("  NeuralPricer -- PR #1: Limpieza Estadística de Outliers")
    print("  Grupo Almerco | Fórmula: sigma = sqrt(sum(xi - mu)^2 / N)")
    print("=" * 70)
 
    # PASO 1 — Cargar y validar columnas (RF-01)
    df = cargar_dataset(DATA_RAW_PATH)
 
    # PASO 2 — Normalizar nombres de columnas y tipos de datos
    df = normalizar_columnas(df)
 
    # PASO 3 — Calcular mu y sigma por categoría con fórmula formal
    df = calcular_estadisticas_por_categoria(df)
 
    # PASO 4 — Eliminar outliers con mu +- 3*sigma (RF-02)
    df_limpio = eliminar_outliers(df, threshold=OUTLIER_THRESHOLD)
 
    # PASO 5 — Guardar CSV limpio
    guardar_resultado(df_limpio, DATA_PROCESSED_PATH)
 
    print("\n" + "=" * 70)
    print("  PR #1 completado exitosamente")
    print(f"  Output: {DATA_PROCESSED_PATH}")
    print("=" * 70 + "\n")