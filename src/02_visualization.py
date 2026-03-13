"""
================================================================================
NeuralPricer — PR #2: Visualización de la Campana de Gauss
================================================================================
Proyecto  : NeuralPricer — Radar de Precios de Mercado
Cliente   : Grupo Almerco
Autor     : [Tu nombre]
Fecha     : Marzo 2026
PR        : #2
--------------------------------------------------------------------------------
OBJETIVO:
    Generar gráficos de distribución de precios para que el gerente de ventas
    pueda ver visualmente dónde está el mercado para cada categoría.
 
    Se generan DOS tipos de gráficos por categoría:
        1. Histograma — muestra cómo se distribuyen los precios (campana de Gauss)
        2. Boxplot    — muestra la mediana, cuartiles y outliers restantes
 
QUÉ ES LA CAMPANA DE GAUSS:
    Cuando tienes muchos precios de productos similares, naturalmente
    se agrupan alrededor de un valor central (la media μ).
    Pocos productos son muy baratos o muy caros.
    Muchos productos tienen precios "normales".
    Eso forma la forma de campana característica.
 
    La línea roja vertical que agregamos marca exactamente dónde está
    ese centro (μ = precio promedio del mercado).
 
GRÁFICOS GENERADOS:
    - outputs/plots/histograma_<categoria>.png
    - outputs/plots/boxplot_<categoria>.png
 
CATEGORÍAS ANALIZADAS (según el documento):
    - Laptops
    - Monitores (Monitors)
 
ENTRADAS:
    - data/processed/clean.csv  (generado por PR #1)
 
SALIDAS:
    - outputs/plots/  (gráficos PNG)
 
DEPENDENCIAS:
    pip install pandas matplotlib seaborn python-dotenv
================================================================================
"""
 
# ------------------------------------------------------------------------------
# IMPORTACIONES
# ------------------------------------------------------------------------------
import os
import sys
 
import pandas as pd
import matplotlib.pyplot as plt   # Librería base de gráficos en Python
import matplotlib.ticker as mtick # Para formatear los ejes (ej: precios con $)
import seaborn as sns             # Librería de gráficos estadísticos sobre matplotlib
import numpy as np
 
from dotenv import load_dotenv
 
 
# ------------------------------------------------------------------------------
# CONFIGURACIÓN — variables desde .env
# ------------------------------------------------------------------------------
load_dotenv()
 
DATA_PROCESSED_PATH = os.getenv("DATA_PROCESSED_PATH", "data/processed/clean.csv")
PLOTS_PATH          = os.getenv("PLOTS_PATH",          "outputs/plots/")
 
# Resolución de los gráficos en DPI (dots per inch)
# 150 DPI = buena calidad para pantalla y presentaciones
PLOT_DPI = 150
 
# Categorías a analizar según el documento del proyecto
# Buscamos estas palabras clave dentro del nombre de la categoría
CATEGORIAS_OBJETIVO = ["Laptop", "Monitor"]
 
# Estilo visual de los gráficos
# "whitegrid" = fondo blanco con líneas de cuadrícula grises
sns.set_theme(style="whitegrid", palette="muted")
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 1: cargar_datos_limpios
# ------------------------------------------------------------------------------
def cargar_datos_limpios(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset limpio generado por el PR #1.
 
    Este archivo ya fue procesado — no tiene outliers de precio
    y tiene las columnas estándar del proyecto.
 
    Parámetros:
        ruta (str): Ruta al CSV limpio.
 
    Retorna:
        pd.DataFrame: Dataset limpio listo para graficar.
    """
 
    if not os.path.exists(ruta):
        print(f"\nERROR: No se encontró el archivo en: {ruta}")
        print("   Asegúrate de haber ejecutado primero el PR #1:")
        print("   python src/01_outlier_cleaning.py")
        sys.exit(1)
 
    print(f"\nCargando dataset limpio desde: {ruta}")
    df = pd.read_csv(ruta, low_memory=False)
    print(f"   Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"   Columnas: {', '.join(df.columns.tolist())}")
 
    return df
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 2: filtrar_categoria
# ------------------------------------------------------------------------------
def filtrar_categoria(df: pd.DataFrame, palabra_clave: str) -> pd.DataFrame:
    """
    Filtra el DataFrame para obtener solo los productos de una categoría.
 
    Usa búsqueda por palabra clave (case-insensitive) porque los nombres
    de categoría en el dataset son largos y variados.
 
    Ejemplo:
        "Laptop" encuentra: "Computers | Laptops", "Laptops & Accessories", etc.
 
    Parámetros:
        df           (pd.DataFrame): Dataset completo.
        palabra_clave (str)        : Palabra a buscar en la columna 'category'.
 
    Retorna:
        pd.DataFrame: Solo las filas de esa categoría.
    """
 
    # str.contains() busca la palabra dentro del string de categoría
    # case=False = no distingue mayúsculas/minúsculas
    # na=False   = las celdas vacías no causan error
    mascara = df["category"].str.contains(palabra_clave, case=False, na=False)
    df_filtrado = df[mascara].copy()
 
    print(f"\n   Categoría '{palabra_clave}': {len(df_filtrado):,} productos encontrados")
 
    if len(df_filtrado) == 0:
        print(f"   ADVERTENCIA: No se encontraron productos con '{palabra_clave}'")
        print(f"   Categorías disponibles (muestra):")
        for cat in df["category"].unique()[:10]:
            print(f"     · {cat}")
 
    return df_filtrado
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 3: calcular_estadisticas
# ------------------------------------------------------------------------------
def calcular_estadisticas(precios: pd.Series) -> dict:
    """
    Calcula las estadísticas descriptivas de una serie de precios.
 
    Estas estadísticas se usan para:
        - Dibujar la línea roja en μ (precio promedio del mercado)
        - Mostrar información en el título del gráfico
        - Entender la distribución de precios
 
    FÓRMULAS:
        μ (media)   = suma(xi) / N
        σ (std)     = sqrt( suma(xi - μ)² / N )
        mediana     = valor central cuando los datos están ordenados
        Q1          = percentil 25 (25% de precios están por debajo)
        Q3          = percentil 75 (75% de precios están por debajo)
 
    Parámetros:
        precios (pd.Series): Serie de precios numéricos.
 
    Retorna:
        dict: Diccionario con todas las estadísticas calculadas.
    """
 
    estadisticas = {
        "media"   : precios.mean(),           # μ — precio promedio del mercado
        "mediana" : precios.median(),          # valor central (menos sensible a extremos)
        "std"     : precios.std(ddof=0),       # σ — dispersión de precios
        "minimo"  : precios.min(),             # precio más bajo
        "maximo"  : precios.max(),             # precio más alto
        "q1"      : precios.quantile(0.25),   # percentil 25
        "q3"      : precios.quantile(0.75),   # percentil 75
        "count"   : len(precios),              # total de productos
    }
 
    return estadisticas
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 4: graficar_histograma
# ------------------------------------------------------------------------------
def graficar_histograma(
    precios     : pd.Series,
    stats       : dict,
    categoria   : str,
    ruta_salida : str
) -> None:
    """
    Genera un histograma de distribución de precios con:
        - Barras que muestran cuántos productos hay en cada rango de precio
        - Curva KDE (línea suavizada que estima la distribución)
        - Línea vertical ROJA marcando el precio promedio μ
        - Línea vertical AZUL marcando la mediana
        - Anotaciones con los valores exactos
 
    QUÉ ES UN HISTOGRAMA:
        Divide el rango de precios en "bins" (intervalos).
        Cada barra muestra cuántos productos tienen precio en ese intervalo.
        Si la mayoría de productos tiene precios similares, las barras del
        centro serán más altas — formando la campana de Gauss.
 
    QUÉ ES KDE (Kernel Density Estimation):
        Es una línea suavizada que estima la "forma" de la distribución.
        Nos dice qué tan probable es encontrar un producto a cada precio.
        El pico de la curva KDE es donde está la mayor concentración de precios.
 
    Parámetros:
        precios     (pd.Series): Precios de los productos.
        stats       (dict)     : Estadísticas precalculadas (media, std, etc).
        categoria   (str)      : Nombre de la categoría para el título.
        ruta_salida (str)      : Ruta donde guardar la imagen PNG.
    """
 
    # ── Crear el lienzo del gráfico ───────────────────────────────────────────
    # fig = figura completa, ax = área donde se dibuja el gráfico
    # figsize = tamaño en pulgadas (ancho, alto)
    fig, ax = plt.subplots(figsize=(12, 6))
 
    # ── Dibujar el histograma con curva KDE ───────────────────────────────────
    # kde=True       = agrega la curva suavizada encima de las barras
    # bins=50        = divide el rango en 50 intervalos
    # color          = color de las barras
    # alpha          = transparencia (0=invisible, 1=sólido) — para ver la KDE
    # edgecolor      = color del borde de cada barra
    sns.histplot(
        data      = precios,
        kde       = True,          # Curva de densidad suavizada
        bins      = 50,            # Número de barras en el histograma
        color     = "#4C72B0",     # Azul corporativo
        alpha     = 0.6,           # Semitransparente para ver la curva KDE
        edgecolor = "white",       # Borde blanco entre barras
        ax        = ax
    )
 
    # ── Línea vertical ROJA = precio promedio μ (requerimiento del documento) ─
    # Esta es la línea más importante: marca exactamente dónde está el mercado
    ax.axvline(
        x         = stats["media"],   # Posición en el eje X = precio medio
        color     = "red",            # Rojo para destacar (requerimiento del doc)
        linewidth = 2.5,              # Grosor de la línea
        linestyle = "--",             # Línea punteada
        label     = f"Media μ = ${stats['media']:.2f}"  # Texto en la leyenda
    )
 
    # ── Línea vertical VERDE = mediana ────────────────────────────────────────
    # La mediana es el precio "del medio" — 50% de productos cuestan más, 50% menos
    # Si la media y mediana están muy separadas, la distribución está sesgada
    ax.axvline(
        x         = stats["mediana"],
        color     = "green",
        linewidth = 2,
        linestyle = ":",
        label     = f"Mediana = ${stats['mediana']:.2f}"
    )
 
    # ── Anotación con el valor exacto de μ sobre la línea roja ───────────────
    # Esto muestra el número exacto directamente en el gráfico
    ax.annotate(
        text     = f"  μ = ${stats['media']:.2f}",   # Texto a mostrar
        xy       = (stats["media"], ax.get_ylim()[1] * 0.85),  # Posición
        color    = "red",
        fontsize = 11,
        fontweight = "bold"
    )
 
    # ── Títulos y etiquetas ───────────────────────────────────────────────────
    ax.set_title(
        f"Distribución de Precios — {categoria}\n"
        f"μ=${stats['media']:.2f}  |  σ=${stats['std']:.2f}  |  "
        f"n={stats['count']:,} productos",
        fontsize = 14,
        fontweight = "bold",
        pad = 15
    )
    ax.set_xlabel("Precio (USD)", fontsize = 12)
    ax.set_ylabel("Cantidad de Productos", fontsize = 12)
 
    # Formato del eje X con símbolo de dólar
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
 
    # ── Leyenda ───────────────────────────────────────────────────────────────
    ax.legend(fontsize = 11, loc = "upper right")
 
    # ── Ajustar márgenes y guardar ────────────────────────────────────────────
    plt.tight_layout()  # Ajusta automáticamente márgenes para que no se corten
 
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.savefig(ruta_salida, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()  # Cerrar figura para liberar memoria
 
    print(f"   Histograma guardado: {ruta_salida}")
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 5: graficar_boxplot
# ------------------------------------------------------------------------------
def graficar_boxplot(
    precios     : pd.Series,
    stats       : dict,
    categoria   : str,
    ruta_salida : str
) -> None:
    """
    Genera un gráfico de caja (Boxplot) que muestra la distribución
    de precios con cuartiles y valores atípicos.
 
    QUÉ ES UN BOXPLOT:
        Es un resumen visual de 5 números clave:
        ┌─────────────────────────────────────────────┐
        │  Mínimo ─── Q1 ─── Mediana ─── Q3 ─── Máximo  │
        └─────────────────────────────────────────────┘
 
        La "caja" va de Q1 a Q3 (rango intercuartílico = IQR)
            - Q1: 25% de los productos cuestan menos que esto
            - Q3: 75% de los productos cuestan menos que esto
            - IQR = Q3 - Q1 = rango donde está el 50% central
 
        Los "bigotes" se extienden hasta 1.5 × IQR desde la caja
        Los puntos fuera de los bigotes son outliers residuales
 
        La línea roja vertical marca la MEDIA (μ) del mercado
 
    Parámetros:
        precios     (pd.Series): Precios de los productos.
        stats       (dict)     : Estadísticas precalculadas.
        categoria   (str)      : Nombre de la categoría.
        ruta_salida (str)      : Ruta donde guardar la imagen PNG.
    """
 
    fig, ax = plt.subplots(figsize=(12, 5))
 
    # ── Dibujar el boxplot ────────────────────────────────────────────────────
    # orient="h"     = orientación horizontal (más fácil de leer precios)
    # width=0.5      = grosor de la caja
    # flierprops     = estilo de los puntos outliers
    sns.boxplot(
        x          = precios,
        orient     = "h",              # Horizontal
        color      = "#4C72B0",        # Azul
        width      = 0.5,
        flierprops = {                 # Estilo de los puntos outliers
            "marker"          : "o",
            "markerfacecolor" : "red",
            "markersize"      : 4,
            "alpha"           : 0.5
        },
        ax = ax
    )
 
    # ── Línea vertical ROJA = precio promedio μ (requerimiento del documento) ─
    ax.axvline(
        x         = stats["media"],
        color     = "red",
        linewidth = 2.5,
        linestyle = "--",
        label     = f"Media μ = ${stats['media']:.2f}"
    )
 
    # ── Anotaciones de Q1, Mediana, Q3 y Media ───────────────────────────────
    # Mostramos los valores exactos sobre el gráfico para facilitar lectura
    y_pos = 0.35  # Posición vertical de las anotaciones
 
    for valor, texto, color in [
        (stats["q1"],     f"Q1\n${stats['q1']:.0f}",         "#2ecc71"),
        (stats["mediana"],f"Mediana\n${stats['mediana']:.0f}", "#3498db"),
        (stats["media"],  f"Media μ\n${stats['media']:.0f}",  "red"),
        (stats["q3"],     f"Q3\n${stats['q3']:.0f}",         "#e67e22"),
    ]:
        ax.annotate(
            text       = texto,
            xy         = (valor, y_pos),
            ha         = "center",
            va         = "bottom",
            fontsize   = 9,
            color      = color,
            fontweight = "bold"
        )
 
    # ── Títulos y etiquetas ───────────────────────────────────────────────────
    ax.set_title(
        f"Boxplot de Precios — {categoria}\n"
        f"Q1=${stats['q1']:.2f}  |  Mediana=${stats['mediana']:.2f}  |  "
        f"Q3=${stats['q3']:.2f}  |  μ=${stats['media']:.2f}",
        fontsize   = 13,
        fontweight = "bold",
        pad        = 15
    )
    ax.set_xlabel("Precio (USD)", fontsize=12)
    ax.set_yticks([])  # Ocultar eje Y (no tiene significado en boxplot horizontal)
 
    # Formato del eje X con símbolo de dólar
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
 
    ax.legend(fontsize=11)
    plt.tight_layout()
 
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    plt.savefig(ruta_salida, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
 
    print(f"   Boxplot guardado   : {ruta_salida}")
 
 
# ------------------------------------------------------------------------------
# FUNCIÓN 6: imprimir_reporte
# ------------------------------------------------------------------------------
def imprimir_reporte(stats: dict, categoria: str) -> None:
    """
    Imprime en consola un reporte con las estadísticas de la categoría.
 
    Útil para verificar los resultados sin abrir los archivos PNG.
 
    Parámetros:
        stats     (dict): Estadísticas calculadas.
        categoria (str) : Nombre de la categoría.
    """
 
    print(f"\n   {'─'*50}")
    print(f"   Estadísticas: {categoria}")
    print(f"   {'─'*50}")
    print(f"   Productos analizados : {stats['count']:>10,}")
    print(f"   Precio mínimo        : ${stats['minimo']:>10.2f}")
    print(f"   Q1  (percentil 25)   : ${stats['q1']:>10.2f}")
    print(f"   Mediana              : ${stats['mediana']:>10.2f}")
    print(f"   Media μ (mercado)    : ${stats['media']:>10.2f}  <-- línea roja")
    print(f"   Q3  (percentil 75)   : ${stats['q3']:>10.2f}")
    print(f"   Precio máximo        : ${stats['maximo']:>10.2f}")
    print(f"   Desviación std σ     : ${stats['std']:>10.2f}")
    print(f"   {'─'*50}")
 
 
# ------------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == "__main__":
 
    print("=" * 70)
    print("  NeuralPricer -- PR #2: Visualización Campana de Gauss")
    print("  Grupo Almerco | Boxplots + Histogramas con línea roja en μ")
    print("=" * 70)
 
    # PASO 1 — Cargar el dataset limpio del PR #1
    df = cargar_datos_limpios(DATA_PROCESSED_PATH)
 
    # PASO 2 — Crear carpeta de outputs si no existe
    os.makedirs(PLOTS_PATH, exist_ok=True)
 
    # PASO 3 — Generar gráficos para cada categoría objetivo
    graficos_generados = 0
 
    for categoria in CATEGORIAS_OBJETIVO:
 
        print(f"\nProcesando categoría: {categoria}...")
 
        # Filtrar productos de esta categoría
        df_cat = filtrar_categoria(df, categoria)
 
        # Si no hay productos suficientes, saltar esta categoría
        if len(df_cat) < 10:
            print(f"   ADVERTENCIA: Muy pocos productos ({len(df_cat)}) — saltando")
            continue
 
        # Extraer la columna de precios
        precios = df_cat["price"].dropna()
 
        # Calcular estadísticas (μ, σ, mediana, cuartiles)
        stats = calcular_estadisticas(precios)
 
        # Imprimir reporte en consola
        imprimir_reporte(stats, categoria)
 
        # Nombre limpio para el archivo (sin caracteres especiales)
        nombre_archivo = categoria.lower().replace(" ", "_").replace("/", "_")
 
        # Generar y guardar histograma
        ruta_hist = os.path.join(PLOTS_PATH, f"histograma_{nombre_archivo}.png")
        graficar_histograma(precios, stats, categoria, ruta_hist)
 
        # Generar y guardar boxplot
        ruta_box = os.path.join(PLOTS_PATH, f"boxplot_{nombre_archivo}.png")
        graficar_boxplot(precios, stats, categoria, ruta_box)
 
        graficos_generados += 2
 
    # PASO 4 — Resumen final
    print("\n" + "=" * 70)
    print(f"  PR #2 completado exitosamente")
    print(f"  Graficos generados : {graficos_generados}")
    print(f"  Ubicacion          : {PLOTS_PATH}")
    print("=" * 70 + "\n")