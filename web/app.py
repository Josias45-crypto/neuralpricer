"""
================================================================================
NeuralPricer — Web Application (Flask Backend) v2
================================================================================
Mejoras v2:
    1. Categorías limpias — solo la primera parte del nombre
    2. Clústeres descriptivos — marca dominante + precio promedio
    3. Gráficos dinámicos — generados en tiempo real con Matplotlib
================================================================================
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica — necesario en servidor
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Estilo de gráficos oscuro corporativo
sns.set_theme(style="dark", palette="muted")
plt.rcParams.update({
    "figure.facecolor"  : "#111118",
    "axes.facecolor"    : "#0a0a0f",
    "axes.edgecolor"    : "#1e1e2e",
    "axes.labelcolor"   : "#8888aa",
    "text.color"        : "#e8e8f0",
    "xtick.color"       : "#8888aa",
    "ytick.color"       : "#8888aa",
    "grid.color"        : "#1e1e2e",
    "grid.alpha"        : 0.5,
})

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERED_PATH = os.path.join(BASE_DIR, "outputs", "data", "clustered.csv")
SIMILARES_PATH = os.path.join(BASE_DIR, "outputs", "data", "similitud_top.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "outputs", "models", "neuralpricer.keras")

df_clustered  = None
df_similares  = None
modelo        = None
encoder_brand = None
encoder_cat   = None
encoder_mfr   = None
encoder_prim  = None


def limpiar_categoria(categoria_str):
    """Toma solo la primera parte del nombre de categoría."""
    if not categoria_str or str(categoria_str) == "nan":
        return None
    return str(categoria_str).split(",")[0].strip()


def cargar_recursos():
    global df_clustered, df_similares, modelo
    global encoder_brand, encoder_cat, encoder_mfr, encoder_prim

    print("\n" + "="*60)
    print("  NeuralPricer v2 — Iniciando servidor Flask")
    print("="*60)

    if os.path.exists(CLUSTERED_PATH):
        df_clustered = pd.read_csv(CLUSTERED_PATH, low_memory=False)
        df_clustered = df_clustered.dropna(subset=["price", "brand", "category"])
        df_clustered["category_clean"] = df_clustered["category"].apply(limpiar_categoria)
        df_clustered = df_clustered.dropna(subset=["category_clean"])
        print(f"  ✅ Dataset cargado: {len(df_clustered):,} productos")
    else:
        print(f"  ❌ No se encontró: {CLUSTERED_PATH}")
        return False

    if os.path.exists(SIMILARES_PATH):
        df_similares = pd.read_csv(SIMILARES_PATH)
        print(f"  ✅ Similares cargados: {len(df_similares):,} pares")
    else:
        df_similares = pd.DataFrame()

    if os.path.exists(MODEL_PATH):
        modelo = keras.models.load_model(MODEL_PATH)
        print(f"  ✅ Modelo cargado")
    else:
        print(f"  ❌ No se encontró el modelo")
        return False

    encoder_brand = LabelEncoder()
    encoder_brand.fit(df_clustered["brand"].astype(str))

    encoder_cat = LabelEncoder()
    encoder_cat.fit(df_clustered["category"].astype(str))

    if "manufacturer" in df_clustered.columns:
        encoder_mfr = LabelEncoder()
        encoder_mfr.fit(df_clustered["manufacturer"].fillna("Unknown").astype(str))

    if "primary_category" in df_clustered.columns:
        encoder_prim = LabelEncoder()
        encoder_prim.fit(df_clustered["primary_category"].fillna("Unknown").astype(str))

    print(f"  ✅ Encoders construidos — {len(encoder_brand.classes_):,} marcas")
    print("="*60 + "\n")
    return True


def figura_a_base64(fig):
    """Convierte figura Matplotlib a string base64 para enviar al navegador."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def get_metrics():
    if df_clustered is None:
        return jsonify({"error": "Dataset no disponible"}), 500
    return jsonify({
        "total_productos"  : len(df_clustered),
        "total_marcas"     : df_clustered["brand"].nunique(),
        "total_categorias" : df_clustered["category_clean"].nunique(),
        "total_clusters"   : df_clustered["cluster_id"].nunique(),
        "precio_promedio"  : round(df_clustered["price"].mean(), 2),
        "precio_min"       : round(df_clustered["price"].min(), 2),
        "precio_max"       : round(df_clustered["price"].max(), 2),
        "mae"              : 256.50,
        "r2"               : 0.6889,
        "features"         : 5
    })


@app.route("/api/opciones")
def get_opciones():
    if df_clustered is None:
        return jsonify({"error": "Dataset no disponible"}), 500

    categorias = sorted(df_clustered["category_clean"].dropna().unique().tolist())
    marcas     = sorted(df_clustered["brand"].dropna().unique().tolist())

    clusters_info = []
    for cluster_id in sorted(df_clustered["cluster_id"].unique()):
        grupo       = df_clustered[df_clustered["cluster_id"] == cluster_id]
        marca_lider = grupo["brand"].mode().iloc[0] if len(grupo) > 0 else "N/A"
        precio_prom = round(grupo["price"].mean(), 0)
        clusters_info.append({
            "id"         : int(cluster_id),
            "label"      : f"Clúster {cluster_id} — {marca_lider} — ${precio_prom:.0f} prom.",
            "marca_lider": marca_lider,
            "precio_prom": precio_prom,
            "total"      : len(grupo)
        })

    return jsonify({
        "marcas"    : marcas,
        "categorias": categorias,
        "clusters"  : clusters_info
    })


@app.route("/api/clusters")
def get_clusters():
    if df_clustered is None:
        return jsonify({"error": "Dataset no disponible"}), 500

    resumen = df_clustered.groupby("cluster_id").agg(
        total      = ("product_name", "count"),
        precio_mu  = ("price", "mean"),
        precio_min = ("price", "min"),
        precio_max = ("price", "max")
    ).reset_index()

    marca_dominante = df_clustered.groupby("cluster_id")["brand"].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    marca_dominante.columns = ["cluster_id", "marca_dominante"]
    resumen = resumen.merge(marca_dominante, on="cluster_id")

    clusters = []
    for _, row in resumen.iterrows():
        clusters.append({
            "cluster_id"      : int(row["cluster_id"]),
            "total"           : int(row["total"]),
            "precio_promedio" : round(float(row["precio_mu"]), 2),
            "precio_min"      : round(float(row["precio_min"]), 2),
            "precio_max"      : round(float(row["precio_max"]), 2),
            "marca_dominante" : str(row["marca_dominante"])
        })

    return jsonify({"clusters": clusters})


@app.route("/api/similares")
def get_similares():
    if df_similares is None or df_similares.empty:
        return jsonify({"similares": []}), 200

    similares = []
    for _, row in df_similares.head(20).iterrows():
        similares.append({
            "producto_1": str(row.get("producto_1", "")),
            "producto_2": str(row.get("producto_2", "")),
            "similitud" : round(float(row.get("similitud_coseno", 0)), 4),
            "distancia" : round(float(row.get("distancia_coseno", 1)), 4)
        })

    return jsonify({"similares": similares})


@app.route("/api/grafico/categoria/<path:categoria>")
def grafico_categoria(categoria):
    """Genera histograma + boxplot para la categoría solicitada en tiempo real."""
    if df_clustered is None:
        return jsonify({"error": "Dataset no disponible"}), 500

    datos = df_clustered[df_clustered["category_clean"] == categoria]["price"]

    if len(datos) < 5:
        return jsonify({"error": f"Pocos datos para '{categoria}'"}), 400

    mu    = datos.mean()
    sigma = datos.std()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#111118")

    sns.histplot(datos, kde=True, ax=ax1, color="#6ee7b7", alpha=0.6,
                 line_kws={"linewidth": 2})
    ax1.axvline(mu, color="#f87171", linewidth=2, linestyle="--",
                label=f"μ = ${mu:,.2f}")
    ax1.axvline(mu - 3*sigma, color="#fbbf24", linewidth=1, linestyle=":",
                label="μ±3σ")
    ax1.axvline(mu + 3*sigma, color="#fbbf24", linewidth=1, linestyle=":")
    ax1.set_title(f"Distribución — {categoria[:40]}", fontsize=11, pad=12)
    ax1.set_xlabel("Precio (USD)")
    ax1.set_ylabel("Frecuencia")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2.boxplot(datos, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#6ee7b7", color="#6ee7b7", alpha=0.4),
                medianprops=dict(color="#f87171", linewidth=2),
                whiskerprops=dict(color="#8888aa"),
                capprops=dict(color="#8888aa"),
                flierprops=dict(marker="o", color="#8888aa", alpha=0.4, markersize=4))
    ax2.axhline(mu, color="#f87171", linewidth=1.5, linestyle="--",
                label=f"μ = ${mu:,.2f}")
    ax2.set_title(f"Boxplot — {categoria[:40]}", fontsize=11, pad=12)
    ax2.set_ylabel("Precio (USD)")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.legend(fontsize=9)

    fig.text(0.5, -0.02,
             f"n={len(datos):,}  μ=${mu:,.2f}  σ=${sigma:,.2f}  min=${datos.min():,.2f}  max=${datos.max():,.2f}",
             ha="center", fontsize=9, color="#8888aa", fontfamily="monospace")

    plt.tight_layout()

    return jsonify({
        "imagen"   : figura_a_base64(fig),
        "categoria": categoria,
        "n"        : len(datos),
        "mu"       : round(mu, 2),
        "sigma"    : round(sigma, 2),
        "min"      : round(float(datos.min()), 2),
        "max"      : round(float(datos.max()), 2)
    })


@app.route("/api/grafico/cluster/<int:cluster_id>")
def grafico_cluster(cluster_id):
    """Genera histograma + top marcas para el clúster solicitado en tiempo real."""
    if df_clustered is None:
        return jsonify({"error": "Dataset no disponible"}), 500

    datos  = df_clustered[df_clustered["cluster_id"] == cluster_id]
    if len(datos) < 3:
        return jsonify({"error": f"Pocos datos para clúster {cluster_id}"}), 400

    precios     = datos["price"]
    mu          = precios.mean()
    marca_lider = datos["brand"].mode().iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#111118")

    sns.histplot(precios, kde=True, ax=ax1, color="#a78bfa", alpha=0.6,
                 line_kws={"linewidth": 2})
    ax1.axvline(mu, color="#f87171", linewidth=2, linestyle="--",
                label=f"μ = ${mu:,.2f}")
    ax1.set_title(f"Clúster {cluster_id} — Precios\nMarca líder: {marca_lider}",
                  fontsize=11, pad=12)
    ax1.set_xlabel("Precio (USD)")
    ax1.set_ylabel("Frecuencia")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    top_marcas = datos["brand"].value_counts().head(8)
    colores    = ["#6ee7b7"] * len(top_marcas)
    if len(colores) > 0:
        colores[0] = "#f87171"
    ax2.barh(top_marcas.index, top_marcas.values, color=colores, alpha=0.8)
    ax2.set_title(f"Top Marcas — Clúster {cluster_id}", fontsize=11, pad=12)
    ax2.set_xlabel("Productos")
    ax2.invert_yaxis()

    fig.text(0.5, -0.02,
             f"Clúster {cluster_id}  |  n={len(datos):,}  |  μ=${mu:,.2f}  |  Líder: {marca_lider}",
             ha="center", fontsize=9, color="#8888aa", fontfamily="monospace")

    plt.tight_layout()

    return jsonify({
        "imagen"     : figura_a_base64(fig),
        "cluster_id" : cluster_id,
        "n"          : len(datos),
        "mu"         : round(mu, 2),
        "marca_lider": marca_lider
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if modelo is None:
        return jsonify({"error": "Modelo no disponible"}), 500

    datos      = request.get_json()
    marca      = str(datos.get("marca", ""))
    categoria  = str(datos.get("categoria", ""))
    cluster_id = int(datos.get("cluster_id", 0))

    if not marca or not categoria:
        return jsonify({"error": "Marca y categoría son requeridos"}), 400

    try:
        brand_enc = int(encoder_brand.transform([marca])[0]) \
            if marca in encoder_brand.classes_ else 0

        cat_orig_rows = df_clustered[df_clustered["category_clean"] == categoria]
        cat_orig = cat_orig_rows["category"].mode().iloc[0] \
            if len(cat_orig_rows) > 0 else categoria
        cat_enc = int(encoder_cat.transform([cat_orig])[0]) \
            if cat_orig in encoder_cat.classes_ else 0

        mfr_enc = prim_enc = 0

        if encoder_mfr is not None:
            marca_df = df_clustered[df_clustered["brand"] == marca]
            if len(marca_df) > 0 and "manufacturer" in marca_df.columns:
                mfr_val = str(marca_df["manufacturer"].mode().iloc[0])
                if mfr_val in encoder_mfr.classes_:
                    mfr_enc = int(encoder_mfr.transform([mfr_val])[0])

        if encoder_prim is not None:
            cat_df = df_clustered[df_clustered["category_clean"] == categoria]
            if len(cat_df) > 0 and "primary_category" in cat_df.columns:
                prim_val = str(cat_df["primary_category"].mode().iloc[0])
                if prim_val in encoder_prim.classes_:
                    prim_enc = int(encoder_prim.transform([prim_val])[0])

        X = np.array([[brand_enc, cat_enc, cluster_id, mfr_enc, prim_enc]],
                     dtype=np.float32)

        pred_norm    = modelo.predict(X, verbose=0)
        cluster_data = df_clustered[df_clustered["cluster_id"] == cluster_id]

        if len(cluster_data) > 0:
            precio_mu  = cluster_data["price"].mean()
            precio_std = cluster_data["price"].std()
            precio     = max(0, round(float(pred_norm[0][0]) * precio_std + precio_mu, 2))
        else:
            precio = round(float(df_clustered["price"].mean()), 2)

        mae = 256.50
        similares = []
        muestra = cluster_data[cluster_data["brand"].str.lower() == marca.lower()].head(3)
        if len(muestra) == 0:
            muestra = cluster_data.head(3)
        for _, prod in muestra.iterrows():
            similares.append({
                "nombre": str(prod["product_name"])[:60],
                "precio": round(float(prod["price"]), 2),
                "marca" : str(prod["brand"])
            })

        return jsonify({
            "precio_predicho"    : precio,
            "rango_min"          : max(0, round(precio - mae, 2)),
            "rango_max"          : round(precio + mae, 2),
            "marca"              : marca,
            "categoria"          : categoria,
            "cluster_id"         : cluster_id,
            "productos_similares": similares,
            "confianza"          : "69%",
            "mae"                : mae
        })

    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    if cargar_recursos():
        print("  🚀 http://localhost:5000\n")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("\n  ❌ Ejecuta primero los scripts ML en orden")
