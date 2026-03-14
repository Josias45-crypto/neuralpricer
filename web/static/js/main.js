/**
 * NeuralPricer v2 — Frontend JavaScript
 * Mejoras: categorías limpias, clústeres descriptivos, gráficos dinámicos
 */

const API_BASE = "";

// Al cargar la página
document.addEventListener("DOMContentLoaded", () => {
    Promise.all([
        cargarMetricas(),
        cargarOpciones(),
        cargarClusters(),
        cargarSimilares()
    ]);
});

function setActive(elemento) {
    document.querySelectorAll(".nav-item").forEach(i => i.classList.remove("active"));
    elemento.classList.add("active");
}

// ── Métricas ──────────────────────────────────────────────────
async function cargarMetricas() {
    try {
        const res   = await fetch(`${API_BASE}/api/metrics`);
        const datos = await res.json();

        actualizarMetrica("card-productos", datos.total_productos.toLocaleString(), "productos analizados");
        actualizarMetrica("card-r2",        datos.r2.toFixed(4),                    "varianza explicada");
        actualizarMetrica("card-mae",       `$${datos.mae.toFixed(2)}`,             "error promedio USD");
        actualizarMetrica("card-clusters",  datos.total_clusters,                   "grupos K-Means");
        actualizarMetrica("card-marcas",    datos.total_marcas.toLocaleString(),    "marcas únicas");
        actualizarMetrica("card-precio-avg",`$${datos.precio_promedio.toLocaleString()}`, "precio promedio μ");
    } catch(e) { console.error("Error métricas:", e); }
}

function actualizarMetrica(id, valor, sub) {
    const card = document.getElementById(id);
    if (!card) return;
    const v = card.querySelector(".metric-value");
    const s = card.querySelector(".metric-sub");
    if (v) { v.classList.remove("loading"); v.textContent = valor; }
    if (s) s.textContent = sub;
}

// ── Opciones para dropdowns ───────────────────────────────────
async function cargarOpciones() {
    try {
        const res   = await fetch(`${API_BASE}/api/opciones`);
        const datos = await res.json();

        // Marcas
        llenarSelect("select-marca", datos.marcas, "Selecciona una marca");

        // Categorías limpias
        llenarSelect("select-categoria", datos.categorias, "Selecciona una categoría");

        // Clústeres descriptivos con label legible
        const labels  = datos.clusters.map(c => c.label);
        const valores = datos.clusters.map(c => c.id);
        llenarSelect("select-cluster", labels, "Selecciona un clúster", valores);

        // Cuando se elige categoría → mostrar su gráfico automáticamente
        document.getElementById("select-categoria").addEventListener("change", (e) => {
            if (e.target.value) mostrarGraficoCategoria(e.target.value);
        });

        // Cuando se elige clúster → mostrar su gráfico automáticamente
        document.getElementById("select-cluster").addEventListener("change", (e) => {
            if (e.target.value !== "") mostrarGraficoCluster(parseInt(e.target.value));
        });

    } catch(e) { console.error("Error opciones:", e); }
}

function llenarSelect(id, opciones, placeholder, valores = null) {
    const sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = placeholder;
    opt.disabled = true; opt.selected = true;
    sel.appendChild(opt);
    opciones.forEach((op, i) => {
        const o = document.createElement("option");
        o.value = valores ? valores[i] : op;
        o.textContent = op;
        sel.appendChild(o);
    });
}

// ── Gráfico dinámico por categoría ───────────────────────────
async function mostrarGraficoCategoria(categoria) {
    const contenedor = document.getElementById("grafico-categoria");
    if (!contenedor) return;

    contenedor.innerHTML = `<div class="grafico-loading">Generando gráfico para <strong>${categoria}</strong>...</div>`;

    try {
        const res   = await fetch(`${API_BASE}/api/grafico/categoria/${encodeURIComponent(categoria)}`);
        const datos = await res.json();

        if (datos.error) {
            contenedor.innerHTML = `<div class="grafico-error">${datos.error}</div>`;
            return;
        }

        // Mostrar imagen base64 directamente en el navegador
        contenedor.innerHTML = `
            <div class="grafico-stats">
                <span>n = ${datos.n.toLocaleString()}</span>
                <span>μ = $${datos.mu.toLocaleString()}</span>
                <span>σ = $${datos.sigma.toLocaleString()}</span>
                <span>min = $${datos.min.toLocaleString()}</span>
                <span>max = $${datos.max.toLocaleString()}</span>
            </div>
            <img src="data:image/png;base64,${datos.imagen}"
                 alt="Gráfico ${categoria}" class="grafico-img">
        `;
    } catch(e) {
        contenedor.innerHTML = `<div class="grafico-error">Error al generar gráfico</div>`;
    }
}

// ── Gráfico dinámico por clúster ─────────────────────────────
async function mostrarGraficoCluster(clusterId) {
    const contenedor = document.getElementById("grafico-cluster");
    if (!contenedor) return;

    contenedor.innerHTML = `<div class="grafico-loading">Generando gráfico del Clúster ${clusterId}...</div>`;

    try {
        const res   = await fetch(`${API_BASE}/api/grafico/cluster/${clusterId}`);
        const datos = await res.json();

        if (datos.error) {
            contenedor.innerHTML = `<div class="grafico-error">${datos.error}</div>`;
            return;
        }

        contenedor.innerHTML = `
            <div class="grafico-stats">
                <span>Clúster ${datos.cluster_id}</span>
                <span>n = ${datos.n.toLocaleString()}</span>
                <span>μ = $${datos.mu.toLocaleString()}</span>
                <span>Marca líder: ${datos.marca_lider}</span>
            </div>
            <img src="data:image/png;base64,${datos.imagen}"
                 alt="Clúster ${clusterId}" class="grafico-img">
        `;
    } catch(e) {
        contenedor.innerHTML = `<div class="grafico-error">Error al generar gráfico</div>`;
    }
}

// ── Predicción ────────────────────────────────────────────────
async function predecirPrecio() {
    const marca     = document.getElementById("select-marca").value;
    const categoria = document.getElementById("select-categoria").value;
    const clusterId = document.getElementById("select-cluster").value;

    if (!marca || !categoria || clusterId === "") {
        alert("Por favor selecciona marca, categoría y clúster");
        return;
    }

    const btn = document.getElementById("btn-predict");
    btn.classList.add("loading");
    btn.innerHTML = "Calculando...";

    try {
        const res = await fetch(`${API_BASE}/api/predict`, {
            method : "POST",
            headers: { "Content-Type": "application/json" },
            body   : JSON.stringify({
                marca, categoria, cluster_id: parseInt(clusterId)
            })
        });

        const datos = await res.json();
        if (datos.error) throw new Error(datos.error);
        mostrarResultado(datos);

    } catch(e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.classList.remove("loading");
        btn.innerHTML = '<span class="btn-icon">◎</span> Predecir Precio Óptimo';
    }
}

function mostrarResultado(datos) {
    document.getElementById("result-empty").style.display = "none";
    document.getElementById("result-data").style.display  = "block";

    document.getElementById("result-price").textContent =
        `$${datos.precio_predicho.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("result-range-min").textContent =
        `$${datos.rango_min.toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
    document.getElementById("result-range-max").textContent =
        `$${datos.rango_max.toLocaleString("en-US", { minimumFractionDigits: 2 })}`;
    document.getElementById("result-confianza").textContent = datos.confianza || "69%";

    const listEl = document.getElementById("result-similares-list");
    listEl.innerHTML = "";
    if (datos.productos_similares && datos.productos_similares.length > 0) {
        datos.productos_similares.forEach(p => {
            const div = document.createElement("div");
            div.className = "similar-item";
            div.innerHTML = `<span class="similar-name">${p.nombre}</span><span class="similar-price">$${p.precio.toFixed(2)}</span>`;
            listEl.appendChild(div);
        });
    } else {
        listEl.innerHTML = '<div style="color:var(--text-dim);font-size:13px">No hay productos similares</div>';
    }
}

// ── Clústeres ─────────────────────────────────────────────────
async function cargarClusters() {
    try {
        const res   = await fetch(`${API_BASE}/api/clusters`);
        const datos = await res.json();
        const grid  = document.getElementById("clusters-grid");
        grid.innerHTML = "";

        const maxTotal = Math.max(...datos.clusters.map(c => c.total));

        datos.clusters.forEach(c => {
            const pct  = ((c.total / maxTotal) * 100).toFixed(1);
            const card = document.createElement("div");
            card.className = "cluster-card";
            card.innerHTML = `
                <div class="cluster-header">
                    <span class="cluster-id">Clúster ${c.cluster_id}</span>
                    <span class="cluster-total">${c.total.toLocaleString()} productos</span>
                </div>
                <div class="cluster-bar-wrap">
                    <div class="cluster-bar" style="width:0%" data-width="${pct}%"></div>
                </div>
                <div class="cluster-stats">
                    <div class="cluster-stat">
                        <div class="cluster-stat-label">Precio μ</div>
                        <div class="cluster-stat-value">$${c.precio_promedio.toLocaleString()}</div>
                    </div>
                    <div class="cluster-stat">
                        <div class="cluster-stat-label">Mínimo</div>
                        <div class="cluster-stat-value">$${c.precio_min.toLocaleString()}</div>
                    </div>
                    <div class="cluster-stat">
                        <div class="cluster-stat-label">Máximo</div>
                        <div class="cluster-stat-value">$${c.precio_max.toLocaleString()}</div>
                    </div>
                    <div class="cluster-stat">
                        <div class="cluster-stat-label">Marca líder</div>
                        <div class="cluster-stat-value">${c.marca_dominante}</div>
                    </div>
                </div>
            `;
            // Click en tarjeta genera el gráfico de ese clúster
            card.style.cursor = "pointer";
            card.addEventListener("click", () => {
                document.getElementById("grafico-cluster").scrollIntoView({ behavior: "smooth" });
                mostrarGraficoCluster(c.cluster_id);
            });
            grid.appendChild(card);
        });

        setTimeout(() => {
            document.querySelectorAll(".cluster-bar").forEach(b => {
                b.style.width = b.dataset.width;
            });
        }, 100);

    } catch(e) { console.error("Error clústeres:", e); }
}

// ── Similares ─────────────────────────────────────────────────
async function cargarSimilares() {
    try {
        const res   = await fetch(`${API_BASE}/api/similares`);
        const datos = await res.json();
        const tbody = document.getElementById("similares-tbody");
        tbody.innerHTML = "";

        if (!datos.similares || datos.similares.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="loading-msg">No hay datos de similitud</td></tr>';
            return;
        }

        datos.similares.forEach(par => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${par.producto_1 || "—"}</td>
                <td>${par.producto_2 || "—"}</td>
                <td><span class="sim-score">${(par.similitud * 100).toFixed(1)}%</span></td>
                <td style="font-family:var(--font-mono);color:var(--text-secondary)">${par.distancia.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });

    } catch(e) { console.error("Error similares:", e); }
}
