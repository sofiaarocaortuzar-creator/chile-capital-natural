"""
Dashboard — Chile Capital Natural
==================================
Visualización interactiva de cobertura vegetal y cambio de uso de suelo
por región y comuna (MapBiomas Chile, 1999–2024).

Lanzar con:
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Asegurar que el root del proyecto esté en el path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etl.config import COMUNAS_GEOJSON_PATH, COVERAGE_CLASSES, DB_PATH

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chile — Capital Natural",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores por clase (para gráficas)
CLASS_COLORS = {meta["name_es"]: meta["color"] for meta in COVERAGE_CLASSES.values()}
CLASS_NAMES_ES = {cid: meta["name_es"] for cid, meta in COVERAGE_CLASSES.items()}

# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db_connection():
    """Conexión DuckDB persistente (read-only para el dashboard)."""
    if not DB_PATH.exists():
        return None
    con = duckdb.connect(str(DB_PATH), read_only=True)
    # spatial no es necesaria para las consultas del dashboard
    try:
        con.execute("LOAD spatial;")
    except Exception:
        pass
    return con


@st.cache_data
def load_geojson() -> dict | None:
    """Carga el GeoJSON de comunas para el mapa choropleth."""
    if not COMUNAS_GEOJSON_PATH.exists():
        return None
    with open(COMUNAS_GEOJSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_panel_data(_con) -> pd.DataFrame:
    """Carga el panel anual completo desde DuckDB."""
    return _con.execute("""
        SELECT p.*, c.name AS comuna_name, c.region_name, c.area_km2
        FROM panel_anual p
        JOIN comunas c ON c.cut_code = p.cut_code
        ORDER BY p.cut_code, p.year
    """).df()


@st.cache_data
def load_regions(_con) -> list[str]:
    """Lista de regiones disponibles."""
    return [
        r[0] for r in _con.execute(
            "SELECT DISTINCT region_name FROM comunas WHERE region_name IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]


@st.cache_data
def load_comunas_by_region(_con, region: str) -> pd.DataFrame:
    """Comunas de una región."""
    return _con.execute(
        "SELECT cut_code, name FROM comunas WHERE region_name = ? ORDER BY name",
        [region],
    ).df()


@st.cache_data
def load_coverage_detail(_con, cut_code: str) -> pd.DataFrame:
    """Detalle de cobertura por clase y año para una comuna."""
    return _con.execute("""
        SELECT
            vc.year,
            vc.class_id,
            cc.name_es,
            cc.color,
            cc.level1,
            vc.area_ha
        FROM vegetation_coverage vc
        JOIN coverage_classes cc ON cc.class_id = vc.class_id
        WHERE vc.cut_code = ?
        ORDER BY vc.year, vc.area_ha DESC
    """, [cut_code]).df()


# ---------------------------------------------------------------------------
# Helpers de visualización
# ---------------------------------------------------------------------------

def make_choropleth(
    df_year: pd.DataFrame,
    geojson: dict,
    column: str,
    title: str,
    color_scale: str = "Greens",
) -> go.Figure:
    """Mapa choropleth de Chile coloreado por `column`."""
    fig = px.choropleth_mapbox(
        df_year,
        geojson=geojson,
        locations="cut_code",
        featureidkey="properties.cut_code",
        color=column,
        color_continuous_scale=color_scale,
        mapbox_style="carto-positron",
        center={"lat": -35.5, "lon": -71.5},
        zoom=3.8,
        opacity=0.75,
        hover_name="comuna_name",
        hover_data={
            "cut_code": False,
            "region_name": True,
            column: ":.1f",
            "total_ha": ":.0f",
        },
        title=title,
        height=620,
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={"title": column},
    )
    return fig


def make_area_chart(df_comuna: pd.DataFrame, comuna_name: str) -> go.Figure:
    """Área apilada de cobertura por clase a lo largo del tiempo."""
    # Pivotar: años en X, clases en Y
    pivot = df_comuna.pivot_table(
        index="year", columns="name_es", values="area_ha", aggfunc="sum"
    ).fillna(0).reset_index()

    fig = go.Figure()
    for col in pivot.columns:
        if col == "year":
            continue
        color = CLASS_COLORS.get(col, "#888888")
        fig.add_trace(go.Scatter(
            x=pivot["year"],
            y=pivot[col],
            name=col,
            stackgroup="one",
            mode="none",
            fillcolor=color,
            hovertemplate=f"<b>{col}</b><br>%{{y:,.0f}} ha<extra></extra>",
        ))

    fig.update_layout(
        title=f"Cobertura vegetal — {comuna_name} (1999–2024)",
        xaxis_title="Año",
        yaxis_title="Hectáreas",
        hovermode="x unified",
        legend={"orientation": "v", "x": 1.02, "y": 1},
        height=440,
        margin={"r": 180},
    )
    return fig


def make_timeseries(
    df_panel: pd.DataFrame,
    cut_codes: list[str],
    column: str,
    label: str,
    comunas_names: dict[str, str],
) -> go.Figure:
    """Serie temporal de un indicador para múltiples comunas."""
    fig = go.Figure()
    for code in cut_codes:
        sub = df_panel[df_panel["cut_code"] == code].sort_values("year")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["year"],
            y=sub[column],
            name=comunas_names.get(code, code),
            mode="lines+markers",
            hovertemplate="%{x}: %{y:,.1f}<extra></extra>",
        ))
    fig.update_layout(
        title=f"Serie temporal — {label}",
        xaxis_title="Año",
        yaxis_title=label,
        hovermode="x unified",
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Pantalla de bienvenida si no hay DB
# ---------------------------------------------------------------------------

def show_setup_instructions():
    st.title("🌿 Chile — Capital Natural")
    st.error("⚠️ Base de datos no encontrada.")
    st.markdown(f"""
    La base de datos DuckDB aún no existe en `{DB_PATH}`.

    **Para inicializarla, ejecuta en tu terminal:**

    ```bash
    cd "{ROOT}"
    uv run python run.py pipeline
    ```

    El pipeline descargará los datos de MapBiomas Chile (~48 MB)
    y los cargará automáticamente. Tarda aprox. **3–5 minutos**.

    Luego vuelve a lanzar el dashboard:
    ```bash
    uv run streamlit run dashboard/app.py
    ```
    """)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def build_sidebar(con, df_panel: pd.DataFrame) -> dict:
    """Construye el panel lateral y retorna los parámetros de filtro."""
    st.sidebar.title("🌿 Chile Capital Natural")
    st.sidebar.caption("Datos: MapBiomas Chile Col. 2 (1999–2024)")
    st.sidebar.divider()

    # Región
    regions = load_regions(con)
    selected_region = st.sidebar.selectbox(
        "Región", options=["Todas"] + regions, index=0
    )

    # Comunas (filtradas por región)
    if selected_region != "Todas":
        df_c = load_comunas_by_region(con, selected_region)
        comunas_options = dict(zip(df_c["name"], df_c["cut_code"]))
    else:
        df_c = con.execute(
            "SELECT cut_code, name FROM comunas ORDER BY name"
        ).df()
        comunas_options = dict(zip(df_c["name"], df_c["cut_code"]))

    selected_comuna_names = st.sidebar.multiselect(
        "Comunas (para series temporales)",
        options=list(comunas_options.keys()),
        max_selections=6,
        placeholder="Selecciona hasta 6 comunas…",
    )
    selected_cut_codes = [comunas_options[n] for n in selected_comuna_names]

    # Año (para el mapa)
    years_available = sorted(df_panel["year"].unique().tolist())
    selected_year = st.sidebar.select_slider(
        "Año (mapa de cobertura)",
        options=years_available,
        value=years_available[-1],
    )

    # Indicador del mapa
    indicator_options = {
        "% Bosque": "pct_bosque",
        "% Vegetación Natural": "pct_natural",
        "Bosque (ha)": "bosque_ha",
        "Bosque Primario (ha)": "bosque_primario_ha",
        "Agricultura (ha)": "agricultura_ha",
        "Plantación Forestal (ha)": "plantacion_ha",
        "Matorral (ha)": "matorral_ha",
    }
    selected_indicator_label = st.sidebar.selectbox(
        "Indicador (mapa)", options=list(indicator_options.keys())
    )
    selected_indicator = indicator_options[selected_indicator_label]

    st.sidebar.divider()
    st.sidebar.caption(
        f"DB: `{DB_PATH.name}` | "
        f"{len(df_panel['cut_code'].unique())} comunas | "
        f"{len(years_available)} años"
    )

    return {
        "region": selected_region,
        "cut_codes": selected_cut_codes,
        "cut_names": dict(zip(selected_cut_codes, selected_comuna_names)),
        "year": selected_year,
        "indicator": selected_indicator,
        "indicator_label": selected_indicator_label,
    }


# ---------------------------------------------------------------------------
# Tabs del dashboard
# ---------------------------------------------------------------------------

def tab_mapa(df_panel: pd.DataFrame, geojson: dict | None, params: dict):
    """Tab 1: Mapa choropleth de cobertura por comuna."""
    st.subheader(f"🗺️ {params['indicator_label']} por comuna — {params['year']}")

    if geojson is None:
        st.warning("GeoJSON de comunas no encontrado. Ejecuta el pipeline primero.")
        return

    # Filtrar por año (y región si aplica)
    df_year = df_panel[df_panel["year"] == params["year"]].copy()
    if params["region"] != "Todas":
        df_year = df_year[df_year["region_name"] == params["region"]]

    if df_year.empty:
        st.info("No hay datos para el año y región seleccionados.")
        return

    col_scales = {
        "pct_bosque": "Greens",
        "pct_natural": "YlGn",
        "bosque_ha": "Greens",
        "bosque_primario_ha": "Greens",
        "agricultura_ha": "YlOrRd",
        "plantacion_ha": "YlOrBr",
        "matorral_ha": "BrBG",
    }
    scale = col_scales.get(params["indicator"], "Viridis")

    fig = make_choropleth(
        df_year, geojson,
        column=params["indicator"],
        title=f"{params['indicator_label']} — {params['year']}",
        color_scale=scale,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla resumen por región
    st.caption("Resumen por región")
    df_reg = (
        df_year.groupby("region_name")
        .agg(
            comunas=("cut_code", "count"),
            bosque_ha=("bosque_ha", "sum"),
            plantacion_ha=("plantacion_ha", "sum"),
            agricultura_ha=("agricultura_ha", "sum"),
            pct_bosque=("pct_bosque", "mean"),
        )
        .reset_index()
        .sort_values("bosque_ha", ascending=False)
    )
    df_reg.columns = [
        "Región", "Comunas", "Bosque (ha)", "Plantación (ha)",
        "Agricultura (ha)", "% Bosque promedio",
    ]
    st.dataframe(
        df_reg.style.format({
            "Bosque (ha)": "{:,.0f}",
            "Plantación (ha)": "{:,.0f}",
            "Agricultura (ha)": "{:,.0f}",
            "% Bosque promedio": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )


def tab_serie_temporal(df_panel: pd.DataFrame, params: dict):
    """Tab 2: Series temporales para comunas seleccionadas."""
    st.subheader("📈 Series temporales por comuna")

    if not params["cut_codes"]:
        st.info("👈 Selecciona una o más comunas en el panel lateral para ver series temporales.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig_bosque = make_timeseries(
            df_panel, params["cut_codes"],
            column="bosque_ha",
            label="Bosque Natural (ha)",
            comunas_names=params["cut_names"],
        )
        st.plotly_chart(fig_bosque, use_container_width=True)

    with col2:
        fig_pct = make_timeseries(
            df_panel, params["cut_codes"],
            column="pct_bosque",
            label="% Bosque",
            comunas_names=params["cut_names"],
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_agri = make_timeseries(
            df_panel, params["cut_codes"],
            column="agricultura_ha",
            label="Agricultura (ha)",
            comunas_names=params["cut_names"],
        )
        st.plotly_chart(fig_agri, use_container_width=True)

    with col4:
        fig_plant = make_timeseries(
            df_panel, params["cut_codes"],
            column="plantacion_ha",
            label="Plantación Forestal (ha)",
            comunas_names=params["cut_names"],
        )
        st.plotly_chart(fig_plant, use_container_width=True)


def tab_detalle_comuna(con, params: dict):
    """Tab 3: Composición detallada de una comuna (área apilada por clase)."""
    st.subheader("🔍 Composición de cobertura — detalle por clase")

    if not params["cut_codes"]:
        st.info("👈 Selecciona comunas en el panel lateral.")
        return

    selected_name = st.selectbox(
        "Ver detalle de:", options=list(params["cut_names"].values())
    )
    cut_code = {v: k for k, v in params["cut_names"].items()}[selected_name]

    df_detail = load_coverage_detail(con, cut_code)
    if df_detail.empty:
        st.warning("No hay datos de detalle para esta comuna.")
        return

    # Área apilada
    fig_area = make_area_chart(df_detail, selected_name)
    st.plotly_chart(fig_area, use_container_width=True)

    # Snapshot del año seleccionado
    df_snap = df_detail[df_detail["year"] == params["year"]].copy()
    if not df_snap.empty:
        st.caption(f"Distribución en {params['year']}")
        total = df_snap["area_ha"].sum()
        df_snap = df_snap.sort_values("area_ha", ascending=False).copy()
        df_snap["% del total"] = df_snap["area_ha"] / total * 100

        fig_pie = px.pie(
            df_snap,
            names="name_es",
            values="area_ha",
            color="name_es",
            color_discrete_map=CLASS_COLORS,
            title=f"Cobertura en {params['year']} — {selected_name}",
            hole=0.35,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=420, showlegend=False)

        col_pie, col_tbl = st.columns([1, 1])
        with col_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_tbl:
            st.dataframe(
                df_snap[["name_es", "area_ha", "% del total"]].rename(columns={
                    "name_es": "Clase",
                    "area_ha": "Área (ha)",
                    "% del total": "% Total",
                }).style.format({"Área (ha)": "{:,.1f}", "% Total": "{:.1f}%"}),
                use_container_width=True,
                hide_index=True,
            )


def tab_datos(df_panel: pd.DataFrame, params: dict):
    """Tab 4: Tabla de datos y exportación CSV/Parquet."""
    st.subheader("📊 Datos — Panel anual")

    # Filtros
    if params["region"] != "Todas":
        df_show = df_panel[df_panel["region_name"] == params["region"]].copy()
    else:
        df_show = df_panel.copy()

    if params["cut_codes"]:
        df_show = df_show[df_show["cut_code"].isin(params["cut_codes"])]

    st.caption(f"{len(df_show):,} registros ({df_show['cut_code'].nunique()} comunas × {df_show['year'].nunique()} años)")

    # Columnas a mostrar
    display_cols = [
        "cut_code", "comuna_name", "region_name", "year",
        "bosque_ha", "bosque_primario_ha", "plantacion_ha",
        "matorral_ha", "agricultura_ha", "agua_ha",
        "total_ha", "pct_bosque", "pct_natural",
    ]
    st.dataframe(
        df_show[display_cols].style.format({
            "bosque_ha": "{:,.1f}",
            "bosque_primario_ha": "{:,.1f}",
            "plantacion_ha": "{:,.1f}",
            "matorral_ha": "{:,.1f}",
            "agricultura_ha": "{:,.1f}",
            "agua_ha": "{:,.1f}",
            "total_ha": "{:,.1f}",
            "pct_bosque": "{:.2f}%",
            "pct_natural": "{:.2f}%",
        }),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    # Exportación
    col_csv, col_parquet = st.columns([1, 4])
    with col_csv:
        csv_bytes = df_show[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Descargar CSV",
            data=csv_bytes,
            file_name="chile_capital_natural_panel.csv",
            mime="text/csv",
        )
    with col_parquet:
        import io
        buf = io.BytesIO()
        df_show[display_cols].to_parquet(buf, index=False)
        st.download_button(
            label="⬇️ Descargar Parquet",
            data=buf.getvalue(),
            file_name="chile_capital_natural_panel.parquet",
            mime="application/octet-stream",
        )


# ---------------------------------------------------------------------------
# Carga de datos — Deforestación y Riesgo Hídrico
# ---------------------------------------------------------------------------

@st.cache_data
def load_deforestation_data(_con) -> pd.DataFrame:
    try:
        return _con.execute("""
            SELECT d.*, c.name AS comuna_name, c.region_name
            FROM deforestation_events d
            JOIN comunas c ON c.cut_code = d.cut_code
            ORDER BY d.cut_code, d.year_from
        """).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_water_risk_data(_con) -> pd.DataFrame:
    try:
        return _con.execute("""
            SELECT w.*, c.name AS comuna_name, c.region_name, c.area_km2
            FROM water_risk w
            JOIN comunas c ON c.cut_code = w.cut_code
            ORDER BY w.bws_score DESC
        """).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_casen_data(_con) -> pd.DataFrame:
    try:
        return _con.execute("""
            SELECT cs.*, c.name AS comuna_name, c.region_name
            FROM casen_comunal cs
            JOIN comunas c ON c.cut_code = cs.cut_code
            ORDER BY cs.cut_code, cs.year
        """).df()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tabs — Deforestación y Riesgo Hídrico
# ---------------------------------------------------------------------------

TRANSITION_COLORS = {
    "deforestation": "#c0392b",
    "plantation":    "#e67e22",
    "degradation":   "#f39c12",
    "reforestation": "#27ae60",
}
TRANSITION_LABELS = {
    "deforestation": "🔴 Deforestación",
    "plantation":    "🟠 Conversión a plantación",
    "degradation":   "🟡 Degradación forestal",
    "reforestation": "🟢 Reforestación",
}

RISK_INDICATORS = {
    "bws_score": "Estrés hídrico",
    "bwd_score": "Agotamiento de agua",
    "iav_score": "Variabilidad interanual",
    "sev_score": "Variabilidad estacional",
    "drr_score": "Riesgo de sequía",
    "rfr_score": "Riesgo inundación fluvial",
    "cfr_score": "Riesgo inundación costera",
}


def tab_deforestation(df_defor: pd.DataFrame, geojson: dict | None, params: dict):
    """Tab 5: Deforestación y transiciones forestales."""
    st.subheader("🌲 Deforestación y transiciones forestales (1999–2024)")

    if df_defor.empty:
        st.warning(
            "Datos de deforestación no disponibles. "
            "Ejecuta `uv run python run.py pipeline --load-only` para procesarlos."
        )
        return

    # Filtros
    tipo_sel = st.multiselect(
        "Tipo de transición",
        options=list(TRANSITION_LABELS.keys()),
        default=["deforestation", "plantation"],
        format_func=lambda x: TRANSITION_LABELS[x],
    )
    df_f = df_defor[df_defor["transition_type"].isin(tipo_sel)].copy()
    if params["region"] != "Todas":
        df_f = df_f[df_f["region_name"] == params["region"]]

    # --- Serie temporal nacional ---
    st.markdown("##### Hectáreas por año — Total nacional (o región seleccionada)")
    df_ts = (
        df_f.groupby(["year_from", "transition_type"])["area_ha"]
        .sum().reset_index()
        .rename(columns={"year_from": "Año", "area_ha": "Hectáreas"})
    )
    fig_ts = px.bar(
        df_ts, x="Año", y="Hectáreas", color="transition_type",
        color_discrete_map=TRANSITION_COLORS,
        barmode="stack",
        labels={"transition_type": "Tipo"},
        title="Transiciones forestales anuales",
        height=380,
    )
    fig_ts.for_each_trace(lambda t: t.update(name=TRANSITION_LABELS.get(t.name, t.name)))
    st.plotly_chart(fig_ts, use_container_width=True)

    # --- Mapa de deforestación acumulada ---
    col_map, col_rank = st.columns([3, 2])

    with col_map:
        if geojson:
            df_acum = (
                df_f[df_f["transition_type"] == "deforestation"]
                .groupby("cut_code")["area_ha"].sum()
                .reset_index()
                .rename(columns={"area_ha": "defor_ha"})
            )
            if not df_acum.empty:
                fig_map = px.choropleth_mapbox(
                    df_acum, geojson=geojson,
                    locations="cut_code", featureidkey="properties.cut_code",
                    color="defor_ha",
                    color_continuous_scale="Reds",
                    mapbox_style="carto-positron",
                    center={"lat": -35.5, "lon": -71.5}, zoom=3.8,
                    opacity=0.75,
                    title="Deforestación acumulada (ha)",
                    height=480,
                )
                fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
                st.plotly_chart(fig_map, use_container_width=True)

    with col_rank:
        st.markdown("##### Top 15 comunas con más deforestación")
        df_rank = (
            df_f[df_f["transition_type"] == "deforestation"]
            .groupby(["cut_code", "comuna_name", "region_name"])["area_ha"]
            .sum().reset_index()
            .sort_values("area_ha", ascending=False)
            .head(15)
        )
        fig_rank = px.bar(
            df_rank, x="area_ha", y="comuna_name", orientation="h",
            color="region_name",
            labels={"area_ha": "ha deforestadas", "comuna_name": ""},
            height=480,
        )
        fig_rank.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_rank, use_container_width=True)

    # --- Serie temporal por comuna seleccionada ---
    if params["cut_codes"]:
        st.markdown("##### Serie temporal por comunas seleccionadas")
        df_com = df_f[df_f["cut_code"].isin(params["cut_codes"])].copy()
        df_com["comuna"] = df_com["cut_code"].map(params["cut_names"])
        fig_com = px.line(
            df_com.groupby(["year_from", "comuna", "transition_type"])["area_ha"]
            .sum().reset_index(),
            x="year_from", y="area_ha", color="comuna",
            line_dash="transition_type",
            labels={"year_from": "Año", "area_ha": "ha", "transition_type": "Tipo"},
            height=360,
        )
        st.plotly_chart(fig_com, use_container_width=True)

    # --- Exportación ---
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar datos de transiciones (CSV)", csv,
                       "deforestacion_chile.csv", "text/csv")


def tab_water_risk(df_risk: pd.DataFrame, geojson: dict | None, params: dict):
    """Tab 6: Riesgo hídrico (WRI Aqueduct 4.0)."""
    st.subheader("💧 Riesgo hídrico por comuna — WRI Aqueduct 4.0")
    st.caption(
        "Baseline histórico 2000–2019. Score 0–5: "
        "0=Bajo · 1=Bajo-Medio · 2=Medio · 3=Medio-Alto · 4=Alto · 5=Extremadamente alto. "
        "Metodología: promedio ponderado por área de sub-cuencas HydroSHEDS."
    )

    if df_risk.empty:
        st.warning(
            "Datos de riesgo hídrico no disponibles. "
            "Ejecuta `uv run python run.py pipeline --load-only` para cargarlos."
        )
        return

    # Filtro región
    df_f = df_risk.copy()
    if params["region"] != "Todas":
        df_f = df_f[df_f["region_name"] == params["region"]]

    # Selector de indicador
    indicator = st.selectbox(
        "Indicador de riesgo",
        options=list(RISK_INDICATORS.keys()),
        format_func=lambda x: RISK_INDICATORS[x],
    )
    ind_label = RISK_INDICATORS[indicator]

    col_map, col_dist = st.columns([3, 2])

    with col_map:
        if geojson and indicator in df_f.columns:
            df_plot = df_f[["cut_code", indicator]].dropna()
            fig = px.choropleth_mapbox(
                df_plot, geojson=geojson,
                locations="cut_code", featureidkey="properties.cut_code",
                color=indicator,
                range_color=[0, 5],
                color_continuous_scale=[
                    [0.0,  "#2166ac"],  # Bajo
                    [0.25, "#74add1"],
                    [0.5,  "#fee090"],
                    [0.75, "#f46d43"],
                    [1.0,  "#a50026"],  # Extremo
                ],
                mapbox_style="carto-positron",
                center={"lat": -35.5, "lon": -71.5}, zoom=3.8,
                opacity=0.8,
                hover_name="cut_code",
                title=f"{ind_label} (score 0–5)",
                height=500,
            )
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        st.markdown(f"##### Distribución — {ind_label}")
        df_valid = df_f[[indicator, "region_name"]].dropna()
        fig_hist = px.histogram(
            df_valid, x=indicator, color="region_name",
            nbins=20, opacity=0.75,
            labels={indicator: "Score (0–5)", "region_name": "Región"},
            title=f"Distribución por región",
            height=260,
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Top 10 comunas más en riesgo
        st.markdown("##### Top 10 comunas en mayor riesgo")
        top10 = (
            df_f[["comuna_name", "region_name", indicator]]
            .dropna()
            .sort_values(indicator, ascending=False)
            .head(10)
        )
        st.dataframe(
            top10.rename(columns={
                "comuna_name": "Comuna",
                "region_name": "Región",
                indicator: f"Score {ind_label}",
            }).style.format({f"Score {ind_label}": "{:.2f}"}),
            hide_index=True, use_container_width=True,
        )

    # --- Radar: perfil hídrico de comunas seleccionadas ---
    if params["cut_codes"]:
        st.markdown("##### Perfil hídrico comparativo — comunas seleccionadas")
        available_inds = [i for i in RISK_INDICATORS if i in df_risk.columns]
        df_sel = df_risk[df_risk["cut_code"].isin(params["cut_codes"])].copy()
        df_sel["label"] = df_sel["cut_code"].map(params["cut_names"])

        fig_radar = go.Figure()
        categories = [RISK_INDICATORS[i] for i in available_inds]
        for _, row in df_sel.iterrows():
            vals = [row.get(i, 0) or 0 for i in available_inds]
            vals += vals[:1]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + categories[:1],
                fill="toself",
                name=row.get("label", row["cut_code"]),
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="Perfil de riesgo hídrico (score 0–5)",
            height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- Tabla completa ---
    st.markdown("##### Tabla completa de indicadores por comuna")
    display_cols = ["comuna_name", "region_name"] + [
        c for c in RISK_INDICATORS if c in df_f.columns
    ]
    df_tbl = df_f[display_cols].dropna(subset=[indicator]).sort_values(indicator, ascending=False)
    st.dataframe(
        df_tbl.rename(columns={"comuna_name": "Comuna", "region_name": "Región",
                                **{k: v for k, v in RISK_INDICATORS.items()}})
        .style.format({v: "{:.2f}" for v in RISK_INDICATORS.values() if v in df_tbl.columns}),
        use_container_width=True, hide_index=True, height=380,
    )

    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar datos de riesgo hídrico (CSV)", csv,
                       "riesgo_hidrico_chile.csv", "text/csv")


# ---------------------------------------------------------------------------
# Tab — CASEN Socioeconómico
# ---------------------------------------------------------------------------

CASEN_INDICATORS = {
    "tasa_pobreza":       "Pobreza por ingresos (%)",
    "tasa_pobreza_multi": "Pobreza multidimensional (%)",
    "tasa_ocupacion":     "Tasa de ocupación (%)",
    "esc_promedio":       "Escolaridad promedio (años)",
    "ypc_promedio":       "Ingreso per cápita promedio (CLP)",
    "pct_agua_red":       "Acceso a agua de red (%)",
    "pct_alcantarillado": "Acceso a alcantarillado (%)",
    "pct_indigena":       "Población indígena (%)",
    "pct_urbano":         "Población urbana (%)",
}


def tab_casen(df_casen: pd.DataFrame, geojson: dict | None, params: dict):
    if df_casen.empty:
        st.info(
            "Datos CASEN no disponibles. Ejecuta el pipeline para cargarlos: "
            "`python run.py pipeline`"
        )
        return

    st.markdown(
        "Indicadores socioeconómicos comunales a partir de la Encuesta CASEN "
        "(Ministerio de Desarrollo Social). Años disponibles: **2017 · 2020 · 2022**. "
        "Las comunas con muestra < 50 personas se marcan como no representativas."
    )

    # Filtros
    col_ind, col_year, col_repr = st.columns(3)
    with col_ind:
        indicator = st.selectbox(
            "Indicador", list(CASEN_INDICATORS.keys()),
            format_func=lambda k: CASEN_INDICATORS[k], key="casen_ind"
        )
    with col_year:
        years_avail = sorted(df_casen["year"].unique())
        year_sel = st.selectbox("Año CASEN", years_avail,
                                index=len(years_avail) - 1, key="casen_year")
    with col_repr:
        only_repr = st.checkbox("Solo comunas representativas (n ≥ 50)", value=True, key="casen_repr")

    df_f = df_casen[df_casen["year"] == year_sel].copy()
    if only_repr and "representativa" in df_f.columns:
        df_f = df_f[df_f["representativa"] == True]

    ind_label = CASEN_INDICATORS[indicator]

    # Mapa
    col_map, col_dist = st.columns([3, 2])
    with col_map:
        if geojson and indicator in df_f.columns:
            is_pct = "pct" in indicator or "tasa" in indicator
            color_scale = "RdYlGn_r" if "pobreza" in indicator else (
                "RdYlGn" if ("ocupacion" in indicator or "agua" in indicator
                             or "alcantarillado" in indicator or "esc" in indicator) else "Blues"
            )
            fig = px.choropleth_mapbox(
                df_f.dropna(subset=[indicator]),
                geojson=geojson,
                locations="cut_code",
                featureidkey="properties.GID_3",
                color=indicator,
                color_continuous_scale=color_scale,
                mapbox_style="carto-positron",
                zoom=3.5, center={"lat": -37, "lon": -71},
                opacity=0.75,
                hover_name="comuna_name",
                hover_data={"region_name": True, indicator: ":.1f",
                            "cut_code": False, "n_obs": True},
                labels={indicator: ind_label, "n_obs": "Muestra"},
                title=f"{ind_label} — {year_sel}",
                height=450,
            )
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("GeoJSON no disponible para el mapa.")

    with col_dist:
        st.markdown(f"##### Distribución — {ind_label}")
        fig_hist = px.histogram(
            df_f.dropna(subset=[indicator]),
            x=indicator,
            nbins=30,
            color_discrete_sequence=["#3498db"],
            labels={indicator: ind_label},
            height=200,
        )
        fig_hist.update_layout(margin={"t": 10, "b": 40}, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Top 10 con peor / mejor indicador
        ascending = "pobreza" in indicator   # mayor pobreza = peor
        top10 = df_f.dropna(subset=[indicator]).nsmallest(10, indicator) \
            if not ascending else df_f.dropna(subset=[indicator]).nlargest(10, indicator)
        label_top = "Mayor valor" if ascending else "Menor valor"
        st.markdown(f"**{label_top} — top 10 comunas**")
        st.dataframe(
            top10[["comuna_name", "region_name", indicator, "n_obs"]]
            .rename(columns={"comuna_name": "Comuna", "region_name": "Región",
                             indicator: ind_label, "n_obs": "Muestra"})
            .style.format({ind_label: "{:.1f}"}),
            use_container_width=True, hide_index=True, height=280,
        )

    # Evolución temporal — comparar años disponibles
    st.markdown("---")
    st.markdown("##### Evolución entre encuestas — comparación temporal")

    region_sel = params.get("region")
    if region_sel:
        df_trend = df_casen[df_casen["region_name"] == region_sel]
    else:
        df_trend = df_casen.copy()

    if only_repr and "representativa" in df_trend.columns:
        df_trend = df_trend[df_trend["representativa"] == True]

    if indicator in df_trend.columns and not df_trend.empty:
        df_trend_agg = (
            df_trend.groupby("year")[indicator]
            .agg(["mean", "median", "std"])
            .reset_index()
            .rename(columns={"mean": "Promedio", "median": "Mediana", "std": "Desv. estándar"})
        )
        fig_trend = px.line(
            df_trend_agg, x="year", y=["Promedio", "Mediana"],
            markers=True,
            labels={"year": "Año CASEN", "value": ind_label, "variable": ""},
            title=f"{ind_label} — {'Región: ' + region_sel if region_sel else 'Chile'} · evolución",
            color_discrete_sequence=["#2980b9", "#27ae60"],
            height=320,
        )
        fig_trend.update_xaxes(tickvals=years_avail)
        st.plotly_chart(fig_trend, use_container_width=True)

    # Tabla exportable
    st.markdown("##### Datos completos")
    display_cols = ["comuna_name", "region_name", "year", "n_obs", "representativa"] + [
        c for c in CASEN_INDICATORS if c in df_casen.columns
    ]
    df_tbl = df_casen[display_cols].sort_values(["cut_code", "year"])
    st.dataframe(
        df_tbl.rename(columns={"comuna_name": "Comuna", "region_name": "Región",
                                "year": "Año", "n_obs": "Muestra", "representativa": "Representativa",
                                **{k: v for k, v in CASEN_INDICATORS.items()}}),
        use_container_width=True, hide_index=True, height=350,
    )
    csv = df_tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar datos CASEN comunal (CSV)", csv,
                       "casen_comunal_chile.csv", "text/csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    con = get_db_connection()

    if con is None:
        show_setup_instructions()
        return

    geojson  = load_geojson()
    df_panel = load_panel_data(con)

    if df_panel.empty:
        st.error("La base de datos está vacía. Ejecuta `uv run python run.py pipeline`.")
        return

    df_defor = load_deforestation_data(con)
    df_risk  = load_water_risk_data(con)
    df_casen = load_casen_data(con)
    params   = build_sidebar(con, df_panel)

    # Header
    st.title("🌿 Chile — Capital Natural")
    st.caption(
        "Cobertura vegetal · Deforestación · Riesgo hídrico · "
        "MapBiomas Chile Col. 2 + WRI Aqueduct 4.0 · Por comuna"
    )

    # Métricas rápidas
    latest = df_panel[df_panel["year"] == df_panel["year"].max()]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Comunas", f"{df_panel['cut_code'].nunique():,}")
    c2.metric("Serie temporal", f"{df_panel['year'].min()}–{df_panel['year'].max()}")
    c3.metric("Bosque natural", f"{latest['bosque_ha'].sum()/1e6:.2f} M ha")
    c4.metric("Deforestación total",
              f"{df_defor[df_defor['transition_type']=='deforestation']['area_ha'].sum()/1e3:.0f} k ha"
              if not df_defor.empty else "—")
    c5.metric("Comunas con riesgo hídrico alto",
              f"{(df_risk['bws_score'] >= 3).sum()}" if not df_risk.empty else "—")

    st.divider()

    # ---------------------------------------------------------------------------
    # Introducción
    # ---------------------------------------------------------------------------
    with st.expander("📖 ¿Qué es este dashboard y cómo usarlo?", expanded=True):
        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("""
### Chile — Capital Natural

Este dashboard consolida información ambiental de Chile a nivel comunal con el
objetivo de hacer visible el estado y la evolución del **capital natural** del país:
los recursos ecosistémicos que sostienen la economía, la sociedad y el bienestar humano.

**¿Qué datos contiene?**

| Capa | Fuente | Cobertura |
|---|---|---|
| 🌿 Cobertura vegetal y uso de suelo | MapBiomas Chile, Colección 2 | 342 comunas · 1999–2024 |
| 🌲 Deforestación y transiciones | MapBiomas Chile (hoja TRANSITION) | 316 comunas · 1999–2024 |
| 💧 Riesgo hídrico | WRI Aqueduct 4.0 (baseline 2000–2019) | 340 comunas · 7 indicadores |
| 👥 Socioeconómico | CASEN 2017, 2020, 2022 (MIDESO) | ~300 comunas · 9 indicadores |

**¿Cómo navegar?**
- Usa el **panel izquierdo** para filtrar por región, comuna y año.
- Cada pestaña muestra un tema distinto: mapas, series temporales, deforestación y riesgo hídrico.
- La pestaña **📊 Datos y exportación** permite descargar tablas en CSV listas para R, Python o Stata.
""")

        with col_right:
            st.markdown("""
### Uso para análisis cuantitativo

Los datos están diseñados para complementarse con información **socioeconómica**
(CASEN, SII, INE, entre otros) y usarse en modelos econométricos de panel:

- Efectos de la deforestación sobre indicadores económicos locales
- Relación entre riesgo hídrico y productividad agrícola
- Valoración económica de servicios ecosistémicos
- Modelos de diferencias en diferencias (DiD) con shocks ambientales

Cada tabla exportable incluye el **código CUT comunal** como llave de unión
con otras fuentes de datos oficiales de Chile.

---
**Fuentes**
- [MapBiomas Chile](https://chile.mapbiomas.org/) — Col. 2, clasificación LULC 30 m
- [WRI Aqueduct 4.0](https://www.wri.org/aqueduct) — Riesgo hídrico por cuenca
- [GADM 4.1](https://gadm.org/) — Límites comunales
""")

    # Tabs
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🗺️ Mapa de cobertura",
        "📈 Series temporales",
        "🔍 Detalle por clase",
        "🌲 Deforestación",
        "💧 Riesgo hídrico",
        "👥 Socioeconómico (CASEN)",
        "📊 Datos y exportación",
    ])

    with t1: tab_mapa(df_panel, geojson, params)
    with t2: tab_serie_temporal(df_panel, params)
    with t3: tab_detalle_comuna(con, params)
    with t4: tab_deforestation(df_defor, geojson, params)
    with t5: tab_water_risk(df_risk, geojson, params)
    with t6: tab_casen(df_casen, geojson, params)
    with t7: tab_datos(df_panel, params)


if __name__ == "__main__":
    main()
