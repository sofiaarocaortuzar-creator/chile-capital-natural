"""
Panel comunal Chile — Capital Natural
======================================
Consolida todas las capas de datos en un panel (cut_code × año) listo
para análisis econométrico: regresiones, correlaciones, diferencias-en-
diferencias, etc.

Uso:
    uv run python analysis/export_panel.py

Salidas:
    data/panel_comunal_chile.csv    ← panel largo, ideal para R/Stata/Python
    data/panel_comunal_chile.xlsx   ← Excel con 4 hojas (panel + auxiliares)
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

console = Console()

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "chile_capital_natural.duckdb"
OUT_DIR = ROOT / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Conexión
# ─────────────────────────────────────────────────────────────────────────────

def connect() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB no encontrado: {DB_PATH}")
    return duckdb.connect(str(DB_PATH), read_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del panel
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Retorna el panel completo: una fila por (cut_code, año).

    Columnas clave:
      Identificadores   : cut_code, year, comuna, region, area_km2
      Cobertura vegetal : bosque_ha … pct_natural (MapBiomas, anual)
      Deforestación     : defor_ha, defor_bosque_ha, defor_eventos
                          (cambios de natural → no-natural ese año)
      Incendios         : n_incendios, ha_quemada, ha_forestal_quemada
      Riesgo hídrico    : bws_score … cfr_score  (cross-sectional, WRI Aqueduct)
      Calidad del aire  : pm25_mean, pm10_mean   (SINCA, ~50 comunas)
      CASEN             : tasa_pobreza, tasa_pobreza_multi, ypc_promedio,
                          ypc_mediana, tasa_ocupacion, esc_promedio,
                          pct_agua_red, pct_alcantarillado, pct_indigena,
                          pct_urbano   (solo 2017, 2020, 2022)
    """

    # ── 1. Backbone: panel_anual (cobertura vegetal anual por comuna) ─────────
    console.print("  Cargando panel base (cobertura vegetal)…")
    df = con.execute("""
        SELECT
            p.cut_code,
            p.year,
            c.name                  AS comuna,
            c.region_name           AS region,
            c.area_km2,
            -- Cobertura en hectáreas
            p.bosque_ha,
            p.bosque_primario_ha,
            p.bosque_secundario_ha,
            p.plantacion_ha,
            p.matorral_ha,
            p.pradera_ha,
            p.estepa_ha,
            p.humedal_ha,
            p.agricultura_ha,
            p.pastizal_ha,
            p.infraestructura_ha,
            p.agua_ha,
            p.hielo_ha,
            p.total_ha,
            -- Porcentajes (del área comunal total)
            p.pct_bosque,
            p.pct_natural
        FROM panel_anual p
        JOIN comunas c ON c.cut_code = p.cut_code
        ORDER BY p.cut_code, p.year
    """).df()
    console.print(f"  ✓ {len(df):,} filas · {df['cut_code'].nunique()} comunas · {df['year'].nunique()} años")

    # ── 2. Deforestación anual (natural → no-natural) ─────────────────────────
    console.print("  Agregando deforestación anual…")
    df_defor = con.execute("""
        SELECT
            cut_code,
            year_to                                     AS year,
            -- Pérdida neta de cualquier cobertura natural
            SUM(area_ha)                                AS defor_ha,
            -- Sólo pérdida de bosque
            SUM(CASE WHEN from_class_name ILIKE '%bosque%' OR
                          from_class_name ILIKE '%forest%'
                     THEN area_ha ELSE 0 END)           AS defor_bosque_ha,
            -- Sólo pérdida de matorral/shrub
            SUM(CASE WHEN from_class_name ILIKE '%matorral%' OR
                          from_class_name ILIKE '%shrub%'
                     THEN area_ha ELSE 0 END)           AS defor_matorral_ha,
            -- Número de polígonos de cambio
            COUNT(*)                                    AS defor_eventos
        FROM deforestation_events
        WHERE transition_type = 'deforestation'
        GROUP BY cut_code, year_to
    """).df()
    df = df.merge(df_defor, on=["cut_code", "year"], how="left")
    df[["defor_ha", "defor_bosque_ha", "defor_matorral_ha", "defor_eventos"]] = \
        df[["defor_ha", "defor_bosque_ha", "defor_matorral_ha", "defor_eventos"]].fillna(0)
    df["defor_eventos"] = df["defor_eventos"].astype(int)
    console.print(f"  ✓ Deforestación: {df_defor['cut_code'].nunique()} comunas con eventos")

    # ── 3. Incendios (por temporada) ──────────────────────────────────────────
    console.print("  Agregando incendios…")
    df_inc = con.execute("""
        SELECT
            cut_code,
            season_end              AS year,
            n_incendios,
            ha_total                AS ha_quemada,
            ha_forestal             AS ha_forestal_quemada,
            ha_veg_natural          AS ha_veg_natural_quemada,
            ha_plantacion           AS ha_plantacion_quemada
        FROM incendios
    """).df()
    df = df.merge(df_inc, on=["cut_code", "year"], how="left")
    df[["n_incendios", "ha_quemada", "ha_forestal_quemada",
        "ha_veg_natural_quemada", "ha_plantacion_quemada"]] = \
        df[["n_incendios", "ha_quemada", "ha_forestal_quemada",
            "ha_veg_natural_quemada", "ha_plantacion_quemada"]].fillna(0)
    df["n_incendios"] = df["n_incendios"].astype(int)
    console.print(f"  ✓ Incendios: {df_inc['cut_code'].nunique()} comunas con registros")

    # ── 4. Riesgo hídrico (cross-sectional → se repite para todos los años) ───
    console.print("  Uniendo riesgo hídrico…")
    df_water = con.execute("""
        SELECT
            cut_code,
            bws_score, bwd_score, iav_score,
            sev_score, drr_score, rfr_score, cfr_score,
            -- Índice compuesto simple (promedio de los 7 indicadores)
            ROUND((bws_score + bwd_score + iav_score + sev_score +
                   drr_score + rfr_score + cfr_score) / 7.0, 3) AS water_risk_idx
        FROM water_risk
    """).df()
    df = df.merge(df_water, on="cut_code", how="left")
    console.print(f"  ✓ Riesgo hídrico: {len(df_water)} comunas")

    # ── 5. Calidad del aire (PM2.5 / PM10) ───────────────────────────────────
    console.print("  Uniendo calidad del aire…")
    df_aire = con.execute("""
        SELECT cut_code, year, pm25_mean, pm10_mean,
               n_stations_pm25, n_stations_pm10
        FROM calidad_aire
    """).df()
    df = df.merge(df_aire, on=["cut_code", "year"], how="left")
    console.print(f"  ✓ Aire: {df_aire['cut_code'].nunique()} comunas con datos PM")

    # ── 6. CASEN (sólo 2017, 2020, 2022) ─────────────────────────────────────
    console.print("  Uniendo CASEN (2017, 2020, 2022)…")
    df_casen = con.execute("""
        SELECT
            cut_code, year,
            n_obs                   AS casen_n_obs,
            representativa          AS casen_representativa,
            tasa_pobreza,
            tasa_pobreza_multi,
            ypc_promedio            AS ingreso_pc_promedio,
            ypc_mediana             AS ingreso_pc_mediana,
            tasa_ocupacion,
            esc_promedio            AS escolaridad_promedio,
            pct_agua_red,
            pct_alcantarillado,
            pct_indigena,
            pct_urbano
        FROM casen_comunal
    """).df()
    df = df.merge(df_casen, on=["cut_code", "year"], how="left")
    console.print(f"  ✓ CASEN: {df_casen['cut_code'].nunique()} comunas × 3 años")

    # ── 7. Variables derivadas útiles para regresiones ────────────────────────
    console.print("  Calculando variables derivadas…")

    # Tasa de deforestación relativa al área de bosque del año anterior
    df = df.sort_values(["cut_code", "year"])
    df["bosque_ha_lag1"] = df.groupby("cut_code")["bosque_ha"].shift(1)
    df["tasa_defor"] = df["defor_bosque_ha"] / df["bosque_ha_lag1"].replace(0, float("nan"))

    # Pérdida acumulada de bosque desde el primer año disponible (por comuna)
    df["bosque_ha_base"] = df.groupby("cut_code")["bosque_ha"].transform("first")
    df["bosque_perdida_acum_ha"] = df["bosque_ha_base"] - df["bosque_ha"]
    df["bosque_perdida_acum_pct"] = df["bosque_perdida_acum_ha"] / \
        df["bosque_ha_base"].replace(0, float("nan"))

    # Intensidad de incendios (ha quemada / área comunal)
    df["inc_intensidad"] = df["ha_quemada"] / df["area_km2"] / 100  # en fracción de área

    df.drop(columns=["bosque_ha_base"], inplace=True)

    # Reordenar columnas
    id_cols = ["cut_code", "year", "comuna", "region", "area_km2"]
    veg_cols = [c for c in df.columns if c in [
        "bosque_ha", "bosque_primario_ha", "bosque_secundario_ha", "plantacion_ha",
        "matorral_ha", "pradera_ha", "estepa_ha", "humedal_ha", "agricultura_ha",
        "pastizal_ha", "infraestructura_ha", "agua_ha", "hielo_ha",
        "total_ha", "pct_bosque", "pct_natural"
    ]]
    defor_cols = [c for c in df.columns if c.startswith("defor") or c in [
        "bosque_ha_lag1", "tasa_defor", "bosque_perdida_acum_ha", "bosque_perdida_acum_pct"
    ]]
    inc_cols = [c for c in df.columns if "incendio" in c or "quemada" in c or
                c == "inc_intensidad"]
    water_cols = [c for c in df.columns if "_score" in c or c == "water_risk_idx"]
    aire_cols = [c for c in df.columns if c.startswith("pm") or c.startswith("n_station")]
    casen_cols = [c for c in df.columns if c in [
        "casen_n_obs", "casen_representativa",
        "tasa_pobreza", "tasa_pobreza_multi",
        "ingreso_pc_promedio", "ingreso_pc_mediana",
        "tasa_ocupacion", "escolaridad_promedio",
        "pct_agua_red", "pct_alcantarillado", "pct_indigena", "pct_urbano"
    ]]
    ordered = id_cols + veg_cols + defor_cols + inc_cols + water_cols + aire_cols + casen_cols
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Diccionario de variables
# ─────────────────────────────────────────────────────────────────────────────

CODEBOOK = [
    # Identificadores
    ("cut_code",                "ID",       "Código CUT de la comuna (5 dígitos)"),
    ("year",                    "ID",       "Año del registro"),
    ("comuna",                  "ID",       "Nombre de la comuna"),
    ("region",                  "ID",       "Nombre de la región"),
    ("area_km2",                "ID",       "Superficie comunal (km²)"),
    # Cobertura vegetal
    ("bosque_ha",               "VEG",      "Superficie de bosque total (ha), MapBiomas"),
    ("bosque_primario_ha",      "VEG",      "Superficie de bosque primario (ha)"),
    ("bosque_secundario_ha",    "VEG",      "Superficie de bosque secundario/regenerado (ha)"),
    ("plantacion_ha",           "VEG",      "Superficie de plantación forestal (ha)"),
    ("matorral_ha",             "VEG",      "Superficie de matorral/shrubland (ha)"),
    ("pradera_ha",              "VEG",      "Superficie de pradera (ha)"),
    ("estepa_ha",               "VEG",      "Superficie de estepa (ha)"),
    ("humedal_ha",              "VEG",      "Superficie de humedales (ha)"),
    ("agricultura_ha",          "VEG",      "Superficie de uso agrícola (ha)"),
    ("pastizal_ha",             "VEG",      "Superficie de pastizal (ha)"),
    ("infraestructura_ha",      "VEG",      "Superficie urbanizada/infraestructura (ha)"),
    ("agua_ha",                 "VEG",      "Superficie de cuerpos de agua (ha)"),
    ("hielo_ha",                "VEG",      "Superficie de nieve/hielo (ha)"),
    ("total_ha",                "VEG",      "Superficie total mapeada por MapBiomas (ha)"),
    ("pct_bosque",              "VEG",      "Fracción del área comunal cubierta por bosque (0–1)"),
    ("pct_natural",             "VEG",      "Fracción del área comunal con cobertura natural (0–1)"),
    # Deforestación
    ("defor_ha",                "DEFOR",    "Área convertida de cobertura natural a no-natural ese año (ha)"),
    ("defor_bosque_ha",         "DEFOR",    "Área de bosque perdida ese año (ha)"),
    ("defor_matorral_ha",       "DEFOR",    "Área de matorral perdida ese año (ha)"),
    ("defor_eventos",           "DEFOR",    "Número de polígonos de cambio detectados ese año"),
    ("bosque_ha_lag1",          "DEFOR",    "Superficie de bosque el año anterior (ha)"),
    ("tasa_defor",              "DEFOR",    "Tasa de deforestación = defor_bosque_ha / bosque_ha_lag1"),
    ("bosque_perdida_acum_ha",  "DEFOR",    "Pérdida acumulada de bosque desde el primer año disponible (ha)"),
    ("bosque_perdida_acum_pct", "DEFOR",    "Pérdida acumulada de bosque como fracción del stock inicial"),
    # Incendios
    ("n_incendios",             "INC",      "Número de incendios registrados en la temporada (CONAF)"),
    ("ha_quemada",              "INC",      "Superficie total quemada en la temporada (ha)"),
    ("ha_forestal_quemada",     "INC",      "Superficie forestal quemada (ha)"),
    ("ha_veg_natural_quemada",  "INC",      "Superficie de vegetación natural quemada (ha)"),
    ("ha_plantacion_quemada",   "INC",      "Superficie de plantaciones quemadas (ha)"),
    ("inc_intensidad",          "INC",      "Ha quemada / área comunal (fracción)"),
    # Riesgo hídrico
    ("bws_score",               "AGUA",     "Estrés hídrico de referencia — Baseline Water Stress (0–5, WRI Aqueduct)"),
    ("bwd_score",               "AGUA",     "Depleción hídrica — Baseline Water Depletion (0–5)"),
    ("iav_score",               "AGUA",     "Variabilidad interanual — Interannual Variability (0–5)"),
    ("sev_score",               "AGUA",     "Variabilidad estacional (0–5)"),
    ("drr_score",               "AGUA",     "Riesgo de sequía — Drought Risk (0–5)"),
    ("rfr_score",               "AGUA",     "Riesgo de crecida fluvial (0–5)"),
    ("cfr_score",               "AGUA",     "Riesgo de crecida costera (0–5)"),
    ("water_risk_idx",          "AGUA",     "Índice compuesto de riesgo hídrico (promedio de los 7 scores)"),
    # Calidad del aire
    ("pm25_mean",               "AIRE",     "Promedio anual PM2.5 (µg/m³) — SINCA. WHO ≤ 5 µg/m³"),
    ("pm10_mean",               "AIRE",     "Promedio anual PM10 (µg/m³) — SINCA. WHO ≤ 15 µg/m³"),
    ("n_stations_pm25",         "AIRE",     "Número de estaciones con datos PM2.5 ese año"),
    ("n_stations_pm10",         "AIRE",     "Número de estaciones con datos PM10 ese año"),
    # CASEN socioeconómico
    ("casen_n_obs",             "CASEN",    "Observaciones CASEN en la comuna (muestra)"),
    ("casen_representativa",    "CASEN",    "True si la muestra comunal es representativa (n ≥ 50)"),
    ("tasa_pobreza",            "CASEN",    "Tasa de pobreza por ingresos (fracción 0–1) — CASEN"),
    ("tasa_pobreza_multi",      "CASEN",    "Tasa de pobreza multidimensional (fracción 0–1) — CASEN"),
    ("ingreso_pc_promedio",     "CASEN",    "Ingreso per cápita promedio del hogar (CLP corrientes)"),
    ("ingreso_pc_mediana",      "CASEN",    "Ingreso per cápita mediano del hogar (CLP corrientes)"),
    ("tasa_ocupacion",          "CASEN",    "Tasa de ocupación (fracción 0–1)"),
    ("escolaridad_promedio",    "CASEN",    "Años de escolaridad promedio de la población ≥ 18 años"),
    ("pct_agua_red",            "CASEN",    "Fracción de hogares con acceso a agua de red pública"),
    ("pct_alcantarillado",      "CASEN",    "Fracción de hogares con alcantarillado"),
    ("pct_indigena",            "CASEN",    "Fracción de personas que se identifican como pueblo originario"),
    ("pct_urbano",              "CASEN",    "Fracción de personas en zona urbana"),
]

def make_codebook() -> pd.DataFrame:
    return pd.DataFrame(CODEBOOK, columns=["variable", "categoria", "descripcion"])


# ─────────────────────────────────────────────────────────────────────────────
# Hoja de cobertura (qué comunas/años tienen cada capa)
# ─────────────────────────────────────────────────────────────────────────────

def make_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de cobertura de datos por capa."""
    layers = {
        "Cobertura vegetal (MapBiomas)": "bosque_ha",
        "Deforestación":                 "defor_ha",
        "Incendios (CONAF)":             "n_incendios",
        "Riesgo hídrico (WRI)":          "bws_score",
        "Calidad del aire (SINCA)":      "pm25_mean",
        "Pobreza CASEN":                 "tasa_pobreza",
    }
    rows = []
    for layer, col in layers.items():
        if col not in df.columns:
            continue
        sub = df[df[col].notna()]
        rows.append({
            "Capa de datos": layer,
            "Comunas con datos": sub["cut_code"].nunique(),
            "Años con datos": f"{int(sub['year'].min())}–{int(sub['year'].max())}",
            "Total filas": len(sub),
            "Columna clave": col,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Exportar
# ─────────────────────────────────────────────────────────────────────────────

def export(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "panel_comunal_chile.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # utf-8-sig para Excel
    console.print(f"  ✓ CSV: [cyan]{csv_path}[/cyan]  ({csv_path.stat().st_size/1e6:.1f} MB)")

    # ── Excel ─────────────────────────────────────────────────────────────────
    xlsx_path = OUT_DIR / "panel_comunal_chile.xlsx"
    codebook  = make_codebook()
    coverage  = make_coverage_summary(df)

    # Sub-panel CASEN (solo años con datos de pobreza)
    df_casen_sub = (
        df[df["tasa_pobreza"].notna()]
        [["cut_code","year","comuna","region","tasa_pobreza","tasa_pobreza_multi",
          "ingreso_pc_promedio","ingreso_pc_mediana","tasa_ocupacion",
          "escolaridad_promedio","pct_indigena","pct_urbano",
          "bosque_ha","pct_bosque","pct_natural","defor_bosque_ha","tasa_defor",
          "n_incendios","ha_quemada","water_risk_idx","pm25_mean","pm10_mean"]]
        .copy()
    )

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer,         sheet_name="Panel completo",    index=False)
        df_casen_sub.to_excel(writer, sheet_name="Panel CASEN",     index=False)
        codebook.to_excel(writer,   sheet_name="Diccionario",       index=False)
        coverage.to_excel(writer,   sheet_name="Cobertura datos",   index=False)

    console.print(f"  ✓ Excel: [cyan]{xlsx_path}[/cyan]  ({xlsx_path.stat().st_size/1e6:.1f} MB)")
    console.print(f"           4 hojas: Panel completo | Panel CASEN | Diccionario | Cobertura datos")


# ─────────────────────────────────────────────────────────────────────────────
# Resumen estadístico
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    console.print(Rule("[bold]Resumen del panel"))

    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Dimensión", style="bold")
    t.add_column("Valor", justify="right")
    t.add_row("Filas totales",              f"{len(df):,}")
    t.add_row("Comunas",                    f"{df['cut_code'].nunique()}")
    t.add_row("Años (rango)",               f"{int(df['year'].min())}–{int(df['year'].max())}")
    t.add_row("Columnas",                   f"{len(df.columns)}")
    t.add_row("Comunas con datos CASEN",    f"{df[df['tasa_pobreza'].notna()]['cut_code'].nunique()}")
    t.add_row("Comunas con datos PM2.5",    f"{df[df['pm25_mean'].notna()]['cut_code'].nunique()}")
    t.add_row("Comunas con deforestación",  f"{df[df['defor_ha']>0]['cut_code'].nunique()}")
    t.add_row("Comunas con incendios",      f"{df[df['n_incendios']>0]['cut_code'].nunique()}")
    console.print(t)

    # Estadísticas de las variables clave para 2022
    console.print(Rule("Variables clave — año 2022 (o más reciente disponible)"))
    yr = df[df["year"] == 2022]
    desc_cols = ["tasa_pobreza","tasa_defor","bosque_perdida_acum_pct",
                 "water_risk_idx","pm25_mean","n_incendios"]
    desc_cols = [c for c in desc_cols if c in df.columns]
    print(yr[desc_cols].describe().round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Rule("[bold green]Exportación del panel comunal — Chile Capital Natural"))

    con = connect()
    console.print(Rule("Construyendo panel"))
    df = build_panel(con)
    con.close()

    print_summary(df)

    console.print(Rule("Exportando archivos"))
    export(df)

    console.print(Rule("[bold green]¡Listo!"))
    console.print(f"\n  → [bold]data/panel_comunal_chile.csv[/bold]   (panel largo, ideal para R/Stata/Python)")
    console.print(f"  → [bold]data/panel_comunal_chile.xlsx[/bold]  (Excel, 4 hojas)")
    console.print()
    console.print("  Tip para R:")
    console.print('    [dim]panel <- read.csv("data/panel_comunal_chile.csv")[/dim]')
    console.print('    [dim]library(plm); pdata <- pdata.frame(panel, index=c("cut_code","year"))[/dim]')
    console.print()
    console.print("  Tip para Stata:")
    console.print('    [dim]import delimited "panel_comunal_chile.csv", encoding(UTF-8)[/dim]')
    console.print('    [dim]encode cut_code, gen(id_comuna); xtset id_comuna year[/dim]')
