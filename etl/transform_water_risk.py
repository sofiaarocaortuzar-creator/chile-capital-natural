"""
Riesgo hídrico a nivel comunal — WRI Aqueduct 4.0

Metodología:
    1. Descargar shapefile de sub-cuencas WRI Aqueduct 4.0 (global)
    2. Filtrar sub-cuencas que intersectan Chile
    3. Intersección espacial: fragmentos de cuenca dentro de cada comuna
    4. Promedio ponderado por área para cada indicador de riesgo
    5. Cargar en DuckDB tabla water_risk

Nota sobre temporalidad:
    Aqueduct 4.0 entrega un baseline histórico (promedio 2000–2019) para la
    mayoría de indicadores, más proyecciones para 2030, 2040 y 2050 bajo
    tres escenarios climáticos (SSP1-2.6, SSP3-7.0, SSP5-8.5).
    No hay datos anuales individuales para riesgo hídrico.

Indicadores incluidos:
    bws  — Estrés hídrico de línea base (proporción demanda/disponibilidad)
    bwd  — Agotamiento de agua (depleción relativa al flujo natural)
    iav  — Variabilidad interanual del caudal
    sev  — Variabilidad estacional del caudal
    drr  — Riesgo de sequía (días por año con déficit hídrico)
    rfr  — Riesgo de inundación fluvial (población en zona de riesgo)
    cfr  — Riesgo de inundación costera
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rich.console import Console

from etl.config import PROCESSED_DIR, RAW_DIR

console = Console()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

AQUEDUCT_URL = "https://files.wri.org/aqueduct/aqueduct-4-0-water-risk-data.zip"
AQUEDUCT_ZIP = RAW_DIR / "aqueduct40.zip"
AQUEDUCT_DIR = RAW_DIR / "aqueduct40"

# Indicadores de interés y sus etiquetas en español
INDICATORS = {
    "bws": {"label": "Estrés hídrico",           "unit": "ratio demanda/oferta"},
    "bwd": {"label": "Agotamiento de agua",       "unit": "ratio vs flujo natural"},
    "iav": {"label": "Variabilidad interanual",   "unit": "coef. variación"},
    "sev": {"label": "Variabilidad estacional",   "unit": "coef. variación"},
    "drr": {"label": "Riesgo de sequía",          "unit": "score 0-5"},
    "rfr": {"label": "Riesgo inundación fluvial", "unit": "score 0-5"},
    "cfr": {"label": "Riesgo inundación costera", "unit": "score 0-5"},
}

# Columnas raw (valor continuo) para cálculo del promedio ponderado
RAW_COLS   = [f"{k}_raw"   for k in INDICATORS]
SCORE_COLS = [f"{k}_score" for k in INDICATORS]
LABEL_COLS = [f"{k}_label" for k in INDICATORS]

# Escenarios futuros disponibles en Aqueduct 4.0
FUTURE_SCENARIOS = {
    "ssp1": "SSP1-RCP2.6 (optimista)",
    "ssp3": "SSP3-RCP7.0 (intermedio)",
    "ssp5": "SSP5-RCP8.5 (pesimista)",
}
FUTURE_YEARS = [2030, 2040, 2050]

# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def download_aqueduct(force: bool = False) -> Path:
    """Descarga el archivo ZIP de Aqueduct 4.0 usando curl (249 MB)."""
    AQUEDUCT_ZIP.parent.mkdir(parents=True, exist_ok=True)

    if AQUEDUCT_ZIP.exists() and not force:
        console.print(f"[green]✓ Ya existe:[/green] {AQUEDUCT_ZIP.name}")
        return AQUEDUCT_ZIP

    console.print(f"[bold cyan]↓ Descargando WRI Aqueduct 4.0 (~249 MB)…[/bold cyan]")
    console.print(f"  URL: {AQUEDUCT_URL}")

    cmd = [
        "curl", "--location", "--retry", "3", "--retry-delay", "5",
        "--connect-timeout", "30", "--max-time", "900",
        "--progress-bar", "--output", str(AQUEDUCT_ZIP),
        AQUEDUCT_URL,
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if AQUEDUCT_ZIP.exists():
            AQUEDUCT_ZIP.unlink()
        raise RuntimeError(f"curl falló (código {result.returncode})")

    size_mb = AQUEDUCT_ZIP.stat().st_size / 1e6
    console.print(f"  [green]✓ Descargado ({size_mb:.0f} MB)[/green]")
    return AQUEDUCT_ZIP


def extract_aqueduct(force: bool = False) -> Path:
    """Extrae el ZIP de Aqueduct."""
    import zipfile

    if AQUEDUCT_DIR.exists() and any(AQUEDUCT_DIR.iterdir()) and not force:
        console.print(f"[green]✓ Ya extraído:[/green] {AQUEDUCT_DIR.name}")
        return AQUEDUCT_DIR

    AQUEDUCT_DIR.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]📦 Extrayendo Aqueduct…[/bold cyan]")
    try:
        with zipfile.ZipFile(AQUEDUCT_ZIP) as zf:
            zf.extractall(AQUEDUCT_DIR)
    except zipfile.BadZipFile:
        result = subprocess.run(
            ["unzip", "-o", str(AQUEDUCT_ZIP), "-d", str(AQUEDUCT_DIR)],
            capture_output=True, text=True,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"unzip falló: {result.stderr}")
    console.print(f"  [green]✓ Extraído en {AQUEDUCT_DIR}[/green]")
    return AQUEDUCT_DIR


def find_aqueduct_file() -> Path:
    """Encuentra el shapefile o GeoPackage de Aqueduct baseline annual."""
    if not AQUEDUCT_DIR.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {AQUEDUCT_DIR}")

    # Buscar GeoPackage primero (más eficiente), luego shapefile
    for pattern in ("*.gpkg", "*.gdb", "*.shp"):
        candidates = sorted(AQUEDUCT_DIR.rglob(pattern))
        # Priorizar archivos con "annual" o "baseline" en el nombre
        preferred = [p for p in candidates
                     if any(k in p.stem.lower() for k in ("annual", "baseline", "aqueduct"))]
        if preferred:
            return preferred[0]
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"No se encontró shapefile/GeoPackage en {AQUEDUCT_DIR}. "
        "Archivos disponibles: " + str(list(AQUEDUCT_DIR.rglob("*"))[:10])
    )


# ---------------------------------------------------------------------------
# Carga y filtrado de Aqueduct
# ---------------------------------------------------------------------------

def load_aqueduct_chile(aqueduct_path: Path) -> gpd.GeoDataFrame:
    """Carga Aqueduct y filtra solo sub-cuencas dentro o que tocan Chile."""
    console.print(f"[bold cyan]💧 Cargando Aqueduct: {aqueduct_path.name}[/bold cyan]")

    # Leer con bbox de Chile para eficiencia (lon: -76 a -65, lat: -56 a -17)
    bbox_chile = (-76.5, -56.5, -65.5, -17.0)
    gdf = gpd.read_file(aqueduct_path, bbox=bbox_chile)
    console.print(f"  Filas en bbox Chile: {len(gdf):,}")

    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Filtrar explícitamente por país si existe la columna
    if "gid_0" in gdf.columns:
        chile = gdf[gdf["gid_0"] == "CHL"].copy()
        console.print(f"  Filtrado CHL: {len(chile):,} sub-cuencas")
    else:
        # Fallback: usar todas las que cayeron en el bbox
        chile = gdf.copy()
        console.print(f"  Sin columna gid_0 — usando todas las del bbox: {len(chile):,}")

    # Verificar que las columnas de indicadores existen
    available_raw = [c for c in RAW_COLS if c in chile.columns]
    if not available_raw:
        # Intentar detectar columnas con patrones similares
        console.print(f"  [yellow]Columnas disponibles: {list(chile.columns[:30])}[/yellow]")
        raise ValueError(
            "No se encontraron columnas de indicadores de riesgo en el archivo. "
            f"Se esperaban: {RAW_COLS[:3]}…"
        )
    console.print(f"  Indicadores encontrados: {available_raw}")
    return chile


# ---------------------------------------------------------------------------
# Intersección espacial y promedio ponderado
# ---------------------------------------------------------------------------

def area_weighted_join(
    gdf_comunas: gpd.GeoDataFrame,
    gdf_aqueduct: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Calcula el promedio ponderado por área de los indicadores de Aqueduct
    para cada comuna.

    Pasos:
      1. Proyectar ambas capas a UTM 18S (para calcular áreas en m²)
      2. Intersectar geometrías → fragmentos de cuenca dentro de cada comuna
      3. Calcular área de cada fragmento
      4. Para cada (comuna, indicador): sum(valor × área_frag) / sum(área_frag)
      5. Volver a añadir metadatos de la comuna

    Retorna DataFrame con una fila por comuna y columnas por indicador.
    """
    console.print("[bold cyan]📐 Intersección espacial comunas × cuencas…[/bold cyan]")

    # Proyectar a UTM 18S para área correcta
    crs_utm = "EPSG:32718"
    com_utm = gdf_comunas[["cut_code", "name", "region_name", "geometry"]].to_crs(crs_utm)
    aqu_utm = gdf_aqueduct.to_crs(crs_utm)

    # Columnas de valores a ponderar: usar _raw (valor continuo)
    # y _score (0-5 normalizado)
    value_cols = [c for c in RAW_COLS + SCORE_COLS if c in aqu_utm.columns]

    # Reemplazar -9999 (no-data de WRI) con NaN antes de intersectar
    for col in value_cols:
        aqu_utm[col] = aqu_utm[col].replace(-9999, float("nan"))
        aqu_utm[col] = pd.to_numeric(aqu_utm[col], errors="coerce")

    console.print(f"  Intersectando {len(com_utm)} comunas × {len(aqu_utm)} cuencas…")

    # Intersección espacial
    inter = gpd.overlay(
        com_utm[["cut_code", "geometry"]],
        aqu_utm[["geometry"] + value_cols],
        how="intersection",
        keep_geom_type=False,
    )

    if inter.empty:
        raise ValueError("La intersección está vacía — verificar CRS y bbox de los datos.")

    # Área de cada fragmento
    inter["frag_area_m2"] = inter.geometry.area
    inter = inter[inter["frag_area_m2"] > 0]

    console.print(f"  Fragmentos generados: {len(inter):,}")

    # Promedio ponderado por área para cada indicador
    records = []
    for cut_code, group in inter.groupby("cut_code"):
        total_area = group["frag_area_m2"].sum()
        row = {"cut_code": cut_code, "total_basin_area_km2": total_area / 1e6}
        for col in value_cols:
            valid = group[["frag_area_m2", col]].dropna(subset=[col])
            if valid.empty:
                row[col] = float("nan")
            else:
                row[col] = (
                    (valid[col] * valid["frag_area_m2"]).sum()
                    / valid["frag_area_m2"].sum()
                )
        records.append(row)

    df_result = pd.DataFrame(records)

    # Unir con metadatos de comunas
    meta = gdf_comunas[["cut_code", "name", "region_name", "area_km2"]].copy()
    df_result = df_result.merge(meta, on="cut_code", how="left")

    # Renombrar columnas a nombres más limpios
    rename = {}
    for ind in INDICATORS:
        raw_col   = f"{ind}_raw"
        score_col = f"{ind}_score"
        if raw_col   in df_result.columns: rename[raw_col]   = f"{ind}_raw"
        if score_col in df_result.columns: rename[score_col] = f"{ind}_score"
    df_result = df_result.rename(columns=rename)

    matched = df_result["cut_code"].nunique()
    console.print(
        f"  [green]✓ Promedio ponderado calculado para {matched} comunas "
        f"con {len(value_cols)} indicadores[/green]"
    )
    return df_result


# ---------------------------------------------------------------------------
# Pipeline completo de riesgo hídrico
# ---------------------------------------------------------------------------

def build_water_risk_dataframe(gdf_comunas: gpd.GeoDataFrame) -> pd.DataFrame:
    """Pipeline completo: descarga → carga → intersección → retorna DataFrame."""
    console.print("[bold cyan]💧 Procesando riesgo hídrico (WRI Aqueduct 4.0)…[/bold cyan]")

    # Descarga y extracción
    download_aqueduct()
    extract_aqueduct()
    aqu_path = find_aqueduct_file()

    # Cargar datos de Chile
    gdf_aqueduct = load_aqueduct_chile(aqu_path)

    # Intersección espacial con comunas
    df_risk = area_weighted_join(gdf_comunas, gdf_aqueduct)

    console.print(
        f"  [green]✓ Riesgo hídrico listo: {len(df_risk)} comunas | "
        f"indicadores: {list(INDICATORS.keys())}[/green]"
    )
    return df_risk
