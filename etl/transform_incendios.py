"""
ETL Incendios Forestales — CONAF Chile

Descarga el resumen de ocurrencia y daño por comuna (1985–2024) desde
el Centro Documental de CONAF y lo normaliza a una tabla comunal × temporada.

Fuente: CONAF — Resumen de Ocurrencia y Daño por Comuna
https://www.conaf.cl/incendios-forestales/incendios-forestales-en-chile/estadisticas-historicas/

Nota sobre temporadas:
    Los incendios forestales en Chile ocurren principalmente entre octubre y
    abril. El archivo CONAF reporta por "temporada" (ej: 2023-2024), no por
    año calendario. Guardamos season_start y season_end, y usamos season_end
    como año de referencia para joins con otras tablas anuales.
"""
from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rich.console import Console

from etl.config import RAW_DIR
from etl.transform import _normalize   # reutilizamos el normalizador de nombres

console = Console()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CONAF_URL = (
    "https://www.conaf.cl/centro-documental/"
    "resumen-de-ocurrencia-y-dano-por-comuna-1985-2023/"
    "?wpdmdl=56992&ind=1731080080984"
    "&filename=10.-Resumen-de-Ocurrencia-y-Dano-por-Comuna-1985-2024_octubre.xlsx"
)
CONAF_XLSX = RAW_DIR / "conaf_incendios_comunal.xlsx"

# Índices de columna (fijos en todas las hojas recientes)
COL_REGION   = 0
COL_PROV     = 1
COL_COMUNA   = 2
COL_N_INC    = 3
COL_HA_PLANT = 10   # TOTAL PLANTACIONES FORESTALES
COL_HA_VEG   = 14   # TOTAL VEGETACIÓN NATURAL
COL_HA_FOR   = 15   # TOTAL FORESTAL (plantaciones + veg natural)
COL_HA_TOTAL = 19   # TOTAL SUPERFICIE AFECTADA (última columna)

# Correcciones manuales de nombres comunales CONAF → estándar
MANUAL_FIXES: dict[str, str] = {
    "o'higgins":            "general bernardo ohiggins",
    "ohiggins":             "general bernardo ohiggins",
    "alto biobio":          "alto biobío",
    "los alamos":           "los álamos",
    "cañete":               "canete",
    "lebu":                 "lebu",
    "tirua":                "tirúa",
    "curacautin":           "curacautín",
    "lonquimay":            "lonquimay",
    "puren":                "purén",
    "angol":                "angol",
    "collipulli":           "collipulli",
    "ercilla":              "ercilla",
    "lumaco":               "lumaco",
    "traiguen":             "traiguén",
    "victoria":             "victoria",
    "renaico":              "renaico",
    "malleco":              "malleco",
    "nva imperial":         "nueva imperial",
    "nueva imperial":       "nueva imperial",
    "carahue":              "carahue",
    "saavedra":             "saavedra",
    "teodoro schmidt":      "teodoro schmidt",
    "tolten":               "toltén",
    "freire":               "freire",
    "pitrufquen":           "pitrufquén",
    "villarrica":           "villarrica",
    "cunco":                "cunco",
    "melipeuco":            "melipeuco",
    "curarrehue":           "curarrehue",
    "pucon":                "pucón",
    "vilcun":               "vilcún",
    "gorbea":               "gorbea",
    "loncoche":             "loncoche",
    "lago ranqui":          "lago ranquil",
}


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def download_conaf(force: bool = False) -> Path:
    """Descarga el Excel de CONAF y lo guarda en caché."""
    CONAF_XLSX.parent.mkdir(parents=True, exist_ok=True)
    if CONAF_XLSX.exists() and not force:
        console.print(f"  [green]✓ Ya existe:[/green] {CONAF_XLSX.name}")
        return CONAF_XLSX

    console.print("[bold cyan]↓ Descargando CONAF Incendios Comunal (2.5 MB)…[/bold cyan]")
    req = urllib.request.urlopen(CONAF_URL, timeout=60)
    CONAF_XLSX.write_bytes(req.read())
    console.print(f"  [green]✓ Guardado: {CONAF_XLSX.name}[/green]")
    return CONAF_XLSX


# ---------------------------------------------------------------------------
# Parseo de hojas
# ---------------------------------------------------------------------------

def _parse_season(sheet_name: str) -> tuple[int, int] | None:
    """
    Extrae (year_start, year_end) del nombre de la hoja.
    Ejemplos: '2023-2024' → (2023, 2024), '84-85 oK' → (1984, 1985)
    """
    m = re.search(r"(\d{2,4})\s*[-–]\s*(\d{2,4})", sheet_name)
    if not m:
        return None
    y1, y2 = int(m.group(1)), int(m.group(2))
    # Normalizar años de 2 dígitos (84 → 1984)
    if y1 < 100:
        y1 += 1900
    if y2 < 100:
        y2 += 1900
    return y1, y2


def _parse_sheet(df_raw: pd.DataFrame, season: tuple[int, int]) -> pd.DataFrame:
    """
    Parsea una hoja del Excel CONAF.
    Retorna DataFrame con columnas estandarizadas.
    """
    season_start, season_end = season
    n_cols = df_raw.shape[1]

    # Ajustar índices si la hoja tiene menos columnas (hojas antiguas)
    col_ha_total = min(COL_HA_TOTAL, n_cols - 1)
    col_ha_for   = min(COL_HA_FOR,   n_cols - 1)
    col_ha_veg   = min(COL_HA_VEG,   n_cols - 1)
    col_ha_plant = min(COL_HA_PLANT,  n_cols - 1)

    # Saltar filas de encabezado (primeras 6 filas)
    df = df_raw.iloc[6:].copy()
    df.columns = range(n_cols)

    # Forward-fill región y provincia
    df[COL_REGION] = df[COL_REGION].ffill()
    df[COL_PROV]   = df[COL_PROV].ffill()

    # Filtrar filas válidas: COMUNA no nula y no es "TOTAL"
    df = df[
        df[COL_COMUNA].notna() &
        (~df[COL_COMUNA].astype(str).str.upper().str.strip().isin(["TOTAL", "NAN", ""]))
    ].copy()

    if df.empty:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        comuna_raw = str(row[COL_COMUNA]).strip()
        if not comuna_raw or comuna_raw.lower() == "nan":
            continue

        try:
            n_inc   = pd.to_numeric(row[COL_N_INC], errors="coerce")
            ha_total = pd.to_numeric(row[col_ha_total], errors="coerce")
            ha_for   = pd.to_numeric(row[col_ha_for],   errors="coerce")
            ha_veg   = pd.to_numeric(row[col_ha_veg],   errors="coerce")
            ha_plant = pd.to_numeric(row[col_ha_plant],  errors="coerce")
        except Exception:
            continue

        records.append({
            "comuna_raw":    comuna_raw,
            "region_raw":    str(row[COL_REGION]).strip(),
            "season_start":  season_start,
            "season_end":    season_end,
            "n_incendios":   int(n_inc) if pd.notna(n_inc) else 0,
            "ha_total":      float(ha_total) if pd.notna(ha_total) else 0.0,
            "ha_forestal":   float(ha_for)   if pd.notna(ha_for)   else 0.0,
            "ha_veg_natural": float(ha_veg)  if pd.notna(ha_veg)   else 0.0,
            "ha_plantacion": float(ha_plant) if pd.notna(ha_plant) else 0.0,
        })

    return pd.DataFrame(records)


def load_conaf_sheets(xlsx_path: Path) -> pd.DataFrame:
    """Lee todas las hojas del Excel CONAF y retorna DataFrame largo."""
    console.print(f"  Leyendo {xlsx_path.name}…")
    xf = pd.ExcelFile(xlsx_path)
    dfs = []

    for sheet in xf.sheet_names:
        season = _parse_season(sheet)
        if season is None:
            console.print(f"    [dim]Hoja ignorada (no parseable): {sheet}[/dim]")
            continue
        try:
            df_raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
            df_parsed = _parse_sheet(df_raw, season)
            if not df_parsed.empty:
                dfs.append(df_parsed)
        except Exception as e:
            console.print(f"    [yellow]⚠ Error en hoja {sheet}: {e}[/yellow]")

    df = pd.concat(dfs, ignore_index=True)
    console.print(
        f"  ✓ {len(df):,} filas · "
        f"{df['season_end'].nunique()} temporadas · "
        f"{df['comuna_raw'].nunique()} comunas únicas"
    )
    return df


# ---------------------------------------------------------------------------
# Join comunal
# ---------------------------------------------------------------------------

def join_incendios_with_comunas(
    df: pd.DataFrame,
    gdf_comunas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Une el DataFrame de incendios con los cut_codes comunales usando
    normalización de nombres + correcciones manuales + fuzzy matching.
    """
    import difflib

    # Índice de nombres normalizados de comunas GADM
    gadm_norm = {_normalize(r["name"]): r["cut_code"]
                 for _, r in gdf_comunas.iterrows()}

    def match_name(raw: str) -> str | None:
        norm = _normalize(raw)
        # Aplicar correcciones manuales
        norm = MANUAL_FIXES.get(norm, norm)
        # Exacto
        if norm in gadm_norm:
            return gadm_norm[norm]
        # Fuzzy
        candidates = difflib.get_close_matches(norm, gadm_norm.keys(),
                                               n=1, cutoff=0.80)
        return gadm_norm[candidates[0]] if candidates else None

    df = df.copy()
    df["cut_code"] = df["comuna_raw"].map(match_name)

    matched = df["cut_code"].notna().sum()
    total   = len(df)
    console.print(
        f"  Join incendios: {matched:,}/{total:,} filas con cut_code "
        f"({matched/total*100:.1f}%)"
    )

    unmatched = df[df["cut_code"].isna()]["comuna_raw"].unique()
    if len(unmatched) > 0:
        console.print(f"  [yellow]Sin match ({len(unmatched)}): {sorted(unmatched)[:10]}[/yellow]")

    return df[df["cut_code"].notna()].copy()


# ---------------------------------------------------------------------------
# Agregación final
# ---------------------------------------------------------------------------

def build_incendios_dataframe(
    gdf_comunas: gpd.GeoDataFrame,
    force: bool = False,
) -> pd.DataFrame:
    """
    Pipeline completo: descarga → parseo → join comunal → agrega.

    Retorna DataFrame con una fila por (cut_code × season_end):
        cut_code, season_start, season_end, n_incendios,
        ha_total, ha_forestal, ha_veg_natural, ha_plantacion
    """
    console.print("[bold cyan]🔥 Procesando incendios forestales (CONAF 1985–2024)…[/bold cyan]")

    xlsx_path = download_conaf(force=force)
    df_raw    = load_conaf_sheets(xlsx_path)
    df_joined = join_incendios_with_comunas(df_raw, gdf_comunas)

    # Agregar por (cut_code, season_end) — por si hay duplicados
    df_agg = (
        df_joined
        .groupby(["cut_code", "season_start", "season_end"], as_index=False)
        .agg(
            n_incendios   =("n_incendios",   "sum"),
            ha_total      =("ha_total",      "sum"),
            ha_forestal   =("ha_forestal",   "sum"),
            ha_veg_natural=("ha_veg_natural","sum"),
            ha_plantacion =("ha_plantacion", "sum"),
        )
    )

    n_com = df_agg["cut_code"].nunique()
    n_sea = df_agg["season_end"].nunique()
    console.print(
        f"  [green]✓ Incendios listos: {len(df_agg):,} registros · "
        f"{n_com} comunas · {n_sea} temporadas[/green]"
    )
    return df_agg
