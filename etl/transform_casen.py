"""
ETL CASEN — Encuesta de Caracterización Socioeconómica Nacional

Descarga la microdata oficial, une con el archivo comunal (que contiene el
código CUT suprimido en el archivo nacional) y agrega a nivel comunal
usando los factores de expansión.

Años disponibles: 2017, 2020, 2022

Fuente: Ministerio de Desarrollo Social — Observatorio Social
https://observatorio.ministeriodesarrollosocial.gob.cl/

Nota sobre representatividad comunal:
    El diseño muestral de CASEN garantiza representatividad a nivel regional,
    no comunal. Las comunas con n_obs < 50 se marcan con representativa=False
    y deben usarse con cautela en modelos econométricos.
"""
from __future__ import annotations

import difflib
import io
import tempfile
import unicodedata
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from rich.console import Console

from etl.config import RAW_DIR

console = Console()


def _normalize(s: str) -> str:
    """Normaliza nombre de comuna para matching (minúsculas, sin tildes)."""
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s

# ---------------------------------------------------------------------------
# Configuración por año
# ---------------------------------------------------------------------------

CASEN_DIR = RAW_DIR / "casen"

# Mapeo de variables por año encuesta → nombre estándar
# Estructura: {year: {variable_estandar: nombre_en_casen}}
VAR_MAP: dict[int, dict[str, str]] = {
    2022: {
        "ypc":           "ypc",              # ingreso per cápita del hogar
        "pobreza":       "pobreza",
        "pobreza_multi": "pobreza_multi_5d",
        "activ":         "activ",
        "esc":           "esc",
        "agua":          "v20",
        "alcantarillado": "v21",
        "indigena":      "r3",
        "area":          "area",
        "expr":          "expr",
    },
    2020: {
        "ypc":           "ypchtotcor",       # CASEN 2020 usa ypchtotcor, no ypc
        "pobreza":       "pobreza",
        "pobreza_multi": "pobreza_multi_5d", # se ajusta por fallback si no existe
        "activ":         "activ",
        "esc":           "esc",
        "agua":          "v20",
        "alcantarillado": "v21",             # puede no existir en 2020; fallback a None
        "indigena":      "r3",
        "area":          "area",
        "expr":          "expr",
    },
    2017: {
        "ypc":           "ypchtotcor",       # CASEN 2017: ypchtotcor es ingreso pc corregido
        "pobreza":       "pobreza",
        "pobreza_multi": "pobreza_multi_4d", # 2017 usa 4 dimensiones
        "activ":         "activ",
        "esc":           "esc",
        "agua":          "v20",
        "alcantarillado": "v21",
        "indigena":      "r3",
        "area":          "area",
        "expr":          "expr",
    },
}

# URLs de descarga (archivos nacionales con microdata)
CASEN_URLS: dict[int, dict[str, str]] = {
    2022: {
        "data": (
            "https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2022/"
            "Base%20de%20datos%20Casen%202022%20STATA_18%20marzo%202024.dta.zip"
        ),
        "link": (
            "https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2022/"
            "Base%20de%20datos%20provincia%20y%20comuna%20Casen%202022%20STATA.dta.zip"
        ),
    },
    2020: {
        "data": (
            "https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2020/"
            "Casen_en_Pandemia_2020_STATA_revisada2022_09.dta.zip"
        ),
        "link": (
            "https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2020/"
            "casen_en_pandemia_2020_provincia_comuna.dta.zip"
        ),
    },
    2017: {
        "data": (
            "https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2017/"
            "casen_2017.dta.zip"
        ),
        "link": (
            "https://observatorio.ministeriodesarrollosocial.gob.cl/storage/docs/casen/2017/"
            "casen_2017_provincia_comuna.dta.zip"
        ),
    },
}

# Umbral mínimo de observaciones para considerar una comuna representativa
MIN_OBS = 50


# ---------------------------------------------------------------------------
# Descarga y caché
# ---------------------------------------------------------------------------

def _download_dta(url: str, cache_path: Path, label: str) -> Path:
    """Descarga un archivo .dta.zip y guarda el .dta localmente."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        console.print(f"  [green]✓ Ya existe:[/green] {cache_path.name}")
        return cache_path

    console.print(f"  [bold cyan]↓ Descargando {label}…[/bold cyan]")
    req = urllib.request.urlopen(url, timeout=300)
    data = req.read()
    size_mb = len(data) / 1e6
    console.print(f"    {size_mb:.0f} MB descargados")

    zf = zipfile.ZipFile(io.BytesIO(data))
    dta_names = [n for n in zf.namelist() if n.endswith(".dta")]
    if not dta_names:
        raise FileNotFoundError(f"No se encontró .dta en {url}")

    cache_path.write_bytes(zf.read(dta_names[0]))
    console.print(f"  [green]✓ Guardado:[/green] {cache_path.name}")
    return cache_path


def download_casen_year(year: int, force: bool = False) -> tuple[Path, Path]:
    """Descarga microdata + archivo comunal para un año CASEN."""
    urls = CASEN_URLS[year]
    data_path = CASEN_DIR / f"casen_{year}_data.dta"
    link_path = CASEN_DIR / f"casen_{year}_link.dta"

    if force:
        data_path.unlink(missing_ok=True)
        link_path.unlink(missing_ok=True)

    console.print(f"[bold]CASEN {year}[/bold]")
    _download_dta(urls["data"], data_path, f"CASEN {year} microdata")
    _download_dta(urls["link"], link_path, f"CASEN {year} comunas")

    return data_path, link_path


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------

def _read_dta(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Lee un archivo Stata con manejo de encoding."""
    try:
        return pd.read_stata(path, convert_categoricals=False, columns=usecols)
    except UnicodeError:
        return pd.read_stata(
            path, convert_categoricals=False, columns=usecols, encoding="latin-1"
        )


def _resolve_vars(df: pd.DataFrame, year: int) -> dict[str, str]:
    """
    Resuelve el mapeo de variables para un año, con fallbacks si alguna
    columna no existe (p.ej. pobreza_multi_5d no existía en CASEN 2017).
    """
    varmap = dict(VAR_MAP[year])
    cols = set(df.columns)

    # Fallback pobreza multidimensional
    if varmap["pobreza_multi"] not in cols:
        for fallback in ("pobreza_multi_4d", "pobreza_multi_3d"):
            if fallback in cols:
                varmap["pobreza_multi"] = fallback
                console.print(f"    [dim]pobreza_multi → {fallback} (fallback)[/dim]")
                break
        else:
            varmap["pobreza_multi"] = None   # no disponible

    # Fallback alcantarillado (no disponible en CASEN 2020)
    if varmap.get("alcantarillado") and varmap["alcantarillado"] not in cols:
        varmap["alcantarillado"] = None

    # Fallback variable indígena (cambió entre encuestas)
    if varmap["indigena"] not in cols:
        for fallback in ("r_p_c", "r_pcb", "pueblo"):
            if fallback in cols:
                varmap["indigena"] = fallback
                console.print(f"    [dim]indigena → {fallback} (fallback)[/dim]")
                break
        else:
            varmap["indigena"] = None

    return varmap


# ---------------------------------------------------------------------------
# Agregación comunal ponderada
# ---------------------------------------------------------------------------

def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Media ponderada. Ignora NaN."""
    mask = series.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return float("nan")
    return (series[mask] * weights[mask]).sum() / weights[mask].sum()


def _weighted_pct(condition: pd.Series, weights: pd.Series) -> float:
    """Porcentaje ponderado de una condición booleana."""
    mask = condition.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return float("nan")
    return (condition[mask].astype(float) * weights[mask]).sum() / weights[mask].sum() * 100


def _aggregate_comuna(group: pd.DataFrame, varmap: dict[str, str]) -> dict:
    """Agrega un grupo (comuna) a las métricas estándar."""
    w = group[varmap["expr"]]
    row: dict = {
        "n_obs":        len(group),
        "representativa": len(group) >= MIN_OBS,
    }

    # Ingreso per cápita
    if "ypc" in varmap and varmap["ypc"] in group.columns:
        col = group[varmap["ypc"]].replace(0, float("nan"))
        row["ypc_promedio"] = _weighted_mean(col, w)
        # Mediana aproximada (sin ponderar exacta, suficiente para panel)
        row["ypc_mediana"] = col.dropna().median()

    # Pobreza por ingresos
    # CASEN codifica inversamente: valor MAYOR = no pobre (p.ej. 3=No pobreza)
    # El máximo representa siempre "No pobreza" → pobre = NOT at max value.
    if "pobreza" in varmap and varmap["pobreza"] in group.columns:
        pov_col = group[varmap["pobreza"]]
        pov_max = pov_col.max()  # siempre el máximo = "No pobreza"
        row["tasa_pobreza"] = _weighted_pct(pov_col < pov_max, w)

    # Pobreza multidimensional: 0=no pobre, 1=pobre (binario, codificación normal)
    if varmap.get("pobreza_multi") and varmap["pobreza_multi"] in group.columns:
        row["tasa_pobreza_multi"] = _weighted_pct(group[varmap["pobreza_multi"]] > 0, w)
    else:
        row["tasa_pobreza_multi"] = float("nan")

    # Tasa de ocupación (activ == 1, entre quienes tienen activ no nulo)
    if "activ" in varmap and varmap["activ"] in group.columns:
        activ = group[varmap["activ"]]
        in_labor_force = activ.isin([1, 2])   # ocupado o desocupado
        if in_labor_force.sum() > 0:
            row["tasa_ocupacion"] = _weighted_pct(
                (activ == 1)[in_labor_force], w[in_labor_force]
            )
        else:
            row["tasa_ocupacion"] = float("nan")

    # Escolaridad
    if "esc" in varmap and varmap["esc"] in group.columns:
        row["esc_promedio"] = _weighted_mean(
            group[varmap["esc"]].replace(-88, float("nan")).replace(-99, float("nan")),
            w,
        )

    # Agua de red (v20 == 1)
    if "agua" in varmap and varmap["agua"] in group.columns:
        row["pct_agua_red"] = _weighted_pct(group[varmap["agua"]] == 1, w)

    # Alcantarillado (v21 == 1)
    if "alcantarillado" in varmap and varmap["alcantarillado"] in group.columns:
        row["pct_alcantarillado"] = _weighted_pct(group[varmap["alcantarillado"]] == 1, w)

    # Población indígena
    if varmap.get("indigena") and varmap["indigena"] in group.columns:
        row["pct_indigena"] = _weighted_pct(group[varmap["indigena"]] == 1, w)
    else:
        row["pct_indigena"] = float("nan")

    # Zona urbana (area == 1)
    if "area" in varmap and varmap["area"] in group.columns:
        row["pct_urbano"] = _weighted_pct(group[varmap["area"]] == 1, w)

    return row


def _build_casen_crosswalk(
    link_path: Path,
    gdf_comunas: gpd.GeoDataFrame,
) -> dict[str, str]:
    """
    Construye un crosswalk {INE_cut_code → GADM_cut_code} leyendo las
    etiquetas de comuna del archivo link de CASEN y haciendo fuzzy-match
    con los nombres del GeoDataFrame comunal.

    Retorna un dict vacío si no se puede construir (falla silenciosa).
    """
    try:
        # Leer con etiquetas para obtener nombres de comunas
        df_labels = pd.read_stata(
            link_path, convert_categoricals=True,
            columns=["folio", "comuna"]
        )
        df_codes = pd.read_stata(
            link_path, convert_categoricals=False,
            columns=["folio", "comuna"]
        )
        # Construir: INE_cut_code → nombre_etiqueta
        df_labels = df_labels.rename(columns={"comuna": "nombre"})
        df_codes["cut_ine"] = df_codes["comuna"].astype(int).astype(str).str.zfill(5)
        mapping_df = (
            df_labels[["folio", "nombre"]]
            .merge(df_codes[["folio", "cut_ine"]], on="folio")
            .drop_duplicates("cut_ine")
        )

        # Lookup GADM: nombre_normalizado → GADM_cut_code
        gadm_lookup: dict[str, str] = {
            _normalize(str(r["name"])): str(r["cut_code"])
            for _, r in gdf_comunas.iterrows()
        }

        crosswalk: dict[str, str] = {}
        unmatched = []
        for _, row in mapping_df.iterrows():
            norm = _normalize(str(row["nombre"]))
            if norm in gadm_lookup:
                crosswalk[row["cut_ine"]] = gadm_lookup[norm]
            else:
                candidates = difflib.get_close_matches(norm, gadm_lookup.keys(), n=1, cutoff=0.80)
                if candidates:
                    crosswalk[row["cut_ine"]] = gadm_lookup[candidates[0]]
                else:
                    unmatched.append(row["nombre"])

        matched = len(crosswalk)
        total = len(mapping_df)
        console.print(
            f"  Crosswalk CASEN→GADM: {matched}/{total} comunas mapeadas"
        )
        if unmatched:
            console.print(f"  [yellow]Sin match ({len(unmatched)}): {sorted(unmatched)[:8]}[/yellow]")
        return crosswalk
    except Exception as e:
        console.print(f"  [yellow]⚠ Crosswalk CASEN→GADM no disponible: {e}[/yellow]")
        return {}


def aggregate_to_comunas(
    df_data: pd.DataFrame,
    df_link: pd.DataFrame,
    year: int,
    crosswalk: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Une microdata con archivo comunal y agrega a nivel (cut_code, year).

    Si se proporciona crosswalk {INE_cut→GADM_cut}, remapea los cut_codes
    a los del GeoDataFrame comunal para asegurar join correcto con panel_anual.

    Retorna DataFrame con una fila por comuna.
    """
    console.print(f"  Uniendo microdata ({len(df_data):,} personas) con archivo comunal…")

    # Normalizar cut_code: CASEN usa entero (p.ej. 13101), nosotros VARCHAR(5)
    df_link = df_link.copy()
    df_link["cut_code"] = df_link["comuna"].astype(int).astype(str).str.zfill(5)

    # Remap INE → GADM si hay crosswalk disponible
    if crosswalk:
        df_link["cut_code"] = df_link["cut_code"].map(crosswalk).fillna(df_link["cut_code"])

    # Join: folio + id_persona si ambos archivos tienen la columna;
    # si el archivo de datos no tiene id_persona (CASEN 2017), deduplicamos
    # el link por folio (todos los miembros del hogar tienen el mismo cut_code).
    has_id = "id_persona" in df_data.columns and "id_persona" in df_link.columns
    join_keys = ["folio", "id_persona"] if has_id else ["folio"]

    link_use = df_link[["folio"] + (["id_persona"] if has_id else []) + ["cut_code", "expc"]]
    if not has_id:
        # Tomar una fila por hogar (cut_code y expc son iguales para todos los miembros)
        link_use = link_use.drop_duplicates("folio")

    df_merged = df_data.merge(link_use, on=join_keys, how="inner")
    console.print(f"  Registros tras join: {len(df_merged):,} ({len(df_merged)/len(df_data)*100:.1f}%)")

    # Resolver variables
    varmap = _resolve_vars(df_merged, year)

    # Agregar por comuna
    records = []
    for cut_code, group in df_merged.groupby("cut_code"):
        row = _aggregate_comuna(group, varmap)
        row["cut_code"] = cut_code
        row["year"] = year
        records.append(row)

    df_result = pd.DataFrame(records)
    n_repr = df_result["representativa"].sum()
    console.print(
        f"  [green]✓ CASEN {year}: {len(df_result)} comunas "
        f"({n_repr} representativas, {len(df_result)-n_repr} con n<{MIN_OBS})[/green]"
    )
    return df_result


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_casen_dataframe(
    years: list[int] | None = None,
    force: bool = False,
    gdf_comunas: Optional[gpd.GeoDataFrame] = None,
) -> pd.DataFrame:
    """
    Descarga y agrega CASEN para los años indicados.
    Por defecto procesa 2017, 2020 y 2022.

    Si se proporciona gdf_comunas, remapea los cut_codes de INE a los
    cut_codes GADM usados por panel_anual y el resto del pipeline.

    Retorna DataFrame largo con columnas:
        cut_code, year, n_obs, representativa,
        ypc_promedio, ypc_mediana, tasa_pobreza, tasa_pobreza_multi,
        tasa_ocupacion, esc_promedio, pct_agua_red, pct_alcantarillado,
        pct_indigena, pct_urbano
    """
    if years is None:
        years = [2017, 2020, 2022]

    console.print("[bold cyan]📊 Procesando CASEN…[/bold cyan]")
    dfs = []

    for year in years:
        console.print(f"\n[bold]── CASEN {year} ──[/bold]")
        try:
            data_path, link_path = download_casen_year(year, force=force)

            # Construir crosswalk INE→GADM usando los nombres del link file
            crosswalk: dict[str, str] = {}
            if gdf_comunas is not None:
                crosswalk = _build_casen_crosswalk(link_path, gdf_comunas)

            # Cargar solo las columnas necesarias del archivo de datos
            varmap = VAR_MAP[year]
            needed_cols = list(set(varmap.values())) + ["folio"]
            if "id_persona" not in needed_cols:
                needed_cols.append("id_persona")

            console.print(f"  Leyendo {data_path.name}…")
            df_data = _read_dta(data_path)
            # Conservar solo columnas que existen
            keep = [c for c in needed_cols if c in df_data.columns]
            df_data = df_data[keep]

            df_link = _read_dta(link_path)
            df_year = aggregate_to_comunas(df_data, df_link, year, crosswalk=crosswalk)
            dfs.append(df_year)

        except Exception as e:
            console.print(f"  [yellow]⚠ CASEN {year} omitido por error: {e}[/yellow]")

    if not dfs:
        raise RuntimeError("No se pudo procesar ningún año CASEN.")

    df_final = pd.concat(dfs, ignore_index=True)
    console.print(
        f"\n[green]✓ CASEN consolidado: {len(df_final):,} filas "
        f"({df_final['cut_code'].nunique()} comunas × {df_final['year'].nunique()} años)[/green]"
    )
    return df_final
