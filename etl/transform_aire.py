"""
ETL Calidad del Aire — SINCA (Sistema de Información Nacional de Calidad del Aire)

Extrae promedios anuales de PM2.5 y PM10 de las estaciones de monitoreo de SINCA
usando el sistema AIRVIRO (apub.tsindico2.cgi). No requiere registro ni API key.

Fuente: Ministerio del Medio Ambiente — SINCA
https://sinca.mma.gob.cl/

Cobertura:
    ~218 estaciones en todo Chile. No todas tienen PM2.5 (muchas sólo tienen PM10).
    Aprox. 80–120 comunas cubiertas con al menos un contaminante.

Nota sobre representatividad:
    Los datos corresponden a estaciones de monitoreo urbanas/industriales.
    Comunas sin estación quedan con NULL. Para cobertura nacional completa se
    recomienda complementar con datos satelitales (ACAG V6, requiere cuenta AWS).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rich.console import Console

from etl.config import RAW_DIR
from etl.transform import _normalize

console = Console()

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

SINCA_BASE      = "https://sinca.mma.gob.cl"
CGI_HTML        = f"{SINCA_BASE}/cgi-bin/APUB-MMA/apub.htmlindico2.cgi"
CGI_DATA        = f"{SINCA_BASE}/cgi-bin/APUB-MMA/apub.tsindico2.cgi"
AIRVIRO_PATH    = "/usr/airviro/data/CONAMA/"

AIRE_DIR        = RAW_DIR / "aire_sinca"
AIRE_CACHE      = AIRE_DIR / "sinca_anual.parquet"

# Regiones disponibles en SINCA
REGIONS = ["M","I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIV","XV","XVI"]

# Contaminantes de interés (nombre en macropath)
CONTAMINANTES = ["PM25", "PM10"]

# Correcciones manuales de nombres de estaciones/comunas → estándar
MANUAL_FIXES: dict[str, str] = {
    "santiago centro":    "santiago",
    "pudahuel":           "pudahuel",
    "cerrillos":          "cerrillos",
    "quilicura":          "quilicura",
    "independencia":      "independencia",
    "el bosque":          "el bosque",
    "cerro navia":        "cerro navia",
    "la florida":         "la florida",
    "las condes":         "las condes",
    "providencia":        "providencia",
    "talagante":          "talagante",
    "padre hurtado":      "padre hurtado",
}

# Espera entre requests para no saturar el servidor
REQUEST_DELAY = 0.3  # segundos


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,*/*",
    })
    return s


# ---------------------------------------------------------------------------
# Descubrimiento de estaciones
# ---------------------------------------------------------------------------

def get_station_ids(session: requests.Session) -> dict[str, list[str]]:
    """Devuelve {region_code: [station_id, ...]} para todas las regiones."""
    stations: dict[str, list[str]] = {}
    for reg in REGIONS:
        try:
            r = session.get(f"{SINCA_BASE}/index.php/region/index/id/{reg}", timeout=20)
            ids = list(set(re.findall(r"/index\.php/estacion/index/id/(\d+)", r.text)))
            if ids:
                stations[reg] = sorted(ids, key=int)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            console.print(f"    [yellow]⚠ Región {reg}: {e}[/yellow]")
    total = sum(len(v) for v in stations.values())
    console.print(f"  ✓ {total} estaciones en {len(stations)} regiones")
    return stations


def parse_station_page(
    session: requests.Session,
    station_id: str,
) -> dict | None:
    """
    Parsea la página de una estación SINCA.
    Devuelve dict con: station_id, name, commune, region, macropaths_pm25, macropaths_pm10
    """
    try:
        r = session.get(
            f"{SINCA_BASE}/index.php/estacion/index/id/{station_id}",
            timeout=20,
        )
        soup = BeautifulSoup(r.text, "html.parser")

        # Metadata básica de la tabla general
        info: dict[str, str] = {}
        for row in soup.select("table#tablaGeneral tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) == 2:
                info[cells[0].get_text(strip=True)] = cells[1].get_text(strip=True)

        commune  = info.get("Comuna", "").strip()
        region   = info.get("Región", "").strip()

        # Extraer todos los CGI links (ic dropdown values) del HTML
        all_ic_values = re.findall(
            r'value="(\./[^"]+\.ic)"',
            r.text,
        )
        # También buscar en los href
        cgi_links = re.findall(
            r'macropath=([^&"\']+)',
            r.text,
        )

        # Filtrar macropaths para PM25 y PM10
        macropaths: dict[str, list[str]] = {"PM25": [], "PM10": []}
        for ic in all_ic_values:
            for cont in CONTAMINANTES:
                if f"/{cont}/" in ic and "anual" in ic:
                    # Extract the base macropath (without the ic filename part)
                    macropaths[cont].append(ic)

        # Si no encontramos en ic values, buscar en cgi links
        if not any(macropaths.values()):
            for mp in cgi_links:
                for cont in CONTAMINANTES:
                    if f"/Cal/{cont}" in mp:
                        # Construct the ic path
                        ic = f"{mp}//{cont}.diario.anual.ic"
                        if ic not in macropaths[cont]:
                            macropaths[cont].append(ic)

        # Nombre de la estación (del header de la página)
        stn_header = soup.find("label", class_="stn")
        if not stn_header:
            # Try to get name from page title
            title = soup.find("title")
            stn_name = title.get_text(strip=True) if title else f"Estacion_{station_id}"
        else:
            stn_name = stn_header.get_text(strip=True)

        if not commune:
            return None

        return {
            "station_id":  station_id,
            "name":        stn_name,
            "commune":     commune,
            "region":      region,
            "ic_pm25":     macropaths["PM25"][:1],  # First annual PM25 macro
            "ic_pm10":     macropaths["PM10"][:1],  # First annual PM10 macro
        }

    except Exception as e:
        console.print(f"    [yellow]⚠ Estación {station_id}: {e}[/yellow]")
        return None


# ---------------------------------------------------------------------------
# Descarga de datos anuales
# ---------------------------------------------------------------------------

def parse_airviro_txt(text: str) -> list[tuple[int, float]]:
    """
    Parsea el formato de texto de AIRVIRO/psgraph.
    Devuelve lista de (year, value) para registros validados con datos.

    Formato de línea: YYMMDD, HHMM, validated, preliminary, unvalidated,
    """
    records = []
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if line == "#DATA":
            in_data = True
            continue
        if line.startswith("EOF") or line.startswith("#") and in_data and line != "#DATA":
            in_data = False
            continue
        if not in_data or not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        try:
            date_str = parts[0].strip()
            year_2d  = int(date_str[:2])
            year     = 2000 + year_2d

            val_str = parts[2].strip()
            if not val_str:
                continue
            val = float(val_str)
            if val > 0:
                records.append((year, val))
        except (ValueError, IndexError):
            continue

    return records


def download_annual_data(
    session: requests.Session,
    ic_path: str,
    from_date: str = "990101",
    to_date:   str = "241231",
) -> list[tuple[int, float]]:
    """
    Descarga promedios anuales de una estación/contaminante vía AIRVIRO CGI.
    Devuelve lista de (year, pm_value).
    """
    try:
        params = {
            "outtype":   "txt",
            "macro":     ic_path,
            "from":      from_date,
            "to":        to_date,
            "path":      AIRVIRO_PATH,
            "lang":      "esp",
            "rsrc":      "",
            "macropath": "",
        }
        r = session.get(CGI_DATA, params=params, timeout=30)
        if r.status_code != 200 or "psgraph" in r.text[:50]:
            return []
        return parse_airviro_txt(r.text)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Join comunal
# ---------------------------------------------------------------------------

def join_aire_with_comunas(
    df: pd.DataFrame,
    gdf_comunas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Une el DataFrame de aire con cut_codes comunales."""
    import difflib

    gadm_norm = {
        _normalize(r["name"]): r["cut_code"]
        for _, r in gdf_comunas.iterrows()
    }

    def match(raw: str) -> Optional[str]:
        norm = _normalize(raw)
        norm = MANUAL_FIXES.get(norm, norm)
        if norm in gadm_norm:
            return gadm_norm[norm]
        candidates = difflib.get_close_matches(norm, gadm_norm.keys(), n=1, cutoff=0.82)
        return gadm_norm[candidates[0]] if candidates else None

    df = df.copy()
    df["cut_code"] = df["commune"].map(match)

    matched = df["cut_code"].notna().sum()
    total   = len(df)
    console.print(
        f"  Join aire: {matched:,}/{total:,} filas con cut_code "
        f"({matched/total*100:.1f}%)"
    )
    unmatched = df[df["cut_code"].isna()]["commune"].unique()
    if len(unmatched):
        console.print(f"  [yellow]Sin match ({len(unmatched)}): {sorted(unmatched)[:10]}[/yellow]")

    return df[df["cut_code"].notna()].copy()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def build_aire_dataframe(
    gdf_comunas: gpd.GeoDataFrame,
    force: bool = False,
    max_stations: int | None = None,
) -> pd.DataFrame:
    """
    Pipeline completo: descubrir estaciones → descargar datos anuales →
    join comunal → agregar por (cut_code × year).

    Retorna DataFrame con columnas:
        cut_code, year, pm25_mean, pm10_mean,
        n_stations_pm25, n_stations_pm10
    """
    console.print("[bold cyan]🌬 Procesando calidad del aire (SINCA)…[/bold cyan]")

    AIRE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Caché ──────────────────────────────────────────────────────────────
    if AIRE_CACHE.exists() and not force:
        console.print(f"  [green]✓ Cache existente: {AIRE_CACHE.name}[/green]")
        df_cache = pd.read_parquet(AIRE_CACHE)
        return _aggregate_to_comunas(df_cache, gdf_comunas)

    # ── Descubrimiento de estaciones ───────────────────────────────────────
    session = _make_session()
    console.print("  Descubriendo estaciones SINCA…")
    all_stations = get_station_ids(session)

    station_ids_flat = []
    for ids in all_stations.values():
        station_ids_flat.extend(ids)
    if max_stations:
        station_ids_flat = station_ids_flat[:max_stations]

    console.print(f"  Procesando {len(station_ids_flat)} estaciones…")

    # ── Parsear metadata de estaciones ─────────────────────────────────────
    station_meta: list[dict] = []
    for i, sid in enumerate(station_ids_flat):
        meta = parse_station_page(session, sid)
        if meta and (meta["ic_pm25"] or meta["ic_pm10"]):
            station_meta.append(meta)
        if (i + 1) % 20 == 0:
            n_pm25 = sum(1 for m in station_meta if m["ic_pm25"])
            n_pm10 = sum(1 for m in station_meta if m["ic_pm10"])
            console.print(
                f"    {i+1}/{len(station_ids_flat)} estaciones — "
                f"{len(station_meta)} con PM25/PM10 "
                f"(PM2.5: {n_pm25}, PM10: {n_pm10})"
            )
        time.sleep(REQUEST_DELAY)

    console.print(f"  ✓ {len(station_meta)} estaciones con PM2.5 y/o PM10")

    # ── Descargar datos anuales ─────────────────────────────────────────────
    records = []
    for i, meta in enumerate(station_meta):
        station_id = meta["station_id"]
        commune    = meta["commune"]

        # PM2.5
        for ic in meta["ic_pm25"]:
            annual = download_annual_data(session, ic)
            for year, val in annual:
                records.append({
                    "station_id": station_id,
                    "commune":    commune,
                    "year":       year,
                    "contaminante": "pm25",
                    "value":      val,
                })

        # PM10
        for ic in meta["ic_pm10"]:
            annual = download_annual_data(session, ic)
            for year, val in annual:
                records.append({
                    "station_id": station_id,
                    "commune":    commune,
                    "year":       year,
                    "contaminante": "pm10",
                    "value":      val,
                })

        time.sleep(REQUEST_DELAY)

        if (i + 1) % 10 == 0:
            console.print(f"    {i+1}/{len(station_meta)} estaciones descargadas — {len(records)} registros")

    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        raise RuntimeError("No se descargó ningún dato de calidad del aire de SINCA.")

    # Guardar caché
    df_raw.to_parquet(AIRE_CACHE, index=False)
    console.print(
        f"  ✓ Datos descargados: {len(df_raw):,} registros "
        f"| Cache: {AIRE_CACHE.name}"
    )

    return _aggregate_to_comunas(df_raw, gdf_comunas)


def _aggregate_to_comunas(
    df_raw: pd.DataFrame,
    gdf_comunas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Une con comunas y agrega por (cut_code × year)."""
    df_joined = join_aire_with_comunas(df_raw, gdf_comunas)

    # Pivot: separar PM25 y PM10
    df_pm25 = (
        df_joined[df_joined["contaminante"] == "pm25"]
        .groupby(["cut_code", "year"])
        .agg(pm25_mean=("value", "mean"), n_stations_pm25=("station_id", "nunique"))
        .reset_index()
    )
    df_pm10 = (
        df_joined[df_joined["contaminante"] == "pm10"]
        .groupby(["cut_code", "year"])
        .agg(pm10_mean=("value", "mean"), n_stations_pm10=("station_id", "nunique"))
        .reset_index()
    )

    df_agg = pd.merge(df_pm25, df_pm10, on=["cut_code", "year"], how="outer")
    df_agg = df_agg.fillna({"n_stations_pm25": 0, "n_stations_pm10": 0})
    df_agg["n_stations_pm25"] = df_agg["n_stations_pm25"].astype(int)
    df_agg["n_stations_pm10"] = df_agg["n_stations_pm10"].astype(int)

    n_com = df_agg["cut_code"].nunique()
    n_yr  = df_agg["year"].nunique()
    console.print(
        f"  [green]✓ Calidad del aire lista: {len(df_agg):,} registros · "
        f"{n_com} comunas · {n_yr} años[/green]"
    )
    return df_agg
