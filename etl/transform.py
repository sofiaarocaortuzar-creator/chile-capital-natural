"""
Transformación de datos crudos al esquema normalizado.

Flujo:
  1. Lee el XLSX de MapBiomas (hoja COVERAGE) → DataFrame long format
  2. Lee el shapefile de comunas DPA 2023 → GeoDataFrame
  3. Normaliza nombres de comunas y hace el join para obtener código CUT
  4. Retorna DataFrames listos para cargar en DuckDB
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rich.console import Console

from etl.config import (
    COMUNAS_GEOJSON_PATH,
    COVERAGE_CLASSES,
    MAPBIOMAS_XLSX_PATH,
    PROCESSED_DIR,
    YEARS,
)

console = Console()

# ---------------------------------------------------------------------------
# Normalización de nombres para join fuzzy
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Normaliza un nombre de comuna para comparación robusta.

    Pasos: lowercase → sin acentos → sin caracteres especiales → sin espacios extra.
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    # Eliminar acentos / diacríticos
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Reemplazar caracteres especiales (apóstrofes, guiones) por espacio
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ---------------------------------------------------------------------------
# Carga y limpieza del XLSX de MapBiomas
# ---------------------------------------------------------------------------

def load_mapbiomas_xlsx(xlsx_path: Path = MAPBIOMAS_XLSX_PATH) -> pd.DataFrame:
    """Lee la hoja COVERAGE del XLSX y devuelve un DataFrame en formato largo.

    Columnas del resultado:
        region_name, provincia_name, comuna_name, class_id,
        year, area_ha
    """
    console.print("[bold cyan]📄 Leyendo XLSX MapBiomas…[/bold cyan]")
    console.print(f"  Archivo: {xlsx_path} ({xlsx_path.stat().st_size / 1e6:.1f} MB)")
    console.print("  (puede tardar 20-40 segundos)")

    df = pd.read_excel(
        xlsx_path,
        sheet_name="COVERAGE",
        engine="openpyxl",
    )

    console.print(f"  [green]✓ {len(df):,} filas cargadas[/green]")
    console.print(f"  Columnas disponibles: {list(df.columns[:15])} …")

    # Detectar columnas de territorio
    col_map = _detect_territory_columns(df)
    console.print(f"  Columnas territorio detectadas: {col_map}")

    # Detectar columnas de año (patrón y1999, y2000, … o 1999, 2000, …)
    year_cols = _detect_year_columns(df)
    console.print(f"  Años detectados: {year_cols[0]} → {year_cols[-1]} ({len(year_cols)} años)")

    # Renombrar columnas de territorio a nombres canónicos
    df = df.rename(columns=col_map)

    # Columna de clase
    class_col = _detect_class_column(df)
    df = df.rename(columns={class_col: "class_id"})

    # Mantener solo filas de nivel comuna (territory_level_4 no nulo)
    df = df[df["comuna_name"].notna()].copy()

    # Convertir class_id a int
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")
    df = df[df["class_id"].notna()]
    df["class_id"] = df["class_id"].astype(int)

    # Solo clases conocidas
    known_classes = set(COVERAGE_CLASSES.keys())
    df = df[df["class_id"].isin(known_classes)]

    # Melt: pasar de wide a long por año
    id_vars = ["region_name", "provincia_name", "comuna_name", "class_id"]
    df_long = df[id_vars + year_cols].melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year_col",
        value_name="area_ha",
    )

    # Extraer año numérico de la columna (puede ser "y1999" o "1999")
    df_long["year"] = (
        df_long["year_col"]
        .astype(str)
        .str.extract(r"(\d{4})")
        .astype(int)
    )
    df_long = df_long.drop(columns=["year_col"])

    # Limpiar valores nulos/negativos
    df_long["area_ha"] = pd.to_numeric(df_long["area_ha"], errors="coerce").fillna(0.0)
    df_long = df_long[df_long["area_ha"] >= 0]

    # Normalizar nombres para join posterior
    df_long["_norm_name"] = df_long["comuna_name"].map(_normalize)

    console.print(
        f"  [green]✓ DataFrame long: {len(df_long):,} filas "
        f"({df_long['year'].min()}–{df_long['year'].max()}, "
        f"{df_long['class_id'].nunique()} clases, "
        f"{df_long['comuna_name'].nunique()} comunas)[/green]"
    )
    return df_long


def _detect_territory_columns(df: pd.DataFrame) -> dict[str, str]:
    """Detecta y mapea las columnas de territorio a nombres canónicos.

    Solo mapea columnas que empiezan con 'territory_' (no 'class_level_*').
    """
    rename = {}
    for col in df.columns:
        col_lower = col.lower()
        # Solo procesar columnas de territorio, no de clase
        if not col_lower.startswith("territory_"):
            continue
        if col_lower == "territory_level_1":
            rename[col] = "country_name"
        elif col_lower == "territory_level_2":
            rename[col] = "region_name"
        elif col_lower == "territory_level_3":
            rename[col] = "provincia_name"
        elif col_lower == "territory_level_4":
            rename[col] = "comuna_name"
    return rename


def _detect_year_columns(df: pd.DataFrame) -> list[str]:
    """Detecta columnas de año (formato y1999 o 1999)."""
    year_cols = [
        col for col in df.columns
        if re.match(r"^y?\d{4}$", str(col).strip())
        and int(re.search(r"\d{4}", str(col)).group()) in range(1990, 2030)
    ]
    return sorted(year_cols)


def _detect_class_column(df: pd.DataFrame) -> str:
    """Detecta la columna de código de clase de cobertura."""
    for col in df.columns:
        if col.lower() in ("class", "class_id", "mapbiomas_class", "codigo"):
            return col
    raise ValueError(f"No se encontró columna de clase en: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Carga y limpieza del shapefile de comunas
# ---------------------------------------------------------------------------

def load_comunas_geodataframe() -> gpd.GeoDataFrame:
    """Lee el GeoJSON GADM 4.1 nivel 2 de comunas y devuelve un GeoDataFrame limpio.

    Columnas garantizadas:
        cut_code (str), name (str), region_name (str),
        area_km2 (float), geometry (EPSG:4326)

    Fuente: GADM 4.1 – columnas NAME_2 (comuna), NAME_1 (región), GID_2 (id único).
    Nota: GADM no incluye el código CUT del INE. Se genera un ID sintético
          basado en GID_2. El código CUT real se puede agregar más adelante
          vía join con un CSV del INE.
    """
    geojson_file = _find_comunas_file()

    console.print(f"[bold cyan]🗺 Cargando comunas:[/bold cyan] {geojson_file.name}")
    gdf = gpd.read_file(geojson_file)
    console.print(f"  Filas: {len(gdf)} | CRS original: {gdf.crs}")
    console.print(f"  Columnas disponibles: {list(gdf.columns)}")

    # Reproyectar a WGS84 si es necesario
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # --- Detectar columnas según la fuente ---
    # GADM 4.1: NAME_2 = nombre comuna, NAME_1 = nombre región, GID_2 = id
    # INE/DPA:  NOM_COM = nombre, NOM_REG = región, CUT_COM = código
    # GADM nivel 3: NAME_3 = comuna, NAME_1 = región, GID_3 = id único
    # GADM nivel 2: NAME_2 = provincia, NAME_1 = región
    # INE/DPA: NOM_COM, NOM_REG, CUT_COM
    name_col   = _pick_col(gdf, ["NAME_3", "NAME_2", "NOM_COM", "NOMBRE", "COMUNA"])
    region_col = _pick_col(gdf, ["NAME_1", "NOM_REG", "REGION", "NOMBRE_REG"])
    id_col     = _pick_col(gdf, ["GID_3", "GID_2", "CC_3", "CC_2", "CUT_COM", "CUT_COMU", "COD_COM", "CODIGO"])

    console.print(f"  → nombre: '{name_col}' | región: '{region_col}' | id: '{id_col}'")

    if name_col is None:
        raise ValueError(f"No se encontró columna de nombre de comuna en: {list(gdf.columns)}")

    # Construir código de identificación
    if id_col and id_col in ("CUT_COM", "CUT_COMU", "COD_COM", "CODIGO"):
        # CUT real del INE (5 dígitos)
        cut_series = gdf[id_col].astype(str).str.zfill(5)
    elif id_col:
        # GADM GID_2: "CHL.1.1_1" → extraer número ordinal como código sintético
        cut_series = (
            gdf[id_col].astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
            .str.replace(".", "", regex=False)
            .str[:5]
            .str.zfill(5)
        )
    else:
        # Último recurso: índice secuencial
        cut_series = pd.Series(
            [str(i).zfill(5) for i in range(len(gdf))], index=gdf.index
        )

    # Construir GeoDataFrame limpio
    clean = gpd.GeoDataFrame(
        {
            "cut_code":    cut_series,
            "name":        gdf[name_col].astype(str).str.strip(),
            "region_name": gdf[region_col].astype(str).str.strip() if region_col else "Sin región",
            "geometry":    gdf.geometry,
        },
        crs="EPSG:4326",
    )

    # Calcular área en km² usando proyección UTM 18S (apropiada para Chile)
    clean_proj = clean.to_crs(epsg=32718)
    clean["area_km2"] = clean_proj.geometry.area / 1_000_000

    # Nombre normalizado para join con MapBiomas
    clean["_norm_name"] = clean["name"].map(_normalize)

    # Asegurar cut_codes únicos (GADM puede tener duplicados en zonas extremas)
    clean = clean.drop_duplicates(subset="cut_code").reset_index(drop=True)

    console.print(f"  [green]✓ {len(clean)} comunas cargadas[/green]")
    return clean


def _find_comunas_file() -> Path:
    """Encuentra el archivo de comunas en el directorio extraído (JSON o SHP)."""
    from etl.config import COMUNAS_EXTRACT_DIR
    if not COMUNAS_EXTRACT_DIR.exists():
        raise FileNotFoundError(
            f"Directorio no encontrado: {COMUNAS_EXTRACT_DIR}. "
            "Ejecuta primero el pipeline."
        )
    # Buscar GeoJSON primero (GADM), luego shapefile (DPA INE)
    for pattern in ("*.json", "*.geojson", "*.shp"):
        candidates = sorted(COMUNAS_EXTRACT_DIR.rglob(pattern))
        # Priorizar archivos con "comun" o "_2" en el nombre (GADM level 2)
        preferred = [p for p in candidates if "comun" in p.stem.lower() or "_2" in p.stem]
        if preferred:
            return preferred[0]
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No se encontró archivo de comunas en {COMUNAS_EXTRACT_DIR}")


def _pick_col(gdf: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    """Retorna la primera columna candidata que existe en el GeoDataFrame."""
    for col in candidates:
        if col in gdf.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Join: MapBiomas ↔ Comunas
# ---------------------------------------------------------------------------

def join_mapbiomas_with_comunas(
    df_mapbiomas: pd.DataFrame,
    gdf_comunas: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Enriquece el DataFrame de MapBiomas con el código de comuna del GeoDataFrame.

    Estrategia de join (en cascada):
      1. Join exacto por nombre normalizado.
      2. Fixes manuales para grafías conocidas GADM vs MapBiomas.
      3. Fuzzy matching (difflib) para los restantes sin match.
    """
    import difflib

    console.print("[bold cyan]🔗 Uniendo MapBiomas ↔ comunas GADM…[/bold cyan]")

    # Tabla de lookup: norm_name → cut_code, area_km2
    lookup = (
        gdf_comunas[["_norm_name", "cut_code", "area_km2", "name"]]
        .drop_duplicates("_norm_name")
        .set_index("_norm_name")
    )
    gadm_names = list(lookup.index)  # nombres normalizados disponibles en GADM

    df = df_mapbiomas.copy()

    # --- Paso 1: join exacto ---
    df["cut_code"] = df["_norm_name"].map(lookup["cut_code"])
    df["area_km2"] = df["_norm_name"].map(lookup["area_km2"])

    matched = df["cut_code"].notna().sum()
    total   = len(df)
    console.print(f"  Paso 1 — exacto: {matched:,}/{total:,} filas ({matched/total*100:.1f}%)")

    # --- Paso 2: fixes manuales ---
    manual_fixes = _build_manual_fixes()
    for bad_name, good_norm in manual_fixes.items():
        mask = (df["_norm_name"] == bad_name) & df["cut_code"].isna()
        if mask.any() and good_norm in lookup.index:
            df.loc[mask, "cut_code"] = lookup.loc[good_norm, "cut_code"]
            df.loc[mask, "area_km2"] = lookup.loc[good_norm, "area_km2"]

    after_manual = df["cut_code"].notna().sum()
    console.print(f"  Paso 2 — manuales: {after_manual:,}/{total:,} filas ({after_manual/total*100:.1f}%)")

    # --- Paso 3: fuzzy matching para los restantes ---
    unmatched_names = df[df["cut_code"].isna()]["_norm_name"].unique()
    fuzzy_map: dict[str, str] = {}
    for name in unmatched_names:
        candidates = difflib.get_close_matches(name, gadm_names, n=1, cutoff=0.82)
        if candidates:
            fuzzy_map[name] = candidates[0]

    for mb_name, gadm_name in fuzzy_map.items():
        mask = (df["_norm_name"] == mb_name) & df["cut_code"].isna()
        if mask.any():
            df.loc[mask, "cut_code"] = lookup.loc[gadm_name, "cut_code"]
            df.loc[mask, "area_km2"] = lookup.loc[gadm_name, "area_km2"]

    after_fuzzy = df["cut_code"].notna().sum()
    console.print(f"  Paso 3 — fuzzy:    {after_fuzzy:,}/{total:,} filas ({after_fuzzy/total*100:.1f}%)")

    # Reportar comunas aún sin match
    still_unmatched = df[df["cut_code"].isna()]["comuna_name"].unique()
    if len(still_unmatched) > 0:
        console.print(
            f"  [yellow]⚠ Sin match final ({len(still_unmatched)} comunas):[/yellow] "
            f"{list(still_unmatched[:8])}"
            + (" …" if len(still_unmatched) > 8 else "")
        )

    df = df.drop(columns=["_norm_name"])
    df = df[df["cut_code"].notna()].copy()
    console.print(f"  [green]✓ DataFrame final: {len(df):,} filas con ID válido[/green]")
    return df


def _build_manual_fixes() -> dict[str, str]:
    """Mapeo: nombre_normalizado_MapBiomas → nombre_normalizado_GADM.

    GADM 4.1 usa grafías distintas a las del INE/MapBiomas para varios
    topónimos chilenos. Esta tabla cubre los casos más frecuentes.
    """
    return {
        # Acentos y eñes
        "nunoa":                        "nunoa",
        "vina del mar":                 "vina del mar",
        "concepcion":                   "concepcion",
        # Diferencias GADM vs MapBiomas
        "aisen":                        "aysen",
        "rio ibanes":                   "rio ibanez",
        "marchigue":                    "marchihue",
        "paiguano":                     "paihuano",
        "san pedro atacama":            "san pedro de atacama",
        "general lagos":                "general lagos",
        "nueva imperial":               "nueva imperial",
        "padre las casas":              "padre las casas",
        "teodoro schmidt":              "teodoro schmidt",
        "los sauces":                   "los sauces",
        "los alamos":                   "los alamos",
        "alto biobio":                  "alto biobio",
        "los vilos":                    "los vilos",
        "monte patria":                 "monte patria",
        "la ligua":                     "la ligua",
        "llanquihue":                   "llanquihue",
        "cochamo":                      "cochamo",
        "hualaihue":                    "hualaihue",
        "puqueldon":                    "puqueldon",
        "dalcahue":                     "dalcahue",
        "curaco de velez":              "curaco de velez",
        "quinchao":                     "quinchao",
        "chonchi":                      "chonchi",
        "quellon":                      "quellon",
        "chaiten":                      "chaiten",
        "futaleufu":                    "futaleufu",
        "palena":                       "palena",
        "coyhaique":                    "coyhaique",
        "lago verde":                   "lago verde",
        "aysen":                        "aysen",
        "cisnes":                       "cisnes",
        "guaitecas":                    "guaitecas",
        "cochrane":                     "cochrane",
        "ohiggins":                     "o higgins",
        "tortel":                       "tortel",
        "villa o higgins":              "villa o higgins",
        "natales":                      "natales",
        "torres del paine":             "torres del paine",
        "punta arenas":                 "punta arenas",
        "rio verde":                    "rio verde",
        "laguna blanca":                "laguna blanca",
        "san gregorio":                 "san gregorio",
        "cabo de hornos":               "cabo de hornos",
        "antartica":                    "antartica",
        "primavera":                    "primavera",
        "timaukel":                     "timaukel",
        "navarino":                     "navarino",
        "porvenir":                     "porvenir",
        "camarones":                    "camarones",
        "colchane":                     "colchane",
        "huara":                        "huara",
        "pica":                         "pica",
        "pozo almonte":                 "pozo almonte",
        "alto hospicio":                "alto hospicio",
        "iquique":                      "iquique",
        "camina":                       "camina",
        "colchane":                     "colchane",
        "calama":                       "calama",
        "mejillones":                   "mejillones",
        "sierra gorda":                 "sierra gorda",
        "taltal":                       "taltal",
        "antofagasta":                  "antofagasta",
        "ollagüe":                      "ollague",
        "ollague":                      "ollague",
        "san pedro de atacama":         "san pedro de atacama",
        "tocopilla":                    "tocopilla",
        "maria elena":                  "maria elena",
        "diego de almagro":             "diego de almagro",
        "caldera":                      "caldera",
        "chanaral":                     "chanaral",
        "copiapo":                      "copiapo",
        "tierra amarilla":              "tierra amarilla",
        "alto del carmen":              "alto del carmen",
        "freirina":                     "freirina",
        "huasco":                       "huasco",
        "vallenar":                     "vallenar",
        "coquimbo":                     "coquimbo",
        "la serena":                    "la serena",
        "andacollo":                    "andacollo",
        "canela":                       "canela",
        "illapel":                      "illapel",
        "los vilos":                    "los vilos",
        "salamanca":                    "salamanca",
        "ovalle":                       "ovalle",
        "combarbala":                   "combarbala",
        "monte patria":                 "monte patria",
        "punitaqui":                    "punitaqui",
        "rio hurtado":                  "rio hurtado",
        "vicuna":                       "vicuna",
        "paiguano":                     "paiguano",
    }


# ---------------------------------------------------------------------------
# Exportar GeoJSON simplificado para el dashboard
# ---------------------------------------------------------------------------

def export_comunas_geojson(gdf_comunas: gpd.GeoDataFrame) -> Path:
    """Exporta un GeoJSON simplificado de comunas para Plotly/Streamlit.

    Simplifica geometrías para reducir el tamaño del archivo (~5–10 MB).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]💾 Exportando GeoJSON de comunas…[/bold cyan]")

    # Simplificar geometría (tolerancia ~500 m en lat/lon approx)
    gdf_simple = gdf_comunas.copy()
    gdf_simple["geometry"] = gdf_comunas.geometry.simplify(
        tolerance=0.005, preserve_topology=True
    )

    # Solo columnas necesarias para el dashboard
    gdf_simple = gdf_simple[["cut_code", "name", "region_name", "area_km2", "geometry"]]

    gdf_simple.to_file(COMUNAS_GEOJSON_PATH, driver="GeoJSON")
    size_mb = COMUNAS_GEOJSON_PATH.stat().st_size / 1e6
    console.print(
        f"  [green]✓ GeoJSON guardado: {COMUNAS_GEOJSON_PATH} ({size_mb:.1f} MB)[/green]"
    )
    return COMUNAS_GEOJSON_PATH


# ---------------------------------------------------------------------------
# Tabla de clases para DuckDB
# ---------------------------------------------------------------------------

def build_classes_dataframe() -> pd.DataFrame:
    """Construye el DataFrame de metadatos de clases MapBiomas."""
    rows = []
    for class_id, meta in COVERAGE_CLASSES.items():
        rows.append({
            "class_id": class_id,
            "name_es": meta["name_es"],
            "name_en": meta["name_en"],
            "color": meta["color"],
            "level1": meta["level1"],
            "is_natural": meta["is_natural"],
        })
    return pd.DataFrame(rows).sort_values("class_id")
