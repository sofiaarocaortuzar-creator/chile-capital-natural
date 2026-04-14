"""
Configuración central: URLs, rutas y definiciones de clases MapBiomas.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DB_PATH = DATA_DIR / "chile_capital_natural.duckdb"

# ---------------------------------------------------------------------------
# URLs de descarga
# ---------------------------------------------------------------------------
MAPBIOMAS_XLSX_URL = (
    "https://chile.mapbiomas.org/wp-content/uploads/sites/13/2026/03/"
    "statistics_lulc_chile_col2_political_level_234.xlsx"
)

# Límites comunales: GADM 4.1 nivel 2 para Chile (UC Davis, ~1 MB, confiable)
# Alternativa oficial INE: geoportal.cl (inestable para descargas grandes)
COMUNAS_ZIP_URL = (
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_CHL_3.json.zip"
)  # Nivel 3 = 346 comunas (nivel 2 = 55 provincias)

# GeoTIFFs de cobertura anual (30 m, Landsat)
MAPBIOMAS_RASTER_URL_TEMPLATE = (
    "https://storage.googleapis.com/mapbiomas-public/initiatives/chile/"
    "coverage/chile_coverage_{year}.tif"
)
RASTER_YEARS_AVAILABLE = list(range(2000, 2023))  # 2000–2022 confirmados

# ---------------------------------------------------------------------------
# Rutas de archivos locales
# ---------------------------------------------------------------------------
MAPBIOMAS_XLSX_PATH = RAW_DIR / "mapbiomas_chile_col2.xlsx"
COMUNAS_ZIP_PATH    = RAW_DIR / "gadm41_CHL_3.zip"
COMUNAS_EXTRACT_DIR = RAW_DIR / "gadm41_CHL_3"
COMUNAS_GEOJSON_PATH = PROCESSED_DIR / "comunas_chile.geojson"

# ---------------------------------------------------------------------------
# Clases de cobertura MapBiomas Chile (Colección 2)
# ---------------------------------------------------------------------------
COVERAGE_CLASSES: dict[int, dict] = {
    3:  {"name_es": "Bosque Natural",        "name_en": "Forest Formation",      "color": "#1f8d49", "level1": "Formación boscosa",        "is_natural": True},
    9:  {"name_es": "Plantación Forestal",   "name_en": "Forest Plantation",     "color": "#7a5c1e", "level1": "Agropecuario/Silvicultura", "is_natural": False},
    11: {"name_es": "Humedal",               "name_en": "Wetland",               "color": "#519799", "level1": "No boscosa natural",       "is_natural": True},
    12: {"name_es": "Pradera Natural",       "name_en": "Grassland",             "color": "#d6bc74", "level1": "No boscosa natural",       "is_natural": True},
    15: {"name_es": "Pastizal",              "name_en": "Pasture",               "color": "#edde8e", "level1": "Agropecuario/Silvicultura", "is_natural": False},
    18: {"name_es": "Agricultura",           "name_en": "Agriculture",           "color": "#E974ED", "level1": "Agropecuario/Silvicultura", "is_natural": False},
    23: {"name_es": "Arena/Playa/Duna",      "name_en": "Sand, Beach & Dune",    "color": "#ffa07a", "level1": "Área no vegetada",         "is_natural": True},
    24: {"name_es": "Infraestructura",       "name_en": "Infrastructure",        "color": "#d4271e", "level1": "Área no vegetada",         "is_natural": False},
    25: {"name_es": "Área no Vegetada",      "name_en": "Other non-vegetated",   "color": "#db4d4f", "level1": "Área no vegetada",         "is_natural": False},
    27: {"name_es": "No Observado",          "name_en": "Not Observed",          "color": "#d5d5e8", "level1": "No observado",             "is_natural": None},
    29: {"name_es": "Afloramiento Rocoso",   "name_en": "Rocky Outcrop",         "color": "#af2a2a", "level1": "No boscosa natural",       "is_natural": True},
    33: {"name_es": "Río/Lago/Océano",       "name_en": "River, Lake or Ocean",  "color": "#2532e4", "level1": "Cuerpo de agua",           "is_natural": True},
    34: {"name_es": "Hielo y Nieve",         "name_en": "Ice and Snow",          "color": "#9bb4f2", "level1": "Cuerpo de agua",           "is_natural": True},
    59: {"name_es": "Bosque Primario",       "name_en": "Primary Forest",        "color": "#006400", "level1": "Formación boscosa",        "is_natural": True},
    60: {"name_es": "Bosque Secundario",     "name_en": "Secondary Forest",      "color": "#00b050", "level1": "Formación boscosa",        "is_natural": True},
    61: {"name_es": "Salares",               "name_en": "Salt Flat",             "color": "#f5e4d0", "level1": "Área no vegetada",         "is_natural": True},
    63: {"name_es": "Estepa",               "name_en": "Steppe",                "color": "#c5b385", "level1": "No boscosa natural",       "is_natural": True},
    66: {"name_es": "Matorral",              "name_en": "Shrubland",             "color": "#04381d", "level1": "No boscosa natural",       "is_natural": True},
    67: {"name_es": "Bosque Achaparrado",    "name_en": "Dwarf Forest",          "color": "#45a35e", "level1": "Formación boscosa",        "is_natural": True},
}

# Años disponibles en el XLSX (Colección 2)
YEARS = list(range(1999, 2025))  # 1999–2024

# Clases que componen "cobertura boscosa total" para indicadores agregados
FOREST_CLASS_IDS = {3, 59, 60, 67}
NATURAL_VEGETATION_IDS = {3, 11, 12, 29, 34, 59, 60, 63, 66, 67}
ANTHROPIC_IDS = {9, 15, 18, 24, 25}
