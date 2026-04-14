"""
Deforestación a partir de la hoja TRANSITION del XLSX de MapBiomas Chile.

Definición de deforestación usada:
    Transición DESDE una clase boscosa (3, 59, 60, 67)
    HACIA una clase NO boscosa (cualquier otra excepto plantación forestal = 9)

Nota sobre degradación forestal:
    Bosque primario → Bosque secundario (59→60) se registra como
    "degradación forestal", no como deforestación estricta.
    Bosque → Plantación forestal (9) se registra como "conversión a plantación".

Salida:
    DataFrame con columnas:
        cut_code, comuna_name, region_name,
        year_from, year_to,
        from_class_id, from_class_name,
        to_class_id, to_class_name,
        transition_type,   # 'deforestation' | 'degradation' | 'plantation' | 'reforestation'
        area_ha
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rich.console import Console

from etl.config import COVERAGE_CLASSES, FOREST_CLASS_IDS, MAPBIOMAS_XLSX_PATH

console = Console()

# ---------------------------------------------------------------------------
# Constantes de clasificación de transiciones
# ---------------------------------------------------------------------------

PLANTATION_ID  = 9
SECONDARY_ID   = 60
PRIMARY_ID     = 59
ALL_FOREST_IDS = FOREST_CLASS_IDS  # {3, 59, 60, 67}

# ---------------------------------------------------------------------------
# Carga y parsing del sheet TRANSITION
# ---------------------------------------------------------------------------

def load_transition_sheet(xlsx_path: Path = MAPBIOMAS_XLSX_PATH) -> pd.DataFrame:
    """Lee la hoja TRANSITION del XLSX MapBiomas y retorna un DataFrame largo.

    Columnas del resultado:
        region_name, provincia_name, comuna_name,
        from_class_id, to_class_id,
        year_from, year_to, area_ha
    """
    console.print("[bold cyan]📄 Leyendo hoja TRANSITION de MapBiomas…[/bold cyan]")
    console.print("  (puede tardar 30–60 segundos — hoja de 149 MB)")

    df = pd.read_excel(xlsx_path, sheet_name="TRANSITION", engine="openpyxl")
    console.print(f"  ✓ {len(df):,} filas | columnas: {list(df.columns[:10])} …")

    # --- Detectar columnas de territorio ---
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "territory_level_2": rename_map[col] = "region_name"
        elif cl == "territory_level_3": rename_map[col] = "provincia_name"
        elif cl == "territory_level_4": rename_map[col] = "comuna_name"
    df = df.rename(columns=rename_map)

    # --- Detectar columnas de clase origen/destino ---
    from_col = _pick(df, ["class_from", "from_class", "CLASS_FROM"])
    to_col   = _pick(df, ["class_to",   "to_class",   "CLASS_TO"])
    if not from_col or not to_col:
        raise ValueError(f"No se encontraron columnas class_from/class_to: {list(df.columns)}")

    df = df.rename(columns={from_col: "from_class_id", to_col: "to_class_id"})

    # Solo filas a nivel de comuna
    df = df[df["comuna_name"].notna()].copy()

    # Convertir clases a int
    for col in ["from_class_id", "to_class_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["from_class_id", "to_class_id"])
    df["from_class_id"] = df["from_class_id"].astype(int)
    df["to_class_id"]   = df["to_class_id"].astype(int)

    # --- Detectar columnas de períodos anuales consecutivos (p1999_2000 … p2023_2024) ---
    period_cols = _detect_annual_period_cols(df)
    console.print(f"  Períodos anuales detectados: {period_cols[0]} → {period_cols[-1]}")

    # Melt a formato largo
    id_vars = ["region_name", "provincia_name", "comuna_name", "from_class_id", "to_class_id"]
    df_long = df[id_vars + period_cols].melt(
        id_vars=id_vars,
        value_vars=period_cols,
        var_name="period_col",
        value_name="area_ha",
    )

    # Extraer year_from y year_to desde el nombre de columna (ej: p1999_2000)
    years = df_long["period_col"].str.extractall(r"(\d{4})").unstack()
    df_long["year_from"] = years[0][0].astype(int).values
    df_long["year_to"]   = years[0][1].astype(int).values
    df_long = df_long.drop(columns=["period_col"])

    # Limpiar valores
    df_long["area_ha"] = pd.to_numeric(df_long["area_ha"], errors="coerce").fillna(0.0)
    df_long = df_long[df_long["area_ha"] > 0]

    console.print(f"  ✓ {len(df_long):,} transiciones con área > 0")
    return df_long


def _detect_annual_period_cols(df: pd.DataFrame) -> list[str]:
    """Detecta columnas de períodos anuales consecutivos (p1999_2000, p2000_2001, …)."""
    pattern = re.compile(r"^p(\d{4})_(\d{4})$")
    cols = []
    for col in df.columns:
        m = pattern.match(str(col).lower())
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if y2 == y1 + 1:  # solo períodos de 1 año
                cols.append(col)
    return sorted(cols)


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Clasificar transiciones
# ---------------------------------------------------------------------------

def classify_transitions(df_long: pd.DataFrame) -> pd.DataFrame:
    """Añade columna transition_type y filtra transiciones relevantes.

    Tipos:
        deforestation  : bosque → no-bosque (excl. plantación)
        plantation     : bosque → plantación forestal
        degradation    : bosque primario → bosque secundario
        reforestation  : no-bosque → bosque
    """
    from_forest = df_long["from_class_id"].isin(ALL_FOREST_IDS)
    to_forest   = df_long["to_class_id"].isin(ALL_FOREST_IDS)
    to_plant    = df_long["to_class_id"] == PLANTATION_ID
    to_sec      = df_long["to_class_id"] == SECONDARY_ID
    from_prim   = df_long["from_class_id"] == PRIMARY_ID

    conditions = [
        from_forest & from_prim & to_sec,           # degradación forestal
        from_forest & to_plant,                      # conversión a plantación
        from_forest & ~to_forest & ~to_plant,        # deforestación estricta
        ~from_forest & to_forest,                    # reforestación
    ]
    choices = ["degradation", "plantation", "deforestation", "reforestation"]

    df_long = df_long.copy()
    df_long["transition_type"] = pd.NA
    for cond, label in zip(conditions, choices):
        df_long.loc[cond, "transition_type"] = label

    # Mantener solo transiciones clasificadas
    df_long = df_long[df_long["transition_type"].notna()].copy()

    # Añadir nombres de clase
    df_long["from_class_name"] = df_long["from_class_id"].map(
        {k: v["name_es"] for k, v in COVERAGE_CLASSES.items()}
    )
    df_long["to_class_name"] = df_long["to_class_id"].map(
        {k: v["name_es"] for k, v in COVERAGE_CLASSES.items()}
    )

    console.print(
        f"  Transiciones clasificadas: "
        + " | ".join(
            f"{t}: {(df_long['transition_type']==t).sum():,}"
            for t in choices
        )
    )
    return df_long


# ---------------------------------------------------------------------------
# Join con comunas (obtener cut_code)
# ---------------------------------------------------------------------------

def join_deforestation_with_comunas(
    df_transitions: pd.DataFrame,
    gdf_comunas,
) -> pd.DataFrame:
    """Añade cut_code usando el mismo fuzzy join que el pipeline principal."""
    import difflib
    import unicodedata, re as _re

    def normalize(name: str) -> str:
        if not isinstance(name, str): return ""
        name = name.strip().lower()
        name = unicodedata.normalize("NFD", name)
        name = "".join(c for c in name if unicodedata.category(c) != "Mn")
        name = _re.sub(r"[^a-z0-9\s]", " ", name)
        return _re.sub(r"\s+", " ", name).strip()

    lookup = (
        gdf_comunas[["_norm_name", "cut_code", "name"]]
        .drop_duplicates("_norm_name")
        .set_index("_norm_name")
    )
    gadm_names = list(lookup.index)

    df = df_transitions.copy()
    df["_norm"] = df["comuna_name"].map(normalize)

    # Exacto
    df["cut_code"] = df["_norm"].map(lookup["cut_code"])

    # Fuzzy
    unmatched = df[df["cut_code"].isna()]["_norm"].unique()
    fuzzy_map = {}
    for name in unmatched:
        c = difflib.get_close_matches(name, gadm_names, n=1, cutoff=0.82)
        if c: fuzzy_map[name] = c[0]

    for mb, gd in fuzzy_map.items():
        mask = (df["_norm"] == mb) & df["cut_code"].isna()
        if mask.any():
            df.loc[mask, "cut_code"] = lookup.loc[gd, "cut_code"]

    df = df.drop(columns=["_norm"])
    matched = df["cut_code"].notna().mean() * 100
    console.print(f"  Join deforestación: {matched:.1f}% filas con cut_code")
    return df[df["cut_code"].notna()].copy()


# ---------------------------------------------------------------------------
# Pipeline completo de deforestación
# ---------------------------------------------------------------------------

def build_deforestation_dataframe(gdf_comunas) -> pd.DataFrame:
    """Pipeline completo: lee TRANSITION → clasifica → join comunas → retorna."""
    console.print("[bold cyan]🌲 Procesando datos de deforestación…[/bold cyan]")

    df_raw        = load_transition_sheet()
    df_classified = classify_transitions(df_raw)
    df_final      = join_deforestation_with_comunas(df_classified, gdf_comunas)

    cols = [
        "cut_code", "comuna_name", "region_name",
        "year_from", "year_to",
        "from_class_id", "from_class_name",
        "to_class_id", "to_class_name",
        "transition_type", "area_ha",
    ]
    df_final = df_final[cols].copy()
    console.print(
        f"  [green]✓ Deforestación lista: {len(df_final):,} eventos | "
        f"{df_final['cut_code'].nunique()} comunas | "
        f"{df_final['year_from'].min()}–{df_final['year_to'].max()}[/green]"
    )
    return df_final
