"""
Gestión de la base de datos DuckDB.

Crea las tablas, carga DataFrames y expone funciones de consulta
para el dashboard y el análisis econométrico.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
from rich.console import Console

from etl.config import DB_PATH

console = Console()

# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------

def get_connection(db_path: Path = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Retorna una conexión DuckDB con la extensión spatial cargada."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


# ---------------------------------------------------------------------------
# DDL — Creación de tablas
# ---------------------------------------------------------------------------

DDL = """
-- Clases de cobertura MapBiomas
CREATE TABLE IF NOT EXISTS coverage_classes (
    class_id    INTEGER PRIMARY KEY,
    name_es     VARCHAR NOT NULL,
    name_en     VARCHAR NOT NULL,
    color       VARCHAR(7),          -- hex color p. ej. '#1f8d49'
    level1      VARCHAR,
    is_natural  BOOLEAN
);

-- Comunas de Chile (clave geográfica central)
CREATE TABLE IF NOT EXISTS comunas (
    cut_code    VARCHAR(5) PRIMARY KEY,
    name        VARCHAR NOT NULL,
    region_name VARCHAR,
    area_km2    DOUBLE
);

-- Tabla de hechos: cobertura vegetal por (comuna × año × clase)
CREATE TABLE IF NOT EXISTS vegetation_coverage (
    cut_code    VARCHAR(5)  NOT NULL,
    year        INTEGER     NOT NULL,
    class_id    INTEGER     NOT NULL,
    area_ha     DOUBLE      NOT NULL,
    PRIMARY KEY (cut_code, year, class_id),
    FOREIGN KEY (cut_code)  REFERENCES comunas(cut_code),
    FOREIGN KEY (class_id)  REFERENCES coverage_classes(class_id)
);

-- Vista panel: una fila por (comuna × año) con clases como columnas
-- Útil para econometría — se crea como tabla para velocidad
CREATE TABLE IF NOT EXISTS panel_anual (
    cut_code            VARCHAR(5),
    year                INTEGER,
    bosque_ha           DOUBLE,   -- clases 3 + 59 + 60 + 67
    bosque_primario_ha  DOUBLE,   -- clase 59
    bosque_secundario_ha DOUBLE,  -- clase 60
    plantacion_ha       DOUBLE,   -- clase 9
    matorral_ha         DOUBLE,   -- clase 66
    pradera_ha          DOUBLE,   -- clase 12
    estepa_ha           DOUBLE,   -- clase 63
    humedal_ha          DOUBLE,   -- clase 11
    agricultura_ha      DOUBLE,   -- clase 18
    pastizal_ha         DOUBLE,   -- clase 15
    infraestructura_ha  DOUBLE,   -- clase 24
    agua_ha             DOUBLE,   -- clase 33
    hielo_ha            DOUBLE,   -- clase 34
    total_ha            DOUBLE,   -- suma de todas las clases
    pct_bosque          DOUBLE,   -- bosque_ha / total_ha * 100
    pct_natural         DOUBLE,   -- vegetación natural / total * 100
    PRIMARY KEY (cut_code, year)
);

-- Transiciones de cobertura vegetal (deforestación, reforestación, etc.)
CREATE TABLE IF NOT EXISTS deforestation_events (
    cut_code            VARCHAR(5)  NOT NULL,
    year_from           INTEGER     NOT NULL,
    year_to             INTEGER     NOT NULL,
    from_class_id       INTEGER     NOT NULL,
    from_class_name     VARCHAR,
    to_class_id         INTEGER     NOT NULL,
    to_class_name       VARCHAR,
    transition_type     VARCHAR     NOT NULL,  -- deforestation | degradation | plantation | reforestation
    area_ha             DOUBLE      NOT NULL,
    PRIMARY KEY (cut_code, year_from, from_class_id, to_class_id)
);

-- Riesgo hídrico por comuna (WRI Aqueduct 4.0, baseline 2000–2019)
CREATE TABLE IF NOT EXISTS water_risk (
    cut_code            VARCHAR(5) PRIMARY KEY,
    -- Estrés hídrico (demanda / disponibilidad, mayor = más crítico)
    bws_raw             DOUBLE,    -- valor continuo (ratio)
    bws_score           DOUBLE,    -- score 0-5 normalizado WRI
    -- Agotamiento de agua subterránea
    bwd_raw             DOUBLE,
    bwd_score           DOUBLE,
    -- Variabilidad interanual del caudal
    iav_raw             DOUBLE,
    iav_score           DOUBLE,
    -- Variabilidad estacional
    sev_raw             DOUBLE,
    sev_score           DOUBLE,
    -- Riesgo de sequía
    drr_raw             DOUBLE,
    drr_score           DOUBLE,
    -- Riesgo de inundación fluvial
    rfr_raw             DOUBLE,
    rfr_score           DOUBLE,
    -- Riesgo de inundación costera
    cfr_raw             DOUBLE,
    cfr_score           DOUBLE,
    -- Metadata
    n_basins            INTEGER,   -- número de cuencas que intersectan la comuna
    total_basin_area_km2 DOUBLE
);
"""

# ---------------------------------------------------------------------------
# Setup inicial
# ---------------------------------------------------------------------------

def setup_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Crea todas las tablas si no existen y limpia en orden correcto."""
    console.print("[bold cyan]🗄 Configurando esquema DuckDB…[/bold cyan]")
    # Borrar en orden inverso de dependencias (hijos antes que padres)
    for tbl in ["deforestation_events", "water_risk", "panel_anual",
                "vegetation_coverage", "coverage_classes", "comunas"]:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    con.execute(DDL)
    console.print("  [green]✓ Tablas creadas[/green]")


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_classes(con: duckdb.DuckDBPyConnection, df_classes: pd.DataFrame) -> None:
    """Carga la tabla de clases de cobertura."""
    con.execute("DELETE FROM coverage_classes")
    con.execute("INSERT INTO coverage_classes SELECT * FROM df_classes")
    n = con.execute("SELECT COUNT(*) FROM coverage_classes").fetchone()[0]
    console.print(f"  [green]✓ coverage_classes: {n} clases cargadas[/green]")


def load_comunas(con: duckdb.DuckDBPyConnection, gdf: gpd.GeoDataFrame) -> None:
    """Carga la tabla de comunas (sin geometría, solo atributos)."""
    df = pd.DataFrame({
        "cut_code":    gdf["cut_code"],
        "name":        gdf["name"],
        "region_name": gdf["region_name"],
        "area_km2":    gdf["area_km2"],
    })
    con.execute("DELETE FROM comunas")
    con.execute("INSERT INTO comunas SELECT * FROM df")
    n = con.execute("SELECT COUNT(*) FROM comunas").fetchone()[0]
    console.print(f"  [green]✓ comunas: {n} comunas cargadas[/green]")


def load_vegetation_coverage(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    batch_size: int = 500_000,
) -> None:
    """Carga la tabla de cobertura vegetal en batches para eficiencia."""
    console.print(f"  Cargando {len(df):,} registros de cobertura…")
    con.execute("DELETE FROM vegetation_coverage")

    data = (
        df[["cut_code", "year", "class_id", "area_ha"]]
        .copy()
        .assign(
            cut_code=lambda d: d["cut_code"].astype(str),
            year=lambda d: d["year"].astype(int),
            class_id=lambda d: d["class_id"].astype(int),
            area_ha=lambda d: d["area_ha"].astype(float),
        )
        # Agregar duplicados que puedan surgir del join (misma comuna × año × clase)
        .groupby(["cut_code", "year", "class_id"], as_index=False)["area_ha"]
        .sum()
    )

    total = 0
    for start in range(0, len(data), batch_size):
        batch = data.iloc[start : start + batch_size]
        con.execute("INSERT INTO vegetation_coverage SELECT * FROM batch")
        total += len(batch)

    n = con.execute("SELECT COUNT(*) FROM vegetation_coverage").fetchone()[0]
    console.print(f"  [green]✓ vegetation_coverage: {n:,} registros cargados[/green]")


def build_panel(con: duckdb.DuckDBPyConnection) -> None:
    """Construye la tabla panel_anual agregando por (comuna × año)."""
    console.print("  Construyendo tabla panel_anual…")
    con.execute("DELETE FROM panel_anual")
    con.execute("""
        INSERT INTO panel_anual
        SELECT
            cut_code,
            year,
            -- Bosque total (clases 3, 59, 60, 67)
            COALESCE(SUM(CASE WHEN class_id IN (3,59,60,67) THEN area_ha END), 0) AS bosque_ha,
            COALESCE(SUM(CASE WHEN class_id = 59            THEN area_ha END), 0) AS bosque_primario_ha,
            COALESCE(SUM(CASE WHEN class_id = 60            THEN area_ha END), 0) AS bosque_secundario_ha,
            COALESCE(SUM(CASE WHEN class_id = 9             THEN area_ha END), 0) AS plantacion_ha,
            COALESCE(SUM(CASE WHEN class_id = 66            THEN area_ha END), 0) AS matorral_ha,
            COALESCE(SUM(CASE WHEN class_id = 12            THEN area_ha END), 0) AS pradera_ha,
            COALESCE(SUM(CASE WHEN class_id = 63            THEN area_ha END), 0) AS estepa_ha,
            COALESCE(SUM(CASE WHEN class_id = 11            THEN area_ha END), 0) AS humedal_ha,
            COALESCE(SUM(CASE WHEN class_id = 18            THEN area_ha END), 0) AS agricultura_ha,
            COALESCE(SUM(CASE WHEN class_id = 15            THEN area_ha END), 0) AS pastizal_ha,
            COALESCE(SUM(CASE WHEN class_id = 24            THEN area_ha END), 0) AS infraestructura_ha,
            COALESCE(SUM(CASE WHEN class_id = 33            THEN area_ha END), 0) AS agua_ha,
            COALESCE(SUM(CASE WHEN class_id = 34            THEN area_ha END), 0) AS hielo_ha,
            -- Total medido
            COALESCE(SUM(area_ha), 0) AS total_ha,
            -- Porcentaje bosque
            CASE
                WHEN SUM(area_ha) > 0
                THEN SUM(CASE WHEN class_id IN (3,59,60,67) THEN area_ha ELSE 0 END) / SUM(area_ha) * 100
                ELSE 0
            END AS pct_bosque,
            -- Porcentaje vegetación natural
            CASE
                WHEN SUM(area_ha) > 0
                THEN SUM(CASE WHEN class_id IN (3,11,12,29,34,59,60,63,66,67) THEN area_ha ELSE 0 END) / SUM(area_ha) * 100
                ELSE 0
            END AS pct_natural
        FROM vegetation_coverage
        GROUP BY cut_code, year
        ORDER BY cut_code, year
    """)
    n = con.execute("SELECT COUNT(*) FROM panel_anual").fetchone()[0]
    console.print(f"  [green]✓ panel_anual: {n:,} registros ({n // 26} comunas × años)[/green]")


def load_deforestation(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Carga la tabla de eventos de deforestación/transición."""
    console.print(f"  Cargando {len(df):,} eventos de transición forestal…")
    con.execute("DELETE FROM deforestation_events")

    cols = ["cut_code", "year_from", "year_to",
            "from_class_id", "from_class_name",
            "to_class_id", "to_class_name",
            "transition_type", "area_ha"]

    # Asegurar que las columnas existan y agregar duplicados
    data = (
        df[cols].copy()
        .assign(cut_code=lambda d: d["cut_code"].astype(str))
        .groupby(
            ["cut_code", "year_from", "year_to", "from_class_id",
             "from_class_name", "to_class_id", "to_class_name", "transition_type"],
            as_index=False,
        )["area_ha"].sum()
    )

    con.execute("INSERT INTO deforestation_events SELECT * FROM data")
    n = con.execute("SELECT COUNT(*) FROM deforestation_events").fetchone()[0]
    console.print(f"  [green]✓ deforestation_events: {n:,} registros cargados[/green]")


def load_water_risk(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Carga la tabla de riesgo hídrico por comuna."""
    console.print(f"  Cargando riesgo hídrico para {len(df)} comunas…")
    con.execute("DELETE FROM water_risk")

    indicators = ["bws", "bwd", "iav", "sev", "drr", "rfr", "cfr"]
    row_cols = ["cut_code"]
    for ind in indicators:
        for suffix in ["raw", "score"]:
            col = f"{ind}_{suffix}"
            if col not in df.columns:
                df[col] = float("nan")
            row_cols.append(col)

    # Añadir conteo de cuencas y área si existe
    df["n_basins"] = df.get("n_basins", pd.Series([None] * len(df)))
    df["total_basin_area_km2"] = df.get("total_basin_area_km2", pd.Series([None] * len(df)))
    row_cols += ["n_basins", "total_basin_area_km2"]

    data = df[row_cols].drop_duplicates("cut_code").copy()
    con.execute("INSERT INTO water_risk SELECT * FROM data")
    n = con.execute("SELECT COUNT(*) FROM water_risk").fetchone()[0]
    console.print(f"  [green]✓ water_risk: {n} comunas cargadas[/green]")


# ---------------------------------------------------------------------------
# Consultas para el dashboard
# ---------------------------------------------------------------------------

def query_panel(
    con: duckdb.DuckDBPyConnection,
    cut_codes: list[str] | None = None,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Retorna el panel anual filtrado, listo para análisis o dashboard."""
    filters = []
    if cut_codes:
        codes_str = ", ".join(f"'{c}'" for c in cut_codes)
        filters.append(f"p.cut_code IN ({codes_str})")
    if years:
        years_str = ", ".join(str(y) for y in years)
        filters.append(f"p.year IN ({years_str})")

    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    return con.execute(f"""
        SELECT
            p.*,
            c.name       AS comuna_name,
            c.region_name,
            c.area_km2
        FROM panel_anual p
        JOIN comunas c ON c.cut_code = p.cut_code
        {where}
        ORDER BY p.cut_code, p.year
    """).df()


def query_coverage_detail(
    con: duckdb.DuckDBPyConnection,
    cut_code: str,
    year: int | None = None,
) -> pd.DataFrame:
    """Detalle de cobertura por clase para una comuna."""
    year_filter = f"AND vc.year = {year}" if year else ""
    return con.execute(f"""
        SELECT
            vc.year,
            vc.class_id,
            cc.name_es,
            cc.color,
            cc.level1,
            vc.area_ha,
            vc.area_ha / NULLIF(SUM(vc.area_ha) OVER (PARTITION BY vc.year), 0) * 100 AS pct
        FROM vegetation_coverage vc
        JOIN coverage_classes cc ON cc.class_id = vc.class_id
        WHERE vc.cut_code = '{cut_code}' {year_filter}
        ORDER BY vc.year, vc.area_ha DESC
    """).df()


def query_regions_summary(con: duckdb.DuckDBPyConnection, year: int) -> pd.DataFrame:
    """Resumen de cobertura boscosa por región para un año dado."""
    return con.execute(f"""
        SELECT
            c.region_name,
            COUNT(DISTINCT p.cut_code)  AS n_comunas,
            SUM(p.bosque_ha)            AS bosque_ha,
            SUM(p.plantacion_ha)        AS plantacion_ha,
            SUM(p.agricultura_ha)       AS agricultura_ha,
            SUM(p.total_ha)             AS total_ha,
            AVG(p.pct_bosque)           AS pct_bosque_promedio
        FROM panel_anual p
        JOIN comunas c ON c.cut_code = p.cut_code
        WHERE p.year = {year}
        GROUP BY c.region_name
        ORDER BY bosque_ha DESC
    """).df()
