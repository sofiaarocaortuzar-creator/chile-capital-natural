"""
Pipeline principal: orquesta descarga, transformación y carga en DuckDB.

Uso:
    python run.py pipeline                    # Todo el proceso
    python run.py pipeline --download-only
    python run.py pipeline --load-only        # Asume archivos ya descargados
    python run.py pipeline --skip-water-risk  # Omite Aqueduct (249 MB)
"""
from __future__ import annotations

import time

from rich.console import Console
from rich.rule import Rule

from db.schema import (
    build_panel,
    get_connection,
    load_classes,
    load_comunas,
    load_deforestation,
    load_vegetation_coverage,
    load_water_risk,
    setup_schema,
)
from etl.config import COMUNAS_GEOJSON_PATH, DB_PATH
from etl.download import download_all
from etl.transform import (
    build_classes_dataframe,
    export_comunas_geojson,
    join_mapbiomas_with_comunas,
    load_comunas_geodataframe,
    load_mapbiomas_xlsx,
)
from etl.transform_deforestation import build_deforestation_dataframe
from etl.transform_water_risk import build_water_risk_dataframe

console = Console()


def run_pipeline(
    force_download: bool = False,
    download_only: bool = False,
    load_only: bool = False,
    skip_water_risk: bool = False,
) -> None:
    """Ejecuta el pipeline completo ETL → DuckDB."""
    t0 = time.perf_counter()
    console.print(Rule("[bold green]Pipeline Chile Capital Natural"))

    # ------------------------------------------------------------------
    # PASO 1 — Descarga
    # ------------------------------------------------------------------
    if not load_only:
        console.print(Rule("PASO 1 — Descarga de archivos"))
        download_all(force=force_download)

    if download_only:
        console.print(Rule("[green]Descarga completada. Fin (--download-only)."))
        return

    # ------------------------------------------------------------------
    # PASO 2 — Transformación
    # ------------------------------------------------------------------
    console.print(Rule("PASO 2 — Transformación y normalización"))

    df_classes  = build_classes_dataframe()
    gdf_comunas = load_comunas_geodataframe()

    if not COMUNAS_GEOJSON_PATH.exists():
        export_comunas_geojson(gdf_comunas)
    else:
        console.print(f"  ✓ GeoJSON ya existe: {COMUNAS_GEOJSON_PATH.name}")

    # 2a. Cobertura vegetal (MapBiomas COVERAGE)
    df_mapbiomas = load_mapbiomas_xlsx()
    df_coverage  = join_mapbiomas_with_comunas(df_mapbiomas, gdf_comunas)

    # 2b. Deforestación (MapBiomas TRANSITION — ya descargado)
    df_deforestation = build_deforestation_dataframe(gdf_comunas)

    # 2c. Riesgo hídrico (WRI Aqueduct 4.0 — ~249 MB, puede omitirse)
    df_water_risk = None
    if not skip_water_risk:
        try:
            df_water_risk = build_water_risk_dataframe(gdf_comunas)
        except Exception as e:
            console.print(f"  [yellow]⚠ Riesgo hídrico omitido por error: {e}[/yellow]")
    else:
        console.print("  [dim]Riesgo hídrico omitido (--skip-water-risk)[/dim]")

    # ------------------------------------------------------------------
    # PASO 3 — Carga en DuckDB
    # ------------------------------------------------------------------
    console.print(Rule("PASO 3 — Carga en DuckDB"))

    con = get_connection(DB_PATH)
    setup_schema(con)

    load_classes(con, df_classes)
    load_comunas(con, gdf_comunas)
    load_vegetation_coverage(con, df_coverage)
    build_panel(con)
    load_deforestation(con, df_deforestation)
    if df_water_risk is not None:
        load_water_risk(con, df_water_risk)

    con.close()

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    console.print(Rule(f"[bold green]Pipeline completado en {elapsed:.0f} s"))
    console.print(f"  Base de datos: [cyan]{DB_PATH}[/cyan]")
    console.print(f"  GeoJSON:       [cyan]{COMUNAS_GEOJSON_PATH}[/cyan]")
    console.print()
    console.print("  Próximo paso → lanza el dashboard:")
    console.print("  [bold]uv run streamlit run dashboard/app.py[/bold]")
