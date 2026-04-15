"""
CLI principal — Chile Capital Natural

Comandos disponibles:
    python run.py pipeline            Descarga + transforma + carga todo
    python run.py pipeline --download-only
    python run.py pipeline --load-only
    python run.py download            Solo descarga archivos
    python run.py dashboard           Lanza Streamlit
    python run.py status              Muestra estado de la base de datos
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Asegurar que el root esté en sys.path (necesario cuando se llama con uv run)
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from etl.config import COMUNAS_GEOJSON_PATH, DB_PATH, MAPBIOMAS_XLSX_PATH, COMUNAS_ZIP_PATH

app = typer.Typer(
    name="chile-capital-natural",
    help="Pipeline de datos ambientales — Capital Natural Chile",
    add_completion=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Comando: pipeline
# ---------------------------------------------------------------------------

@app.command()
def pipeline(
    force_download:   bool = typer.Option(False, "--force-download",   help="Re-descarga aunque los archivos ya existan"),
    download_only:    bool = typer.Option(False, "--download-only",    help="Solo descarga, no procesa"),
    load_only:        bool = typer.Option(False, "--load-only",        help="Solo procesa (asume archivos descargados)"),
    skip_water_risk:  bool = typer.Option(False, "--skip-water-risk",  help="Omite descarga de Aqueduct (~249 MB)"),
    skip_casen:       bool = typer.Option(False, "--skip-casen",       help="Omite descarga de CASEN (~200 MB total)"),
):
    """Ejecuta el pipeline completo: descarga → transforma → carga en DuckDB."""
    from etl.pipeline import run_pipeline
    run_pipeline(
        force_download=force_download,
        download_only=download_only,
        load_only=load_only,
        skip_water_risk=skip_water_risk,
        skip_casen=skip_casen,
    )


# ---------------------------------------------------------------------------
# Comando: download
# ---------------------------------------------------------------------------

@app.command()
def download(
    force: bool = typer.Option(False, "--force", help="Re-descarga aunque ya existan"),
):
    """Descarga los archivos crudos (XLSX MapBiomas + shapefile comunas)."""
    from etl.download import download_all
    download_all(force=force)


# ---------------------------------------------------------------------------
# Comando: dashboard
# ---------------------------------------------------------------------------

@app.command()
def dashboard():
    """Lanza el dashboard Streamlit en el navegador."""
    dashboard_path = ROOT / "dashboard" / "app.py"
    if not DB_PATH.exists():
        console.print(
            "[red]⚠ Base de datos no encontrada.[/red] "
            "Ejecuta primero: [bold]python run.py pipeline[/bold]"
        )
        raise typer.Exit(1)

    console.print("[bold green]🚀 Lanzando dashboard...[/bold green]")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Comando: status
# ---------------------------------------------------------------------------

@app.command()
def status():
    """Muestra el estado actual de los datos y la base de datos."""
    console.print("\n[bold]Estado — Chile Capital Natural[/bold]\n")

    # Archivos crudos
    table = Table(title="Archivos locales", show_header=True)
    table.add_column("Archivo", style="cyan")
    table.add_column("Existe", justify="center")
    table.add_column("Tamaño")

    files = [
        ("XLSX MapBiomas",       MAPBIOMAS_XLSX_PATH),
        ("ZIP Comunas INE",      COMUNAS_ZIP_PATH),
        ("GeoJSON Comunas",      COMUNAS_GEOJSON_PATH),
        ("DuckDB",               DB_PATH),
    ]
    for name, path in files:
        exists = path.exists()
        size = f"{path.stat().st_size / 1e6:.1f} MB" if exists else "—"
        icon = "✅" if exists else "❌"
        table.add_row(name, icon, size)

    console.print(table)

    # Estadísticas de la DB
    if DB_PATH.exists():
        try:
            import duckdb
            con = duckdb.connect(str(DB_PATH), read_only=True)

            stats = Table(title="Contenido de la base de datos", show_header=True)
            stats.add_column("Tabla", style="cyan")
            stats.add_column("Registros", justify="right")

            for tbl in ["comunas", "coverage_classes", "vegetation_coverage", "panel_anual"]:
                try:
                    n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    stats.add_row(tbl, f"{n:,}")
                except Exception:
                    stats.add_row(tbl, "[red]tabla no encontrada[/red]")

            # Rango de años
            try:
                yr = con.execute(
                    "SELECT MIN(year), MAX(year) FROM panel_anual"
                ).fetchone()
                stats.add_row("Rango temporal", f"{yr[0]}–{yr[1]}")
            except Exception:
                pass

            con.close()
            console.print(stats)

        except Exception as e:
            console.print(f"[red]Error al leer la DB:[/red] {e}")

    console.print()
    console.print("[bold]Próximos pasos:[/bold]")
    if not MAPBIOMAS_XLSX_PATH.exists():
        console.print("  1. [yellow]python run.py pipeline[/yellow]  ← ejecutar pipeline completo")
    elif not DB_PATH.exists():
        console.print("  1. [yellow]python run.py pipeline --load-only[/yellow]  ← datos ya descargados")
    else:
        console.print("  1. [green]python run.py dashboard[/green]  ← abrir dashboard ✅")
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
