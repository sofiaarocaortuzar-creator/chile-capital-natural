"""
Descarga de archivos crudos con barra de progreso.
Usa curl del sistema para archivos grandes (más robusto que httpx en Mac),
y httpx para archivos pequeños.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from etl.config import (
    COMUNAS_EXTRACT_DIR,
    COMUNAS_ZIP_PATH,
    COMUNAS_ZIP_URL,
    MAPBIOMAS_XLSX_PATH,
    MAPBIOMAS_XLSX_URL,
    RAW_DIR,
)

console = Console()

# ---------------------------------------------------------------------------
# Función genérica de descarga (httpx, para archivos pequeños/medianos)
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, label: str, force: bool = False) -> Path:
    """Descarga `url` a `dest` mostrando una barra de progreso.

    Si el archivo ya existe y `force=False`, lo omite.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        console.print(f"[green]✓ Ya existe:[/green] {dest.name} — omitiendo descarga")
        return dest

    console.print(f"[bold cyan]↓ Descargando:[/bold cyan] {label}")
    console.print(f"  URL: {url}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(label, total=None)

        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or None
            progress.update(task, total=total)

            with open(dest, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 64):
                    f.write(chunk)
                    progress.advance(task, len(chunk))

    size_mb = dest.stat().st_size / 1_048_576
    console.print(f"  [green]✓ Guardado en {dest} ({size_mb:.1f} MB)[/green]")
    return dest


def download_file_curl(url: str, dest: Path, label: str, force: bool = False) -> Path:
    """Descarga usando curl del sistema (más confiable para archivos grandes con redirects).

    Valida que el archivo descargado sea un ZIP válido.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        # Validar que el existente sea un ZIP real
        if _is_valid_zip(dest):
            console.print(f"[green]✓ Ya existe:[/green] {dest.name} — omitiendo descarga")
            return dest
        else:
            console.print(f"[yellow]⚠ Archivo existente corrupto, re-descargando…[/yellow]")
            dest.unlink()

    console.print(f"[bold cyan]↓ Descargando con curl:[/bold cyan] {label}")
    console.print(f"  URL: {url}")

    cmd = [
        "curl",
        "--location",          # seguir redirects
        "--retry", "3",        # reintentar 3 veces
        "--retry-delay", "5",
        "--connect-timeout", "30",
        "--max-time", "600",   # máximo 10 minutos
        "--progress-bar",
        "--output", str(dest),
        url,
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"curl falló (código {result.returncode}) descargando {url}")

    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"Descarga vacía: {dest}")

    size_mb = dest.stat().st_size / 1_048_576
    console.print(f"  [green]✓ Guardado en {dest} ({size_mb:.1f} MB)[/green]")
    return dest


def _is_valid_zip(path: Path) -> bool:
    """Retorna True si el archivo es un ZIP con directorio central válido."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.namelist()  # fuerza lectura del directorio central
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Descargas específicas
# ---------------------------------------------------------------------------

def download_mapbiomas_xlsx(force: bool = False) -> Path:
    """Descarga el XLSX de estadísticas MapBiomas Chile (Colección 2)."""
    return download_file(
        url=MAPBIOMAS_XLSX_URL,
        dest=MAPBIOMAS_XLSX_PATH,
        label="MapBiomas Chile — Estadísticas por comuna (1999–2024)",
        force=force,
    )


def download_comunas_shapefile(force: bool = False) -> Path:
    """Descarga límites comunales GADM 4.1 nivel 2 para Chile (~1 MB)."""
    zip_path = download_file(
        url=COMUNAS_ZIP_URL,
        dest=COMUNAS_ZIP_PATH,
        label="Límites comunales Chile — GADM 4.1 nivel 2 (UC Davis)",
        force=force,
    )
    _extract_zip(zip_path, COMUNAS_EXTRACT_DIR)
    return zip_path


def _extract_zip(zip_path: Path, extract_to: Path, force: bool = False) -> None:
    """Extrae un ZIP si el directorio destino no existe aún."""
    if extract_to.exists() and any(extract_to.iterdir()) and not force:
        console.print(f"[green]✓ Ya extraído:[/green] {extract_to}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]📦 Extrayendo:[/bold cyan] {zip_path.name} → {extract_to}")

    # Usar unzip del sistema si Python zipfile falla (más tolerante a ZIPs no estándar)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
    except zipfile.BadZipFile:
        console.print("  [yellow]zipfile falló, intentando con unzip del sistema…[/yellow]")
        result = subprocess.run(
            ["unzip", "-o", str(zip_path), "-d", str(extract_to)],
            capture_output=True, text=True,
        )
        if result.returncode not in (0, 1):  # 1 = warnings menores, OK
            raise RuntimeError(f"unzip falló:\n{result.stderr}")

    console.print(f"  [green]✓ Extraído en {extract_to}[/green]")


def find_comunas_shapefile() -> Path:
    """Busca el shapefile de comunas dentro del directorio extraído.

    Retorna el primer .shp cuyo nombre contenga 'comun' (case-insensitive).
    """
    if not COMUNAS_EXTRACT_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio {COMUNAS_EXTRACT_DIR}. "
            "Ejecuta primero `download_comunas_shapefile()`."
        )

    candidates = sorted(COMUNAS_EXTRACT_DIR.rglob("*.shp"))
    # Priorizar archivos con "comun" en el nombre
    comunas_shp = [p for p in candidates if "comun" in p.stem.lower()]

    if comunas_shp:
        found = comunas_shp[0]
    elif candidates:
        # Fallback: primer shapefile disponible
        found = candidates[0]
        console.print(
            f"[yellow]⚠ No se encontró shapefile 'comunas', usando: {found.name}[/yellow]"
        )
    else:
        raise FileNotFoundError(
            f"No hay shapefiles en {COMUNAS_EXTRACT_DIR}. "
            "Vuelve a descargar el ZIP con `download_comunas_shapefile(force=True)`."
        )

    console.print(f"[green]✓ Shapefile comunas:[/green] {found.name}")
    return found


def download_all(force: bool = False) -> None:
    """Descarga todos los archivos necesarios para el pipeline."""
    console.rule("[bold]Descarga de datos crudos")
    download_mapbiomas_xlsx(force=force)
    download_comunas_shapefile(force=force)
    console.rule("[bold green]Descarga completada")
