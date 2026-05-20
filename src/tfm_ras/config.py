"""Utilidades para cargar la configuración del proyecto."""

from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    """
    Carga el fichero de configuración del proyecto.

    Parameters
    ----------
    path : str or Path
        Ruta al fichero YAML de configuración.

    Returns
    -------
    dict
        Diccionario con la configuración.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de configuración: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    """Devuelve la raíz del proyecto, asumiendo que esta función vive en src/tfm_ras/."""
    return Path(__file__).resolve().parent.parent.parent
