"""Curado de mutaciones COSMIC.

Este modulo contiene la interfaz que deben usar los notebooks. Las funciones
principales siguen siendo tareas del alumno, pero los nombres de columnas y el
esquema de salida estan fijados para mantener reproducibilidad.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAS_GENES = ("KRAS", "HRAS", "NRAS")
UNIPROT_BY_GENE = {
    "KRAS": "P01116",
    "HRAS": "P01112",
    "NRAS": "P01111",
}
EXPECTED_COSMIC_COLUMNS = {
    "GENE_NAME",
    "Mutation AA",
    "Mutation somatic status",
    "Mutation Description",
    "ID_sample",
    "Primary site",
    "Histology",
}
CURATED_COLUMNS = [
    "gene",
    "uniprot_id",
    "position",
    "wt_aa",
    "mut_aa",
    "hgvs_p",
    "sample_count",
    "tumour_types",
    "primary_tissues",
    "cosmic_version",
]
AA_CHANGE_RE = re.compile(r"^p\.([A-Z])(\d+)([A-Z])$")


def load_cosmic_raw(path: str | Path) -> pd.DataFrame:
    """
    Carga el fichero en bruto descargado de COSMIC.

    Parameters
    ----------
    path : str or Path
        Ruta al fichero COSMIC (típicamente un TSV).

    Returns
    -------
    pd.DataFrame
        DataFrame con todas las columnas originales.
    """
    # TODO(alumno): cargar TSV con pandas, comprobar columnas esperadas y
    # documentar cualquier mapeo de nombres usado para COSMIC real.
    raise NotImplementedError("Implementar carga del fichero COSMIC")


def filter_somatic_missense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra mutaciones somáticas confirmadas y de tipo missense.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de COSMIC en bruto.

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado.
    """
    # TODO(alumno): usar los filtros declarados en configs/config.yaml y guardar
    # conteos n_inicial -> n_somaticas -> n_missense -> n_final.
    raise NotImplementedError


def deduplicate_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados manteniendo una mutacion por muestra y cambio proteico.
    """
    raise NotImplementedError


def parse_aa_change(hgvs_p: str) -> tuple[str, int, str]:
    """
    Parsea una notación HGVS proteica (p.G12D) a (residuo_wt, posicion, residuo_mut).

    Examples
    --------
    >>> parse_aa_change("p.G12D")
    ('G', 12, 'D')
    """
    if not isinstance(hgvs_p, str):
        raise TypeError("hgvs_p debe ser un string.")

    match = AA_CHANGE_RE.fullmatch(hgvs_p.strip())
    if match is None:
        raise ValueError("Formato esperado: p.G12D, p.G13V, p.Q61R, etc.")

    wt_aa, position, mut_aa = match.groups()
    return wt_aa, int(position), mut_aa


def build_curated_dataset(df_filtered: pd.DataFrame, gene: str) -> pd.DataFrame:
    """
    Construye el DataFrame curado final.

    La salida debe contener exactamente, como minimo, las columnas definidas en
    ``CURATED_COLUMNS``. Consulta ``docs/contratos_datos.md`` antes de cambiar
    este esquema.
    """
    raise NotImplementedError


def missing_expected_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve columnas COSMIC esperadas que no aparecen en ``df``."""
    return sorted(EXPECTED_COSMIC_COLUMNS - set(df.columns))
