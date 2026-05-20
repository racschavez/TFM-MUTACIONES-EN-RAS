"""Caracterizacion estructural: SASA, distancia al sitio activo y conservacion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from Bio.PDB import Structure

NUCLEOTIDE_RESNAMES = {"GDP", "GTP", "GNP", "GCP"}
MAGNESIUM_RESNAMES = {"MG", "MG2", "MG2+"}
SASA_COLUMNS = ["gene", "position", "residue", "sasa_abs", "sasa_rel"]
DISTANCE_COLUMNS = ["gene", "position", "residue", "distance_to_active_site"]
CONSERVATION_COLUMNS = ["msa_position", "conservation_entropy"]
MASTER_FEATURE_COLUMNS = [
    "gene",
    "position",
    "msa_position",
    "sample_count",
    "sasa_rel",
    "distance_to_active_site",
    "conservation_entropy",
    "is_recurrent",
]


def fetch_pdb(pdb_id: str, output_dir: str | Path) -> Path:
    """Descarga un PDB y devuelve la ruta local.

    Debe reutilizar ficheros ya descargados para que el notebook sea
    reproducible sin red una vez exista cache en ``data/external/pdb/``.
    """
    raise NotImplementedError


def parse_structure(pdb_path: str | Path) -> Structure:
    """Parsea una estructura PDB con Biopython y falla si el fichero no existe."""
    # TODO(alumno): empezar por DSSP si esta instalado. Si no, usar un fallback
    # razonable y dejarlo documentado en el notebook.
    raise NotImplementedError


def compute_sasa(
    structure: Structure,
    chain_id: str = "A",
    method: str = "dssp",
    relative: bool = True,
) -> pd.DataFrame:
    """
    Calcula la accesibilidad al solvente por residuo.

    Parameters
    ----------
    structure : Bio.PDB.Structure
    chain_id : str
        Cadena a analizar.
    method : str
        "dssp" o "freesasa".
    relative : bool
        Si True, normaliza por SASA del residuo en estado extendido (Tien et al. 2013).

    Returns
    -------
    pd.DataFrame con columnas: position, residue, sasa_abs, sasa_rel.
    """
    # TODO(alumno): definir active_site_atoms de forma trazable. No imputar
    # residuos ausentes; devolver NaN cuando no haya coordenadas.
    raise NotImplementedError


def distance_to_active_site(
    structure: Structure,
    active_site_atoms: list,
    chain_id: str = "A",
) -> pd.DataFrame:
    """
    Distancia mínima de cada residuo a los átomos del sitio activo.

    Parameters
    ----------
    active_site_atoms : list
        Lista de Bio.PDB.Atom que definen el sitio activo (nucleótido + Mg²⁺).

    Returns
    -------
    pd.DataFrame con columnas: position, residue, min_distance_angstrom.
    """
    raise NotImplementedError


def shannon_entropy_per_position(msa) -> pd.DataFrame:
    """Calcula la entropia de Shannon por columna del MSA.

    La salida debe contener ``CONSERVATION_COLUMNS``.
    """
    raise NotImplementedError


def merge_features(
    mutations_df: pd.DataFrame,
    sasa_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    conservation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el DataFrame maestro integrando todos los descriptores.

    Returns
    -------
    pd.DataFrame
        Una fila por (gen, posicion) con las columnas minimas de
        ``MASTER_FEATURE_COLUMNS``.
    """
    raise NotImplementedError
