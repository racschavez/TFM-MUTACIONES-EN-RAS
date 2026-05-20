"""Alineamiento multiple y mapping de posiciones.

Las funciones de este modulo son el nucleo del hito 2. Mantienen contratos de
entrada/salida estables para que los notebooks y tests puedan validar G12, G13 y
Q61 sin depender de detalles internos de implementacion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from Bio.Align import MultipleSeqAlignment

POSITION_MAP_COLUMNS = ["msa_position", "KRAS", "HRAS", "NRAS"]
HOTSPOT_POSITIONS = (12, 13, 61)


def fetch_uniprot_sequence(uniprot_id: str) -> str:
    """Descarga o recupera de cache la secuencia canonica desde UniProt.

    Debe devolver una cadena de aminoacidos sin cabecera FASTA. Si no hay red,
    se permite leer una copia cacheada en ``data/external/sequences/`` y
    documentarlo en el notebook.
    """
    raise NotImplementedError


def run_msa(sequences: dict[str, str], algorithm: str = "clustalo") -> MultipleSeqAlignment:
    """
    Ejecuta MSA usando el algoritmo especificado.

    Parameters
    ----------
    sequences : dict
        Mapping {gene_name: sequence}.
    algorithm : str
        "clustalo", "mafft" o "muscle".

    Returns
    -------
    MultipleSeqAlignment
    """
    # TODO(alumno): usar clustalo/mafft si estan disponibles. Para KRAS, HRAS y
    # NRAS canonicos se puede justificar una comprobacion ungapped si las tres
    # secuencias tienen la misma longitud y no hay indels en el dominio G.
    raise NotImplementedError


def build_position_map(msa: MultipleSeqAlignment) -> pd.DataFrame:
    """
    Construye la tabla de equivalencias posición-a-posición a partir del MSA.

    Returns
    -------
    pd.DataFrame
        Debe contener las columnas ``POSITION_MAP_COLUMNS``. Las posiciones son
        1-indexed. Los gaps se representan como NaN.
    """
    raise NotImplementedError


def find_recurrent_positions(
    mutations_df: pd.DataFrame,
    position_map: pd.DataFrame,
    min_members: int = 2,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Identifica posiciones del MSA mutadas en al menos ``min_members`` miembros
    con al menos ``min_samples`` muestras COSMIC en cada uno.

    La salida debe incluir, como minimo: ``msa_position``, ``members_mutated``,
    ``total_sample_count`` y ``genes``.
    """
    raise NotImplementedError
