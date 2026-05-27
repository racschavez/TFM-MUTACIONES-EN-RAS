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
    df = pd.read_csv(path, sep="\t")
    if missing_expected_columns(df) != []:
        raise ValueError(
            f"Faltan columnas esperadas en el fichero COSMIC: {missing_expected_columns(df)}"
        )
    else:
        return df

    # TODO(alumno): cargar TSV con pandas, comprobar columnas esperadas y
    # documentar cualquier mapeo de nombres usado para COSMIC real.


def filter_somatic_missense(df: pd.DataFrame, cfg_filters: dict) -> pd.DataFrame:
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
    from IPython.display import display

    n_inicial = len(df)

    df_somaticas = df[df["Mutation somatic status"] == cfg_filters['somatic_status']]
    n_somaticas = len(df_somaticas)

    df_filtrado = df_somaticas[df_somaticas["Mutation Description"] == cfg_filters['mutation_description_contains']]
    
    n_missense = len(df_filtrado)
    n_final = len(df_filtrado)

    summary = pd.DataFrame([{
        "Conteo Inicial": n_inicial,
        "Mutaciones Somáticas": n_somaticas,
        "Mutaciones Missense": n_missense,
        "Conteo Final": n_final
    }])

    display(summary.style.hide(axis="index"))

    return df_filtrado

    # TODO(alumno): usar los filtros declarados en configs/config.yaml y guardar
    # conteos n_inicial -> n_somaticas -> n_missense -> n_final.


def deduplicate_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados manteniendo una mutacion por muestra y cambio proteico.
    """
    df_dedup = df.drop_duplicates(subset=["ID_sample", "Mutation AA"])

    return df_dedup


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


def build_curated_dataset(df_filtered: pd.DataFrame, cfg_version: str) -> pd.DataFrame:
    """
    Construye el DataFrame curado final.

    La salida debe contener exactamente, como minimo, las columnas definidas en
    ``CURATED_COLUMNS``. Consulta ``docs/contratos_datos.md`` antes de cambiar
    este esquema.
    """

    #Construcción del dataframe columna por columna
    df_curated = pd.DataFrame()

        #Columna "gene"
    df_curated['gene'] = df_filtered['GENE_NAME'].astype(str)

        #Columna "uniprot_id"
    df_curated['uniprot_id'] = df_filtered['GENE_NAME'].map(UNIPROT_BY_GENE).astype(str)
    
        #Columna "position", "wt_aa", "mut_aa"

    df_curated['position'] = df_filtered['Mutation AA'].apply(lambda x: parse_aa_change(x)[1]).astype(int)
    df_curated['wt_aa'] = df_filtered['Mutation AA'].apply(lambda x: parse_aa_change(x)[0]).astype(str)
    df_curated['mut_aa'] = df_filtered['Mutation AA'].apply(lambda x: parse_aa_change(x)[2]).astype(str)

        #Columna "hgvs_p"   
    df_curated['hgvs_p']= df_filtered['Mutation AA'].astype(str)

        #Columna "sample_count"
    df_curated['sample_count'] = len(df_curated)

        #Columna "tumour_types"
    df_curated['tumour_types'] = df_filtered['Histology'].astype(str)

        #Columna "primary_tissues"
    df_curated['primary_tissues'] = df_filtered['Primary site'].astype(str)

        #Columna "cosmic_version"
    df_curated['cosmic_version'] = str(cfg_version)
    
    return df_curated


def missing_expected_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve columnas COSMIC esperadas que no aparecen en ``df``."""
    return sorted(EXPECTED_COSMIC_COLUMNS - set(df.columns))
