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
EXPECTED_COSMIC_COLUMNS = [
    "GENE_SYMBOL",
    "MUTATION_AA",
    "MUTATION_SOMATIC_STATUS",
    "MUTATION_DESCRIPTION",
    "COSMIC_SAMPLE_ID",
    "PRIMARY_SITE",
    "PRIMARY_HISTOLOGY",
]
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


def load_cosmic_raw(*paths) -> list[pd.DataFrame]:
    """
    Carga el fichero en bruto descargado de COSMIC.
    Parameters
    ----------
    path : str or Path
        Ruta al fichero COSMIC (típicamente un TSV).
    Returns
    -------
    list[pd.DataFrame]
        Lista de DataFrames cargados.
    """
    #Lectura de los DataFrames cargados con pandas
    dfs = [pd.read_csv(path, sep="\t", low_memory=False) for path in paths]

    #Comprobación de columnas esperadas en los DataFrames cargados
    missing = missing_expected_columns(*dfs)

    if missing:
        raise ValueError(
            f"Faltan columnas esperadas en el fichero COSMIC: {missing}"
        )
    
    else:
        return dfs

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

    #Filtrando por genes RAS
    df = df[df["GENE_SYMBOL"].isin(RAS_GENES)]
    #Conteo inicial de mutaciones en genes RAS
    n_inicial = len(df)

    #Filtrando por mutaciones somáticas confirmadas
    df_somaticas = df[df["MUTATION_SOMATIC_STATUS"] == cfg_filters['somatic_status']]
    #Conteo de mutaciones somáticas confirmadas
    n_somaticas = len(df_somaticas)

    #Filtrando por mutaciones missense
    df_filtrado = df_somaticas[(
        df_somaticas["MUTATION_DESCRIPTION"] == cfg_filters['mutation_description_contains'])
        &
        (df_somaticas["MUTATION_AA"].str.match(r"^p\.([A-Z])(\d+)([A-Z])$", na=False))
    ]
    
    #Conteo de mutaciones missense
    n_missense = len(df_filtrado)
    n_final = len(df_filtrado)

    #Mostrar resumen de conteos
    summary = pd.DataFrame([{
        "Conteo Inicial": n_inicial,
        "Mutaciones Somáticas": n_somaticas,
        "Mutaciones Missense": n_missense,
        "Conteo Final": n_final
    }])

    display(summary.style.hide(axis="index"))

    #DataFrame filtrado
    return df_filtrado

    # TODO(alumno): usar los filtros declarados en configs/config.yaml y guardar
    # conteos n_inicial -> n_somaticas -> n_missense -> n_final.


def deduplicate_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados manteniendo una mutacion por muestra y cambio proteico.
    """
    df_dedup = df.drop_duplicates(subset=["COSMIC_SAMPLE_ID", "MUTATION_AA"])

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
    df_curated['gene'] = df_filtered['GENE_SYMBOL'].astype(str)

        #Columna "uniprot_id"
    df_curated['uniprot_id'] = df_filtered['GENE_SYMBOL'].map(UNIPROT_BY_GENE).astype(str)
    
        #Columna "position", "wt_aa", "mut_aa"

    df_curated['position'] = df_filtered['MUTATION_AA'].apply(lambda x: parse_aa_change(x)[1]).astype(int)
    df_curated['wt_aa'] = df_filtered['MUTATION_AA'].apply(lambda x: parse_aa_change(x)[0]).astype(str)
    df_curated['mut_aa'] = df_filtered['MUTATION_AA'].apply(lambda x: parse_aa_change(x)[2]).astype(str)

        #Columna "hgvs_p"   
    df_curated['hgvs_p']= df_filtered['MUTATION_AA'].astype(str)

        #Columna "sample_count"
    df_curated['sample_count'] = len(df_curated)

        #Columna "tumour_types"
    df_curated['tumour_types'] = df_filtered['PRIMARY_HISTOLOGY'].astype(str)

        #Columna "primary_tissues"
    df_curated['primary_tissues'] = df_filtered['PRIMARY_SITE'].astype(str)

        #Columna "cosmic_version"
    df_curated['cosmic_version'] = str(cfg_version)
    
    return df_curated


def missing_expected_columns(*dfs) -> list:
    """Devuelve columnas COSMIC esperadas ausentes en los DataFrames proporcionados."""
    combined_columns = set().union(*(set(df.columns) for df in dfs))

    missing = set(EXPECTED_COSMIC_COLUMNS) - combined_columns

    return sorted(missing)
