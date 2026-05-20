"""
Tests unitarios mínimos.

El alumno debe ampliar estos tests para cubrir las funciones críticas.
Los tests sobre hotspots conocidos (G12, G13, Q61) son OBLIGATORIOS.
"""

import pytest
import pandas as pd

from tfm_ras.cosmic_loader import (
    CURATED_COLUMNS,
    EXPECTED_COSMIC_COLUMNS,
    missing_expected_columns,
    parse_aa_change,
)


def test_parse_aa_change_g12d():
    """G12D debe parsearse a (G, 12, D)."""
    wt, pos, mut = parse_aa_change("p.G12D")
    assert wt == "G"
    assert pos == 12
    assert mut == "D"


def test_parse_aa_change_q61r():
    """Q61R debe parsearse correctamente."""
    wt, pos, mut = parse_aa_change("p.Q61R")
    assert wt == "Q"
    assert pos == 61
    assert mut == "R"


def test_parse_aa_change_rejects_invalid_format():
    """El parser no debe aceptar formatos ambiguos."""
    with pytest.raises(ValueError):
        parse_aa_change("G12D")


def test_expected_cosmic_columns_are_documented():
    """Las columnas criticas de entrada deben estar fijadas en codigo."""
    assert {"GENE_NAME", "Mutation AA", "ID_sample"}.issubset(
        EXPECTED_COSMIC_COLUMNS
    )


def test_missing_expected_columns_reports_absent_columns():
    df = pd.DataFrame({"GENE_NAME": ["KRAS"], "Mutation AA": ["p.G12D"]})
    missing = missing_expected_columns(df)
    assert "ID_sample" in missing
    assert "GENE_NAME" not in missing


def test_curated_schema_contains_minimum_columns():
    assert [
        "gene",
        "uniprot_id",
        "position",
        "wt_aa",
        "mut_aa",
        "hgvs_p",
        "sample_count",
    ] == CURATED_COLUMNS[:7]


# --- Tests adicionales obligatorios para el TFM ---
# El alumno debe completar:
#
# def test_mapping_g12_kras_to_hras():
#     """En el MSA, la posición de G12 en KRAS debe equivaler a G12 en HRAS."""
#     ...
#
# def test_mapping_q61_across_family():
#     """Q61 debe ser equivalente en los tres miembros."""
#     ...
#
# def test_active_site_residues_close_to_ligand():
#     """Los residuos del P-loop deben tener distancia <5 Å al GDP."""
#     ...
