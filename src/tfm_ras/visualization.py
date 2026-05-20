"""Visualizacion: heatmaps, scatter plots y vistas 3D."""

import pandas as pd
import matplotlib.pyplot as plt

FIGURE_REQUIREMENTS = {
    "heatmap": ["gene", "msa_position", "sample_count"],
    "scatter": ["sasa_rel", "distance_to_active_site", "sample_count"],
    "three_d": ["gene", "position"],
}


def heatmap_mutations_per_position(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """
    Heatmap con posiciones del MSA en eje X y miembros de la familia en eje Y.
    El color codifica la frecuencia de mutación (log o lineal).

    Debe devolver un ``matplotlib.figure.Figure`` para que pueda guardarse desde
    el notebook sin depender de estado global.
    """
    raise NotImplementedError


def scatter_sasa_distance(df: pd.DataFrame, **kwargs) -> plt.Figure:
    """
    Scatter SASA vs distancia al sitio activo, coloreado por frecuencia.

    Incluir etiquetas suficientes para distinguir hotspots principales.
    """
    raise NotImplementedError


def view_3d_with_hotspots(
    pdb_id: str,
    hotspot_positions: list[int],
    color_by: str = "frequency",
):
    """
    Vista 3D interactiva con py3Dmol resaltando los hotspots.

    Parameters
    ----------
    pdb_id : str
    hotspot_positions : list[int]
    color_by : str
        "frequency", "conservation" o "sasa".

    Returns
    -------
    py3Dmol.view
    """
    raise NotImplementedError
