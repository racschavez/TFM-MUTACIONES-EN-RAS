"""
Módulo de visualización para el análisis comparativo estructural de los parálogos RAS.

Genera las figuras finales del pipeline bioinformático del TFM, incluyendo:

  - **Heatmap de mutaciones somáticas** (heatmap_mutations_per_position): muestra la
    distribución de carga mutacional a lo largo de las posiciones MSA de KRAS, HRAS y
    NRAS en escala logarítmica, facilitando la comparación directa entre parálogos.

  - **Clustermap de patrones recurrentes** (clustermap_top_positions): aplica
    agrupamiento jerárquico sobre las posiciones más frecuentemente mutadas para revelar
    si existen bloques de posiciones con perfiles de incidencia similares entre genes.

  - **Scatter SASA vs distancia al sitio activo** (scatter_sasa_distance): representa
    el contexto estructural de cada posición mutada relacionando su accesibilidad al
    solvente (SASA) con su proximidad funcional al sitio de unión a GTP/GDP.

  - **Vista 3D interactiva de un único parálogo** (view_3d_with_hotspots): genera un
    fichero HTML auto-contenido con un visor WebGL (py3Dmol) que colorea las esferas de
    los hotspots según una métrica cuantitativa (p. ej. número de muestras).

  - **Vista 3D comparativa de los tres parálogos** (comparative_3d_view): superpone
    estructuralmente KRAS, HRAS y NRAS sobre los átomos Cα del dominio G (residuos 1-166)
    y los representa en el mismo visor con colores diferenciados por gen.

  - **Panel resumen** (summary_panel): figura compuesta multipanel (A–F) que integra
    todos los hallazgos clave del TFM en una única imagen lista para incluir en la memoria.

Dependencias externas requeridas:
    matplotlib, seaborn, numpy, pandas, py3Dmol, biopython (Bio.PDB).
"""

from __future__ import annotations  # Permite anotaciones de tipo diferidas (PEP 563)

from io import StringIO  # Buffer de texto en memoria; usado para escribir PDB sin disco
from pathlib import Path  # Manejo de rutas de forma orientada a objetos (independiente de SO)

import matplotlib.pyplot as plt  # Motor de renderizado de figuras 2D
import numpy as np  # Operaciones numéricas vectorizadas (log10, clip, etc.)
import pandas as pd  # Manipulación de DataFrames tabulares
import py3Dmol  # Visor molecular WebGL embebible en HTML (sin servidor)
import seaborn as sns  # Figuras estadísticas de alto nivel sobre matplotlib
from Bio.PDB import PDBIO, PDBParser, Superimposer  # Lectura, escritura y superposición de estructuras PDB

from tfm_ras.config import project_root  # Devuelve la ruta raíz del proyecto (definida en config.py)

# ---------------------------------------------------------------------------
# Constantes globales de configuración visual y biológica
# ---------------------------------------------------------------------------

# Paleta de colores canónica para los tres parálogos RAS.
# Usar siempre los mismos colores en todas las figuras garantiza coherencia visual
# en la memoria del TFM y facilita la identificación inmediata de cada gen.
# Los valores son códigos hexadecimales inspirados en la paleta Tableau-10.
GENE_COLORS = {"KRAS": "#4C78A8", "HRAS": "#F58518", "NRAS": "#54A24B"}

# Posiciones oncogénicas canónicas de las proteínas RAS.
# G12 (Gly12) y G13 (Gly13) pertenecen al P-loop (loop de unión a fosfato, residuos 10-17)
# y su mutación impide la hidrólisis del GTP por mecanismo estérico con Gln61.
# Q61 (Gln61) forma parte del Switch II y es el residuo catalítico que coordina el agua
# nucleofílica en la GTPasa; su mutación bloquea la hidrólisis intrínseca y la mediada por GAP.
# El diccionario mapea posición UniProt (int) → etiqueta de texto para las figuras.
HOTSPOTS = {12: "G12", 13: "G13", 61: "Q61", 146: "A146", 117: "K117", 59:"A59"}

# Nombres de residuos de ligandos en los ficheros PDB que se colorearán de forma
# diferenciada en las vistas 3D para contextualizar el sitio de unión a nucleótido:
#   GDP  = guanosina difosfato (estado inactivo)
#   GTP  = guanosina trifosfato (estado activo)
#   GNP  = guanilil-imidodifosfato (análogo no hidrolizable de GTP; GMPPNP)
#   GCP  = guanilil-metileno-difosfato (análogo no hidrolizable; GMPPCP)
#   MG   = ion magnesio coordinado por el nucleótido (esencial para la actividad GTPasa)
LIGAND_RESNAMES = ["GDP", "GTP", "GNP", "GCP", "MG"]

STRUCTURE_REGISTRY = {
    "KRAS": {
        "GTP": {"pdb_id": "5UK9", "chain": "B"},
        "GDP": {"pdb_id": "4OBE", "chain": "A"},
    },
    "HRAS": {
        "GTP": {"pdb_id": "3K8Y", "chain": "A"},
        "GDP": {"pdb_id": "4Q21", "chain": "A"},
    },
    "NRAS": {
        "GTP": {"pdb_id": "5UHV", "chain": "A"},
        "GDP": {"pdb_id": "3CON", "chain": "A"},
    },
}
# ===========================================================================
# Clase auxiliar: visor py3Dmol con JavaScript embebido (sin CDN)
# ===========================================================================

# Ruta al fichero 3Dmol-min.js descargado localmente.
# JupyterLab ≥ 3 impone una Content Security Policy (CSP) estricta que bloquea
# la carga dinámica de scripts desde CDNs externos (como cloudflare o 3dmol.org).
# La solución es incrustar el código JavaScript directamente en el HTML del output
# de la celda, evitando así cualquier petición de red en el navegador.
_3DMOL_JS_PATH = Path(__file__).parent.parent.parent / "assets" / "3Dmol-min.js"


class _OfflineView:
    """
    Envuelve un objeto py3Dmol.view para incrustar el JavaScript de 3Dmol.js
    directamente en el HTML de salida, en lugar de cargarlo desde una CDN.

    Esto es necesario porque JupyterLab ≥ 3 bloquea scripts externos por CSP.
    El método _repr_html_() parchea el HTML generado por py3Dmol:
      1. Sustituye la llamada loadScriptAsync(CDN_URL) por un bloque inline
         que inserta el contenido completo de 3Dmol-min.js si aún no está cargado.
      2. Define $3Dmolpromise como una Promise ya resuelta para que el callback
         .then(...) del visor se ejecute de inmediato.
    """

    def __init__(self, view: "py3Dmol.view"):
        self._view = view  # Objeto view original de py3Dmol

    def _repr_html_(self) -> str:
        """Genera el HTML del visor con 3Dmol.js incrustado en lugar de CDN."""
        html = self._view._make_html()  # HTML estándar de py3Dmol (con CDN)

        # Cargamos el contenido de 3Dmol-min.js para incrustarlo.
        # Si el fichero no existe (estudiante aún no lo ha descargado), usamos el CDN
        # como fallback pero mostramos una advertencia en la salida.
        if _3DMOL_JS_PATH.exists():
            js_code = _3DMOL_JS_PATH.read_text(encoding="utf-8")
            # Código de reemplazo: define $3Dmol inline si no está ya en el entorno
            # y resuelve $3Dmolpromise inmediatamente (sin esperar carga de red).
            inline_block = (
                "if(typeof $3Dmol === 'undefined') {\n"
                "  var savedexports = (typeof exports !== 'undefined') ? exports : (exports = {});\n"
                "  var savedmodule  = (typeof module  !== 'undefined') ? module  : (module  = {});\n"
                + js_code + "\n"
                "  exports = savedexports; module = savedmodule;\n"
                "}\n"
                "if(typeof $3Dmolpromise === 'undefined') {\n"
                "  $3Dmolpromise = Promise.resolve();\n"
                "}"
            )
        else:
            # Fallback: mantener la carga desde CDN con un aviso en consola
            inline_block = (
                "console.warn('3Dmol-min.js no encontrado en assets/. "
                "Cargando desde CDN (requiere internet).');\n"
                "if(typeof $3Dmolpromise === 'undefined') {\n"
                "  $3Dmolpromise = loadScriptAsync('https://cdnjs.cloudflare.com"
                "/ajax/libs/3Dmol/2.4.2/3Dmol-min.js');\n"
                "}"
            )

        # Buscamos el bloque que py3Dmol genera para cargar el script desde CDN.
        # La plantilla de py3Dmol produce exactamente este patrón (ver fuente de py3Dmol):
        #   if(typeof $3Dmolpromise === 'undefined') {
        #   $3Dmolpromise = null;
        #     $3Dmolpromise = loadScriptAsync('URL');
        #   }
        import re  # módulo de expresiones regulares (importación local para no contaminar namespace)
        # Patrón que captura el bloque completo de carga del CDN
        cdn_pattern = re.compile(
            r"if\(typeof \$3Dmolpromise === 'undefined'\) \{\n"
            r"\$3Dmolpromise = null;\n"
            r"  \$3Dmolpromise = loadScriptAsync\('[^']*'\);\n"
            r"\}",
            re.MULTILINE,
        )
        # Usamos una función lambda como reemplazo para que re.sub no interprete
        # las secuencias de escape del contenido de 3Dmol.js (p.ej. \s, \d, etc.)
        # como patrones de referencia de grupo. Con una función callable, el string
        # se devuelve literalmente sin ningún procesamiento de escape.
        patched = cdn_pattern.sub(lambda _: inline_block, html)
        return patched

    def __getattr__(self, name):
        # Delega cualquier otro atributo/método al view original de py3Dmol
        return getattr(self._view, name)


# ===========================================================================
# Funciones públicas de visualización
# ===========================================================================

def heatmap_mutations_per_position(master_df, output_basename, **kwargs):
    """
    Genera un heatmap de la carga mutacional somática por posición MSA y gen.

    Cada celda representa el número total de muestras de cáncer con mutación en esa
    posición para ese gen, transformado a escala log10(x+1) para comprimir el rango
    dinámico: KRAS concentra miles de muestras en G12/G13 mientras la mayoría de
    posiciones tienen <10, por lo que la escala lineal quedaría completamente dominada
    por los hotspots y el resto sería indiferenciable.

    La paleta **cividis** es perceptualmente uniforme (intensidad aumenta monótonamente
    con el valor) y está optimizada para daltonismo (deuteranopia/protanopia), lo que la
    hace adecuada para publicaciones y presentaciones académicas.

    Las líneas blancas verticales y etiquetas sobre los hotspots canónicos (G12, G13, Q61)
    guían la lectura hacia las posiciones biológicamente más relevantes.

    Args:
        master_df (pd.DataFrame): DataFrame maestro con columnas 'gene', 'msa_position'
            y 'total_samples' (al menos). Típicamente generado por el módulo de análisis.
        output_basename (str | Path): Ruta base para los ficheros de salida (sin extensión,
            o con .png/.svg que se eliminará automáticamente).
        **kwargs: Parámetros opcionales; actualmente soporta 'figsize' (tuple) para
            controlar las dimensiones de la figura en pulgadas.

    Returns:
        dict: Diccionario con claves 'png' y 'svg' apuntando a los ficheros generados
            (objetos Path).
    """
def heatmap_mutations_per_position(master_df, output_basename, **kwargs):

    output_base = _output_base(output_basename)

    df_single_state = master_df.query("state == 'GTP'")
    
    matrix = master_df.pivot_table(
        index="gene",
        columns="msa_position",
        values="total_samples",
        aggfunc="sum",
        fill_value=0,
    ).reindex(["KRAS", "HRAS", "NRAS"])

    matrix = np.log10(matrix + 1)

    # Figura
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (14, 4.0)))

    sns.heatmap(
        matrix,
        cmap="cividis",
        cbar_kws={"label": "log₁₀(muestras+1)", "shrink": 0.8},
        ax=ax,
        linecolor="white",
    )

    ax.set_xlabel("Carga de mutaciones somáticas en los parálogos RAS", labelpad=25)
    ax.set_ylabel("")


    # =========================================================
    # HOTSPOTS
    # =========================================================

    HOTSPOTS = {
        12: "G12",
        13: "G13",
        59: "A59",
        61: "Q61",
        117: "K117",
        146: "A146",
    }

    # offsets manuales para evitar choques entre posiciones cercanas
    y_offsets = {
        12: -0.25,
        13: -0.05,
        59: -0.25,
        61: -0.05,
        117: -0.05,
        146: -0.05,
    }

    for position, label in HOTSPOTS.items():

        # línea vertical (heatmap usa índice de columnas → position - 1)
        ax.axvline(position - 0.5, color="black", linewidth=0.8, linestyle="--")

        # etiqueta desplazada para evitar solapamiento
        ax.text(
            position - 0.5,
            y_offsets.get(position, -0.15),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="gray",
                linewidth=0.5,
                alpha=0.9,
            ),
        )

    ax.grid(False)


    fig.subplots_adjust(top=0.88, bottom=0.30)

    fig.tight_layout()

    return _save_png_svg(fig, output_base)


def clustermap_top_positions(master_df, top_n=30, output_basename=None):
    """
    Genera un clustermap (heatmap + dendrograma) de las posiciones más frecuentemente
    mutadas en los tres parálogos RAS.

    A diferencia del heatmap simple, el **agrupamiento jerárquico** en filas (posiciones)
    revela si ciertas posiciones comparten perfiles de incidencia similares entre genes.
    Por ejemplo, si G12 y G13 se agrupan juntas es porque sus patrones de afectación
    relativa en KRAS/HRAS/NRAS son similares; si Q61 se separa, su distribución entre
    parálogos es distinta. Esto permite identificar bloques funcionalmente análogos más
    allá de la simple magnitud de las mutaciones.

    El agrupamiento de **columnas está desactivado** (col_cluster=False) para mantener
    el orden fijo KRAS > HRAS > NRAS y facilitar la comparación con otras figuras del
    TFM.

    La escala logarítmica log10(x+1) se aplica por la misma razón que en el heatmap
    simple: evitar que unos pocos hotspots de KRAS dominen visualmente toda la figura.
    La paleta **viridis** (amarillo=alto, morado=bajo) es la referencia científica
    estándar por ser perceptualmente uniforme y reproducible en escala de grises.

    Args:
        master_df (pd.DataFrame): DataFrame maestro con columnas 'msa_position', 'gene'
            y 'total_samples'.
        top_n (int): Número de posiciones con mayor carga total que se incluirán.
            Por defecto 30, suficiente para ver el paisaje mutacional global sin saturar
            la figura.
        output_basename (str | Path | None): Ruta base de salida. Si es None se usa
            'figures/04_clustermap_patterns' relativo a la raíz del proyecto.

    Returns:
        dict: Diccionario con claves 'png' y 'svg' (objetos Path).
    """
    # Si no se proporciona nombre de salida, usa la ruta por defecto dentro del proyecto
    output_base = _output_base(output_basename or "figures/04_clustermap_patterns")

    df_single_state = master_df.query("state == 'GTP'")

    # Selecciona las top_n posiciones con mayor carga mutacional total (sumando los tres genes).
    # groupby + sum colapsa todas las filas de la misma posición; sort_values ordena descendente.
    top_positions = (
        df_single_state.groupby("msa_position")["total_samples"]
        .sum().sort_values(ascending=False).head(top_n).index  # Índices (posiciones) de las top_n más mutadas
    )

    # Filtra el DataFrame original para quedarse solo con las posiciones seleccionadas
    subset = df_single_state.loc[master_df["msa_position"].isin(top_positions)]

    # Construye la matriz posición × gen con suma de muestras por celda
    matrix = subset.pivot_table(
        index="msa_position",          # Filas: posición en el alineamiento
        columns="gene",                # Columnas: gen (KRAS, HRAS, NRAS)
        values="total_samples",        # Valor: suma de muestras
        aggfunc="sum",                 # Función de agregación: suma
        fill_value=0,                  # Sin datos → 0 (pseudocuenta implícita antes del log)
    ).reindex(columns=["KRAS", "HRAS", "NRAS"])  # Orden canónico de columnas

    # Transformación log10(x+1) para comprimir el rango dinámico (igual que en heatmap)
    matrix = np.log10(matrix + 1)

    # sns.clustermap combina heatmap + dendrogramas de agrupamiento jerárquico.
    # col_cluster=False: no agrupar columnas (mantiene orden KRAS/HRAS/NRAS fijo).
    # La altura de la figura escala con el número de posiciones para legibilidad.
    grid = sns.clustermap(
        matrix,                                        # Matriz de datos transformados
        cmap="viridis",                                # Paleta perceptualmente uniforme (referencia científica)
        figsize=(6, max(4, len(matrix) * 0.35)),       # Altura mínima 4 pulgadas; crece con las filas
        cbar_kws={"label": "log10(muestras+1)"},       # Etiqueta de la barra de color
        col_cluster=False,                             # Deshabilita agrupamiento de columnas (genes)
    )
    grid.fig.suptitle("Patrones de mutación más recurrentes", y=1.02)  # Título por encima del dendrograma

    png = output_base.with_suffix(".png")  # Ruta del fichero PNG
    svg = output_base.with_suffix(".svg")  # Ruta del fichero SVG

    # clustermap devuelve un objeto ClusterGrid cuya figura interna es grid.fig
    grid.fig.savefig(png, dpi=300, bbox_inches="tight")  # Guarda PNG a 300 dpi (calidad de publicación)
    grid.fig.savefig(svg, bbox_inches="tight")            # Guarda SVG vectorial (editable en Inkscape/Illustrator)
    plt.close(grid.fig)  # Libera la memoria de la figura del clustermap
    return {"png": png, "svg": svg}  # Devuelve las rutas generadas para su uso por el pipeline


def scatter_sasa_distance(master_df, output_basename, **kwargs):
    """
    Genera una figura de 3x2 paneles (filas=gen, columnas=estado) comparando
    SASA relativa vs distancia al sitio activo para cada combinación
    gen-estado (KRAS/HRAS/NRAS × GTP/GDP).

    Replica la lógica de scatter_sasa_distance pero con:
      - Ejes X/Y compartidos entre todos los paneles (mismo rango), para que
        el desplazamiento de puntos entre GTP y GDP sea directamente comparable.
      - Escala de tamaño de punto (log10(total_samples+1)) idéntica en todos
        los paneles, usando el mismo factor de escalado que la figura original.
      - Barra de color de entropía compartida (mismo vmin/vmax) en vez de una
        por panel.

    Nota: total_samples y shannon_entropy son propiedades de (gen, posición)
    independientes del estado estructural, por lo que el tamaño y color de
    los puntos serán iguales entre el panel GTP y GDP de un mismo gen; lo que
    cambia entre columnas es la posición (x, y), reflejando cómo varía el
    contexto estructural (SASA, distancia al sitio activo) según la
    conformación.

    Args:
        master_df (pd.DataFrame): DataFrame maestro con columnas 'gene', 'state',
            'sasa_rel', 'dist_active_site_angstrom', 'total_samples',
            'shannon_entropy', 'uniprot_position', 'aa_wt'.
        output_basename (str | Path): Ruta base para los ficheros de salida.
        **kwargs: 'figsize' opcional (tuple).

    Returns:
        dict: Diccionario con claves 'png' y 'svg' (objetos Path).
    """
    output_base = _output_base(output_basename)

    genes = ["KRAS", "HRAS", "NRAS"]
    states = ["GTP", "GDP"]

    # Rango de ejes compartido entre todos los paneles, calculado sobre el
    # dataset completo para que la comparación GTP vs GDP sea directa
    x_min, x_max = master_df["sasa_rel"].min(), master_df["sasa_rel"].max()
    y_min, y_max = master_df["dist_active_site_angstrom"].min(), master_df["dist_active_site_angstrom"].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    # Rango de color compartido para la entropía de Shannon
    vmin, vmax = master_df["shannon_entropy"].min(), master_df["shannon_entropy"].max()

    fig, axes = plt.subplots(
        nrows=len(genes), ncols=len(states),
        figsize=kwargs.get("figsize", (9, 11)),
        sharex=True, sharey=True,
    )

    scatter_ref = None  # guarda una referencia para la colorbar compartida

    for i, gene in enumerate(genes):
        for j, state in enumerate(states):
            ax = axes[i, j]
            subset = master_df.query("gene == @gene and state == @state").copy()

            subset["point_size"] = np.clip(np.log10(subset["total_samples"] + 1) * 55, 10, 250)

            scatter = ax.scatter(
                subset["sasa_rel"],
                subset["dist_active_site_angstrom"],
                s=subset["point_size"],
                c=subset["shannon_entropy"],
                cmap="viridis",
                vmin=vmin, vmax=vmax,
                alpha=0.76,
                edgecolor="white",
                linewidth=0.35,
            )
            scatter_ref = scatter

            ax.axvline(0.25, color="#555555", linestyle="--", linewidth=0.8)
            ax.axhline(5.0, color="#555555", linestyle="--", linewidth=0.8)

            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

            ax.set_title(f"{gene} — {state}", fontsize=10, fontweight="bold")

            if i == len(genes) - 1:
                ax.set_xlabel("SASA relativa")
            if j == 0:
                ax.set_ylabel("Distancia al sitio activo (Å)")

            # Anota solo hotspots canónicos con muestras reales
            annotated = subset.loc[
                subset["uniprot_position"].isin(HOTSPOTS) & subset["total_samples"].gt(0)
            ]
            for row in annotated.itertuples():
                ax.annotate(
                    f"{row.aa_wt}{row.uniprot_position}",
                    (row.sasa_rel, row.dist_active_site_angstrom),
                    ha="center", va="center", fontsize=7, fontweight = "bold", color="black",
                    xytext=(0,8), textcoords="offset points", bbox=dict(
                        boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", linewidth=0.5, alpha=0.9
                    )
                )

    fig.suptitle("Contexto estructural de las mutaciones RAS por isoforma y estado", fontsize=13, y=1.01)
    fig.colorbar(scatter_ref, ax=axes, label="Entropía de Shannon", shrink=0.6, pad=0.02)

    return _save_png_svg(fig, output_base)


def view_3d_with_hotspots(pdb_id, hotspot_data, output_html, color_by="sample_count", chain="A"):
    """
    Genera una vista 3D interactiva de la estructura de un parálogo RAS con los
    hotspots mutacionales coloreados cuantitativamente, exportada como HTML.
 
 
    Args:
        pdb_id (str | Path): Identificador PDB o ruta directa al fichero .pdb.
        hotspot_data (pd.DataFrame | list[dict]): Datos de los hotspots.
        output_html (str | Path): Ruta del fichero HTML de salida.
        color_by (str): Columna que determina el color/tamaño de las esferas.
        chain (str): Cadena de la estructura a representar (por defecto "A";
            usar "B" para KRAS-GTP/5UK9).
 
    Returns:
        tuple[Path, py3Dmol.view]: Ruta al HTML y el visor (usar como última
            expresión de la celda de Jupyter para verlo inline con zoom/rotación).
    """
    pdb_path = _resolve_pdb_path(pdb_id)
    pdb_text = pdb_path.read_text(encoding="utf-8")
 
    hotspot_data = _normalize_hotspot_data(hotspot_data)
    max_value = max(float(hotspot_data[color_by].max()), 1.0) if not hotspot_data.empty else 1.0
 
    view = py3Dmol.view(width=820, height=560)
    view.addModel(pdb_text, "pdb")

    view.setStyle({}, {})
 
    view.setStyle({"chain": chain}, {"cartoon": {"color": "lightgray"}})
    view.addStyle({"chain": chain, "resn": LIGAND_RESNAMES}, {"stick": {"colorscheme": "greenCarbon"}})
 
    for row in hotspot_data.itertuples():
        value = float(getattr(row, color_by))
        view.addStyle(
            {"chain": chain, "resi": int(row.uniprot_position)},
            {
                "sphere": {
                    "color": _metric_color(value, max_value),
                    "radius": 0.75 + min(value / max_value, 1.0) * 0.55,
                }
            },
        )
 
    view.zoomTo()
 
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    view.write_html(str(output_html))
    return output_html, _OfflineView(view)


def comparative_3d_view(pdb_paths_dict, hotspot_data, output_html, chains=None, hotspot_color="#BE61E6", show_labels=True, opacity=0.88):
    """
    Genera una vista 3D comparativa de varios parálogos RAS (o de varias conformaciones) superpuestos estructuralmente en un único visor WebGL.
 
    Args:
        pdb_paths_dict (dict[str, str | Path]): {etiqueta: id_pdb o ruta}.
            El primer elemento en orden de inserción es la referencia fija
            de la superposición.
        hotspot_data (pd.DataFrame | list[dict] | None): hotspots a marcar
            (columnas 'gene' y 'uniprot_position'). None → sin hotspots.
        output_html (str | Path): Ruta del fichero HTML de salida.
        chains (dict[str, str] | None): {etiqueta: cadena}. None → "A" para
            todas las etiquetas.
        hotspot_color (str): color de la cinta en las posiciones hotspot.
        show_labels (bool): si True (por defecto), añade el nombre de cada
            hotspot sobre la estructura de referencia.
 
    Returns:
        tuple[Path, py3Dmol.view]: Ruta al HTML y el visor (usar como última
            expresión de la celda de Jupyter).
    """
    aligned_models = _aligned_pdb_texts(pdb_paths_dict, chains=chains)
    chains = chains or {}
 
    if hotspot_data is None:
        hotspot_data = pd.DataFrame(columns=["gene", "uniprot_position"])
    hotspot_data = _normalize_hotspot_data(hotspot_data)
 
    view = py3Dmol.view(width=920, height=620)
 
    # 1) Carga todos los modelos primero
    for pdb_text in aligned_models.values():
        view.addModel(pdb_text, "pdb")
 
    # 2) Limpia de golpe el estilo de líneas por defecto de TODOS los modelos
    view.setStyle({}, {})
 
    # 3) Aplica el estilo visible modelo a modelo
    legend_entries = []
    for model_index, gene in enumerate(aligned_models.keys()):
        gene_chain = chains.get(gene, "A")
        gene_color = GENE_COLORS.get(gene, "lightgray")
        legend_entries.append((gene, gene_color))
 
        view.setStyle(
            {"model": model_index, "chain": gene_chain},
            {"cartoon": {"color": gene_color, "opacity": opacity}},
        )
        view.addStyle(
            {"model": model_index, "chain": gene_chain, "resn": LIGAND_RESNAMES},
            {"stick": {"colorscheme": "greenCarbon"}},
        )
 
        gene_hotspots = (
            hotspot_data.loc[hotspot_data["gene"].eq(gene)]
            if "gene" in hotspot_data.columns
            else hotspot_data.iloc[0:0]
        )
        for row in gene_hotspots.itertuples():
            view.addStyle(
                {"model": model_index, "chain": gene_chain, "resi": int(row.uniprot_position)},
                {"cartoon": {"color": hotspot_color}},
            )
 
    if not hotspot_data.empty:
        legend_entries.append(("Hotspot", hotspot_color))
    _add_legend(view, legend_entries)
 
    if show_labels and not hotspot_data.empty:
        reference_gene = next(iter(aligned_models))
        reference_chain = chains.get(reference_gene, "A")
        _add_position_labels(
            view, model_index=0, chain=reference_chain,
            positions=hotspot_data["uniprot_position"].unique(),
            border_color=hotspot_color,
        )
 
    view.zoomTo()
 
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    view.write_html(str(output_html))
    return output_html, _OfflineView(view)


def compare_states_3d(gene, output_html, gtp_pdb=None, gdp_pdb=None, gtp_chain=None, gdp_chain=None, hotspot_data=None, gtp_color="#2166AC", gdp_color="#8C8C8C", hotspot_color="#BE61E6", show_labels=True):

    """
    Superpone las conformaciones GTP (activa) y GDP (inactiva) de UN MISMO
    parálogo RAS sobre los átomos Cα del dominio G (residuos 1-166), para
    visualizar el desplazamiento conformacional de Switch I/II asociado al
    ciclo de activación/inactivación de la GTPasa.
 
    Si no se especifican gtp_pdb/gdp_pdb/gtp_chain/gdp_chain, se toman de
    STRUCTURE_REGISTRY[gene], que ya contiene tus pares validados.
 
    El emparejamiento de átomos Cα para la superposición se hace por número
    de residuo (ver _aligned_pdb_texts), por lo que la comparación es robusta
    aunque una de las dos estructuras tenga residuos ausentes (p. ej. 5UK9
    carece de backbone en 62-64).
 
    Args:
        gene (str): 'KRAS', 'HRAS' o 'NRAS'.
        output_html (str | Path): ruta del fichero HTML de salida.
        gtp_pdb, gdp_pdb (str | Path | None): id PDB o ruta directa. None →
            se toma de STRUCTURE_REGISTRY[gene].
        gtp_chain, gdp_chain (str | None): cadena de cada estructura. None →
            se toma de STRUCTURE_REGISTRY[gene].
        hotspot_data (pd.DataFrame | None): hotspots a marcar (columnas
            'gene' y 'uniprot_position'); se filtra automáticamente por
            `gene` si la columna existe. Opcional.
        gtp_color, gdp_color (str): colores de cinta para cada estado.
 
    Returns:
        tuple[Path, py3Dmol.view]: ruta al HTML y el visor (usar como última
            expresión de la celda de Jupyter para verlo inline con zoom/rotación).
    """
    registry_entry = STRUCTURE_REGISTRY.get(gene, {})
 
    gtp_pdb = gtp_pdb or registry_entry.get("GTP", {}).get("pdb_id")
    gdp_pdb = gdp_pdb or registry_entry.get("GDP", {}).get("pdb_id")
    gtp_chain = gtp_chain or registry_entry.get("GTP", {}).get("chain", "A")
    gdp_chain = gdp_chain or registry_entry.get("GDP", {}).get("chain", "A")
 
    if gtp_pdb is None or gdp_pdb is None:
        raise ValueError(
            f"No hay estructuras GTP/GDP para '{gene}' ni en los argumentos "
            f"ni en STRUCTURE_REGISTRY."
        )
 
    pdb_specs = {"GTP": gtp_pdb, "GDP": gdp_pdb}
    chains = {"GTP": gtp_chain, "GDP": gdp_chain}
    aligned = _aligned_pdb_texts(pdb_specs, chains=chains)
 
    state_colors = {"GTP": gtp_color, "GDP": gdp_color}
    state_opacity = {"GTP": 0.85, "GDP": 0.55}
    state_ligand_scheme = {"GTP": "greenCarbon", "GDP": "grayCarbon"}
    state_labels = {"GTP": "GTP (activa)", "GDP": "GDP (inactiva)"}
 
    view = py3Dmol.view(width=820, height=580)
 
    # 1) Carga ambos modelos
    for pdb_text in aligned.values():
        view.addModel(pdb_text, "pdb")
 
    # 2) Limpia el estilo de líneas por defecto de ambos modelos
    view.setStyle({}, {})
 
    if hotspot_data is not None:
        hotspot_data = _normalize_hotspot_data(hotspot_data)
        if "gene" in hotspot_data.columns:
            hotspot_data = hotspot_data.loc[hotspot_data["gene"].eq(gene)]
 
    # 3) Aplica el estilo visible modelo a modelo
    for model_index, state in enumerate(aligned.keys()):
        state_chain = chains[state]
 
        view.setStyle(
            {"model": model_index, "chain": state_chain},
            {"cartoon": {"color": state_colors[state], "opacity": state_opacity[state]}},
        )
        view.addStyle(
            {"model": model_index, "chain": state_chain, "resn": LIGAND_RESNAMES},
            {"stick": {"colorscheme": state_ligand_scheme[state]}},
        )
 
        if hotspot_data is not None:
            for row in hotspot_data.itertuples():
                view.addStyle(
                    {"model": model_index, "chain": state_chain, "resi": int(row.uniprot_position)},
                    {"cartoon": {"color": hotspot_color}},
                )
 
    legend_entries = [(state_labels[s], state_colors[s]) for s in aligned.keys()]
    if hotspot_data is not None and not hotspot_data.empty:
        legend_entries.append(("Hotspot", hotspot_color))
    _add_legend(view, legend_entries)
 
    if show_labels and hotspot_data is not None and not hotspot_data.empty:
        reference_state = next(iter(aligned))  # "GTP", primero en pdb_specs
        _add_position_labels(
            view, model_index=0, chain=chains[reference_state],
            positions=hotspot_data["uniprot_position"].unique(),
            border_color=hotspot_color,
        )
 
    view.zoomTo()
 
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    view.write_html(str(output_html))
    return output_html, _OfflineView(view)
 


def compare_isoforms_3d(state, output_html, hotspot_data=None, genes=("KRAS", "HRAS", "NRAS"), pdb_paths=None, chains=None, hotspot_color="#BE61E6", show_labels=True):
    
    """
    Superpone la MISMA conformación (GTP o GDP) de los tres parálogos RAS.
 
    Es el complemento de compare_states_3d: esa función fija el gen y compara
    sus dos estados; esta función fija el estado y compara los tres genes
    entre sí (p. ej. las tres estructuras GTP superpuestas, y por separado
    las tres GDP), reutilizando la superposición estructural ya implementada
    en comparative_3d_view.
 
    Si no se especifican pdb_paths/chains, se construyen desde
    STRUCTURE_REGISTRY para el `state` indicado.
 
    Args:
        state (str): 'GTP' o 'GDP'.
        output_html (str | Path): ruta del fichero HTML de salida.
        hotspot_data (pd.DataFrame | None): hotspots a marcar (columnas
            'gene' y 'uniprot_position'). Opcional.
        genes (tuple[str]): genes a incluir; el primero es la referencia fija
            de la superposición.
        pdb_paths (dict[str, str|Path] | None): {gen: id_pdb o ruta}. None →
            se construye desde STRUCTURE_REGISTRY[gen][state].
        chains (dict[str, str] | None): {gen: cadena}. None → se construye
            desde STRUCTURE_REGISTRY[gen][state].
 
    Returns:
        tuple[Path, py3Dmol.view]: ruta al HTML y el visor (usar como última
            expresión de la celda de Jupyter).
    """
    if pdb_paths is None:
        pdb_paths = {gene: STRUCTURE_REGISTRY[gene][state]["pdb_id"] for gene in genes}
    if chains is None:
        chains = {gene: STRUCTURE_REGISTRY[gene][state].get("chain", "A") for gene in genes}
 
    return comparative_3d_view(
        pdb_paths, hotspot_data, output_html, chains=chains,
        hotspot_color=hotspot_color, show_labels=show_labels, opacity=0.88
    )


def verify_structure_registry(registry=None, genes=("KRAS", "HRAS", "NRAS"),
                               states=("GTP", "GDP")):
    """
    Comprueba que las estructuras/cadenas configuradas en STRUCTURE_REGISTRY
    son las que realmente esperas, ANTES de generar ninguna figura 3D.
 
    Por cada combinación gen/estado verifica:
      - Que el fichero PDB existe en data/external/pdb/ (o donde lo tengas).
      - Que la cadena indicada tiene residuos Cα en el dominio G (1-166) —
        un número muy bajo o cero suele indicar que la cadena está mal
        (p. ej. usar "A" en una estructura donde la cadena real es "B").
      - Que el ligando presente en esa cadena es coherente con el estado
        declarado: GTP espera GTP/GNP/GCP (análogos no hidrolizables
        incluidos); GDP espera GDP. Si no coincide, probablemente hay un
        PDB ID mal asignado en el registro (p. ej. un "GTP" que en
        realidad es la forma GDP-bound de esa estructura).
 
    Args:
        registry (dict | None): registro a comprobar. None → STRUCTURE_REGISTRY.
        genes (tuple[str]): genes a comprobar.
        states (tuple[str]): estados a comprobar.
 
    Returns:
        pd.DataFrame: una fila por combinación gen/estado, con columnas
            'gene', 'state', 'pdb_id', 'chain', 'file_found',
            'ca_domain_g' (nº de residuos Cα encontrados en 1-166),
            'ligands_in_chain', 'ligand_matches_state'.
    """
    registry = registry or STRUCTURE_REGISTRY
    parser = PDBParser(QUIET=True)
 
    expected_ligands = {
        "GTP": {"GTP", "GNP", "GCP"},  # incluye análogos no hidrolizables
        "GDP": {"GDP"},
    }
 
    rows = []
    for gene in genes:
        for state in states:
            entry = registry.get(gene, {}).get(state, {})
            pdb_id = entry.get("pdb_id")
            chain_id = entry.get("chain", "A")
 
            row = {
                "gene": gene, "state": state, "pdb_id": pdb_id, "chain": chain_id,
                "file_found": False, "ca_domain_g": 0,
                "ligands_in_chain": [], "ligand_matches_state": None,
            }
 
            if pdb_id is None:
                rows.append(row)
                continue
 
            pdb_path = _resolve_pdb_path(pdb_id)
            row["file_found"] = pdb_path.exists()
            if not row["file_found"]:
                rows.append(row)
                continue
 
            structure = parser.get_structure(pdb_id, str(pdb_path))
            model = next(structure.get_models())
 
            if chain_id not in model:
                rows.append(row)
                continue
 
            row["ca_domain_g"] = len(_ca_atoms(model, chain=chain_id))
 
            ligands_found = sorted({
                residue.resname
                for residue in model[chain_id]
                if residue.resname in LIGAND_RESNAMES
            })
            row["ligands_in_chain"] = ligands_found
            row["ligand_matches_state"] = bool(set(ligands_found) & expected_ligands.get(state, set()))
 
            rows.append(row)
 
    return pd.DataFrame(rows)

# ===========================================================================
# Funciones auxiliares internas (prefijo _ indica uso privado al módulo)
# ===========================================================================

def _save_png_svg(fig, output_base):
    """
    Guarda una figura matplotlib en formato PNG (300 dpi) y SVG (vectorial),
    crea los directorios necesarios y cierra la figura para liberar memoria.

    Los dos formatos son complementarios: PNG para visualización directa y para
    incluir en documentos Word/LaTeX, SVG para edición vectorial en Inkscape o
    Adobe Illustrator (líneas nítidas a cualquier escala, ideal para figuras de tesis).

    Args:
        fig (matplotlib.figure.Figure): Figura ya construida lista para guardar.
        output_base (Path): Ruta base sin extensión (p. ej. project_root/figures/01_heatmap).

    Returns:
        dict: Diccionario {'png': Path, 'svg': Path} con las rutas generadas.
    """
    png = output_base.with_suffix(".png")  # Añade extensión .png a la ruta base
    svg = output_base.with_suffix(".svg")  # Añade extensión .svg a la ruta base
    png.parent.mkdir(parents=True, exist_ok=True)  # Crea el directorio de salida recursivamente si no existe
    fig.savefig(png, dpi=300, bbox_inches="tight")  # Guarda PNG a 300 dpi; bbox_inches="tight" elimina márgenes en blanco
    fig.savefig(svg, bbox_inches="tight")            # Guarda SVG vectorial; también recorta márgenes
    plt.close(fig)                                   # Libera la figura de la memoria de matplotlib
    return {"png": png, "svg": svg}                  # Devuelve ambas rutas para uso posterior en el pipeline


def _output_base(output_basename):
    """
    Normaliza una cadena o Path de salida eliminando extensiones conocidas y
    convirtiéndola en ruta absoluta bajo project_root() si no es ya absoluta.

    Permite que los llamadores pasen tanto 'figures/01_heatmap' como
    'figures/01_heatmap.png' y obtengan el mismo resultado, evitando que
    _save_png_svg genere ficheros con doble extensión como '01_heatmap.png.png'.

    Args:
        output_basename (str | Path): Nombre base con o sin extensión, absoluto o relativo.

    Returns:
        Path: Ruta absoluta sin extensión lista para añadir .png/.svg/.html.
    """
    path = Path(output_basename)  # Convierte a objeto Path para usar la API pathlib

    # Si el path tiene una extensión de imagen o HTML conocida, la elimina para obtener la base limpia
    if path.suffix.lower() in {".png", ".svg", ".html"}:
        path = path.with_suffix("")  # Elimina la extensión (p. ej. .png → sin extensión)

    # Si la ruta no es absoluta, la ancla en la raíz del proyecto para garantizar coherencia
    if not path.is_absolute():
        path = project_root() / path  # Concatena con la raíz del proyecto (definida en config.py)

    return path  # Devuelve la ruta absoluta sin extensión


def _resolve_pdb_path(pdb_id):
    """
    Resuelve la ruta al fichero PDB a partir de un identificador o ruta directa.

    Si pdb_id es una ruta que existe en el sistema de ficheros, la devuelve directamente.
    En caso contrario, construye la ruta convencional del proyecto:
    <project_root>/data/external/pdb/<PDB_ID_EN_MAYÚSCULAS>.pdb

    Esto permite que las funciones de visualización acepten tanto rutas absolutas
    (útil en tests y notebooks) como identificadores PDB de cuatro letras (útil
    en el pipeline automatizado).

    Args:
        pdb_id (str | Path): Identificador PDB (p. ej. '4obe', '4OBE', '4OBE.pdb')
            o ruta directa al fichero.

    Returns:
        Path: Ruta al fichero PDB resuelto.
    """
    path = Path(str(pdb_id))  # Convierte a Path (str() cubre el caso en que sea ya un Path)
    if path.exists():         # Si la ruta ya existe en el sistema de ficheros, la usa directamente
        return path

    # Construye la ruta convencional: elimina sufijo .PDB si lo tiene, convierte a mayúsculas
    pdb_name = str(pdb_id).upper().removesuffix(".PDB")  # "4obe.pdb" → "4OBE", "4OBE" → "4OBE"
    return project_root() / "data" / "external" / "pdb" / f"{pdb_name}.pdb"  # Ruta canónica del proyecto


def _normalize_hotspot_data(hotspot_data):
    """
    Estandariza un DataFrame o lista de dicts de hotspots asegurando que las
    columnas esperadas ('sample_count' y 'uniprot_position') estén presentes.

    Distintas partes del pipeline pueden generar hotspot_data con nombres de columna
    ligeramente distintos ('total_samples' en lugar de 'sample_count', 'position'
    en lugar de 'uniprot_position'). Esta función crea alias para garantizar
    compatibilidad sin modificar los datos originales.

    Args:
        hotspot_data (pd.DataFrame | list[dict] | dict): Datos de hotspots en
            cualquier formato que pd.DataFrame() acepte.

    Returns:
        pd.DataFrame: Copia del DataFrame con columnas 'sample_count' y
            'uniprot_position' garantizadas.
    """
    data = pd.DataFrame(hotspot_data).copy()  # Convierte a DataFrame y copia para no modificar el original

    # Si no existe 'sample_count' pero sí 'total_samples', crea el alias
    # (permite usar esta función tanto con salidas del pipeline como con datos externos)
    if "sample_count" not in data.columns and "total_samples" in data.columns:
        data["sample_count"] = data["total_samples"]  # Alias: sample_count ← total_samples

    # Si no existe 'uniprot_position' pero sí 'position', crea el alias
    if "uniprot_position" not in data.columns and "position" in data.columns:
        data["uniprot_position"] = data["position"]  # Alias: uniprot_position ← position

    return data  # Devuelve el DataFrame normalizado con los alias añadidos


def _metric_color(value, max_value):
    """
    Convierte un valor numérico normalizado en un color HTML hexadecimal
    interpolando linealmente entre azul oscuro (valor bajo) y rojo (valor alto).

    La interpolación es manual (no usa matplotlib colormap) para generar colores
    HTML directamente compatibles con la API de py3Dmol, que acepta cadenas '#RRGGBB'.

    Esquema de color:
      - Canal rojo:  45 (valor=0, azul puro) → 230 (valor=máximo, rojo intenso)
      - Canal verde: fijo en 0x4c (76/255 ≈ 30%), aporta un tono cálido neutro
      - Canal azul:  180 (valor=0) → 45 (valor=máximo); inversamente proporcional al rojo

    El resultado es una rampa perceptual de azul cobalto (#2d4cb4-like para valores bajos)
    a rojo tomate (#e64c2d-like para valores altos), pasando por tonos violáceos intermedios.

    Args:
        value (float): Valor de la métrica para este residuo.
        max_value (float): Valor máximo observado en el dataset (para normalizar a [0, 1]).

    Returns:
        str: Color en formato '#RRGGBB' (hexadecimal de 6 dígitos).
    """
    # Calcula la fracción normalizada entre 0.0 (mínimo) y 1.0 (máximo).
    # min/max garantizan que nunca se sale del rango [0, 1] aunque value > max_value.
    fraction = min(max(value / max_value, 0.0), 1.0)

    # Interpolación lineal del canal rojo: 45 cuando fraction=0, 230 cuando fraction=1
    red = int(230 * fraction + 45 * (1 - fraction))

    # Interpolación lineal del canal azul: inversa al rojo (alto rojo = bajo azul)
    blue = int(45 * fraction + 180 * (1 - fraction))

    # Canal verde fijo en 0x4c (76); proporciona un tono intermedio constante
    return f"#{red:02x}4c{blue:02x}"  # Formato '#RRGGBB'; :02x garantiza dos dígitos hex siempre


def _aligned_pdb_texts(pdb_paths_dict, chains=None):
    """
    Lee estructuras PDB (de distintos parálogos, o de distintos estados de un
    mismo parálogo), las superpone sobre los átomos Cα del dominio G
    (residuos 1-166) usando el algoritmo de Kabsch, y devuelve el texto PDB
    de cada una ya transformada.
 
    El primer elemento de `pdb_paths_dict` (en orden de inserción) se usa
    como referencia fija; los demás se rotan/trasladan para superponerse a
    ella. El emparejamiento de átomos Cα se hace por número de residuo
    (intersección de residuos presentes en ambas estructuras, ordenados),
    no por posición en la lista — esto evita desplazamientos silenciosos
    cuando una estructura tiene huecos de backbone (p. ej. 5UK9 en 62-64).
 
    Se requieren al menos 10 residuos Cα comunes para considerar la
    superposición válida; si no se alcanza, se emite un warning y esa
    estructura se deja sin transformar (se conserva su orientación original).
 
    Args:
        pdb_paths_dict (dict[str, str | Path]): {etiqueta: id_pdb o ruta}.
            La etiqueta puede ser un nombre de gen ('KRAS') o un estado
            ('GTP'), según el caso de uso.
        chains (dict[str, str] | None): {etiqueta: cadena}. None o etiqueta
            ausente → cadena "A" por defecto.
 
    Returns:
        dict[str, str]: {etiqueta: texto_pdb} con el contenido PDB de cada
            estructura en las coordenadas ya alineadas.
    """
    chains = chains or {}
    parser = PDBParser(QUIET=True)
 
    structures = {
        label: parser.get_structure(label, str(_resolve_pdb_path(path)))
        for label, path in pdb_paths_dict.items()
    }
 
    reference_label = next(iter(structures))
    reference_model = next(structures[reference_label].get_models())
    reference_chain = chains.get(reference_label, "A")
    reference_atoms = _ca_atoms(reference_model, chain=reference_chain)
 
    aligned = {}
 
    for label, structure in structures.items():
        model = next(structure.get_models())
        label_chain = chains.get(label, "A")
 
        if label != reference_label:
            mobile_atoms = _ca_atoms(model, chain=label_chain)
 
            # Solo los residuos presentes en AMBAS estructuras, ordenados
            # por número de residuo (no por posición en la lista).
            common_residues = sorted(set(reference_atoms) & set(mobile_atoms))
 
            if len(common_residues) >= 10:
                ref_subset = [reference_atoms[r] for r in common_residues]
                mob_subset = [mobile_atoms[r] for r in common_residues]
 
                superimposer = Superimposer()
                superimposer.set_atoms(ref_subset, mob_subset)
                superimposer.apply(list(model.get_atoms()))
            else:
                import warnings
                warnings.warn(
                    f"Superposición de '{label}' (cadena {label_chain}) sobre "
                    f"'{reference_label}' (cadena {reference_chain}) omitida: "
                    f"solo {len(common_residues)} residuos Cα comunes en el "
                    f"dominio G (<10). Revisa las cadenas seleccionadas."
                )
 
        handle = StringIO()
        io = PDBIO()
        io.set_structure(model)
        io.save(handle)
 
        aligned[label] = handle.getvalue()
 
    return aligned


def _ca_atoms(model, chain="A"):
    """
    Extrae los átomos Cα de la cadena indicada de un modelo PDB, limitados a
    los residuos 1-166 del dominio G catalítico de las proteínas RAS,
    indexados por número de residuo.
 
    Se devuelve un diccionario {num_residuo: átomo} en lugar de una lista
    para poder emparejar átomos equivalentes ENTRE DOS ESTRUCTURAS POR
    NÚMERO DE RESIDUO, no por posición en la lista. Esto es crítico cuando
    una de las estructuras tiene residuos ausentes en el fichero PDB (p. ej.
    5UK9 carece de backbone en 62-64): emparejar por posición desplazaría
    todos los residuos posteriores al hueco y produciría una superposición
    silenciosamente incorrecta.
 
    Args:
        model (Bio.PDB.Model.Model): Modelo de una estructura Bio.PDB.
        chain (str): Identificador de cadena a usar (p. ej. "A" o "B"; KRAS-
            GTP/5UK9 requiere "B").
 
    Returns:
        dict[int, Bio.PDB.Atom.Atom]: {número_residuo: átomo Cα}, restringido
            al dominio G (residuos 1-166).
    """
    atoms = {}
    chain_obj = model[chain]
 
    for residue in chain_obj:
        if residue.id[0] == " " and 1 <= residue.id[1] <= 166 and "CA" in residue:
            atoms[residue.id[1]] = residue["CA"]
 
    return atoms


def _add_legend(view, entries, x_offset=20, y_start=20, y_step=24, font_size=13):
    """
    Añade una leyenda de color fija en la esquina superior izquierda del
    visor 3D. A diferencia de una etiqueta normal de 3Dmol.js (que está
    anclada a una posición 3D y se mueve/rota con la molécula), esta usa
    `useScreen=True` para quedar fija en coordenadas de PANTALLA: no cambia
    de sitio aunque rotes, hagas zoom o desplaces la vista. Es lo que
    permite saber de un vistazo qué color corresponde a qué estructura
    (p. ej. GTP vs GDP, o KRAS/HRAS/NRAS).
 
    Args:
        view (py3Dmol.view): visor sobre el que añadir la leyenda.
        entries (list[tuple[str, str]]): pares (etiqueta, color_hex), en el
            orden en que se apilan verticalmente.
        x_offset, y_start, y_step (int): posición y espaciado en píxeles.
        font_size (int): tamaño de fuente de la leyenda.
    """
    for i, (label, color) in enumerate(entries):
        view.addLabel(
            label,
            {
                "position": {"x": 0, "y": 0, "z": 0},
                "useScreen": True,
                "screenOffset": {"x": x_offset, "y": y_start + i * y_step},
                "fontColor": "white",
                "backgroundColor": color,
                "backgroundOpacity": 0.85,
                "fontSize": font_size,
                "borderThickness": 0,
                "inFront": True,
            },
        )


def _add_position_labels(view, model_index, chain, positions, label_color="black", label_bg="white", border_color="#333333", font_size=11):
    """
    Añade una etiqueta de texto con el nombre de cada hotspot (p. ej. "G12",
    "Q61", tomado de HOTSPOTS) ANCLADA A LA POSICIÓN 3D REAL de ese residuo.
 
    A diferencia de _add_legend (fija en pantalla, no se mueve), estas
    etiquetas SÍ giran y se desplazan junto con la molécula al rotar o
    hacer zoom, porque en vez de darle una coordenada de pantalla se le da
    una selección de átomos (chain + resi) y 3Dmol.js calcula la posición
    a partir de ahí.
 
    Se añade una única etiqueta por posición (no una por modelo/estructura),
    anclada al modelo de referencia (`model_index`), para no duplicar el
    mismo texto varias veces cuando hay varias estructuras superpuestas casi
    en el mismo sitio.
 
    Args:
        view (py3Dmol.view): visor sobre el que añadir las etiquetas.
        model_index (int): índice del modelo al que anclar las etiquetas
            (normalmente el primero/de referencia de la superposición).
        chain (str): cadena de ese modelo.
        positions (Iterable[int]): posiciones UniProt de los hotspots.
        label_color, label_bg, border_color (str): estilo del texto.
        font_size (int): tamaño de fuente.
    """
    for position in sorted({int(p) for p in positions}):
        label_text = HOTSPOTS.get(position, str(position))
        view.addLabel(
            label_text,
            {
                "fontColor": label_color,
                "backgroundColor": label_bg,
                "backgroundOpacity": 0.85,
                "borderColor": border_color,
                "borderThickness": 1.0,
                "fontSize": font_size,
                "inFront": True,
                "screenOffset": {"x": 0, "y": -16},
            },
            {"model": model_index, "chain": chain, "resi": position},
        )