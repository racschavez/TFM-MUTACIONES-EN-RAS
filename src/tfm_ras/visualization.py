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

            ax.axvline(0.2, color="#555555", linestyle="--", linewidth=0.8)
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


def view_3d_with_hotspots(pdb_id, hotspot_data, output_html, color_by="sample_count"):
    """
    Genera una vista 3D interactiva de la estructura de un parálogo RAS con los
    hotspots mutacionales coloreados cuantitativamente, exportada como HTML.

    **py3Dmol** es una librería Python que genera código JavaScript para el visor
    molecular 3Dmol.js (WebGL). La función write_html() produce un fichero HTML
    auto-contenido con todo el JavaScript incrustado: se abre directamente en cualquier
    navegador moderno sin instalar nada, lo que lo hace ideal para incluir como anexo
    interactivo en la memoria digital del TFM o compartir con el tribunal.

    El protocolo de visualización es:
      1. La cadena principal (backbone) se muestra en representación de cinta
         (cartoon) gris claro para no distraer del detalle de los hotspots.
      2. Los ligandos (GDP/GTP y Mg) se muestran como varillas (stick) con esquema
         de color greenCarbon para distinguirlos del esqueleto proteico.
      3. Cada residuo hotspot se representa como una esfera cuyo color varía de
         azul (valor bajo) a rojo (valor alto) mediante interpolación lineal en RGB
         (función _metric_color), y cuyo radio también aumenta con el valor para
         un doble código visual (color + tamaño).

    Args:
        pdb_id (str | Path): Identificador PDB (p. ej. '4OBE') o ruta directa al
            fichero .pdb. Si es un identificador se busca en data/external/pdb/.
        hotspot_data (pd.DataFrame | list[dict]): Datos de los hotspots con columnas
            'uniprot_position' (o 'position') y la métrica indicada en color_by
            (o 'total_samples' como alias de 'sample_count').
        output_html (str | Path): Ruta del fichero HTML de salida.
        color_by (str): Columna de hotspot_data que determina el color/tamaño de las
            esferas. Por defecto 'sample_count' (número de muestras oncogénicas).

    Returns:
        tuple[Path, py3Dmol.view]: Ruta al fichero HTML generado y el objeto view
            para visualización inline en Jupyter (usar como última expresión de la celda).
    """
    pdb_path = _resolve_pdb_path(pdb_id)        # Resuelve la ruta al fichero PDB local
    pdb_text = pdb_path.read_text(encoding="utf-8")  # Lee el contenido del PDB como texto plano

    hotspot_data = _normalize_hotspot_data(hotspot_data)  # Estandariza nombres de columnas

    # Calcula el valor máximo de la métrica para normalizar colores entre 0 y 1.
    # Se fuerza un mínimo de 1.0 para evitar división por cero cuando hotspot_data está vacío.
    max_value = max(float(hotspot_data[color_by].max()), 1.0) if not hotspot_data.empty else 1.0

    view = py3Dmol.view(width=820, height=560)  # Inicializa el visor con dimensiones en píxeles

    view.addModel(pdb_text, "pdb")  # Carga el texto PDB en el visor (formato "pdb" activa el parser de 3Dmol.js)

    # Representación base: cinta gris claro para toda la proteína
    view.setStyle({"cartoon": {"color": "lightgray"}})

    # Sobrescribe el estilo de los ligandos con varillas de carbono verde para destacarlos
    view.addStyle({"resn": LIGAND_RESNAMES}, {"stick": {"colorscheme": "greenCarbon"}})

    # Itera sobre cada hotspot y le asigna una esfera con color y radio proporcionales al valor
    for row in hotspot_data.itertuples():
        value = float(getattr(row, color_by))  # Valor de la métrica para este hotspot (float)
        view.addStyle(
            {"chain": "A", "resi": int(row.uniprot_position)},  # Selector: cadena A, número de residuo
            {
                "sphere": {
                    "color": _metric_color(value, max_value),                    # Color azul→rojo según valor
                    "radius": 0.75 + min(value / max_value, 1.0) * 0.55,         # Radio: 0.75 Å (mín) a 1.30 Å (máx)
                }
            },
        )

    view.zoomTo()  # Centra y ajusta el zoom para encuadrar toda la proteína

    output_html = Path(output_html)                      # Convierte a objeto Path
    output_html.parent.mkdir(parents=True, exist_ok=True)  # Crea el directorio de salida si no existe
    view.write_html(str(output_html))  # Exporta el visor como HTML auto-contenido (JavaScript incrustado)
    return output_html, _OfflineView(view)  # Devuelve la ruta del HTML y un wrapper offline para visualización inline


def comparative_3d_view(pdb_paths_dict, hotspot_data, output_html):
    """
    Genera una vista 3D comparativa de los tres parálogos RAS superpuestos
    estructuralmente en un único visor WebGL exportado como HTML.

    **Superposición estructural (structural alignment):** Los tres parálogos comparten
    el dominio G (residuos 1–166) con >85% de identidad de secuencia. La función
    _aligned_pdb_texts() superpone HRAS y NRAS sobre KRAS como referencia minimizando
    el RMSD de los átomos Cα de ese dominio catalítico conservado. Esto permite
    comparar visualmente las diferencias conformacionales en los loops Switch I y II,
    que son los determinantes moleculares de la especificidad de señalización de cada
    parálogo.

    Cada gen se colorea con su color canónico (GENE_COLORS) en transparencia moderada
    (opacity=0.72) para que las tres cintas puedan verse simultáneamente. Los hotspots
    de cada gen se marcan con esferas del mismo color que su cinta.

    El visor HTML resultante es adecuado para incluir como **Figura Suplementaria
    Interactiva** en el repositorio del TFM o como enlace desde la memoria digital.

    Args:
        pdb_paths_dict (dict[str, str | Path]): Diccionario {gen: ruta_pdb},
            p. ej. {"KRAS": "data/external/pdb/4OBE.pdb", "HRAS": ..., "NRAS": ...}.
            El primer gen en orden de inserción se usará como referencia de superposición.
        hotspot_data (pd.DataFrame | list[dict]): Datos de hotspots con columnas
            'gene' y 'uniprot_position' (o 'position').
        output_html (str | Path): Ruta del fichero HTML de salida.

    Returns:
        tuple[Path, py3Dmol.view]: Ruta al fichero HTML generado y el objeto view
            para visualización inline en Jupyter (usar como última expresión de la celda).
    """
    # Superpone las tres estructuras y devuelve sus textos PDB alineados
    aligned_models = _aligned_pdb_texts(pdb_paths_dict)

    hotspot_data = _normalize_hotspot_data(hotspot_data)  # Estandariza nombres de columnas

    view = py3Dmol.view(width=920, height=620)  # Visor más grande para acomodar tres estructuras

    # Añade cada estructura como un modelo independiente en el visor
    # py3Dmol indexa los modelos por orden de inserción (0, 1, 2, ...)
    for model_index, (gene, pdb_text) in enumerate(aligned_models.items()):
        view.addModel(pdb_text, "pdb")  # Añade el texto PDB del gen actual al visor

        # Representación en cinta con el color canónico del gen y ligera transparencia
        view.setStyle(
            {"model": model_index},  # Selector: solo afecta a este modelo
            {"cartoon": {"color": GENE_COLORS.get(gene, "lightgray"), "opacity": 0.72}},  # Cinta semitransparente
        )

        # Ligandos en varillas de carbono verde (igual que en view_3d_with_hotspots)
        view.addStyle(
            {"model": model_index, "resn": LIGAND_RESNAMES},  # Selector: ligandos de este modelo
            {"stick": {"colorscheme": "greenCarbon"}},        # Varillas con esquema de carbono verde
        )

        # Marca los hotspots de este gen con esferas del color canónico del gen
        gene_hotspots = hotspot_data.loc[hotspot_data["gene"].eq(gene)]  # Filtra hotspots de este gen
        for row in gene_hotspots.itertuples():  # Itera sobre los hotspots del gen actual
            view.addStyle(
                {"model": model_index, "chain": "A", "resi": int(row.uniprot_position)},  # Selector de residuo
                {"sphere": {"color": GENE_COLORS.get(gene, "red"), "radius": 0.85}},      # Esfera de radio uniforme
            )

    view.zoomTo()  # Centra el zoom sobre todos los modelos cargados

    output_html = Path(output_html)                        # Convierte a objeto Path
    output_html.parent.mkdir(parents=True, exist_ok=True)  # Crea el directorio si no existe
    view.write_html(str(output_html))  # Exporta el visor comparativo como HTML auto-contenido
    return output_html, _OfflineView(view)  # Devuelve la ruta del HTML y un wrapper offline para visualización inline


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


def _aligned_pdb_texts(pdb_paths_dict):
    """
    Lee las estructuras PDB de los tres parálogos RAS, las superpone usando el
    algoritmo de mínimos cuadrados sobre átomos Cα del dominio G (residuos 1–166),
    y devuelve el texto PDB de cada estructura ya transformada.

    **Bio.PDB.Superimposer** implementa el algoritmo de Kabsch (1976) para encontrar
    la rotación y traslación óptimas que minimizan el RMSD entre dos conjuntos de
    átomos equivalentes. Se usa el primer gen del diccionario como estructura de
    referencia fija; los demás se transforman (rotan y trasladan) para maximizar la
    superposición.

    **Limitación a residuos 1–166:** El dominio G catalítico compartido por KRAS,
    HRAS y NRAS comprende los residuos 1–166. Los residuos 167–189 corresponden al
    hipervariable C-terminal (HVR), que difiere mucho entre parálogos y no debe
    incluirse en la superposición para evitar sesgar la alineación del núcleo
    conservado.

    Se requieren al menos 10 átomos Cα comunes para que la superposición sea
    estadísticamente válida (n_atoms >= 10).

    **StringIO + PDBIO:**
    PDBIO es la clase de Biopython para escribir estructuras PDB. Normalmente escribe
    a disco, pero aquí se usa un buffer StringIO (fichero de texto en memoria) para
    obtener el texto PDB sin crear ficheros temporales. Esto evita contaminación del
    sistema de ficheros y es más rápido para estructuras en memoria.

    Args:
        pdb_paths_dict (dict[str, str | Path]): Diccionario {gen: ruta_pdb} con las
            rutas a los ficheros PDB de los parálogos. El primer gen en orden de
            inserción será la referencia de superposición.

    Returns:
        dict[str, str]: Diccionario {gen: texto_pdb} con el contenido PDB de cada
            estructura en las coordenadas alineadas.
    """
    parser = PDBParser(QUIET=True)  # Inicializa el parser de PDB; QUIET=True suprime advertencias menores

    # Lee todas las estructuras PDB y las almacena en un diccionario {gen: Structure}
    structures = {
        gene: parser.get_structure(gene, str(path))  # str(path) por compatibilidad con versiones antiguas de Biopython
        for gene, path in pdb_paths_dict.items()
    }

    reference_gene = next(iter(structures))  # Primer gen del diccionario → referencia fija de superposición

    # Obtiene el primer modelo de la estructura de referencia (PDB puede tener múltiples modelos NMR)
    reference_model = next(structures[reference_gene].get_models())

    # Extrae los átomos Cα del dominio G (residuos 1–166, cadena A) de la referencia
    reference_atoms = _ca_atoms(reference_model)

    aligned = {}  # Diccionario de salida: {gen: texto_pdb_alineado}

    for gene, structure in structures.items():
        model = next(structure.get_models())  # Primer modelo de la estructura actual

        if gene != reference_gene:  # La referencia no se transforma; solo los demás genes
            mobile_atoms = _ca_atoms(model)  # Átomos Cα del dominio G de la estructura móvil

            # Usa el mínimo de átomos disponibles entre referencia y móvil para el par equivalente
            n_atoms = min(len(reference_atoms), len(mobile_atoms))

            if n_atoms >= 10:  # Mínimo estadístico: al menos 10 átomos para una superposición válida
                superimposer = Superimposer()  # Inicializa el superimposor (algoritmo de Kabsch)
                # set_atoms define los pares de átomos equivalentes: referencia[:n] ↔ móvil[:n]
                superimposer.set_atoms(reference_atoms[:n_atoms], mobile_atoms[:n_atoms])
                # apply aplica la rotación/traslación calculada a TODOS los átomos del modelo móvil
                superimposer.apply(list(model.get_atoms()))

        # Serializa la estructura transformada a texto PDB usando un buffer en memoria.
        # StringIO actúa como un fichero de texto virtual (sin E/S en disco).
        handle = StringIO()
        io = PDBIO()             # Instancia el escritor de PDB de Biopython
        io.set_structure(model)  # Asocia el modelo (ya transformado) al escritor
        io.save(handle)          # Escribe el PDB en el buffer de texto en memoria

        aligned[gene] = handle.getvalue()  # Extrae el texto PDB completo del buffer

    return aligned  # Devuelve {gen: texto_pdb} para todos los parálogos en coordenadas alineadas


def _ca_atoms(model):
    """
    Extrae los átomos Cα (carbono alfa) de la cadena A de un modelo PDB,
    limitados a los residuos 1–166 del dominio G catalítico de las proteínas RAS.

    Solo se incluyen residuos HETATM=espacio en blanco (residuos estándar, no ligandos
    ni moléculas de agua) que contengan el átomo 'CA'.

    El límite 1–166 corresponde al dominio G (GTPase domain) que es estructuralmente
    conservado entre KRAS, HRAS y NRAS (>85% identidad). Los residuos 167+ pertenecen
    al linker y a la región hipervariable C-terminal, que tiene longitud y secuencia
    muy distintas entre parálogos y no debe usarse para la superposición.

    Args:
        model (Bio.PDB.Model.Model): Modelo de una estructura Bio.PDB con al menos
            la cadena 'A' presente.

    Returns:
        list[Bio.PDB.Atom.Atom]: Lista de átomos Cα en orden de numeración secuencial,
            filtrados al dominio G (residuos 1–166).
    """
    atoms = []            # Lista acumuladora de átomos Cα válidos
    chain = model["A"]    # Selecciona la cadena A (cadena principal en los PDB de RAS)

    for residue in chain:  # Itera sobre todos los residuos de la cadena A
        # residue.id es una tupla (hetfield, seqnum, icode).
        # hetfield == " " (espacio) indica residuo de aminoácido estándar
        # (los ligandos tienen hetfield == "H_XXX" y las aguas == "W").
        # seqnum entre 1 y 166 restringe al dominio G conservado.
        # "CA" in residue verifica que el residuo tiene átomo Cα (algunos residuos
        # en extremos de cadena o con desorden pueden carecer de él).
        if residue.id[0] == " " and 1 <= residue.id[1] <= 166 and "CA" in residue:
            atoms.append(residue["CA"])  # Añade el átomo Cα del residuo a la lista

    return atoms  # Devuelve la lista de Cα del dominio G para superposición
