"""
Módulo de alineamiento de secuencias y mapeo de posiciones para la familia RAS.

Contexto biológico
------------------
Las proteínas KRAS, HRAS y NRAS pertenecen a la familia RAS de GTPasas pequeñas.
Las tres isoformas comparten una elevada identidad de secuencia (~85-90 %) en la
región N-terminal catalítica (dominios G1-G5), pero difieren en la región
hipervariable C-terminal que determina su localización subcelular. Al ser productos
de genes parálogos, acumulan diferencias puntuales que pueden alterar la actividad
intrínseca de GTPasa o la interacción con efectores.

Un alineamiento múltiple de secuencias (MSA, Multiple Sequence Alignment) dispone
las tres secuencias en columnas de manera que cada columna agrupa residuos que
comparten posición evolutiva u ocupan el mismo lugar en la estructura tridimensional.
El MSA es el punto de partida para:
  1. Construir un mapa de equivalencias posición-a-posición, de modo que "KRAS G12"
     se corresponde con "HRAS G12" y "NRAS G12" aunque los índices en la secuencia
     original puedan diferir si alguna isoforma tiene inserciones o deleciones.
  2. Identificar posiciones recurrentemente mutadas en las tres isoformas, lo que
     sugiere restricción funcional y es relevante para el diseño de inhibidores
     pan-RAS.

Este módulo implementa:
  - Descarga y caché de secuencias desde la base de datos UniProt.
  - Ejecución de herramientas externas de MSA (ClustalΩ, MAFFT, MUSCLE).
  - Construcción del mapa posicional KRAS-HRAS-NRAS.
  - Identificación de posiciones recurrentes con mutaciones oncogénicas.
  - Visualización del alineamiento con anotación de regiones funcionales.
  - Funciones auxiliares de análisis (fracción de identidad, secuencia consenso).
"""

from __future__ import annotations  # Permite usar anotaciones de tipo diferidas (PEP 563); necesario para compatibilidad con Python < 3.10

import shutil          # Utilidades de sistema de ficheros (e.g. buscar ejecutables en el PATH)
import subprocess      # Lanzar procesos externos del sistema operativo (herramientas de MSA)
import tempfile        # Crear directorios y ficheros temporales seguros en disco
import warnings        # Emitir advertencias de ejecución sin interrumpir el flujo del programa

from collections import Counter  # Contador eficiente para contar ocurrencias de elementos (usado en consenso)
from pathlib import Path          # Rutas de ficheros orientadas a objetos, multiplataforma

import matplotlib.pyplot as plt   # Generación de figuras y gráficos científicos (visualización del MSA)
import numpy as np                # Operaciones numéricas sobre arrays; usado para matrices de colores y NaN
import pandas as pd               # Manipulación de datos tabulares (DataFrames para los mapas posicionales)
import requests                   # Peticiones HTTP para descargar secuencias desde UniProt

from Bio import AlignIO           # Lectura/escritura de ficheros de alineamiento en distintos formatos (BioPython)
from Bio.Align import MultipleSeqAlignment  # Clase que representa un alineamiento múltiple en BioPython
from Bio.Seq import Seq           # Clase que representa una secuencia biológica (ADN, ARN o proteína)
from Bio.SeqRecord import SeqRecord  # Secuencia anotada con identificador y metadatos

from tfm_ras.config import project_root  # Función que devuelve la ruta raíz del proyecto (para rutas absolutas portables)


# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

# Nombres canónicos de los tres genes de la familia RAS analizados en este TFM.
# Se usan como claves en diccionarios y para normalizar identificadores de secuencia.
GENES = ("KRAS", "HRAS", "NRAS")

# Nombres de las columnas del DataFrame que representa el mapa de equivalencias posicionales.
# Cada fila corresponde a una columna del MSA:
#   msa_position   → índice 1-based de la columna en el alineamiento
#   kras_position  → posición 1-based en la secuencia de KRAS (NaN si hay gap)
#   hras_position  → posición 1-based en la secuencia de HRAS (NaN si hay gap)
#   nras_position  → posición 1-based en la secuencia de NRAS (NaN si hay gap)
#   kras_aa        → aminoácido de KRAS en esa columna ('-' si es gap)
#   hras_aa        → aminoácido de HRAS en esa columna ('-' si es gap)
#   nras_aa        → aminoácido de NRAS en esa columna ('-' si es gap)
#   conserved      → True si el mismo aminoácido aparece en las tres isoformas sin gaps
POSITION_MAP_COLUMNS = [
    "msa_position", "kras_position", "hras_position", "nras_position",
    "kras_aa", "hras_aa", "nras_aa", "conserved",
]

# Nombres de las columnas del DataFrame que reporta posiciones recurrentemente mutadas.
# Cada fila corresponde a una posición del MSA donde al menos min_members isoformas
# acumulan suficientes muestras mutadas (>= min_samples):
#   msa_position      → posición en el alineamiento
#   kras_pos          → posición en KRAS (entero o NaN)
#   hras_pos          → posición en HRAS (entero o NaN)
#   nras_pos          → posición en NRAS (entero o NaN)
#   n_members_mutated → número de isoformas que superan el umbral min_samples
#   total_samples     → suma de muestras mutadas en las tres isoformas
#   kras_samples      → número de muestras mutadas en KRAS
#   hras_samples      → número de muestras mutadas en HRAS
#   nras_samples      → número de muestras mutadas en NRAS
#   top_mutations     → cadena con la mutación más frecuente de cada isoforma (e.g. "KRAS:p.G12D|NRAS:p.Q61R")
RECURRENT_COLUMNS = [
    "msa_position", "kras_pos", "hras_pos", "nras_pos",
    "n_members_mutated", "total_samples",
    "kras_samples", "hras_samples", "nras_samples", "top_mutations",
]

# Regiones funcionales de KRAS4B (numeración estándar del dominio G).
# Los números corresponden a posiciones en la secuencia de KRAS (coordenadas 1-based).
# Estas regiones son esenciales para la actividad GTPasa y la interacción con efectores:
#   P-loop (motivo G1, residuos 10-17):  bucle de unión al fosfato del GTP/GDP.
#   Switch I (residuos 30-38):           cambia de conformación al hidrolizar GTP;
#                                        interactúa con efectores (RAF, PI3K, RalGDS).
#   Switch II (residuos 60-76):          segundo bucle conformacionalmente activo;
#                                        contiene Q61 crítico para la hidrólisis del GTP.
#   NKxD (residuos 116-119):             motivo de reconocimiento de la base de guanina.
#   SAK (residuos 145-147):              segundo motivo de especificidad de guanina.
FUNCTIONAL_REGIONS = {
    "P-loop": (10, 17),
    "Switch I": (30, 38),
    "Switch II": (60, 76),
    "NKxD": (116, 119),
    "SAK": (145, 147),
}

# Clasificación fisicoquímica de los aminoácidos en grupos funcionales.
# Esta agrupación se utiliza para colorear el alineamiento y detectar cambios
# de propiedades (e.g. una mutación hidrofóbica→cargada puede alterar la estructura).
# Los grupos reflejan las propiedades de la cadena lateral:
#   hydrophobic → apolares; forman el núcleo hidrofóbico de las proteínas
#   polar       → polares sin carga neta; participan en puentes de hidrógeno
#   positive    → cargados positivamente a pH fisiológico (básicos)
#   negative    → cargados negativamente a pH fisiológico (ácidos)
#   special     → Gly (G) y Pro (P) con propiedades conformacionales únicas
AA_GROUPS = {
    "hydrophobic": set("AVLIMFWY"),  # Ala, Val, Leu, Ile, Met, Phe, Trp, Tyr
    "polar": set("STNQC"),           # Ser, Thr, Asn, Gln, Cys
    "positive": set("KRH"),          # Lys, Arg, His
    "negative": set("DE"),           # Asp, Glu
    "special": set("GP"),            # Gly (máxima flexibilidad), Pro (restricción de φ)
}

# Colores asociados a cada grupo fisicoquímico para la visualización del MSA.
# Los colores siguen una paleta perceptualmente distinguible (estilo Vega/Altair):
#   hydrophobic → azul oscuro
#   polar       → verde azulado
#   positive    → naranja
#   negative    → rojo
#   special     → verde
#   gap         → gris claro (posiciones sin residuo)
AA_GROUP_COLORS = {
    "hydrophobic": "#4C78A8",
    "polar": "#72B7B2",
    "positive": "#F58518",
    "negative": "#E45756",
    "special": "#54A24B",
    "gap": "#D9D9D9",
}


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------

def fetch_uniprot_sequence(uniprot_id):
    """
    Descarga la secuencia proteica de un identificador UniProt y la almacena en caché local.

    UniProt (Universal Protein Resource, https://www.uniprot.org) es la base de datos
    de referencia para secuencias y anotaciones de proteínas. Cada entrada tiene un
    identificador único (accession), como P01116 (KRAS humano), P01112 (HRAS) o P01111 (NRAS).

    El formato FASTA es el estándar de facto para secuencias biológicas. Consiste en:
      - Una línea de cabecera que comienza con '>' seguida de un identificador y descripción.
      - Una o más líneas con la secuencia de aminoácidos (o nucleótidos) en código de una letra.

    La función implementa una estrategia de caché: si el fichero FASTA ya existe en disco
    con contenido no vacío, lo lee directamente sin realizar ninguna petición de red.
    Esto evita llamadas repetidas a la API durante el desarrollo del TFM.

    Parámetros
    ----------
    uniprot_id : str
        Identificador de acceso UniProt (e.g. 'P01116' para KRAS humano).

    Devuelve
    --------
    str
        Secuencia de aminoácidos en código de una letra, sin cabecera FASTA.

    Raises
    ------
    requests.HTTPError
        Si la petición HTTP falla (código 4xx o 5xx), indicando que el ID no existe
        o que el servidor de UniProt no está disponible.

    Ejemplos
    --------
    >>> seq = fetch_uniprot_sequence("P01116")
    >>> seq[:5]
    'MTEYK'
    """
    # Construir la ruta del directorio de caché relativa a la raíz del proyecto.
    # project_root() devuelve un objeto Path apuntando al directorio raíz del TFM.
    cache_dir = project_root() / "data" / "external" / "sequences"

    # Crear el directorio de caché y todos sus padres si no existen;
    # exist_ok=True evita error si el directorio ya existe.
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Ruta completa del fichero de caché para este UniProt ID.
    cache_path = cache_dir / f"{uniprot_id}.fasta"

    # Si el fichero de caché existe y no está vacío (st_size > 0), leerlo directamente.
    # Esto evita peticiones de red innecesarias.
    if cache_path.exists() and cache_path.stat().st_size > 0:
        # Leer el contenido del fichero de caché y extraer la secuencia sin la cabecera.
        return _sequence_from_fasta(cache_path.read_text(encoding="utf-8"))

    # Construir la URL de la API REST de UniProt para obtener el FASTA.
    # El endpoint /uniprotkb/{id}.fasta devuelve directamente el fichero FASTA.
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"

    # Realizar la petición HTTP GET con un timeout de 60 segundos.
    # timeout evita que el programa se quede bloqueado indefinidamente si la red falla.
    response = requests.get(url, timeout=60)

    # Lanzar una excepción si el servidor devuelve un código de error HTTP (4xx, 5xx).
    response.raise_for_status()

    # Guardar la respuesta en el fichero de caché para usos futuros.
    cache_path.write_text(response.text, encoding="utf-8")

    # Extraer y devolver la secuencia a partir del texto FASTA descargado.
    return _sequence_from_fasta(response.text)


def run_msa(sequences, algorithm="clustalo"):
    """
    Ejecuta un alineamiento múltiple de secuencias (MSA) con la herramienta indicada.

    Un MSA coloca múltiples secuencias en columnas tales que los residuos homólogos
    quedan alineados. Los algoritmos disponibles son:

    - ClustalΩ (clustalo): utiliza perfiles HMM y matrices de distancias para
      alinear progresivamente. Es robusto para proteínas con alta identidad
      (>30 %) y es el más habitual en estudios de familias de proteínas.
    - MAFFT: emplea transformadas de Fourier rápidas para encontrar regiones
      similares. Muy rápido en proteínas largas y con precisión comparable a
      ClustalΩ para familias bien conservadas.
    - MUSCLE (muscle): algoritmo iterativo que refina el alineamiento en varias
      rondas. Útil cuando se necesita máxima precisión a costa de mayor tiempo.

    En el caso particular de KRAS, HRAS y NRAS, si las tres secuencias tienen la
    misma longitud (lo que ocurre en la región catalítica), la función puede omitir
    la herramienta externa y construir directamente un alineamiento sin gaps, ya que
    las secuencias son prácticamente isomórficas en longitud.

    Parámetros
    ----------
    sequences : dict[str, str]
        Diccionario {nombre_gen → secuencia_aminoácidos}. Las claves deben incluir
        "KRAS", "HRAS" y "NRAS" (u otras si se amplia el análisis).
    algorithm : str, optional
        Herramienta de MSA a utilizar. Valores válidos: 'clustalo', 'mafft', 'muscle'.
        Por defecto 'clustalo'.

    Devuelve
    --------
    Bio.Align.MultipleSeqAlignment
        Objeto BioPython con el alineamiento múltiple resultante. Cada registro
        contiene la secuencia con gaps ('-') insertados para mantener el alineamiento.

    Raises
    ------
    ValueError
        Si el diccionario de secuencias está vacío o el algoritmo no es válido.
    RuntimeError
        Si el ejecutable del algoritmo no se encuentra en el PATH y las secuencias
        tienen longitudes distintas (imposible alinear sin herramienta externa),
        o si el proceso externo termina con código de error.
    """
    # Verificar que el diccionario de secuencias no está vacío.
    if not sequences:
        raise ValueError("At least one sequence is required for MSA.")

    # Normalizar el nombre del algoritmo a minúsculas para comparaciones robustas.
    algorithm = algorithm.lower()

    # Validar que el algoritmo solicitado es uno de los soportados.
    if algorithm not in {"clustalo", "mafft", "muscle"}:
        raise ValueError("algorithm must be one of: clustalo, mafft, muscle")

    # Calcular el conjunto de longitudes únicas entre todas las secuencias.
    # Si todas tienen la misma longitud, len(lengths) == 1.
    lengths = {len(sequence) for sequence in sequences.values()}

    # Caso especial: las tres secuencias RAS tienen la misma longitud.
    # En este caso, KRAS/HRAS/NRAS se pueden alinear sin herramienta externa,
    # ya que no hay necesidad de insertar gaps para emparejar posiciones homólogas.
    if len(lengths) == 1 and set(sequences) == set(GENES):
        warnings.warn(
            "Secuencias KRAS/HRAS/NRAS de igual longitud; usando alineamiento sin gaps.",
            stacklevel=2,  # stacklevel=2 apunta la advertencia al llamador, no a esta función
        )
        # Construir el alineamiento directamente sin gaps a partir de las secuencias.
        return _alignment_from_sequences(sequences)

    # shutil.which busca el ejecutable del algoritmo en el PATH del sistema,
    # de forma similar al comando `which` en bash. Devuelve None si no se encuentra.
    executable = shutil.which(algorithm)

    # Si el ejecutable no está disponible en el sistema:
    if executable is None:
        # Si todas las secuencias tienen la misma longitud, podemos omitir la herramienta
        # y construir el alineamiento sin gaps emitiendo solo una advertencia.
        if len(lengths) == 1:
            warnings.warn(f"{algorithm!r} no encontrado; usando alineamiento sin gaps.", stacklevel=2)
            return _alignment_from_sequences(sequences)
        # Si las longitudes son distintas, necesitamos la herramienta sí o sí.
        raise RuntimeError(f"Ejecutable MSA no encontrado: {algorithm}")

    # Crear un directorio temporal que se eliminará automáticamente al salir del bloque.
    # tempfile.TemporaryDirectory gestiona la creación y limpieza de forma segura,
    # evitando dejar residuos en disco aunque ocurra una excepción.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)  # Convertir la cadena del directorio temporal a Path

        # Rutas de los ficheros de entrada y salida para la herramienta de MSA.
        input_fasta = tmpdir_path / "input.fasta"
        output_fasta = tmpdir_path / "output.fasta"

        # Escribir las secuencias en formato FASTA en el fichero de entrada.
        _write_fasta(sequences, input_fasta)

        if algorithm == "clustalo":
            # Construir el comando de ClustalΩ:
            #   -i  → fichero de entrada (secuencias sin alinear)
            #   -o  → fichero de salida (secuencias alineadas)
            #   --force     → sobreescribir el fichero de salida si existe
            #   --outfmt fasta → formato de salida FASTA
            cmd = [executable, "-i", str(input_fasta), "-o", str(output_fasta), "--force", "--outfmt", "fasta"]
            # subprocess.run lanza el proceso externo y espera a que termine.
            # capture_output=True captura stdout y stderr en lugar de mostrarlos en pantalla.
            # check=False evita que Python lance una excepción automática si el proceso falla
            # (lo comprobaremos manualmente con result.returncode).
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        elif algorithm == "mafft":
            # MAFFT escribe el alineamiento en stdout en lugar de en un fichero de salida.
            # --auto selecciona automáticamente la estrategia de alineamiento más adecuada.
            cmd = [executable, "--auto", str(input_fasta)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                # Guardar el stdout de MAFFT en el fichero de salida para que AlignIO lo lea.
                output_fasta.write_text(result.stdout, encoding="utf-8")

        else:
            # Para MUSCLE, se usa una función auxiliar que soporta tanto la versión 5
            # (con --align/--output) como la versión 3 (con -in/-out).
            result = _run_muscle(executable, input_fasta, output_fasta)

        # Comprobar si el proceso externo terminó con error (código != 0).
        if result.returncode != 0:
            raise RuntimeError(f"{algorithm} falló con código {result.returncode}:\n{result.stderr}")

        # Leer el fichero FASTA alineado y devolver el objeto MultipleSeqAlignment de BioPython.
        return AlignIO.read(output_fasta, "fasta")


def build_position_map(msa):
    """
    Construye el mapa de equivalencias posición-a-posición a partir del MSA.

    El mapa posicional es una tabla donde cada fila corresponde a una columna del
    alineamiento múltiple. Para cada columna se registra:
      - La posición en el alineamiento (índice 1-based).
      - La posición en la secuencia de cada isoforma (o NaN si hay un gap en esa isoforma).
      - El aminoácido de cada isoforma en esa columna.
      - Si el residuo está conservado (mismo aminoácido en las tres isoformas sin gaps).

    Este mapa es la pieza clave para comparar numeración entre isoformas. Por ejemplo,
    la posición oncogénica G12 de KRAS corresponde a G12 en HRAS y NRAS, pero tras
    insertar gaps por diferencias en la región hipervariable, los índices absolutos
    en el alineamiento pueden diferir.

    La función también valida que los hotspots oncogénicos más importantes (G12, G13, Q61)
    estén correctamente alineados: si el alineamiento coloca un aminoácido distinto en
    alguna isoforma para estas posiciones, se lanza una excepción.

    Parámetros
    ----------
    msa : Bio.Align.MultipleSeqAlignment
        Alineamiento múltiple con al menos los registros de KRAS, HRAS y NRAS.
        Los identificadores de los registros deben contener el nombre del gen.

    Devuelve
    --------
    pd.DataFrame
        DataFrame con las columnas definidas en POSITION_MAP_COLUMNS.
        Tiene tantas filas como columnas tenga el alineamiento.

    Raises
    ------
    ValueError
        Si alguna de las tres isoformas no está presente en el alineamiento.
    AssertionError
        Si los hotspots G12, G13 o Q61 no están alineados correctamente.
    """
    # Obtener un diccionario {gen → SeqRecord} para acceder a cada secuencia por nombre.
    records = _records_by_gene(msa)

    # Verificar que el alineamiento contiene registros para los tres genes requeridos.
    missing = set(GENES) - set(records)
    if missing:
        raise ValueError(f"MSA no contiene registros para: {', '.join(sorted(missing))}")

    # Longitud total del alineamiento (número de columnas, incluyendo las posiciones con gaps).
    alignment_length = msa.get_alignment_length()

    # Contadores de posición para cada gen (inicializados a 0).
    # Se incrementan cada vez que se encuentra un residuo real (no gap) en esa isoforma.
    counters = dict.fromkeys(GENES, 0)

    # Lista donde se acumularán los diccionarios de cada fila del mapa posicional.
    rows = []

    # Iterar sobre cada columna del alineamiento (índice 0-based).
    for column_index in range(alignment_length):
        # Inicializar la fila con la posición MSA en formato 1-based.
        row = {"msa_position": column_index + 1}

        # Diccionario auxiliar para almacenar los aminoácidos de cada gen en esta columna.
        aas = {}

        for gene in GENES:
            # Obtener el aminoácido del gen en esta columna del alineamiento.
            # str(...)[column_index] accede al carácter en esa posición de la secuencia alineada.
            # .upper() normaliza a mayúsculas por si alguna herramienta devuelve minúsculas.
            aa = str(records[gene].seq[column_index]).upper()

            if aa != "-":
                # Si no es un gap, incrementar el contador de posición para este gen
                # y registrar la posición 1-based en la secuencia original.
                counters[gene] += 1
                position = counters[gene]
            else:
                # Si es un gap ('-'), la posición en la secuencia es indefinida.
                # NaN (Not a Number) de NumPy se usa para representar valores nulos en pandas.
                position = np.nan

            # Guardar el aminoácido y la posición para este gen en la fila actual.
            aas[gene] = aa
            row[f"{gene.lower()}_position"] = position  # e.g. "kras_position"
            row[f"{gene.lower()}_aa"] = aa              # e.g. "kras_aa"

        # Determinar si la columna está conservada:
        # se consideran solo los genes sin gap y se comprueba que todos tienen el mismo aminoácido.
        non_gap_aas = [aas[gene] for gene in GENES if aas[gene] != "-"]

        # Conservada si las tres isoformas tienen residuo (sin gap) y es el mismo en las tres.
        row["conserved"] = len(non_gap_aas) == 3 and len(set(non_gap_aas)) == 1

        # Añadir la fila completada a la lista de filas.
        rows.append(row)

    # Construir el DataFrame a partir de la lista de filas, con las columnas en orden definido.
    position_map = pd.DataFrame(rows, columns=POSITION_MAP_COLUMNS)

    # Validar que los hotspots oncogénicos clave (G12, G13, Q61) están correctamente alineados.
    _validate_hotspot_alignment(position_map)

    return position_map


def find_recurrent_positions(mutations_df, position_map, min_members=2, min_samples=10):
    """
    Identifica posiciones del MSA recurrentemente mutadas en múltiples isoformas RAS.

    Una posición se considera recurrente si, en la base de datos de mutaciones,
    aparece mutada en al menos ``min_members`` isoformas con un mínimo de
    ``min_samples`` muestras cada una. Este criterio biológico refleja la presión
    selectiva convergente: si tres genes distintos adquieren mutaciones en la misma
    posición estructural en tumores independientes, esa posición tiene relevancia
    funcional y probablemente forme parte del sitio activo o de una interfaz proteína-proteína.

    Las mutaciones oncogénicas más conocidas de RAS (G12, G13, Q61) son recurrentes
    en las tres isoformas porque afectan directamente a la actividad GTPasa intrínseca
    o a la interacción con la proteína GAP (GTPase-Activating Protein) que normalmente
    estimula la hidrólisis del GTP.

    Parámetros
    ----------
    mutations_df : pd.DataFrame
        DataFrame con columnas mínimas:
          - 'gene'         : nombre del gen ('KRAS', 'HRAS', 'NRAS'), insensible a mayúsculas.
          - 'position'     : posición numérica del aminoácido mutado (int).
          - 'sample_count' : número de muestras/tumores en que se observa esa mutación.
          - 'hgvs_p'       : notación HGVS de la mutación (e.g. 'p.G12D').
    position_map : pd.DataFrame
        Mapa posicional generado por :func:`build_position_map`.
    min_members : int, optional
        Número mínimo de isoformas que deben estar mutadas en esa posición MSA.
        Por defecto 2. Usar 3 para encontrar solo las posiciones pan-RAS.
    min_samples : int, optional
        Número mínimo de muestras en cada isoforma para considerar que está
        "recurrentemente mutada" en esa isoforma. Por defecto 10.

    Devuelve
    --------
    pd.DataFrame
        DataFrame con las columnas definidas en RECURRENT_COLUMNS, ordenado de mayor
        a menor número total de muestras mutadas. Puede estar vacío si ninguna posición
        cumple los criterios.
    """
    # Lista donde se irán acumulando las filas del resultado.
    rows = []

    # Iterar sobre cada fila del mapa posicional.
    # itertuples() es más eficiente que iterrows() para acceso por columnas nombradas,
    # devolviendo named tuples donde cada campo es accesible como atributo (row.kras_position).
    for map_row in position_map.itertuples(index=False):
        # Diccionario que almacenará el recuento de muestras mutadas por isoforma.
        gene_samples = {}

        # Lista de cadenas con la mutación más frecuente de cada isoforma (para la columna top_mutations).
        top_mutations = []

        for gene in GENES:
            gene_lower = gene.lower()  # Versión en minúsculas para acceder a columnas del mapa (e.g. "kras")

            # Obtener la posición de esta isoforma en la fila actual del mapa posicional.
            # getattr(object, name) es equivalente a object.name pero permite nombres dinámicos.
            position = getattr(map_row, f"{gene_lower}_position")

            samples = 0  # Por defecto, 0 muestras si la posición es un gap o no hay mutaciones.

            # Solo procesar si la posición no es NaN (i.e., no hay gap en esta isoforma).
            if pd.notna(position):
                position = int(position)  # Convertir de float a int (pandas almacena NaN como float)

                # Filtrar el DataFrame de mutaciones para este gen y esta posición.
                # .astype(str).str.upper().eq(gene) normaliza el nombre del gen antes de comparar.
                # .eq(position) selecciona solo las filas con esa posición concreta.
                subset = mutations_df.loc[
                    mutations_df["gene"].astype(str).str.upper().eq(gene)
                    & mutations_df["position"].eq(position)
                ]

                # Sumar el total de muestras si hay mutaciones en este gen/posición.
                samples = int(subset["sample_count"].sum()) if not subset.empty else 0

                # Si supera el umbral de muestras, registrar la mutación más frecuente.
                if samples >= min_samples:
                    # Ordenar por sample_count descendente y tomar la primera fila (la más frecuente).
                    top_row = subset.sort_values("sample_count", ascending=False).iloc[0]
                    # Formatear como "GEN:notación_HGVS" (e.g. "KRAS:p.G12D").
                    top_mutations.append(f"{gene}:{top_row.hgvs_p}")

            # Guardar el número de muestras para esta isoforma.
            gene_samples[gene_lower] = samples

        # Contar cuántas isoformas superan el umbral min_samples en esta posición.
        n_members_mutated = sum(samples >= min_samples for samples in gene_samples.values())

        # Solo incluir esta posición en el resultado si cumple el criterio de recurrencia.
        if n_members_mutated >= min_members:
            rows.append({
                "msa_position": int(map_row.msa_position),              # Posición en el alineamiento
                "kras_pos": _nullable_int(map_row.kras_position),        # Posición en KRAS (int o NaN)
                "hras_pos": _nullable_int(map_row.hras_position),        # Posición en HRAS (int o NaN)
                "nras_pos": _nullable_int(map_row.nras_position),        # Posición en NRAS (int o NaN)
                "n_members_mutated": int(n_members_mutated),             # Número de isoformas mutadas
                "total_samples": int(sum(gene_samples.values())),        # Suma total de muestras
                "kras_samples": gene_samples["kras"],                    # Muestras mutadas en KRAS
                "hras_samples": gene_samples["hras"],                    # Muestras mutadas en HRAS
                "nras_samples": gene_samples["nras"],                    # Muestras mutadas en NRAS
                "top_mutations": "|".join(top_mutations),                # Mutaciones más frecuentes separadas por '|'
            })

    # Construir el DataFrame con las columnas en el orden especificado.
    recurrent = pd.DataFrame(rows, columns=RECURRENT_COLUMNS)

    # Si no hay posiciones recurrentes, devolver el DataFrame vacío sin intentar ordenarlo.
    if recurrent.empty:
        return recurrent

    # Ordenar por total de muestras (descendente) y por posición MSA (ascendente)
    # para que los hotspots más frecuentes aparezcan primero.
    return recurrent.sort_values(["total_samples", "msa_position"], ascending=[False, True]).reset_index(drop=True)


def visualize_msa(msa, position_map, output_path):
    """
    Genera una imagen del alineamiento múltiple coloreado por grupos fisicoquímicos.

    La visualización muestra las tres secuencias alineadas como una matriz de colores
    donde cada celda representa un residuo y su color indica el grupo fisicoquímico
    (hidrofóbico, polar, cargado positivo/negativo, especial o gap). Sobre la matriz se
    superponen:
      - Regiones funcionales sombreadas (P-loop, Switch I/II, NKxD, SAK).
      - Líneas verticales y etiquetas para los hotspots oncogénicos G12, G13 y Q61.
      - Leyenda de colores en la parte inferior.

    Esta figura es útil para identificar visualmente posiciones conservadas (columnas
    monocromas), regiones con alta variabilidad entre isoformas y la localización de
    mutaciones oncogénicas en el contexto del alineamiento.

    Parámetros
    ----------
    msa : Bio.Align.MultipleSeqAlignment
        Alineamiento múltiple con al menos KRAS, HRAS y NRAS.
    position_map : pd.DataFrame
        Mapa posicional generado por :func:`build_position_map`, necesario para
        localizar las regiones funcionales y hotspots en coordenadas MSA.
    output_path : str or Path
        Ruta del fichero de imagen de salida. Se admite cualquier formato soportado
        por Matplotlib (PNG, SVG, PDF). El directorio padre se crea si no existe.

    Devuelve
    --------
    Path
        Ruta del fichero de imagen generado.
    """
    # Obtener el diccionario {gen → SeqRecord} para acceder a cada secuencia por nombre.
    records = _records_by_gene(msa)

    # Asegurar que output_path sea un objeto Path (acepta tanto str como Path como entrada).
    output_path = Path(output_path)

    # Crear el directorio de salida y sus padres si no existen.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lista ordenada de los nombres de grupos de color (define el índice en la paleta discreta).
    color_names = list(AA_GROUP_COLORS)

    # Diccionario inverso: {nombre_grupo → índice} para asignar valores enteros a la matriz.
    # Matplotlib necesita índices numéricos para usar un colormap discreto (ListedColormap).
    color_lookup = {name: index for index, name in enumerate(color_names)}

    # Construir la matriz de colores como lista de listas de índices enteros.
    matrix = []
    labels = []  # Etiquetas de las filas (nombres de genes)

    for gene in GENES:
        labels.append(gene)  # Añadir el nombre del gen como etiqueta de fila

        # Para cada aminoácido de la secuencia alineada, obtener el índice de color correspondiente.
        # _aa_group(aa) clasifica el aminoácido en su grupo fisicoquímico.
        matrix.append([color_lookup[_aa_group(aa)] for aa in str(records[gene].seq)])

    # Convertir la lista de listas a un array NumPy 2D (filas=genes, columnas=posiciones MSA).
    matrix_array = np.asarray(matrix)

    # Crear un colormap discreto (ListedColormap) con los colores en el orden de color_names.
    cmap = plt.matplotlib.colors.ListedColormap([AA_GROUP_COLORS[name] for name in color_names])

    # Calcular el ancho de la figura de forma proporcional a la longitud del alineamiento.
    # División entre 12 para que 12 residuos ocupen aproximadamente 1 pulgada.
    # max(12, ...) garantiza un ancho mínimo de 12 pulgadas.
    fig_width = max(12, msa.get_alignment_length() / 12)

    # Crear la figura y el eje de Matplotlib con las dimensiones calculadas.
    fig, ax = plt.subplots(figsize=(fig_width, 3.4))

    # Dibujar la matriz de colores como una imagen (imshow):
    #   aspect="auto"          → no fuerza celdas cuadradas; se adapta al espacio disponible
    #   interpolation="nearest" → sin suavizado entre celdas (cada píxel = un residuo)
    #   cmap                   → colormap discreto construido arriba
    ax.imshow(matrix_array, aspect="auto", interpolation="nearest", cmap=cmap)

    # Configurar las marcas del eje Y con los nombres de los genes.
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Etiquetas de los ejes y título.
    ax.set_xlabel("Posición MSA")
    ax.set_title("Alineamiento múltiple de la familia RAS")

    # Crear marcas en el eje X cada 20 posiciones MSA (de 0 a la longitud total).
    tick_positions = np.arange(0, msa.get_alignment_length(), 20)
    ax.set_xticks(tick_positions)
    # Las etiquetas son 1-based (sumamos 1 a los índices 0-based).
    ax.set_xticklabels(tick_positions + 1)

    # Dibujar rectángulos sombreados para cada región funcional definida en FUNCTIONAL_REGIONS.
    for region_name, (start, end) in FUNCTIONAL_REGIONS.items():
        # Localizar las filas del mapa posicional donde kras_position está dentro del rango.
        # .between(start, end) es inclusivo en ambos extremos.
        region_rows = position_map.loc[position_map["kras_position"].between(start, end)]

        # Si la región no aparece en el mapa (por truncamiento de secuencia), saltar.
        if region_rows.empty:
            continue

        # Calcular los límites en coordenadas MSA (0-based para imshow):
        # x0 → borde izquierdo (msa_position mínima - 1.5 para incluir el margen izquierdo)
        # x1 → borde derecho  (msa_position máxima - 0.5)
        x0 = int(region_rows["msa_position"].min()) - 1.5
        x1 = int(region_rows["msa_position"].max()) - 0.5

        # Dibujar un rectángulo semitransparente (alpha=0.08) sobre la región funcional.
        ax.axvspan(x0, x1, color="black", alpha=0.08)

        # Añadir el nombre de la región centrado bajo el sombreado.
        # y=-0.72 coloca la etiqueta ligeramente por debajo del eje superior.
        ax.text((x0 + x1) / 2, -0.72, region_name, ha="center", va="bottom", fontsize=8)

    # Añadir líneas verticales y etiquetas para los hotspots oncogénicos más importantes.
    # Los números son posiciones en KRAS; las etiquetas son la nomenclatura estándar.
    for position, label in [(12, "G12"), (13, "G13"), (61, "Q61")]:
        # Buscar la fila del mapa posicional correspondiente a esta posición de KRAS.
        rows = position_map.loc[position_map["kras_position"].eq(position)]

        # Si la posición no está en el mapa, saltar.
        if rows.empty:
            continue

        # Obtener la posición MSA (0-based para imshow, por eso restamos 1).
        x = int(rows.iloc[0]["msa_position"]) - 1

        # Dibujar una línea vertical negra fina en la posición del hotspot.
        ax.axvline(x, color="black", linewidth=1)

        # Añadir la etiqueta del hotspot en la parte superior de la figura.
        # y=len(labels)-0.05 coloca la etiqueta justo encima de la última fila.
        ax.text(x, len(labels) - 0.05, label, ha="center", va="bottom", fontsize=8)

    # Construir las marcas de leyenda como parches de color con el nombre del grupo.
    handles = [plt.matplotlib.patches.Patch(color=AA_GROUP_COLORS[name], label=name) for name in color_names]

    # Colocar la leyenda centrada debajo del gráfico para no solapar la imagen.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=6)

    # Ajustar los márgenes para que todos los elementos quepan en la figura.
    fig.tight_layout()

    # Guardar la figura en disco con resolución de 300 DPI (adecuada para publicaciones).
    fig.savefig(output_path, dpi=300)

    # Cerrar la figura para liberar memoria (importante en bucles que generan muchas figuras).
    plt.close(fig)

    return output_path


def identity_fraction(msa):
    """
    Calcula la fracción de identidad media por pares entre todas las secuencias del MSA.

    La fracción de identidad mide qué proporción de posiciones alineadas tienen
    exactamente el mismo aminoácido en dos secuencias. Para KRAS, HRAS y NRAS se
    esperan valores superiores al 85 % en la región catalítica, reflejando su origen
    como genes parálogos duplicados durante la evolución de los vertebrados.

    Solo se comparan posiciones donde ambas secuencias tienen un residuo real
    (se excluyen las columnas donde alguna tiene un gap), de modo que los gaps
    no penalizan artificialmente la identidad.

    Parámetros
    ----------
    msa : Bio.Align.MultipleSeqAlignment
        Alineamiento múltiple de al menos dos secuencias.

    Devuelve
    --------
    float
        Fracción de identidad media (entre 0.0 y 1.0) sobre todos los pares posibles.
        Devuelve NaN si no hay ninguna posición comparable (e.g. secuencias completamente
        de gaps).
    """
    # Lista para acumular la fracción de identidad de cada par de secuencias.
    identities = []

    # Iterar sobre todos los pares únicos de secuencias (combinaciones sin repetición).
    # enumerate(msa) devuelve (índice, secuencia); msa[i+1:] son las secuencias posteriores.
    for i, first in enumerate(msa):
        for second in msa[i + 1:]:
            matches = 0   # Número de posiciones con el mismo aminoácido
            compared = 0  # Número de posiciones comparadas (sin gaps en ninguna de las dos)

            # Comparar residuo a residuo las dos secuencias alineadas.
            for aa_1, aa_2 in zip(str(first.seq), str(second.seq)):
                # Saltar posiciones donde alguna de las dos secuencias tiene un gap.
                if aa_1 == "-" or aa_2 == "-":
                    continue
                compared += 1          # Posición válida para comparar
                matches += aa_1 == aa_2  # Suma 1 si son iguales, 0 si no (bool → int en Python)

            # Solo añadir la fracción si hay al menos una posición comparada.
            if compared:
                identities.append(matches / compared)

    # Devolver la media de todas las fracciones de identidad por pares,
    # o NaN si la lista está vacía (no hubo pares comparables).
    return float(np.mean(identities)) if identities else float("nan")


def consensus_sequence(msa):
    """
    Calcula la secuencia consenso del alineamiento múltiple.

    La secuencia consenso se construye seleccionando en cada columna del alineamiento
    el aminoácido más frecuente entre todas las secuencias (excluyendo gaps). Si todas
    las secuencias tienen un gap en esa columna, se usa '-' en el consenso.

    En familias proteicas altamente conservadas como RAS, la secuencia consenso es
    muy similar a cualquiera de las tres isoformas, lo que confirma su alta identidad.
    El consenso puede usarse como secuencia de referencia al diseñar inhibidores pan-RAS.

    Parámetros
    ----------
    msa : Bio.Align.MultipleSeqAlignment
        Alineamiento múltiple de cualquier número de secuencias.

    Devuelve
    --------
    str
        Secuencia consenso de longitud igual al número de columnas del alineamiento.
        Contiene '-' en posiciones donde todas las secuencias tienen gap.
    """
    # Lista donde se acumularán los aminoácidos del consenso, uno por columna.
    consensus = []

    # Iterar sobre cada columna del alineamiento (índice 0-based).
    for column_index in range(msa.get_alignment_length()):
        # Counter cuenta las ocurrencias de cada carácter en esta columna.
        # La expresión generadora extrae el aminoácido de cada secuencia en column_index.
        counts = Counter(str(record.seq[column_index]) for record in msa)

        # Eliminar los gaps del conteo; no queremos que un gap "gane" la votación mayoritaria.
        counts.pop("-", None)  # None evita KeyError si no hay gaps

        # Seleccionar el aminoácido más frecuente (most_common(1) devuelve [(aa, count)]).
        # Si counts está vacío (solo había gaps), usar '-' como carácter de consenso.
        consensus.append(counts.most_common(1)[0][0] if counts else "-")

    # Unir la lista de caracteres en una única cadena de texto.
    return "".join(consensus)


# ---------------------------------------------------------------------------
# Funciones privadas (auxiliares internas, no forman parte de la API pública)
# ---------------------------------------------------------------------------

def _sequence_from_fasta(text):
    """
    Extrae la secuencia de aminoácidos de un texto en formato FASTA.

    El formato FASTA consiste en una o más líneas de cabecera (que comienzan con '>')
    seguidas de líneas de secuencia. Esta función elimina las líneas de cabecera y
    concatena el resto para obtener la secuencia completa como una sola cadena.

    Parámetros
    ----------
    text : str
        Contenido completo de un fichero FASTA (una sola entrada).

    Devuelve
    --------
    str
        Secuencia de aminoácidos sin espacios ni cabecera.
    """
    # Dividir el texto en líneas, eliminar espacios en blanco iniciales/finales y descartar líneas vacías.
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Concatenar solo las líneas que no empiezan por '>' (excluir cabeceras FASTA).
    return "".join(line for line in lines if not line.startswith(">"))


def _alignment_from_sequences(sequences):
    """
    Crea un objeto MultipleSeqAlignment de BioPython a partir de un diccionario de secuencias.

    Esta función construye un alineamiento "trivial" (sin gaps) útil cuando las secuencias
    tienen exactamente la misma longitud y no es necesario ejecutar una herramienta externa.
    Para KRAS/HRAS/NRAS, el dominio catalítico tiene 189 residuos en las tres isoformas.

    Parámetros
    ----------
    sequences : dict[str, str]
        Diccionario {nombre → secuencia} con las secuencias a alinear.

    Devuelve
    --------
    Bio.Align.MultipleSeqAlignment
        Alineamiento sin gaps con un SeqRecord por cada entrada del diccionario.
    """
    # Construir una lista de SeqRecord y envolverla en MultipleSeqAlignment.
    # Seq(sequence) crea el objeto secuencia de BioPython.
    # SeqRecord(..., id=name, description="") crea un registro con identificador pero sin descripción.
    return MultipleSeqAlignment([
        SeqRecord(Seq(sequence), id=name, description="")
        for name, sequence in sequences.items()
    ])


def _write_fasta(sequences, path):
    """
    Escribe un diccionario de secuencias en formato FASTA en un fichero.

    Parámetros
    ----------
    sequences : dict[str, str]
        Diccionario {nombre → secuencia}.
    path : Path
        Ruta del fichero de salida donde se escribirá el FASTA.
    """
    lines = []  # Lista de líneas a escribir
    for name, sequence in sequences.items():
        # Cada entrada FASTA consiste en una línea de cabecera '>nombre' y la línea de secuencia.
        lines.extend([f">{name}", sequence])

    # Escribir todas las líneas unidas por saltos de línea, con un salto final.
    # encoding="utf-8" garantiza compatibilidad multiplataforma.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_muscle(executable, input_fasta, output_fasta):
    """
    Ejecuta MUSCLE soportando tanto la versión 5 (nueva sintaxis) como la versión 3 (antigua).

    MUSCLE v5 usa la sintaxis: muscle -align input.fasta -output output.fasta
    MUSCLE v3 usa la sintaxis: muscle -in input.fasta -out output.fasta

    La función intenta primero la versión 5 y, si falla (returncode != 0), reintenta
    con la sintaxis de la versión 3. Esto garantiza compatibilidad con instalaciones
    antiguas de MUSCLE sin requerir que el usuario identifique manualmente la versión.

    Parámetros
    ----------
    executable : str
        Ruta completa al ejecutable de MUSCLE (obtenida con shutil.which).
    input_fasta : Path
        Ruta del fichero FASTA de entrada con las secuencias a alinear.
    output_fasta : Path
        Ruta del fichero FASTA de salida donde MUSCLE escribirá el alineamiento.

    Devuelve
    --------
    subprocess.CompletedProcess
        Resultado del proceso que tuvo éxito (o del segundo intento si el primero falló).
    """
    # Intentar primero con la sintaxis de MUSCLE v5.
    result = subprocess.run(
        [executable, "-align", str(input_fasta), "-output", str(output_fasta)],
        capture_output=True,  # Capturar stdout y stderr para no contaminar la salida del programa
        text=True,            # Decodificar la salida como texto (no bytes)
        check=False,          # No lanzar excepción automáticamente; comprobar manualmente
    )

    # Si MUSCLE v5 tuvo éxito, devolver el resultado directamente.
    if result.returncode == 0:
        return result

    # Si MUSCLE v5 falló, reintentar con la sintaxis de MUSCLE v3.
    return subprocess.run(
        [executable, "-in", str(input_fasta), "-out", str(output_fasta)],
        capture_output=True,
        text=True,
        check=False,
    )


def _records_by_gene(msa):
    """
    Construye un diccionario {gen → SeqRecord} a partir del alineamiento múltiple.

    Los registros en el MSA tienen identificadores que pueden incluir información
    adicional (e.g. "sp|P01116|RASK_HUMAN"). Esta función busca si el nombre de alguno
    de los genes de GENES aparece como subcadena en el identificador (tras convertir a
    mayúsculas) para asociar cada registro con su gen correspondiente.

    Parámetros
    ----------
    msa : Bio.Align.MultipleSeqAlignment
        Alineamiento múltiple cuyos registros tienen identificadores que contienen
        el nombre del gen ('KRAS', 'HRAS' o 'NRAS').

    Devuelve
    --------
    dict[str, SeqRecord]
        Diccionario {nombre_gen → SeqRecord}. Puede estar incompleto si algún gen
        no tiene registro con su nombre en el identificador.
    """
    records = {}  # Resultado a construir

    for record in msa:
        # Normalizar el identificador a mayúsculas para comparar sin distinción de caso.
        record_id = record.id.upper()

        # Comprobar si alguno de los nombres de genes aparece en el identificador.
        for gene in GENES:
            if gene in record_id:
                records[gene] = record  # Asociar el gen con este registro
                break  # Cada registro se asocia al primer gen que coincida

    return records


def _validate_hotspot_alignment(position_map):
    """
    Verifica que los hotspots oncogénicos clave están correctamente alineados en el MSA.

    Los hotspots G12, G13 y Q61 son las posiciones más frecuentemente mutadas en
    tumores humanos en todas las isoformas RAS. Si el alineamiento es correcto,
    estas posiciones deben tener el mismo aminoácido en las tres isoformas (Gly en
    12 y 13, Gln en 61) y sus índices en KRAS, HRAS y NRAS deben coincidir.

    Si el alineamiento contiene errores (por ejemplo, si la herramienta MSA introdujo
    gaps incorrectos), esta validación lo detectará y lanzará una excepción descriptiva,
    evitando que el análisis continúe con datos incorrectos.

    Parámetros
    ----------
    position_map : pd.DataFrame
        Mapa posicional generado por :func:`build_position_map`.

    Raises
    ------
    AssertionError
        Si algún hotspot no aparece en el mapa o si el aminoácido/posición observada
        no coincide con el esperado para alguna isoforma.
    """
    # Diccionario {posición_KRAS → aminoácido_esperado} para los tres hotspots principales.
    # G12 y G13 son glicinas críticas en el P-loop; Q61 es la glutamina catalítica en Switch II.
    expected = {12: "G", 13: "G", 61: "Q"}

    for position, aa in expected.items():
        # Buscar la fila del mapa posicional correspondiente a esta posición de KRAS.
        rows = position_map.loc[position_map["kras_position"].eq(position)]

        # Si la posición no aparece en el mapa, el alineamiento está truncado o es incorrecto.
        if rows.empty:
            raise AssertionError(f"La posición KRAS {position} no aparece en el mapa MSA.")

        # Tomar la primera (y única) fila correspondiente a esta posición.
        row = rows.iloc[0]

        # Verificar el alineamiento para cada isoforma.
        for gene in GENES:
            gene_lower = gene.lower()

            # Obtener la posición y aminoácido observados para este gen en esta columna del MSA.
            observed_position = row[f"{gene_lower}_position"]
            observed_aa = row[f"{gene_lower}_aa"]

            # Comprobar que tanto la posición como el aminoácido coinciden con lo esperado.
            # Para un alineamiento correcto de RAS, G12 de KRAS debe alinearse con G12 de HRAS y NRAS.
            if int(observed_position) != position or observed_aa != aa:
                raise AssertionError(
                    f"Hotspot {aa}{position} no está alineado para {gene}: {observed_aa}{observed_position}"
                )


def _nullable_int(value):
    """
    Convierte un valor a int si no es NaN, o devuelve NaN en caso contrario.

    En pandas, cuando un DataFrame contiene columnas enteras con valores NaN,
    Python los almacena como float (ya que int no puede representar NaN de forma nativa
    hasta el tipo Int64 de pandas). Esta función convierte explícitamente a int
    los valores válidos y preserva np.nan para las posiciones de gap,
    lo que es útil al construir el DataFrame de posiciones recurrentes.

    Parámetros
    ----------
    value : float or int or NaN
        Valor a convertir.

    Devuelve
    --------
    int or float
        El valor convertido a int, o np.nan si el valor original era NaN.
    """
    # pd.isna() devuelve True tanto para np.nan como para None y pd.NaT.
    return np.nan if pd.isna(value) else int(value)


def _aa_group(aa):
    """
    Determina el grupo fisicoquímico de un aminoácido según la clasificación en AA_GROUPS.

    Esta función se usa para asignar un color a cada residuo en la visualización del MSA.
    Los aminoácidos no reconocidos (caracteres no estándar como 'X', 'B', 'Z' u otros)
    se clasifican como 'polar' por defecto, evitando errores en la visualización.

    Parámetros
    ----------
    aa : str
        Carácter de un solo aminoácido en código de una letra (e.g. 'G', 'A', '-').
        El carácter '-' indica un gap en el alineamiento.

    Devuelve
    --------
    str
        Nombre del grupo fisicoquímico: 'hydrophobic', 'polar', 'positive',
        'negative', 'special' o 'gap' para el carácter '-'.
    """
    # Normalizar a mayúsculas por si la secuencia alineada contiene minúsculas.
    aa = aa.upper()

    # Los gaps se clasifican en su propio grupo para mostrarlos en gris en la visualización.
    if aa == "-":
        return "gap"

    # Buscar el aminoácido en cada grupo fisicoquímico definido en AA_GROUPS.
    for group, residues in AA_GROUPS.items():
        if aa in residues:
            return group  # Devolver el nombre del grupo en cuanto se encuentre

    # Si el aminoácido no pertenece a ningún grupo conocido (e.g. 'X' para desconocido),
    # clasificarlo como 'polar' para que la visualización no genere un error de color.
    return "polar"