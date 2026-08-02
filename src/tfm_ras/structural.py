"""
structural.py — Utilidades de bioinformática estructural para proteínas RAS
===========================================================================

Contexto biológico
------------------
Las proteínas RAS (KRAS, HRAS, NRAS) son GTPasas pequeñas que actúan como
interruptores moleculares: en su forma unida a GTP están activas y transmiten
señales de proliferación celular; al hidrolizar GTP a GDP se inactivan. Las
mutaciones oncogénicas en posiciones clave (G12, G13, Q61 del dominio G)
bloquean la GTPasa en estado activo, promoviendo el cáncer.

Este módulo proporciona funciones para calcular cuatro propiedades
estructurales a partir de ficheros PDB descargados del banco de datos RCSB:

1. **SASA (Superficie Accesible al Solvente)**
   El SASA (en inglés, Solvent-Accessible Surface Area) mide el área, en
   ångströms cuadrados (Å²), de la superficie de un residuo que puede ser
   contactada por una molécula de agua (representada como una sonda esférica
   de radio ~1,4 Å). Un residuo con SASA alto está expuesto al disolvente
   (superficie), mientras que uno con SASA bajo está enterrado en el núcleo
   hidrofóbico de la proteína. El SASA relativo normaliza el valor absoluto
   respecto al máximo teórico de cada aminoácido en un péptido extendido
   (escala de Tien et al., 2013), produciendo valores entre 0 (completamente
   enterrado) y ≈1 (completamente expuesto).

2. **Distancia al sitio activo**
   El sitio activo de las proteínas RAS es el bolsillo de unión al nucleótido
   (GDP/GTP) en el dominio G. Está flanqueado por los lazos P-loop, Switch I y
   Switch II, y alberga un ion Mg²⁺ esencial para la coordinación del fosfato.
   La distancia mínima de un residuo a los átomos del ligando nucleotídico y
   del Mg²⁺ es un descriptor de la accesibilidad funcional: residuos con
   distancias pequeñas (<6 Å) se consideran en la primera esfera de
   coordinación del sitio activo.

3. **Entropía de Shannon por posición del MSA**
   El alineamiento múltiple de secuencias (MSA) permite medir la conservación
   evolutiva columna a columna. La entropía de Shannon (H, en bits) de una
   columna del MSA se calcula como:

       H = -Σ p_i · log₂(p_i)

   donde p_i es la frecuencia del aminoácido i en la columna (excluyendo
   huecos). H = 0 indica conservación absoluta (un solo aminoácido en todas
   las secuencias); H > 0 indica variabilidad. La unidad son bits porque se
   usa logaritmo en base 2.

4. **Tabla maestra de features**
   Combina mutaciones clínicas (COSMIC), SASA, distancia al sitio activo y
   entropía de Shannon en un único DataFrame por gen y posición del dominio G
   (residuos 1–166). Esta tabla es la entrada principal para los modelos
   estadísticos y de aprendizaje automático del TFM.

Uso típico
----------
::

    from tfm_ras.structural import fetch_pdb, parse_structure, compute_sasa
    pdb_path  = fetch_pdb("4OBE", output_dir="data/raw/pdb")
    model     = parse_structure(pdb_path)
    sasa_df   = compute_sasa(model, chain_id="A")

Dependencias externas
---------------------
- biopython  : parseo de PDB, DSSP, Shrake-Rupley, MSA
- numpy      : álgebra lineal (distancias) y aritmética vectorizada
- pandas     : DataFrames para almacenar resultados por posición
- matplotlib : visualización de la tabla maestra
- requests   : descarga de ficheros PDB desde RCSB

Referencias
-----------
* Tien, M. Z., et al. (2013). PLoS ONE, 8(11), e80635.
  Escala de SASA máximo teórico para los 20 aminoácidos estándar.
* Shannon, C. E. (1948). Bell System Technical Journal, 27(3), 379-423.
  Definición original de la entropía de la información.
* Prior, I. A., et al. (2020). Cancer Research, 80(14), 2969-2974.
  Frecuencias de mutaciones oncogénicas en KRAS, HRAS y NRAS.
"""

# ---------------------------------------------------------------------------
# Importaciones de la biblioteca estándar de Python
# ---------------------------------------------------------------------------

# 'annotations' de __future__ activa la evaluación diferida de las
# anotaciones de tipo (PEP 563).  Esto permite usar tipos aún no definidos
# en anotaciones sin que Python los evalúe en tiempo de importación.
from __future__ import annotations

# 'math' proporciona funciones matemáticas de precisión de punto flotante
# definidas por el estándar C: logaritmos, trigonometría, constantes (π, e).
# Aquí se usa 'math.log2' para calcular la entropía de Shannon en bits.
import math

# 'shutil' (shell utilities) ofrece operaciones de alto nivel sobre ficheros
# y directorios.  'shutil.which' localiza un ejecutable en el PATH del sistema,
# útil para comprobar si el binario 'mkdssp' (o 'dssp') está instalado.
import shutil

# 'Counter' es una subclase de dict especializada en contar elementos
# hashables.  Counter("AABBA") devuelve {"A": 3, "B": 2}.  Se usa para
# calcular las frecuencias de aminoácidos en cada columna del MSA.
from collections import Counter

# 'Path' ofrece una API orientada a objetos para rutas del sistema de
# ficheros, portable entre Linux, Mac y Windows.  Permite concatenar
# directorios con '/', comprobar existencia y leer/escribir ficheros.
from pathlib import Path

# ---------------------------------------------------------------------------
# Importaciones de bibliotecas de terceros
# ---------------------------------------------------------------------------

# 'matplotlib.pyplot' es la interfaz de alto nivel de Matplotlib para crear
# figuras, subplots, ejes y guardar gráficos en formatos de imagen (PNG, SVG).
import matplotlib.pyplot as plt

# 'numpy' (numerical Python) proporciona el tipo ndarray y operaciones
# vectorizadas de álgebra lineal, estadística y manipulación de arrays.
# Es la base de prácticamente todo el ecosistema científico de Python.
import numpy as np

# 'pandas' proporciona el DataFrame (tabla bidimensional con etiquetas) y la
# Series (columna con etiqueta).  Se usa aquí para almacenar y combinar los
# datos estructurales por residuo y gen.
import pandas as pd

# 'requests' es la biblioteca estándar de facto para hacer peticiones HTTP en
# Python.  Aquí se usa para descargar ficheros PDB del servidor RCSB mediante
# una petición GET.
import requests

# 'IUPACData' de Biopython contiene diccionarios con datos del estándar IUPAC
# para biología molecular: tablas de codones, propiedades de aminoácidos, etc.
# En concreto, 'protein_letters_3to1' convierte códigos de tres letras
# (ej. "Ala") a de una letra (ej. "A").
from Bio.Data import IUPACData

# 'DSSP' es una clase de Biopython que invoca el programa externo DSSP/mkdssp
# para calcular estructura secundaria y accesibilidad al solvente a partir de
# un fichero PDB.  Es más rápido que Shrake-Rupley pero requiere el binario.
# 'PDBParser' lee ficheros en formato PDB y construye la jerarquía SMCRA.
# 'ShrakeRupley' implementa en Python puro el algoritmo de rodamiento de
# esfera para estimar el SASA sin necesidad de programas externos.
from Bio.PDB import DSSP, PDBParser, ShrakeRupley

# 'Structure' es la clase raíz de la jerarquía SMCRA (Structure→Model→Chain→
# Residue→Atom) de Biopython.  Se importa aquí para anotar tipos, aunque no
# se usa explícitamente en las anotaciones de este módulo.
from Bio.PDB.Structure import Structure

# ---------------------------------------------------------------------------
# Constantes globales del módulo
# ---------------------------------------------------------------------------

# Tupla con los tres genes de la familia RAS que analiza este TFM.
# Se usa una tupla (inmutable) para señalar que el conjunto de genes no
# debe cambiar en tiempo de ejecución.
GENES = ("KRAS", "HRAS", "NRAS")

# Posición final del dominio G (GTPase domain) en la numeración UniProt.
# Los residuos 1–166 corresponden al dominio catalítico conservado entre los
# tres parálogos; a partir del residuo 167 comienza la región hipervariable
# (HVR), que difiere significativamente entre KRAS, HRAS y NRAS y se excluye
# del análisis comparativo.
DOMAIN_G_END = 166

# Lista ordenada de columnas de la tabla maestra.  Define el contrato de
# salida de la función merge_features y garantiza un orden reproducible.
# Columna por columna:
#   "gene"                     : identificador del gen RAS (KRAS, HRAS o NRAS)
#   "uniprot_position"         : posición del residuo en la secuencia UniProt
#                                (numeración canónica de referencia)
#   "msa_position"             : posición equivalente en el alineamiento múltiple
#                                de secuencias (MSA) de los tres parálogos
#   "aa_wt"                    : aminoácido silvestre (wild-type) en esa posición,
#                                en código de una letra (ej. "G" para Gly12)
#   "total_samples"            : número total de muestras tumorales (base COSMIC)
#                                con alguna mutación en ese residuo
#   "top_mutation"             : notación HGVS proteica (ej. "p.Gly12Val") de la
#                                mutación más frecuente en ese residuo
#   "sasa_abs"                 : SASA absoluto del residuo en Å² (calculado con
#                                DSSP o Shrake-Rupley sobre la estructura PDB)
#   "sasa_rel"                 : SASA relativo (sasa_abs / SASA_max_teórico),
#                                sin unidades; 0 = enterrado, ≈1 = expuesto
#   "dist_active_site_angstrom": distancia mínima en Å entre cualquier átomo
#                                pesado del residuo y cualquier átomo pesado del
#                                nucleótido ligando (GDP/GTP) o del ion Mg²⁺
#   "shannon_entropy"          : entropía de Shannon en bits de la columna del
#                                MSA correspondiente; mide la variabilidad
#                                evolutiva del residuo (0 = totalmente conservado)
#   "recurrent"                : booleano; True si la mutación aparece en ≥2 de
#                                los 3 parálogos con ≥10 muestras tumorales
#                                (criterio de mutación recurrente pan-RAS)
#   "n_members_mutated"        : número de parálogos (0–3) con ≥10 muestras
#                                tumorales mutadas en esa posición
MASTER_COLUMNS = [
    "gene", "state", "uniprot_position", "msa_position", "aa_wt",
    "total_samples", "top_mutation",
    "sasa_abs", "sasa_rel",
    "dist_active_site_angstrom",
    "shannon_entropy",
    "recurrent", "n_members_mutated",
]

# Diccionario con el SASA máximo teórico (en Å²) para cada uno de los 20
# aminoácidos estándar.  Los valores provienen de Tien et al. (2013), que
# estimaron el SASA de cada residuo X en el tripéptido Gly-X-Gly en
# conformación extendida, usando una sonda de agua de radio 1,4 Å.
#
# Este valor se usa como denominador para normalizar el SASA absoluto medido
# en la estructura PDB y obtener el SASA relativo (fracción de superficie
# expuesta respecto al máximo posible para ese tipo de aminoácido).
#
# Se eligió la escala de Tien 2013 porque:
#   (a) está calculada con el mismo radio de sonda que usa DSSP y Shrake-Rupley
#       en Biopython (1,4 Å), lo que hace la normalización internamente
#       consistente;
#   (b) usa como referencia el tripéptido Gly-X-Gly, que minimiza los efectos
#       de contexto de secuencia y proporciona el área máxima accesible real;
#   (c) es la escala recomendada por la documentación de Biopython para SASA
#       relativo, garantizando reproducibilidad frente a otras publicaciones.
#
# Las claves son los códigos de una letra de los aminoácidos; los valores son
# los SASA máximos en Å² con dos cifras significativas.
TIEN_2013_MAX_SASA = {
    "A": 129.0,   # Alanina      (Ala) — cadena lateral metilo, pequeña y no polar
    "R": 274.0,   # Arginina     (Arg) — cadena lateral larga y cargada positivamente
    "N": 195.0,   # Asparagina   (Asn) — cadena lateral amida polar
    "D": 193.0,   # Aspartato    (Asp) — cadena lateral carboxilato, cargada negativamente
    "C": 167.0,   # Cisteína     (Cys) — cadena lateral tiol, participa en puentes disulfuro
    "Q": 225.0,   # Glutamina    (Gln) — cadena lateral amida polar más larga; Q61 es hotspot
    "E": 223.0,   # Glutamato    (Glu) — cadena lateral carboxilato más larga, cargada
    "G": 104.0,   # Glicina      (Gly) — sin cadena lateral, máximo conformacional pequeño; G12/G13 son hotspots
    "H": 224.0,   # Histidina    (His) — cadena lateral imidazol, puede estar protonada o no
    "I": 197.0,   # Isoleucina   (Ile) — cadena lateral ramificada, hidrofóbica
    "L": 201.0,   # Leucina      (Leu) — cadena lateral ramificada, hidrofóbica
    "K": 236.0,   # Lisina       (Lys) — cadena lateral amina larga, cargada positivamente
    "M": 224.0,   # Metionina    (Met) — cadena lateral tioéter, flexible
    "F": 240.0,   # Fenilalanina (Phe) — cadena lateral aromática, hidrofóbica
    "P": 159.0,   # Prolina      (Pro) — nitrógeno de cadena lateral en el anillo, rigidez conformacional
    "S": 155.0,   # Serina       (Ser) — cadena lateral hidroximetilo, polar pequeña
    "T": 172.0,   # Treonina     (Thr) — cadena lateral hidroxietilo, polar y ramificada
    "W": 285.0,   # Triptófano   (Trp) — cadena lateral indol bicíclica, la más grande
    "Y": 263.0,   # Tirosina     (Tyr) — cadena lateral fenol, aromática y polar
    "V": 174.0,   # Valina       (Val) — cadena lateral isopropilo, hidrofóbica y compacta
}

# Diccionario de conversión de código de tres letras a código de una letra
# para los 20 aminoácidos estándar, construido a partir del diccionario
# canónico de Biopython 'IUPACData.protein_letters_3to1'.
#
# El estándar IUPAC define dos sistemas de nomenclatura para aminoácidos:
#   - Código de 3 letras: "Ala", "Arg", "Asn", ... (primera letra mayúscula)
#   - Código de 1 letra:  "A",   "R",   "N",  ...
#
# Los ficheros PDB almacenan los residuos con el código de 3 letras en
# mayúsculas (ej. "ALA", "GLY", "GLN").  La comprensión de diccionario
# convierte las claves a mayúsculas con '.upper()' para que la búsqueda
# ("ALA" en THREE_TO_ONE) funcione independientemente del caso con que
# Biopython devuelva el nombre del residuo.
THREE_TO_ONE = {
    key.upper(): value                          # "ALA" -> "A", "GLY" -> "G", etc.
    for key, value in IUPACData.protein_letters_3to1.items()  # itera el dict IUPAC original
}


# ===========================================================================
# Funciones públicas
# ===========================================================================

def fetch_pdb(pdb_id, output_dir):
    """Descarga un fichero PDB del banco de datos RCSB y lo guarda en disco.

    El RCSB PDB (Research Collaboratory for Structural Bioinformatics Protein
    Data Bank, https://www.rcsb.org) es el archivo global de estructuras 3D
    de macromoléculas biológicas determinadas por cristalografía de rayos X,
    resonancia magnética nuclear (RMN) o cryo-EM.  Cada entrada se identifica
    por un código de 4 caracteres alfanuméricos (ej. "4OBE" para KRAS·GDP).

    Un fichero PDB es un fichero de texto plano con un formato de columnas
    fijo definido por el wwPDB.  Contiene:
      - Metadatos de la estructura (HEADER, TITLE, REMARK)
      - Coordenadas atómicas (registros ATOM y HETATM) en Å
      - Conectividades (CONECT) y matrices de transformación (SEQRES, CRYST1)

    Si el fichero ya existe en disco con tamaño > 0, se valida parseándolo y
    se reutiliza sin hacer ninguna petición de red.

    Parameters
    ----------
    pdb_id : str
        Identificador PDB de 4 caracteres (ej. "4OBE").  Se convierte
        automáticamente a mayúsculas.
    output_dir : str or Path
        Directorio donde se guardará el fichero descargado.  Se crea si no
        existe.

    Returns
    -------
    Path
        Ruta absoluta al fichero PDB descargado o reutilizado.

    Raises
    ------
    requests.HTTPError
        Si el servidor RCSB devuelve un código de error HTTP (ej. 404 si el
        PDB ID no existe).
    ValueError
        Si el fichero descargado no es un PDB válido (lo detecta PDBParser).

    Examples
    --------
    >>> pdb_path = fetch_pdb("4OBE", output_dir="data/raw/pdb")
    >>> print(pdb_path)
    data/raw/pdb/4OBE.pdb
    """
    # Normalizamos el identificador PDB a mayúsculas: los IDs son
    # alfanuméricos y el servidor RCSB los acepta en cualquier caso, pero
    # los ficheros locales se nombran siempre en mayúsculas por convención.
    pdb_id = pdb_id.upper()  # ej. "4obe" -> "4OBE"

    # Convertimos output_dir a objeto Path para poder usar la API de pathlib
    # (operador '/', .exists(), .stat(), etc.) independientemente de si el
    # llamante pasó una cadena o un Path.
    output_dir = Path(output_dir)  # ej. "data/raw/pdb" -> PosixPath("data/raw/pdb")

    # Creamos el directorio de salida y todos los directorios intermedios que
    # falten.  'parents=True' crea también los antecesores; 'exist_ok=True'
    # no lanza error si el directorio ya existe.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construimos la ruta esperada del fichero PDB local.  El operador '/'
    # de pathlib concatena el directorio con el nombre de fichero.
    output_path = output_dir / f"{pdb_id}.pdb"  # ej. "data/raw/pdb/4OBE.pdb"

    # Creamos una instancia de PDBParser con QUIET=True para suprimir los
    # avisos que Biopython imprime al encontrar residuos no estándar o
    # discontinuidades de cadena (frecuentes en estructuras reales).
    parser = PDBParser(QUIET=True)

    # --- Caso 1: el fichero ya existe y no está vacío (caché local) ----------
    if output_path.exists() and output_path.stat().st_size > 0:
        # '.stat().st_size' devuelve el tamaño del fichero en bytes; un valor
        # > 0 descarta ficheros corruptos de descargas anteriores incompletas.

        # Parseamos el fichero para validar que es un PDB bien formado.
        # Si estuviera corrupto, PDBParser lanzaría una excepción aquí, lo
        # que permite detectar el problema antes de usarlo en cálculos.
        parser.get_structure(pdb_id, output_path)  # valida el PDB en disco

        # Devolvemos la ruta al fichero ya existente sin hacer ninguna
        # petición de red, ahorrando tiempo y ancho de banda.
        return output_path  # ruta al fichero PDB en caché

    # --- Caso 2: hay que descargar el fichero del servidor RCSB ---------------

    # Construimos la URL de descarga directa del fichero PDB en formato texto.
    # RCSB ofrece los ficheros en: https://files.rcsb.org/download/<ID>.pdb
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    # Hacemos la petición HTTP GET con un tiempo máximo de espera de 60 s.
    # 'timeout=60' evita que la función se bloquee indefinidamente si el
    # servidor no responde.
    response = requests.get(url, timeout=60)  # petición GET al servidor RCSB

    # 'raise_for_status()' lanza requests.HTTPError si el código HTTP es ≥400
    # (ej. 404 Not Found si el PDB ID no existe, 500 Internal Server Error).
    # Así la función falla de forma explícita y descriptiva en lugar de
    # guardar silenciosamente un HTML de error como si fuera un PDB.
    response.raise_for_status()  # aborta si la descarga no fue exitosa

    # Escribimos el texto recibido en el fichero de salida con codificación
    # UTF-8.  Los ficheros PDB son texto ASCII puro pero especificamos UTF-8
    # por robustez ante posibles extensiones del formato.
    output_path.write_text(response.text, encoding="utf-8")  # guarda el PDB en disco

    # Validamos el fichero recién descargado parseándolo con Biopython.
    # Si la descarga fue incompleta o el servidor devolvió contenido
    # inesperado, esta línea lanzará una excepción.
    parser.get_structure(pdb_id, output_path)  # valida el PDB recién descargado

    # Devolvemos la ruta al fichero guardado para que el llamante pueda
    # pasarla a otras funciones del módulo.
    return output_path  # ruta al PDB descargado


def parse_structure(pdb_path):
    """Parsea un fichero PDB y devuelve el primer modelo de la estructura.

    Biopython organiza las estructuras macromoleculares siguiendo la jerarquía
    **SMCRA** (Structure → Model → Chain → Residue → Atom):

    - **Structure**: objeto raíz que agrupa todos los modelos de la entrada
      PDB.  Cada estructura tiene un identificador (ej. "4OBE").
    - **Model**: un conjunto de coordenadas atómicas.  En cristalografía de
      rayos X normalmente hay un único modelo (modelo 0); en RMN puede haber
      docenas de conformaciones alternativas (modelos 0..N).
    - **Chain**: una cadena polipeptídica o de ácido nucleico identificada por
      una letra (ej. cadena "A" para la proteína RAS).
    - **Residue**: un aminoácido (o nucleótido, ligando, agua…) identificado
      por su nombre de tres letras y su número de secuencia.
    - **Atom**: átomo individual con coordenadas (x, y, z) en Å, factor de
      temperatura (B-factor) y ocupancia.

    Esta función extrae el primer modelo (índice 0) porque los cálculos de
    SASA y distancias operan sobre un único juego de coordenadas.

    Parameters
    ----------
    pdb_path : str or Path
        Ruta al fichero PDB en disco.

    Returns
    -------
    Bio.PDB.Model.Model
        Primer modelo de la jerarquía SMCRA, listo para ser pasado a
        'compute_sasa' o 'distance_to_active_site'.

    Examples
    --------
    >>> model = parse_structure("data/raw/pdb/4OBE.pdb")
    >>> chains = list(model.get_chains())
    >>> print([chain.id for chain in chains])
    ['A']
    """
    # Instanciamos el parser de Biopython con modo silencioso para no
    # contaminar la salida estándar con avisos de formato.
    parser = PDBParser(QUIET=True)

    # 'Path(pdb_path).stem' extrae el nombre del fichero sin extensión
    # (ej. "/data/4OBE.pdb" -> "4OBE") para usarlo como identificador de
    # la estructura en Biopython.  '.upper()' asegura que el ID quede en
    # mayúsculas aunque el nombre de fichero esté en minúsculas.
    structure = parser.get_structure(Path(pdb_path).stem.upper(), pdb_path)
    
    # 'structure.get_models()' devuelve un generador que produce objetos Model
    # en orden.  'next(...)' consume el primer elemento del generador, que
    # corresponde al modelo 0 (el único en estructuras de rayos X y el de
    # referencia en estructuras de RMN).
    return next(structure.get_models())  # primer modelo de la jerarquía SMCRA


def get_active_site_atoms(structure, ligand_resnames=("GDP", "GTP", "GNP", "GCP"), cofactor="MG", max_mg_distance=3.5):
    """Extrae los átomos del sitio activo de la proteína RAS de la estructura.

    El sitio activo de las proteínas RAS es el bolsillo de unión al nucleótido
    de guanina, localizado en el dominio G (residuos 1–166).  Está formado por:

    - **Nucleótido de guanina**: GDP (guanosín difosfato) cuando la proteína
      está inactiva, GTP (guanosín trifosfato) cuando está activa.  En
      estructuras cristalográficas también aparecen análogos no hidrolizables:
        - GNP (GDPNP, β,γ-iminaguanosín trifosfato): mimetiza GTP sin poder
          hidrolizarse.
        - GCP (GDPCP, β,γ-metilenguanosín trifosfato): otro análogo de GTP
          no hidrolizable.
    - **Ion Mg²⁺ (cofactor MG)**: coordinado por los fosfatos del nucleótido
      y por los residuos Ser17 y Thr35, es esencial para la actividad GTPasa
      y para la interacción con efectores como RAF1 y PI3K.

    Los residuos G12, G13 (P-loop) y Q61 (Switch II) rodean directamente el
    bolsillo de nucleótido; sus mutaciones oncogénicas impiden la hidrólisis
    del GTP.

    Parameters
    ----------
    structure : Bio.PDB.Model.Model
        Modelo estructural (devuelto por 'parse_structure').
    ligand_resnames : tuple of str, optional
        Nombres de residuos PDB del ligando nucleotídico.  Por defecto incluye
        GDP, GTP y los análogos no hidrolizables GNP y GCP.
    cofactor : str, optional
        Nombre del cofactor metálico en el PDB.  Por defecto "MG" (ion Mg²⁺).

    Returns
    -------
    list of Bio.PDB.Atom.Atom
        Lista de todos los átomos (de la clase Atom de Biopython) que
        pertenecen al nucleótido ligando o al ion Mg²⁺.  Se pasan directamente
        a 'distance_to_active_site'.

    Raises
    ------
    ValueError
        Si la estructura no contiene ningún residuo con los nombres indicados
        (p.ej. si se proporcionó un PDB sin ligando o con nombres distintos).

    Examples
    --------
    >>> model = parse_structure("data/raw/pdb/4OBE.pdb")
    >>> atoms = get_active_site_atoms(model)
    >>> print(len(atoms))
    32
    """
    # Convertimos la tupla de nombres de ligandos a un conjunto (set) en
    # mayúsculas.  Usar un set en lugar de una lista hace que la comprobación
    # 'resname in ligand_resnames' sea O(1) en lugar de O(n).
    ligand_resnames = {name.upper() for name in ligand_resnames}  # {"GDP", "GTP", "GNP", "GCP"}

    # Normalizamos el nombre del cofactor a mayúsculas para la comparación.
    cofactor = cofactor.upper()  # "MG" -> "MG"

    # Extraemos los átomos del nucleótido
    nucleotide_atoms = []
    for residue in structure.get_residues():
        if residue.resname.strip().upper() in ligand_resnames:
            nucleotide_atoms.extend(list(residue.get_atoms()))

    if not nucleotide_atoms:
        raise ValueError("No se encontró nucleótido en la estructura.")

    nucleotide_coords = np.array([a.coord for a in nucleotide_atoms])

    # Añadimos solo el MG que esté cerca del nucleótido
    mg_atoms = []
    for residue in structure.get_residues():

        if residue.resname.strip().upper() == cofactor:
            for atom in residue.get_atoms():
                # Distancia mínima de este átomo MG a cualquier átomo del nucleótido
                dists = np.linalg.norm(nucleotide_coords - atom.coord, axis=1)
                if dists.min() <= max_mg_distance:
                    mg_atoms.append(atom)  # solo el MG funcional

    atoms = nucleotide_atoms + mg_atoms

    if not atoms:
        raise ValueError("No se encontraron átomos del nucleótido ligando ni del cofactor Mg.")

    return atoms


def compute_sasa(structure, chain_id="A", relative=True):
    """Calcula la Superficie Accesible al Solvente (SASA) por residuo.

    Utiliza el algoritmo Shrake-Rupley implementado en Biopython.

    Parameters
    ----------
    structure : Bio.PDB.Model.Model
        Modelo estructural de Biopython.
    chain_id : str
        Identificador de la cadena a analizar.
    relative : bool
        Si True, calcula SASA relativa normalizada con la escala
        de Tien et al. 2013.

    Returns
    -------
    pandas.DataFrame
        Columnas:
        - position
        - residue
        - sasa_abs
        - sasa_rel
    """

    result = _compute_sasa_shrake_rupley(
        structure,
        chain_id=chain_id,
        relative=relative,
    )

    return result, "Shrake-Rupley"


def distance_to_active_site(structure, active_site_atoms, chain_id):
    """Calcula la distancia mínima de cada residuo al sitio activo de RAS.

    Para cada residuo estándar de la cadena indicada, se calcula la distancia
    mínima en Å entre cualquiera de sus átomos pesados (no hidrógenos) y
    cualquiera de los átomos pesados del sitio activo (nucleótido + Mg²⁺).

    Se usa la **distancia mínima por pares** (en lugar de la distancia entre
    centroides de masas) porque es la métrica biológicamente más relevante para
    determinar si un residuo está en contacto directo con el ligando:
      - Dos átomos a <3,5 Å pueden formar enlaces de hidrógeno.
      - Átomos a <4 Å están en contacto de Van der Waals.
      - La distancia mínima captura el extremo más próximo de la cadena lateral.

    Los hidrógenos se excluyen porque suelen estar ausentes en estructuras de
    rayos X de resolución media/baja (no se refinen en la densidad electrónica)
    y porque sus posiciones son menos precisas que las de los átomos pesados.

    La distancia se calcula vectorialmente usando broadcasting de NumPy para
    evitar un triple bucle Python explícito (ver notas en el código).

    Parameters
    ----------
    structure : Bio.PDB.Model.Model
        Modelo estructural (devuelto por 'parse_structure').
    active_site_atoms : list of Bio.PDB.Atom.Atom
        Átomos del sitio activo (devueltos por 'get_active_site_atoms').
    chain_id : str, optional
        Identificador de la cadena proteica a analizar (por defecto "A").

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas: "position", "residue", "min_distance_angstrom".
        Una fila por residuo estándar de la cadena; NaN si el residuo no tiene
        átomos pesados.

    Raises
    ------
    ValueError
        Si 'active_site_atoms' no contiene ningún átomo pesado.
    KeyError
        Si 'chain_id' no existe en la estructura.

    Examples
    --------
    >>> model  = parse_structure("data/raw/pdb/4OBE.pdb")
    >>> atoms  = get_active_site_atoms(model)
    >>> dist_df = distance_to_active_site(model, atoms)
    >>> print(dist_df[dist_df.position == 12])
       position residue  min_distance_angstrom
    ...       12       G              3.21...
    """
    # Obtenemos la cadena proteica de interés usando la función auxiliar
    # privada '_get_chain', que lanza un KeyError descriptivo si no existe.
    chain = _get_chain(structure, chain_id)  # objeto Chain de Biopython

    # Construimos un array NumPy 2D de forma (N_activo, 3) con las coordenadas
    # (x, y, z) en Å de todos los átomos pesados del sitio activo.
    #
    # 'getattr(atom, "element", "").upper() != "H"' filtra los hidrógenos:
    #   - 'getattr(atom, "element", "")' accede al atributo 'element' del
    #     átomo de forma segura (devuelve "" si el atributo no existe, evitando
    #     AttributeError en estructuras con formato incompleto).
    #   - '.upper()' normaliza el símbolo del elemento a mayúsculas.
    #   - '!= "H"' descarta el hidrógeno (símbolo "H").
    active_coords = np.asarray(
        [atom.coord for atom in active_site_atoms      # coordenadas (x,y,z) de cada átomo
         if getattr(atom, "element", "").upper() != "H"],  # solo átomos pesados (no H)
        dtype=float,  # aseguramos tipo de dato float64 para precisión numérica
    )  # resultado: array de forma (N_activo, 3)

    # Verificamos que hay al menos un átomo pesado en el sitio activo.
    if active_coords.size == 0:  # array vacío -> ningún átomo pesado encontrado
        raise ValueError("active_site_atoms no contiene átomos pesados.")

    # Lista donde acumularemos las filas del DataFrame de resultados.
    rows = []  # cada elemento será un dict con position, residue y min_distance

    # Iteramos sobre los residuos de la cadena proteica.
    for residue in chain:  # cada 'residue' es un objeto Residue de Biopython

        # Descartamos residuos no estándar (agua, iones, ligandos, residuos
        # con marca de inserción) usando la función auxiliar '_is_standard_residue'.
        if not _is_standard_residue(residue):
            continue  # saltamos a la siguiente iteración del bucle

        # Construimos un array (N_residuo, 3) con las coordenadas de los
        # átomos pesados del residuo actual, usando el mismo filtro de
        # hidrógenos que se aplicó al sitio activo.
        heavy_coords = np.asarray(
            [atom.coord for atom in residue.get_atoms()        # coordenadas de cada átomo
             if getattr(atom, "element", "").upper() != "H"],  # solo átomos pesados
            dtype=float,  # tipo float64 para cálculos de distancia precisos
        )  # resultado: array de forma (N_residuo, 3) o (0, 3) si no hay átomos pesados

        # Si el residuo no tiene átomos pesados (caso patológico pero posible
        # en estructuras con errores de formato), registramos NaN.
        if heavy_coords.size == 0:  # array vacío -> no hay átomos pesados en el residuo
            min_distance = np.nan  # NaN indica valor ausente en pandas/numpy
        else:
            # Calculamos la matriz de distancias por pares usando broadcasting
            # de NumPy (evita el triple bucle explícito y es mucho más rápido):
            #
            # 'heavy_coords[:, np.newaxis, :]'  -> forma (N_residuo, 1, 3)
            # 'active_coords[np.newaxis, :, :]' -> forma (1, N_activo, 3)
            #
            # 'np.newaxis' es equivalente a 'None' y sirve para insertar un
            # nuevo eje (dimensión de tamaño 1) en la posición indicada.  Esto
            # es la forma estándar de preparar arrays para broadcasting 2D.
            # NumPy expande automáticamente (broadcast) ambos arrays a la forma
            # común (N_residuo, N_activo, 3) antes de restar, produciendo un
            # array de diferencias vectoriales (x, y, z) para cada par de
            # átomos.
            #
            # 'np.linalg.norm(..., axis=2)' calcula la norma euclidiana (módulo
            # del vector diferencia, es decir, la distancia en Å) a lo largo del
            # eje de las coordenadas (dimensión 2, que tiene los 3 valores xyz).
            # El resultado es un array de forma (N_residuo, N_activo) donde cada
            # elemento [i, j] es la distancia entre el átomo i del residuo y el
            # átomo j del sitio activo.
            distances = np.linalg.norm(
                heavy_coords[:, np.newaxis, :] - active_coords[np.newaxis, :, :], axis=2
            )  # array (N_residuo, N_activo) con distancias por pares en Å

            # Tomamos la distancia mínima del array completo: el mínimo sobre
            # todos los pares (i_residuo, j_activo).  'np.min' recorre el array
            # aplanado si no se especifica eje.  Convertimos a float Python
            # con 'float(...)' para evitar que sea un np.float64 en el dict.
            min_distance = float(np.min(distances))  # distancia mínima átomo-átomo en Å

        # Añadimos la fila con los datos de este residuo a la lista.
        rows.append({
            "position": int(residue.id[1]),         # número de secuencia del residuo (ej. 12)
            "residue": _residue_one_letter(residue), # código de 1 letra (ej. "G" para Gly)
            "min_distance_angstrom": min_distance,   # distancia mínima al sitio activo en Å
        })

    # Construimos y devolvemos el DataFrame con todas las filas acumuladas.
    return pd.DataFrame(rows)  # DataFrame (N_residuos, 3)


def shannon_entropy_per_position(msa):
    """Calcula la entropía de Shannon por columna de un alineamiento múltiple.

    La entropía de Shannon mide la incertidumbre (o variabilidad) en la
    distribución de aminoácidos de una columna del MSA.  En el contexto de
    conservación evolutiva:

      - **H = 0 bits**: la columna es completamente conservada (un único
        aminoácido en todas las secuencias).  El residuo es probablemente
        crítico para la estructura o la función.
      - **H > 0 bits**: la columna es variable.  Valores altos indican que
        muchos aminoácidos distintos aparecen con frecuencias similares,
        lo que suele indicar baja importancia funcional o alta tolerancia a
        sustituciones.
      - **H máx ≈ 4,32 bits**: cuando los 20 aminoácidos estándar están
        equidistribuidos (log₂ 20 ≈ 4,32).

    La fórmula es:
        H = -Σᵢ p_i · log₂(p_i)

    donde p_i = frecuencia del aminoácido i en la columna (excluyendo huecos
    "-").  Se usa log₂ para obtener la entropía en bits (en lugar de nats,
    que se obtendrían con log natural).

    Los huecos ("-") se excluyen del cálculo de frecuencias pero se cuentan
    por separado (columna "n_gaps") para detectar regiones con inserción/
    deleción frecuente, que tienen su propia interpretación biológica.

    Parameters
    ----------
    msa : Bio.Align.MultipleSeqAlignment
        Objeto de alineamiento múltiple de secuencias de Biopython.  Puede
        cargarse con 'Bio.AlignIO.read(fichero, "fasta")'.

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas: "msa_position" (1-based), "entropy_bits" y
        "n_gaps".  Una fila por columna del alineamiento.

    Examples
    --------
    >>> from Bio import AlignIO
    >>> msa = AlignIO.read("data/interim/ras_msa.fasta", "fasta")
    >>> entropy_df = shannon_entropy_per_position(msa)
    >>> print(entropy_df[entropy_df.msa_position == 12])
       msa_position  entropy_bits  n_gaps
    ...           12           0.0       0
    """
    # Lista donde acumularemos un dict por cada columna del MSA.
    rows = []  # se llenará con N_columnas diccionarios

    # 'msa.get_alignment_length()' devuelve el número de columnas del
    # alineamiento (longitud de todas las secuencias, que es igual para todas
    # porque el MSA rellena con huecos "-" para igualar longitudes).
    for column_index in range(msa.get_alignment_length()):  # itera de 0 a L-1

        # Extraemos la columna actual: para cada secuencia del alineamiento
        # tomamos el carácter en la posición 'column_index' y lo convertimos
        # a mayúsculas para que "a" y "A" se cuenten como el mismo aminoácido.
        # 'record.seq' es el objeto Seq de Biopython; '[column_index]' devuelve
        # el carácter en esa posición como cadena de 1 carácter.
        column = [str(record.seq[column_index]).upper() for record in msa]
        # column es una lista de tantos caracteres como secuencias hay en el MSA

        # Contamos los huecos ("-") en la columna.  'list.count("-")' devuelve
        # el número de apariciones del carácter "-" en la lista.
        n_gaps = column.count("-")  # número de secuencias con hueco en esta posición

        # Filtramos los huecos para calcular la distribución de aminoácidos
        # solo sobre las secuencias que tienen un residuo real en esta posición.
        residues = [aa for aa in column if aa != "-"]  # lista de aminoácidos sin huecos

        # Calculamos la entropía solo si hay al menos un residuo no hueco.
        if residues:  # lista no vacía -> hay al menos una secuencia sin hueco

            # 'Counter(residues)' crea un diccionario {aminoácido: frecuencia_absoluta}.
            # Por ejemplo, Counter(["G", "G", "A"]) -> Counter({"G": 2, "A": 1}).
            counts = Counter(residues)  # frecuencias absolutas de cada aminoácido

            # Número total de residuos no hueco (denominador para las frecuencias
            # relativas).
            total = sum(counts.values())  # suma de todas las cuentas absolutas

            # Calculamos la entropía de Shannon en bits con la fórmula:
            #   H = -Σᵢ p_i · log₂(p_i)
            # donde p_i = count_i / total.
            #
            # 'math.log2(x)' calcula el logaritmo en base 2 de x.  Se usa log₂
            # (en lugar de math.log para logaritmo natural) porque el bit es la
            # unidad natural de información para procesos binarios/discretos y
            # es la convención estándar en bioinformática de secuencias.
            #
            # La expresión generadora '(count/total)*math.log2(count/total)
            # for count in counts.values()' produce un valor por aminoácido;
            # 'sum(...)' los acumula y el '-' delante invierte el signo para
            # obtener un valor positivo (los logaritmos de fracciones son
            # negativos).
            entropy = -sum(
                (count / total) * math.log2(count / total)  # p_i · log₂(p_i)
                for count in counts.values()                 # itera sobre frecuencias absolutas
            )  # resultado en bits; 0 si un solo aminoácido, máx ≈ 4,32 si 20 equiprobables

        else:
            # La columna está formada únicamente por huecos (región de
            # inserción/deleción sin ningún residuo): la entropía no está
            # definida y se registra como NaN.
            entropy = np.nan  # valor ausente estándar de NumPy/pandas

        # Añadimos la fila para esta columna del MSA.
        # Nota: 'msa_position' es 1-based (column_index es 0-based) para
        # ser consistente con las convenciones de biología molecular donde
        # las posiciones de secuencia se cuentan desde 1.
        rows.append({
            "msa_position": column_index + 1,  # posición 1-based en el MSA
            "entropy_bits": entropy,            # entropía de Shannon en bits
            "n_gaps": n_gaps,                   # número de secuencias con hueco en esta posición
        })

    # Construimos y devolvemos el DataFrame completo.
    return pd.DataFrame(rows)  # DataFrame (N_columnas, 3)


def merge_features(mutations_df, sasa_df, distance_df, conservation_df, position_map=None):
    """Construye la tabla maestra de features combinando todas las fuentes de datos.

    La tabla maestra es el DataFrame central del TFM: une en una sola tabla
    plana, para cada combinación (gen, posición), los datos de:
      - Mutaciones clínicas: base COSMIC (somatic mutations), número de
        muestras tumorales y mutación más frecuente.
      - Estructura 3D: SASA absoluto y relativo, distancia al sitio activo.
      - Conservación evolutiva: entropía de Shannon del MSA.
      - Criterio de recurrencia pan-RAS: si la posición está mutada en ≥2
        de los 3 parálogos (KRAS, HRAS, NRAS) con ≥10 muestras tumorales.

    El DataFrame resultante tiene el esquema definido en MASTER_COLUMNS y
    cubre todas las posiciones del dominio G (residuos 1–DOMAIN_G_END=166)
    para los tres genes RAS.

    Parameters
    ----------
    mutations_df : pandas.DataFrame
        DataFrame con las mutaciones clínicas.  Debe tener columnas: "gene",
        "position", "sample_count", "hgvs_p".
    sasa_df : pandas.DataFrame
        DataFrame con el SASA por posición y gen.  Columnas: "gene",
        "position", "sasa_abs", "sasa_rel".
    distance_df : pandas.DataFrame
        DataFrame con distancias al sitio activo.  Columnas: "gene",
        "position", "min_distance_angstrom".
    conservation_df : pandas.DataFrame
        DataFrame con entropías por columna del MSA (devuelto por
        'shannon_entropy_per_position').  Columnas: "msa_position",
        "entropy_bits".
    position_map : pandas.DataFrame or None
        DataFrame de mapeo entre posición UniProt (por gen) y posición MSA.
        Columnas: "msa_position", "kras_position", "kras_aa",
        "hras_position", "hras_aa", "nras_position", "nras_aa".
        Obligatorio; lanza ValueError si es None.

    Returns
    -------
    pandas.DataFrame
        Tabla maestra con columnas MASTER_COLUMNS, una fila por (gen, posición)
        con posición presente en 'position_map'.

    Raises
    ------
    ValueError
        Si 'position_map' es None.

    Examples
    --------
    >>> master_df = merge_features(mut_df, sasa_df, dist_df, cons_df, pos_map)
    >>> print(master_df.shape)
    (498, 12)  # 166 posiciones × 3 genes = 498 filas aproximadamente
    """
    # Verificamos que se ha proporcionado el mapa de posiciones, ya que es
    # imprescindible para alinear las coordenadas UniProt con el MSA.
    if position_map is None:  # el valor centinela None indica que no se pasó
        raise ValueError("position_map es necesario para construir la tabla maestra.")

    # Construimos los diccionarios de búsqueda rápida (lookup tables) para
    # evitar buscar en el DataFrame de mutaciones en cada iteración del bucle.
    # Esto reduce la complejidad de O(N²) a O(N).
    recurrent_lookup = _recurrent_lookup(mutations_df)  # dict {posicion: {recurrent, n_members}}
    mutation_lookup = _mutation_lookup(mutations_df)     # dict {(gen, posicion): {total_samples, top_mutation}}

    # Lista donde acumularemos las filas de la tabla maestra.
    rows = []  # se añadirá un dict por cada (gen, posición) procesado

    # Iteramos sobre los tres parálogos RAS y sobre cada posición del dominio G.
    for gene in GENES:  # "KRAS", "HRAS", "NRAS"
        for state in ["GTP", "GDP"]:

            # Nombre del gen en minúsculas, usado como prefijo de columna en
            # 'position_map' (ej. "kras_position", "kras_aa").
            gene_lower = gene.lower()  # "KRAS" -> "kras", "HRAS" -> "hras", etc.

            # Iteramos sobre todas las posiciones del dominio G (1 a 166 inclusive).
            for uniprot_position in range(1, DOMAIN_G_END + 1):  # 1, 2, ..., 166

                # --- Buscar la fila de mapeo de posición -------------------------

                # Filtramos el DataFrame de mapeo para encontrar la fila(s) donde
                # la posición UniProt de este gen coincide con 'uniprot_position'.
                # '.eq(x)' es equivalente a '== x' pero funciona con NaN de forma
                # predecible (NaN.eq(NaN) es False, == devuelve NaN).
                map_rows = position_map.loc[
                    position_map[f"{gene_lower}_position"].eq(uniprot_position)
                ]  # DataFrame filtrado; puede tener 0, 1 o más filas

                # Si no hay fila para esta posición y gen (puede faltar en el mapa
                # si la posición está en una región sin alinear), la omitimos.
                if map_rows.empty:  # sin filas -> posición no mapeada
                    continue  # saltamos al siguiente valor de uniprot_position

                # Tomamos la primera fila del resultado como Serie de pandas.
                # '.iloc[0]' accede por posición entera (0 = primera fila) sin
                # importar el índice del DataFrame original.
                map_row = map_rows.iloc[0]  # Serie con los datos de mapeo de esta posición

                # Extraemos la posición MSA y el aminoácido wild-type de este gen.
                msa_position = int(map_row["msa_position"])        # posición en el MSA (int)
                aa_wt = map_row[f"{gene_lower}_aa"]                # aminoácido wild-type en código 1 letra

                # --- Buscar información de mutaciones ----------------------------

                # El lookup devuelve un dict vacío {} si no hay mutaciones en esta
                # posición y gen, lo que hace que .get() devuelva los valores por
                # defecto en las líneas siguientes.
                mutation_info = mutation_lookup.get((gene, uniprot_position), {})
                # mutation_info contiene {"total_samples": int, "top_mutation": str} o {}

                # --- Buscar SASA y distancia al sitio activo ---------------------

                # Función auxiliar que filtra el DataFrame y devuelve un dict con
                # los valores de la primera fila coincidente, o {} si no hay.
                sasa_info = _single_row_lookup(
                    sasa_df,
                    gene=gene,
                    state=state,
                    position=uniprot_position,
                )

                distance_info = _single_row_lookup(
                    distance_df,
                    gene=gene,
                    state=state,
                    position=uniprot_position,
                )
                
                # --- Buscar entropía de conservación del MSA ---------------------

                # Filtramos el DataFrame de conservación por posición MSA.
                conservation_rows = conservation_df.loc[
                    conservation_df["msa_position"].eq(msa_position)
                ]  # DataFrame filtrado; debería tener 0 o 1 fila

                # Extraemos la entropía si existe; de lo contrario usamos NaN.
                entropy = (
                    float(conservation_rows.iloc[0]["entropy_bits"])  # valor de la primera fila
                    if not conservation_rows.empty                     # solo si hay filas
                    else np.nan                                        # NaN si no hay datos
                )

                # --- Buscar criterio de recurrencia pan-RAS ----------------------

                # El criterio de recurrencia es a nivel de posición (no de gen):
                # buscamos por posición UniProt sin distinguir gen.
                rec_info = recurrent_lookup.get(uniprot_position, {})
                # rec_info contiene {"recurrent": bool, "n_members_mutated": int} o {}

                # --- Construir la fila de la tabla maestra -----------------------

                # Añadimos un dict con todos los features de esta (gen, posición).
                rows.append({                                         # estado GTP o GDP
                    "gene": gene, 
                    "state": state,                                            # identificador del gen
                    "uniprot_position": uniprot_position,                    # posición en UniProt (1–166)
                    "msa_position": msa_position,                            # posición en el MSA
                    "aa_wt": aa_wt,                                          # aminoácido wild-type
                    "total_samples": int(mutation_info.get("total_samples", 0)),  # total muestras mutadas (0 si ninguna)
                    "top_mutation": mutation_info.get("top_mutation", ""),   # mutación más frecuente (cadena vacía si ninguna)
                    "sasa_abs": sasa_info.get("sasa_abs", np.nan),           # SASA absoluto en Å² (NaN si no hay PDB)
                    "sasa_rel": sasa_info.get("sasa_rel", np.nan),           # SASA relativo 0-1 (NaN si no hay PDB)
                    "dist_active_site_angstrom": distance_info.get("min_distance_angstrom", np.nan),  # distancia al sitio activo en Å
                    "shannon_entropy": entropy,                              # entropía de Shannon en bits
                    "recurrent": bool(rec_info.get("recurrent", False)),     # True si mutación pan-RAS recurrente
                    "n_members_mutated": int(rec_info.get("n_members_mutated", 0)),  # número de parálogos con mutación
                })
    
    
    # Construimos el DataFrame final con las columnas en el orden canónico
    # definido por MASTER_COLUMNS.  Pasar 'columns=MASTER_COLUMNS' garantiza
    # el orden de columnas aunque el dict de cada fila las tenga en otro orden.
    return pd.DataFrame(rows, columns=MASTER_COLUMNS)  # tabla maestra (N_filas, 12)


def plot_features_overview(master_df, output_path, state=None):
    """Genera y guarda un panel de cuatro subgráficos de la tabla maestra.

    Produce una figura de alta resolución (300 dpi) con cuatro paneles
    apilados verticalmente, compartiendo el eje X (posición MSA):

    1. SASA relativo por posición MSA (línea por gen).
    2. Distancia al sitio activo en Å (línea por gen).
    3. Entropía de Shannon en bits (barras en gris, una por posición MSA).
    4. log₁₀(muestras tumorales + 1) (barras apiladas/solapadas por gen).

    Las regiones de los lazos Switch I (10–17), P-loop (30–38) y Switch II
    (60–76) se sombrean con una capa gris translúcida.  Las posiciones
    hotspot canónicas (12, 13, 61) se marcan con líneas verticales punteadas.

    Parameters
    ----------
    master_df : pandas.DataFrame
        Tabla maestra devuelta por 'merge_features'.
    output_path : str or Path
        Ruta donde se guardará la figura (ej. "results/figures/overview.png").
        El directorio padre se crea automáticamente si no existe.
    state : str or None
        Si se especifica ("GTP" o "GDP"), filtra la tabla maestra a ese
        estado antes de graficar. Si es None, se usan ambos estados.

    Returns
    -------
    Path
        Ruta al fichero de imagen guardado.

    Examples
    --------
    >>> out = plot_features_overview(master_df, "results/figures/overview.png")
    >>> print(out)
    results/figures/overview.png
    """
    # Convertimos la ruta de salida a objeto Path para poder usar
    # '.parent.mkdir()' de forma portable.
    output_path = Path(output_path)  # ej. "results/figures/overview.png"

    # Creamos el directorio padre si no existe (y todos los intermedios).
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if state is not None:
        master_df = master_df.loc[master_df["state"] == state].copy()

    # Creamos la figura con 4 subplots en columna.  'sharex=True' hace que
    # todos los ejes compartan el mismo eje X, de forma que al hacer zoom en
    # uno se ajustan todos.  'figsize=(13, 9)' define el tamaño en pulgadas.
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)

    # Paleta de colores por gen: se usa la paleta de Vega-Lite para que los
    # colores sean perceptualmente distinguibles y reproducibles entre figuras.
    colors = {"KRAS": "#4C78A8", "HRAS": "#F58518", "NRAS": "#54A24B"}

    # --- Trazado de datos por gen --------------------------------------------
    linestyles = {
        "GDP": "--",
        "GTP": "-"
    }

    for gene in GENES:

        # --- Panel 3: muestras tumorales (independiente del estado GTP/GDP) --
        # 'total_samples' no varía entre GTP y GDP para un mismo gen y posición
        # (viene de COSMIC, no de la estructura), así que se calcula UNA sola
        # vez por gen, fuera del bucle de estados, para evitar:
        #   (a) dibujarlo dos veces superpuesto cuando state=None, y
        #   (b) que salga vacío cuando se filtra a un solo estado y ese
        #       estado no coincide con la última iteración del bucle interno.
        gene_samples = (
            master_df[master_df["gene"] == gene]
            .drop_duplicates("msa_position")
            .sort_values("msa_position")
        )

        for current_state in ["GDP", "GTP"]:

            subset = (
                master_df[
                    (master_df["gene"] == gene) &
                    (master_df["state"] == current_state)
                ]
                .sort_values("msa_position")
            )

            axes[0].plot(
                subset["msa_position"],
                subset["sasa_rel"],
                color=colors[gene],
                linestyle=linestyles[current_state],
                label=f"{gene} {current_state}",
            )

            axes[1].plot(
                subset["msa_position"],
                subset["dist_active_site_angstrom"],
                color=colors[gene],
                linestyle=linestyles[current_state],  # FIX: antes era linestyles[state]
            )

        # Panel 3: log₁₀(muestras + 1).  Se usa log porque los recuentos de
        # mutaciones tienen distribuciones muy sesgadas (G12D en KRAS tiene
        # miles de muestras vs. pocas en otras posiciones).  Se suma 1 antes
        # del log para evitar log(0) cuando no hay muestras ('total_samples=0').
        # 'alpha=0.35' hace las barras semitransparentes para que se vean las
        # de los tres genes solapadas.
        # FIX: se dibuja una sola vez por gen (fuera del bucle de estados),
        # usando 'gene_samples' en vez de 'subset'.
        axes[3].bar(
            gene_samples["msa_position"],
            np.log10(gene_samples["total_samples"] + 1),
            alpha=0.35,
            label=gene,
            color=colors[gene],
        )

    # --- Panel 2: entropía de Shannon (independiente del gen) ----------------

    # La entropía del MSA es la misma para los tres genes en una posición dada
    # (ya que se calcula a partir del MSA multi-gen compartido), por lo que
    # eliminamos duplicados de msa_position antes de graficar.
    # '.drop_duplicates("msa_position")' conserva solo la primera fila de cada
    # valor de msa_position; '.sort_values("msa_position")' ordena por posición
    # para que las barras aparezcan de izquierda a derecha.
    entropy = master_df.drop_duplicates("msa_position").sort_values("msa_position")

    # Panel 2: barras de entropía en gris neutro (no codificadas por gen).
    axes[2].bar(entropy["msa_position"], entropy["shannon_entropy"], color="#9D9D9D")

    # --- Etiquetas de los ejes Y ---------------------------------------------
    axes[0].set_ylabel("SASA rel.")                   # fracción de superficie expuesta
    axes[1].set_ylabel("Dist. sitio activo (Å)")      # distancia mínima en ångströms
    axes[2].set_ylabel("Entropía (bits)")             # entropía de Shannon en bits
    axes[3].set_ylabel("log10(muestras+1)")           # escala logarítmica de muestras tumorales

    # Etiqueta del eje X solo en el panel inferior (los demás comparten el eje).
    axes[3].set_xlabel("Posición MSA")

    # --- Elementos de anotación biológica ------------------------------------
    for ax in axes:  # aplica a los cuatro paneles

        # Sombreado gris claro (alpha=0.05) para regiones funcionalmente
        # importantes del dominio G de RAS:
        # P-loop (motivo G1, residuos 10-17)
        # Switch I (residuos 30-38)
        # Switch II (residuos 60-76)
        # NKxD (residuos 116-119)
        # SAK (residuos 145-147)

        FUNCTIONAL_REGIONS = {
            "P-loop": (10, 17),
            "Switch I": (30, 38),
            "Switch II": (60, 76),
            "NKxD": (116, 119),
            "SAK": (145, 147),
            }

        for region, (start, end) in FUNCTIONAL_REGIONS.items():

            # Región sombreada
            ax.axvspan(start, end, color="black", alpha=0.05)

            # Etiqueta de la región
            ax.text(
                (start + end)/2,
                0.98,
                region,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="dimgray",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5)
                )

        # Líneas verticales punteadas en los hotspots oncogénicos canónicos
        for position in (12, 13, 61, 146, 117, 59):

            ax.axvline(position,
                    color="black",
                    linewidth=0.8,
                    linestyle="--")

            ax.text(
                position,
                0.03,                     # 3 % desde abajo
                str(position),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
                bbox=dict(facecolor="white",
                        alpha=0.6,
                        edgecolor="none",
                        pad=0.2)
            )

        # Cuadrícula gris muy suave para facilitar la lectura de valores.
        ax.grid(alpha=0.2)

    # Leyenda en el panel 0 con los tres genes, en 3 columnas para ahorrar espacio.
    axes[0].legend(ncol=3, loc="upper right")

    # 'tight_layout()' ajusta automáticamente los márgenes para evitar que las
    # etiquetas de los ejes se superpongan entre paneles.
    if state is None:
        fig.suptitle("RAS - Todos los estados", fontsize=15)
    else:
        fig.suptitle(f"RAS - Estado {state}", fontsize=15)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Guardamos la figura en el fichero de salida a 300 dpi (resolución
    # adecuada para publicación científica).  Matplotlib infiere el formato
    # (PNG, PDF, SVG…) a partir de la extensión del fichero.
    fig.savefig(output_path, dpi=300)

    # Cerramos la figura para liberar memoria.  Sin esto, matplotlib acumula
    # figuras en memoria si la función se llama múltiples veces.
    plt.close(fig)

    # Devolvemos la ruta al fichero generado para que el llamante pueda
    # registrarla en un log o pasarla a otro proceso.
    return output_path  # ruta al fichero de imagen guardado


# ===========================================================================
# Funciones privadas (auxiliares, no forman parte de la API pública)
# ===========================================================================

def _compute_sasa_shrake_rupley(structure, chain_id, relative):
    """Calcula el SASA usando el algoritmo de Shrake-Rupley (Python puro).

    El algoritmo de Shrake-Rupley triangula la superficie de Van der Waals de
    la proteína con una malla uniforme de puntos (sonda esférica rodante) y
    cuenta los puntos expuestos al solvente.  Está implementado en Python puro
    dentro de Biopython y no requiere programas externos.

    Parameters
    ----------
    structure : Bio.PDB.Model.Model
        Modelo estructural de Biopython.
    chain_id : str
        Identificador de la cadena a extraer.
    relative : bool
        Si True, incluye la columna 'sasa_rel' normalizada con Tien 2013.

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas: "position", "residue", "sasa_abs", "sasa_rel".

    Notes
    -----
    Esta función es privada (prefijo '_') y solo se llama desde 'compute_sasa'.
    """
    # Calculamos el SASA para todos los residuos de la estructura al nivel "R"
    # (Residue).  'level="R"' significa que el resultado se acumula por residuo
    # (en lugar de "A" = por átomo, o "C" = por cadena).  El resultado se
    # almacena en el atributo '.sasa' de cada objeto Residue de la estructura.
    ShrakeRupley().compute(structure, level="R")  # añade atributo .sasa a cada Residue

    # Lista donde acumularemos las filas del resultado.
    rows = []

    # Obtenemos la cadena de interés.
    chain = _get_chain(structure, chain_id)  # objeto Chain de Biopython

    # Iteramos sobre los residuos de la cadena.
    for residue in chain:  # cada 'residue' es un objeto Residue de Biopython

        # Descartamos residuos no estándar (agua, ligandos, iones).
        if not _is_standard_residue(residue):
            continue  # saltamos residuos no estándar

        # Obtenemos el código de 1 letra del aminoácido.
        aa = _residue_one_letter(residue)  # ej. "G" para Gly12

        # Leemos el SASA absoluto calculado por Shrake-Rupley y almacenado
        # como atributo '.sasa' del objeto Residue.  'getattr(residue, "sasa",
        # np.nan)' accede de forma segura: si por algún motivo el atributo no
        # existe, devuelve np.nan en lugar de lanzar AttributeError.
        sasa_abs = float(getattr(residue, "sasa", np.nan))  # SASA absoluto en Å²

        # Calculamos el SASA relativo dividiendo por el máximo de Tien 2013.
        max_sasa = TIEN_2013_MAX_SASA.get(aa, np.nan)  # SASA máximo para este aminoácido

        # Solo calculamos el relativo si se solicitó (relative=True) y si
        # tenemos un valor de máximo válido (max_sasa no es np.nan ni 0).
        # La condición 'and max_sasa' es falsa si max_sasa es 0 o np.nan,
        # lo que evita una división por cero.
        sasa_rel = sasa_abs / max_sasa if relative and max_sasa else np.nan
        # sasa_rel es un float entre ~0 (enterrado) y ~1 (expuesto), o NaN

        # Añadimos la fila con los datos de este residuo.
        rows.append({
            "position": int(residue.id[1]),  # número de secuencia del residuo
            "residue": aa,                    # aminoácido en código de 1 letra
            "sasa_abs": sasa_abs,             # SASA absoluto en Å²
            "sasa_rel": sasa_rel,             # SASA relativo (NaN si no aplica)
        })

    # Construimos y devolvemos el DataFrame.
    return pd.DataFrame(rows)  # DataFrame (N_residuos, 4)


def _get_chain(structure, chain_id):
    """Devuelve el objeto Chain de una estructura dado su identificador.

    Parameters
    ----------
    structure : Bio.PDB.Model.Model
        Modelo estructural de Biopython.
    chain_id : str
        Identificador de la cadena (ej. "A").

    Returns
    -------
    Bio.PDB.Chain.Chain
        Objeto Chain de Biopython.

    Raises
    ------
    KeyError
        Con mensaje descriptivo si la cadena no existe, listando las cadenas
        disponibles para facilitar el diagnóstico.

    Notes
    -----
    Esta función es privada (prefijo '_') y se llama desde varias funciones
    del módulo.
    """
    try:
        # En la jerarquía SMCRA de Biopython, los objetos Model, Chain y
        # Residue son de tipo MutableMapping y soportan el operador de
        # suscripción [key] para acceder a sus hijos.  'structure[chain_id]'
        # devuelve el Chain con ese identificador.
        return structure[chain_id]  # devuelve el objeto Chain si existe

    except KeyError as exc:
        # Si la cadena no existe, construimos un mensaje de error que lista
        # las cadenas disponibles para ayudar al usuario a identificar el
        # identificador correcto (ej. la cadena proteica puede ser "B" en
        # estructuras con simetría no cristalográfica).

        # 'structure.get_chains()' es un generador de todos los objetos Chain
        # del modelo; '.id' es el identificador de cadena de un carácter.
        chains = ", ".join(chain.id for chain in structure.get_chains())
        # chains es una cadena como "A, B" o "A" según la estructura

        # Lanzamos la excepción con un mensaje descriptivo que incluye la
        # cadena buscada y las disponibles, encadenando la excepción original
        # con 'from exc' para preservar el traceback completo.
        raise KeyError(
            f"Cadena {chain_id!r} no encontrada. Cadenas disponibles: {chains}"
        ) from exc  # encadena la excepción original para el traceback completo


def _is_standard_residue(residue):
    """Determina si un residuo de Biopython es un aminoácido estándar.

    En los ficheros PDB, el campo heterátomo (hetatm flag) del ID del residuo
    distingue entre:
      - " " (espacio): aminoácido o nucleótido estándar de la cadena principal.
      - "H_XXX": heteroátomo (ligando, cofactor, ión, modificación postrad.).
      - "W": molécula de agua (HOH).

    Para ser considerado "estándar", el residuo debe:
    1. Tener el hetatm flag igual a " " (espacio).
    2. Tener un nombre de tres letras presente en el diccionario THREE_TO_ONE
       (es decir, ser uno de los 20 aminoácidos canónicos).

    Parameters
    ----------
    residue : Bio.PDB.Residue.Residue
        Residuo de Biopython.

    Returns
    -------
    bool
        True si es un aminoácido estándar; False en caso contrario.

    Notes
    -----
    Esta función es privada (prefijo '_').
    """
    # 'residue.id' es una tupla de tres elementos: (hetatm_flag, seq_id, icode)
    # donde:
    #   - residue.id[0]: hetatm flag (" " para estándar, "H_XXX" para hetero,
    #                    "W" para agua)
    #   - residue.id[1]: número de secuencia entero (ej. 12 para Gly12)
    #   - residue.id[2]: código de inserción (ej. " " o "A" en estructuras con
    #                    inserciones numeradas como 34A, 34B)
    return (
        residue.id[0] == " "                              # es aminoácido estándar (no hetero, no agua)
        and residue.resname.strip().upper() in THREE_TO_ONE  # su nombre está en el diccionario de 20 aa
    )


def _residue_one_letter(residue):
    """Convierte el nombre de tres letras de un residuo PDB a código de una letra.

    Parameters
    ----------
    residue : Bio.PDB.Residue.Residue
        Residuo de Biopython.

    Returns
    -------
    str
        Código de una letra del aminoácido (ej. "G" para Gly), o "X" si el
        residuo no está en el diccionario de los 20 aminoácidos estándar
        (convención estándar para aminoácido desconocido/no estándar).

    Notes
    -----
    Esta función es privada (prefijo '_').
    """
    # '.resname' es el nombre de 3 letras del residuo en el PDB (ej. "GLY").
    # '.strip()' elimina espacios (algunos PDBs tienen padding).
    # '.upper()' normaliza a mayúsculas para la búsqueda en THREE_TO_ONE.
    # '.get(..., "X")' devuelve "X" si la clave no está en el diccionario
    # (convenio bioinformático para aminoácido no estándar o desconocido).
    return THREE_TO_ONE.get(residue.resname.strip().upper(), "X")
    # devuelve "G", "A", "Q", etc., o "X" para residuos no estándar


def _mutation_lookup(mutations_df):
    """Construye un diccionario de búsqueda de mutaciones por (gen, posición).

    Para cada combinación (gen, posición), calcula el total de muestras
    tumorales y la mutación más frecuente (la de mayor 'sample_count').

    Parameters
    ----------
    mutations_df : pandas.DataFrame
        DataFrame con columnas: "gene", "position", "sample_count", "hgvs_p".

    Returns
    -------
    dict
        Diccionario de la forma {(gene, position): {"total_samples": int,
        "top_mutation": str}}.

    Notes
    -----
    Esta función es privada (prefijo '_') y se llama desde 'merge_features'.
    """
    # Diccionario resultado que se llenará durante la iteración.
    lookup = {}  # resultado vacío inicial

    # 'mutations_df.groupby(["gene", "position"])' agrupa el DataFrame por las
    # columnas "gene" y "position" y devuelve un GroupBy iterable.
    # Al iterar, cada iteración produce una tupla ((gene, position), group_df)
    # donde group_df es el sub-DataFrame con todas las mutaciones de ese grupo.
    for (gene, position), group in mutations_df.groupby(["gene", "position"]):

        # Ordenamos el grupo por número de muestras de mayor a menor para que
        # la mutación más frecuente quede en la primera fila.
        # 'ascending=False' invierte el orden por defecto (ascendente).
        group = group.sort_values("sample_count", ascending=False)

        # Tomamos la primera fila del grupo ordenado = la mutación más frecuente.
        # '.iloc[0]' accede a la primera fila por posición entera.
        top = group.iloc[0]  # Serie pandas con los datos de la mutación más frecuente

        # Almacenamos en el diccionario la información agregada de este grupo.
        lookup[(str(gene).upper(), int(position))] = {
            "total_samples": int(group["sample_count"].sum()),  # suma de todas las muestras del grupo
            "top_mutation": str(top["hgvs_p"]),                 # notación HGVS de la mutación top
        }

    # Devolvemos el diccionario de búsqueda completo.
    return lookup  # dict {(gene_str, position_int): {...}}


def _recurrent_lookup(mutations_df, min_samples=10):
    """Construye un diccionario de recurrencia pan-RAS por posición.

    Una posición se considera "recurrente" (pan-RAS) si está mutada con
    ≥ min_samples muestras tumorales en ≥ 2 de los 3 parálogos RAS.
    Este criterio identifica posiciones funcionalmente críticas que son
    hotspot en más de un miembro de la familia.

    Parameters
    ----------
    mutations_df : pandas.DataFrame
        DataFrame con columnas: "gene", "position", "sample_count".
    min_samples : int, optional
        Umbral mínimo de muestras tumorales por gen para contar el gen como
        "mutado" en esa posición.  Por defecto 10.

    Returns
    -------
    dict
        Diccionario {position: {"recurrent": bool, "n_members_mutated": int}}.

    Notes
    -----
    Esta función es privada (prefijo '_') y se llama desde 'merge_features'.
    """
    # Diccionario resultado.
    lookup = {}  # resultado vacío inicial

    # Agregamos el total de muestras por (gen, posición).
    # 'groupby(..., as_index=False)' realiza el grupo sin convertir las
    # columnas de agrupación en índice, lo que facilita el acceso posterior.
    # '[..].sum()' suma la columna "sample_count" dentro de cada grupo.
    grouped = mutations_df.groupby(["gene", "position"], as_index=False)["sample_count"].sum()
    # grouped tiene columnas: "gene", "position", "sample_count" (total por gen-posición)

    # Iteramos sobre las posiciones únicas del DataFrame agregado.
    # 'grouped.groupby("position")' produce (position, sub_df) donde sub_df
    # contiene una fila por gen que tiene mutaciones en esa posición.
    for position, group in grouped.groupby("position"):

        # Contamos cuántos miembros de la familia tienen ≥ min_samples muestras
        # en esta posición.  'group["sample_count"] >= min_samples' produce una
        # Serie booleana; '.sum()' cuenta los True (equivalente a contar filas
        # donde la condición se cumple).
        n_members = int((group["sample_count"] >= min_samples).sum())
        # n_members es 0, 1, 2 o 3 según cuántos parálogos tienen mutación significativa

        # Almacenamos si la posición es recurrente (mutada en ≥2 parálogos)
        # y cuántos miembros están afectados.
        lookup[int(position)] = {
            "recurrent": n_members >= 2,     # True si mutación en ≥2 parálogos
            "n_members_mutated": n_members,  # número de parálogos con mutación significativa
        }

    # Devolvemos el diccionario de recurrencia completo.
    return lookup  # dict {position_int: {"recurrent": bool, "n_members_mutated": int}}


def _single_row_lookup(df, gene, state, position):
    """Busca y devuelve como dict la primera fila que coincide con gen, estado y posición."""
    
    subset = df.loc[
        df["gene"].eq(gene)
        & df["state"].eq(state)
        & df["position"].eq(position)
    ]

    if subset.empty:
        return {}

    return subset.iloc[0].to_dict()