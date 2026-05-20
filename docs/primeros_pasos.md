# Primeros pasos

Este repositorio es una plantilla de trabajo, no una solucion completa. El objetivo de la primera semana es dejar el entorno funcionando, entender el flujo de datos y producir el primer CSV curado.

## 1. Preparar el entorno

```bash
conda env create -f environment.yml
conda activate tfm-ras
pip install -e .
```

Comprueba que Python encuentra el paquete:

```bash
python -c "import tfm_ras; print(tfm_ras.__version__)"
```

## 2. Abrir el notebook de verificacion

```bash
jupyter lab notebooks/00_setup.ipynb
```

Ejecuta todas las celdas. Si falla una conexion externa, anota el error exacto y continua con los datos de ejemplo siempre que el entorno Python funcione.

## 3. Trabajar primero con el ejemplo minimo

Antes de usar COSMIC real, usa `data/example/cosmic_minimal_example.tsv` para implementar y probar el curado basico:

- carga TSV;
- filtrado de mutaciones somaticas confirmadas;
- filtrado de missense;
- deduplicacion por muestra y cambio aminoacidico;
- parseo de cambios tipo `p.G12D`;
- salida con el esquema descrito en `docs/contratos_datos.md`.

## 4. Incorporar COSMIC real

Cuando tengas acceso autorizado a COSMIC, coloca el fichero en:

```text
data/raw/cosmic/CosmicMutantCensus.tsv
```

No subas nunca el fichero bruto a Git. Anota en la memoria la version exacta de COSMIC y la fecha de descarga.

## 5. Orden de trabajo recomendado

1. `00_setup.ipynb`: entorno, rutas y acceso a servicios externos.
2. `01_data_curation.ipynb`: curado de mutaciones y PDBs.
3. `02_alignment_mapping.ipynb`: secuencias, MSA y equivalencias de posicion.
4. `03_structural_features.ipynb`: SASA, distancia al sitio activo y conservacion.
5. `04_results_visualization.ipynb`: tablas finales, figuras y vistas 3D.

## 6. Como pedir ayuda

Cuando comuniques un bloqueo, incluye:

- comando o celda ejecutada;
- error completo;
- fichero de entrada usado;
- resultado esperado;
- resultado obtenido.
