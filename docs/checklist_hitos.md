# Checklist de hitos

Este checklist sirve para controlar avance técnico.

## Hito 1: curado de datos

- [ ] `00_setup.ipynb` ejecuta sin errores críticos.
- [ ] `01_data_curation.ipynb` carga el TSV de ejemplo.
- [ ] El parser acepta cambios tipo `p.G12D`, `p.G13V`, `p.Q61R`.
- [ ] Los filtros reportan conteos antes y después.
- [ ] La salida `cosmic_curated.csv` cumple `docs/contratos_datos.md`.
- [ ] Los PDBs principales se descargan o se documenta por que no se han podido descargar.

## Hito 2: alineamiento y mapping

- [ ] Las tres secuencias canonicas se obtienen desde UniProt o cache local.
- [ ] El MSA se guarda en `data/processed/`.
- [ ] `position_map.csv` permite localizar G12, G13 y Q61 en los tres genes.
- [ ] Hay tests unitarios para G12, G13 y Q61.
- [ ] `recurrent_positions.csv` usa el umbral declarado en `configs/config.yaml`.

## Hito 3: rasgos estructurales

- [ ] Cada estructura PDB se parsea con Biopython.
- [ ] La definición de sitio activo esta escrita en el notebook.
- [ ] SASA y distancia al sitio activo se calculan por residuo.
- [ ] Los residuos faltantes aparecen como `NaN`, no como valores inventados.
- [ ] `master_features.csv` integra mutaciones, mapping y rasgos estructurales.

## Hito 4: visualización e integración

- [ ] Hay una tabla de mutaciones principales por gen.
- [ ] Hay un heatmap de mutaciones por posición equivalente y gen.
- [ ] Hay un scatter SASA vs distancia al sitio activo.
- [ ] Las vistas 3D resaltan hotspots y estructuras usadas.

