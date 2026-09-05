# Contratos de datos

Este documento define los esquemas mínimos que deben producir los notebooks.

## Entrada COSMIC esperada

El loader debe poder trabajar con un TSV que contenga, como mínimo, estas columnas o equivalentes claramente mapeadas:

| Columna | Uso |
| --- | --- |
| `GENE_NAME` | Filtrar KRAS, HRAS y NRAS |
| `Mutation AA` | Cambio proteico tipo `p.G12D` |
| `Mutation somatic status` | Filtrar variantes somáticas confirmadas |
| `Mutation Description`  | Filtrar sustituciones missense |
| `ID_sample` | Deduplicar llamadas por muestra |
| `Primary site`  | Resumen por tejido primario |
| `Histology` | Resumen por histología |
| `COSMIC version`  | Trazabilidad de versión, si existe |

Si la version descargada de COSMIC usa otros nombres, crea un mapeo explícito y documentalo en el notebook 01.

## `data/processed/cosmic_curated.csv`

Columnas obligatorias:

| Columna | Tipo esperado | Descripción |
| --- | --- | --- |
| `gene` | string | `KRAS`, `HRAS` o `NRAS` |
| `uniprot_id` | string | Identificador UniProt canónico |
| `position` | int | Posición aminoacidica UniProt |
| `wt_aa` | string | Residuo wild type, código de una letra |
| `mut_aa` | string | Residuo mutado, código de una letra |
| `hgvs_p` | string | Cambio proteico, por ejemplo `p.G12D` |
| `sample_count` | int | Número de muestras únicas tras deduplicar |
| `tumour_types` | string | Histologías principales concatenadas |
| `primary_tissues` | string | Tejidos primarios principales concatenados |
| `cosmic_version` | string | Versión o etiqueta del dataset |

## `data/processed/position_map.csv`

Columnas obligatorias:

| Columna | Descripción |
| --- | --- |
| `msa_position` | Posición 1-indexed en el MSA |
| `KRAS` | Posición UniProt KRAS o vacio si gap |
| `HRAS` | Posición UniProt HRAS o vacio si gap |
| `NRAS` | Posición UniProt NRAS o vacio si gap |

## `data/processed/recurrent_positions.csv`

Columnas obligatorias:

| Columna | Descripción |
| --- | --- |
| `msa_position` | Posición equivalente en el MSA |
| `members_mutated` | Número de miembros con mutación recurrente |
| `total_sample_count` | Suma de muestras en los miembros incluidos |
| `genes` | Genes que cumplen el umbral |

## `data/processed/master_features.csv`

Columnas obligatorias:

| Columna | Descripción |
| --- | --- |
| `gene` | Gen RAS |
| `position` | Posición UniProt |
| `msa_position` | Posición equivalente en el MSA |
| `sample_count` | Frecuencia agregada de mutación |
| `sasa_rel` | Accesibilidad relativa al solvente |
| `distance_to_active_site` | Distancia mínima al sitio activo en Angstrom |
| `conservation_entropy` | Entropía de Shannon por columna del MSA |
| `is_recurrent` | Booleano o 0/1 segun el criterio definido |
