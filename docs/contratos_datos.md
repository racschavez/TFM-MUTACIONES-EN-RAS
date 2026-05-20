# Contratos de datos

Este documento define los esquemas minimos que deben producir los notebooks. Puedes añadir columnas, pero no eliminar ni cambiar el significado de las columnas obligatorias.

## Entrada COSMIC esperada

El loader debe poder trabajar con un TSV que contenga, como minimo, estas columnas o equivalentes claramente mapeadas:

| Columna | Uso |
| --- | --- |
| `GENE_NAME` | Filtrar KRAS, HRAS y NRAS |
| `Mutation AA` | Cambio proteico tipo `p.G12D` |
| `Mutation somatic status` | Filtrar variantes somaticas confirmadas |
| `Mutation Description` | Filtrar sustituciones missense |
| `ID_sample` | Deduplicar llamadas por muestra |
| `Primary site` | Resumen por tejido primario |
| `Histology` | Resumen por histologia |
| `COSMIC version` | Trazabilidad de version, si existe |

Si la version descargada de COSMIC usa otros nombres, crea un mapeo explicito y documentalo en el notebook 01.

## `data/processed/cosmic_curated.csv`

Columnas obligatorias:

| Columna | Tipo esperado | Descripcion |
| --- | --- | --- |
| `gene` | string | `KRAS`, `HRAS` o `NRAS` |
| `uniprot_id` | string | Identificador UniProt canonico |
| `position` | int | Posicion aminoacidica UniProt |
| `wt_aa` | string | Residuo wild type, codigo de una letra |
| `mut_aa` | string | Residuo mutado, codigo de una letra |
| `hgvs_p` | string | Cambio proteico, por ejemplo `p.G12D` |
| `sample_count` | int | Numero de muestras unicas tras deduplicar |
| `tumour_types` | string | Histologias principales concatenadas |
| `primary_tissues` | string | Tejidos primarios principales concatenados |
| `cosmic_version` | string | Version o etiqueta del dataset |

## `data/processed/position_map.csv`

Columnas obligatorias:

| Columna | Descripcion |
| --- | --- |
| `msa_position` | Posicion 1-indexed en el MSA |
| `KRAS` | Posicion UniProt KRAS o vacio si gap |
| `HRAS` | Posicion UniProt HRAS o vacio si gap |
| `NRAS` | Posicion UniProt NRAS o vacio si gap |

## `data/processed/recurrent_positions.csv`

Columnas obligatorias:

| Columna | Descripcion |
| --- | --- |
| `msa_position` | Posicion equivalente en el MSA |
| `members_mutated` | Numero de miembros con mutacion recurrente |
| `total_sample_count` | Suma de muestras en los miembros incluidos |
| `genes` | Genes que cumplen el umbral |

## `data/processed/master_features.csv`

Columnas obligatorias:

| Columna | Descripcion |
| --- | --- |
| `gene` | Gen RAS |
| `position` | Posicion UniProt |
| `msa_position` | Posicion equivalente en el MSA |
| `sample_count` | Frecuencia agregada de mutacion |
| `sasa_rel` | Accesibilidad relativa al solvente |
| `distance_to_active_site` | Distancia minima al sitio activo en Angstrom |
| `conservation_entropy` | Entropia de Shannon por columna del MSA |
| `is_recurrent` | Booleano o 0/1 segun el criterio definido |
