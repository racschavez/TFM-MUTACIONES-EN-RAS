# TFM — Análisis comparativo de mutaciones somáticas en la familia RAS

Pipeline en Python para curar, integrar y analizar estructuralmente mutaciones somáticas de COSMIC en los miembros de la familia RAS humana (KRAS, HRAS, NRAS).

> **Estado:** plantilla reforzada. El alumno debe completar los módulos y notebooks 01–04 según el documento de encargo. La plantilla incluye datos de ejemplo mínimos, contratos de salida y checklists técnicos, pero no contiene la solución del análisis.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/           # Datos en bruto (NO subir a Git: ver .gitignore)
│   ├── example/       # TSV mínimo para desarrollo sin COSMIC real
│   ├── processed/     # Datos derivados curados
│   └── external/      # PDBs, MSAs, etc. descargados
├── notebooks/         # Notebooks numerados 01-04
├── src/tfm_ras/       # Módulos Python reutilizables
├── tests/             # Tests unitarios (pytest)
├── figures/           # Figuras finales para la memoria
├── configs/           # Ficheros de configuración (YAML)
├── docs/              # Documentación adicional
├── environment.yml    # Entorno conda
├── pyproject.toml     # Configuración del paquete
├── README.md          # Este fichero
└── .gitignore
```

## Cómo arrancar

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd tfm-ras
```

### 2. Crear el entorno conda

```bash
conda env create -f environment.yml
conda activate tfm-ras
```

### 3. Instalar el paquete en modo desarrollo

```bash
pip install -e .
```

### 4. Verificar instalación

```bash
jupyter lab notebooks/00_setup.ipynb
```

Ejecuta todas las celdas: si todas pasan, el entorno está listo.

### 5. Leer la guía de arranque

Antes de tocar COSMIC real, lee:

- `docs/primeros_pasos.md`
- `docs/recursos_tecnicos.md`
- `docs/contratos_datos.md`
- `docs/checklist_hitos.md`

### 6. Ejecutar los notebooks en orden

1. `01_data_curation.ipynb` — Curado de COSMIC y descarga de PDBs
2. `02_alignment_mapping.ipynb` — MSA y tabla de equivalencias
3. `03_structural_features.ipynb` — SASA, distancia al sitio activo, conservación
4. `04_results_visualization.ipynb` — Mapas de calor y vistas 3D

## Datos

Los datos en bruto de COSMIC NO están versionados en este repositorio (licencia académica). Deben descargarse del portal oficial de COSMIC con una cuenta autorizada.

Los ficheros de COSMIC se colocan en `data/raw/cosmic/`. El notebook `01_data_curation.ipynb` produce el dataset curado en `data/processed/`.

Para empezar sin esperar a COSMIC, usa:

```text
data/example/cosmic_minimal_example.tsv
```

Ese fichero solo sirve para desarrollar y probar el flujo; no debe usarse para conclusiones biologicas.

## Tests

```bash
pytest tests/
```

Los tests mínimos empiezan por el parseo de cambios aminoacidicos. El alumno debe ampliarlos para verificar el mapping de hotspots clásicos (G12, G13, Q61) y los contratos de datos.

## Reproducibilidad

- Todas las semillas aleatorias están fijadas en los notebooks (`numpy`, `random`).
- El entorno completo está congelado en `environment.yml`.
- Los pasos exactos para reproducir cada figura están documentados en cada notebook.

## Licencia

Código: MIT (ver `LICENSE`).
Datos derivados: respetando la licencia académica de COSMIC.

## Cita

Si usas este pipeline en tu trabajo, cita:

> [Apellidos, Nombre del alumno]. (2026). Análisis comparativo de mutaciones somáticas recurrentes en la familia RAS. TFM, Máster en Bioinformática, [Universidad]. DOI: [pendiente, Zenodo]

## Contacto

- Autor/a: [Nombre del alumno] — [email institucional]
- Tutor/a: [Nombre del tutor] — [email institucional]
