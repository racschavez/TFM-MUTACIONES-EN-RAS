# TFM — Análisis comparativo de mutaciones somáticas en la familia RAS

Pipeline en Python para curar, integrar y analizar estructuralmente mutaciones somáticas de COSMIC v104 en los miembros de la familia de proteínas RAS humana (KRAS, HRAS, NRAS).

> **Estado:** Pipeline completo de análisis con 5 cuadernos ejecutables, notebooks 00-04; para la obtención de datasets curados a nivel mutacional y estructural. 
Incluye gráficos representativos de los datos, y figuras 2D y 3D para la representaciones estructurales.

## Estructura del repositorio

```
.
├── data/
│   ├── raw/           # Datos en bruto
│   ├── example/       # TSV mínimo para desarrollo sin datos de COSMIC reales
│   ├── processed/     # Datos derivados curados
│   └── external/      # PDBs y secuencias FASTA. Archivos descargados
├── notebooks/         # Notebooks numerados 01-04
├── src/tfm_ras/       # Módulos Python reutilizables
├── figures/           # Figuras finales para la memoria
├── configs/           # Ficheros de configuración (YAML)
├── docs/              # Documentación adicional
├── environment.yml    # Entorno conda
├── pyproject.toml     # Configuración del paquete
├── README.md          # Fichero inicial para aclaraciones del pipeline
├── LICENSE            # Licencia de uso para el pipeline
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

1. `01_data_curation.ipynb` — Dataset curado con datos de COSMIC v104
2. `02_alignment_mapping.ipynb` — MSA y mapa posicional
3. `03_structural_features.ipynb` — SASA, distancia al sitio activo, conservación evolutiva
4. `04_results_visualization.ipynb` — Gráficos y figuras finales

## Datos

Los datos en bruto de COSMIC NO están versionados en este repositorio (licencia académica). Deben descargarse del portal oficial de COSMIC con una cuenta autorizada.

Los ficheros de COSMIC se colocan en `data/raw/cosmic/`. El notebook `01_data_curation.ipynb` produce el dataset curado en `data/processed/`.

Para empezar sin esperar a COSMIC, usa:

```text
data/example/cosmic_minimal_example.tsv
```

Ese fichero solo sirve para desarrollar y probar el flujo; no debe usarse para conclusiones biológicas.


## Reproducibilidad

- Todas las semillas aleatorias están fijadas en los notebooks (`numpy`, `random`).
- El entorno completo está congelado en `environment.yml`.
- Los pasos exactos para reproducir cada figura están documentados en cada notebook.

## Licencia

Código: MIT (ver `LICENSE`).
Datos derivados: respetando la licencia académica de COSMIC.

## Cita

Si usas este pipeline en tu trabajo, cita:

> [Chávez Sosa, Rachel A.]. (2026). Análisis comparativo de mutaciones somáticas recurrentes en la familia de proteínas RAS. TFM, Máster en Bioinformática, [Universidad de Nebrija]. DOI: [pendiente, Zenodo]

## Contacto

- Autor/a: [Rachel A. Chávez Sosa] — [rchavezs1@alumnos.nebrija.es]
- Tutor/a: [Álvaro Serrano Navarro] — [aserrann@nebrija.es]
