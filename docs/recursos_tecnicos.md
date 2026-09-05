# Recursos técnicos

## Familia RAS

| Gen | UniProt canónico | Nota |
| --- | --- | --- |
| KRAS | `P01116-1` | Isoforma 4B como referencia principal |
| HRAS | `P01112` | Secuencia canónica humana |
| NRAS | `P01111` | Secuencia canónica humana |

## Dominio de análisis

El análisis se restringe al dominio G, aproximadamente residuos 1-166 en numeración UniProt. La región hipervariable C-terminal queda fuera del alcance principal porque suele estar ausente en estructuras experimentales y difiere entre isoformas.

## Hotspots de referencia

| Posición | Residuo WT | Contexto |
| --- | --- | --- |
| 12 | G | P-loop, hotspot oncogénico clásico |
| 13 | G | P-loop, hotspot oncogénico clásico |
| 61 | Q | Switch II, residuo catalítico clave |

Estos hotspots sirven como controles biológicos y técnicos. Si no aparecen tras el curado y el mapping, revisa filtros, nomenclatura y equivalencias de posición.

## Estructuras PDB sugeridas

| Gen | GTP | Ligando esperado | GDP |
| --- | --- | --- | --- |
| KRAS | `5UK9` | GCP | `4OBE` |
| HRAS | `3K8Y` | GNP | `4Q21` |
| NRAS | `5UHV` | GNP | `3CON` |

Las estructuras pueden tener residuos faltantes. Estos valores serán representados como `NaN`.

## Sitio activo

Para el calculo estructural, definimos el sitio activo como el conjunto de átomos del ligando (GDP, GTP, o análogos no hidrolizables como GNP/GppNHp) más el ion Mg²⁺ coordinado. 
Las distancias de cada residuo se calculan como la distancia mínima a cualquier átomo de ese conjunto.


## Umbral inicial de recurrencia

Como umbral de recurrencia de mutaciones se estableció:

- posición mutada en al menos 2 de los 3 genes;
- mínimo 10 muestras por gen afectado.


