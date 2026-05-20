# Recursos tecnicos

## Familia RAS

| Gen | UniProt canonico | Nota |
| --- | --- | --- |
| KRAS | `P01116-1` | Isoforma 4B como referencia principal |
| HRAS | `P01112` | Secuencia canonica humana |
| NRAS | `P01111` | Secuencia canonica humana |

## Dominio de analisis

El analisis se restringe al dominio G, aproximadamente residuos 1-166 en numeracion UniProt. La region hipervariable C-terminal queda fuera del alcance principal porque suele estar ausente en estructuras experimentales y difiere entre isoformas.

## Hotspots de referencia

| Posicion | Residuo WT | Contexto |
| --- | --- | --- |
| 12 | G | P-loop, hotspot oncogenico clasico |
| 13 | G | P-loop, hotspot oncogenico clasico |
| 61 | Q | Switch II, residuo catalitico clave |

Estos hotspots sirven como controles biologicos y tecnicos. Si no aparecen tras el curado y el mapping, revisa filtros, nomenclatura y equivalencias de posicion.

## Estructuras PDB sugeridas

| Gen | Principal | Ligando esperado | Secundaria |
| --- | --- | --- | --- |
| KRAS | `4OBE` | GDP | `5UK9` |
| HRAS | `3K8Y` | analogo de GTP | `2RGE` |
| NRAS | `5UHV` | analogo de GTP | `3CON` |

Las estructuras pueden tener residuos faltantes. No imputes esos valores: represéntalos como `NaN` y documenta la limitacion.

## Sitio activo

Para el calculo estructural, define el sitio activo de forma reproducible. Dos opciones aceptables:

- atomos del nucleotido (`GDP`, `GTP`, `GNP` u otros analogos presentes) y el ion magnesio;
- residuos de P-loop, Switch I y Switch II que contacten el ligando por debajo de un umbral definido.

Elige una definicion, justificala y usala de forma consistente.

## Umbral inicial de recurrencia

Usa como criterio inicial:

- posicion mutada en al menos 2 de los 3 genes;
- minimo 10 muestras por gen afectado.

Si los datos reales aconsejan cambiar el umbral, documenta el cambio y muestra como afecta al ranking.
