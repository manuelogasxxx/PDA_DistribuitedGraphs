# Analítica distribuida de grafos IoT — Equipo T800
## Guía de ejecución (pasos 1 a 3)

Estado del proyecto: **datos sintéticos, detector centralizado y particionado con halos: listos.**
Pendiente: workers distribuidos reales, instrumentación, orquestador de experimentos y análisis
estadístico (ver [Qué sigue](#9-qué-sigue)).

---

## 1. Qué contiene el paquete

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `iot_graph_generator.py` | Genera el grafo (nodos, aristas, coordenadas, comunidades de referencia) |
| 1 | `signals.py` | Genera lecturas `X[t,n]`, anomalías, datos faltantes y la verdad de referencia |
| 2 | `detect.py` | Detector de anomalías e imputación, centralizado y diseñado para distribuirse |
| 2 | `metrics.py` | AUC, AP, F1, umbral, MAE/RMSE de imputación |
| 3 | `partition.py` | Particionadores, halos por worker, plan de comunicación, métricas y verificación |
| — | `example_pipeline.py` | Recorrido completo de los pasos 1-2 con calibración de umbral y demo de halo |
| — | `test_*.py` (4 archivos) | 71 tests (70 se ejecutan sin extras; 1 requiere `pymetis`) |
| — | `requirements.txt` | Dependencias con las versiones verificadas |

**Cómo se conectan los módulos**

```
iot_graph_generator ──► signals ──► detect ──► metrics
        │                              ▲
        └──────────► partition ────────┘   (partition usa detect.adjacency_hops/induced_edges;
                                            detect define cuánto halo se necesita)
```

## 2. Instalación

Requiere Python 3.10 o superior (verificado con 3.12). Coloca **los archivos `.py` en una misma
carpeta**, porque se importan entre sí.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Comprobar que todo funciona

```bash
pytest -q                            # esperado: 70 passed, 1 skipped
```

El test omitido (`test_metis_optional`) es normal si no instalaste `pymetis`. Tarda unos 10 segundos.
Para ejecutar solo una parte:

```bash
pytest -q test_partition.py                       # solo particionado
pytest -q -k "centralized_result"                 # solo la prueba de equivalencia con halo
pytest -q -x                                      # detenerse en el primer fallo
```

| Archivo de tests | Casos | Qué protege |
|---|---|---|
| `test_iot_graph_generator.py` | 14 | Determinismo, conectividad, IDs permutados, peso como similitud, calibración de grado |
| `test_signals.py` | 11 | Convenciones de la verdad de referencia, tasas objetivo, escenarios pareados |
| `test_detect.py` | 17 | Métricas con casos calculados a mano, detectores, equivalencia con halo |
| `test_partition.py` | 28 + 1 omitido | Validez y balance, teoría del hash, ausencia de fuga, invariantes del plan, equivalencia por worker |

## 4. Ejecución rápida

### 4.1 Demo completa en un comando

```bash
python example_pipeline.py            # n=3000, unos segundos
python example_pipeline.py --n 10000
```

Genera datos, calibra el umbral en otra semilla, evalúa los tres detectores y muestra cómo un
worker con su porción del grafo iguala al centralizado solo si el halo es suficiente.

### 4.2 Flujo por línea de comandos (4 pasos)

```bash
# 1. Grafo.  --k-neighbors 8 da grado medio ~10
python iot_graph_generator.py --mode hybrid --n 10000 --seed 1 --k-neighbors 8 --out ./data

# 2. Señales sobre ese grafo
python signals.py --graph ./data/hybrid_n10000_seed1 --seed 1 \
    --anomaly both --missing structured

# 3. Detector centralizado + métricas
python detect.py --graph ./data/hybrid_n10000_seed1 \
    --signals ./data/hybrid_n10000_seed1/signals_seed1_both_structured.npz \
    --hops 1 --impute-iters 2 --tau 4.0

# 4. Particiones, halos y estimación de comunicación (verificando los planes)
python partition.py --graph ./data/hybrid_n10000_seed1 \
    --methods centralized hash spatial community community_refined \
    --k 2 4 8 --depth 2 --seed 1 --T 96 --verify --out ./partitions
```

Salida esperada del paso 3 (valores de la verificación; los tuyos deben coincidir):

```
halo requerido: 2 salto(s) | tiempo total: 0.25s
[spatial ] AUC=0.809 AP=0.542 F1=0.635 (tau=4.00, P=0.92, R=0.49)
[temporal] AUC=0.789 AP=0.221 F1=0.202 (tau=4.00, P=0.71, R=0.12)
[combined] AUC=0.845 AP=0.539 F1=0.618 (tau=4.00, P=0.84, R=0.49)
[imputación] MAE=0.461 RMSE=0.656 sobre 48713 lecturas faltantes
```

Salida esperada del paso 4 (extracto, n=10000, halo de 2 saltos):

```
método               k  corte  desbal frontera  replic desbal_ef  MB halo  t_part
hash                 8  0.876   1.053    1.000    7.81      7.88     32.7    0.00
spatial              8  0.021   1.000    0.067    1.14      1.21      0.7    0.00
community            8  0.010   1.051    0.042    1.13      1.18      0.6    0.02
community_refined    8  0.007   1.051    0.037    1.13      1.18      0.6    0.02
```

> Si encadenas un comando con `| head`, puede aparecer `BrokenPipeError`. No es un fallo:
> `head` cierra la tubería antes de que el programa termine de escribir.

### 4.3 Dónde queda cada cosa

```
data/hybrid_n10000_seed1/
    graph.npz                              aristas, pesos, coordenadas, comunidad de referencia
    manifest.json                          parámetros, versiones, hash del grafo, validación
    signals_seed1_both_structured.npz      X, X_clean, labels, mask
    signals_seed1_both_structured.json     parámetros y reporte de las señales
partitions/hybrid_n10000_seed1/
    part_<método>_k<k>_seed<s>.npy         id de worker por nodo
    metrics_seed1_depth2.json              una fila de métricas por (método, k)
```

## 5. Uso como librería

```python
from iot_graph_generator import generate
from signals import generate_signals
from detect import DetectorConfig, run_centralized, adjacency_hops, spatial_scores
from partition import make_plan, verify_plan
from metrics import evaluate

g   = generate("hybrid", n=5000, seed=1, k_neighbors=8)
sig = generate_signals(g, seed=1, anomaly="both", missing="structured")
cfg = DetectorConfig(hops=1, impute_iters=2)            # halo requerido = 2 saltos

# Detector centralizado + métricas
res = run_centralized(g, sig, cfg)
fila = evaluate(res.scores("spatial"), res.X_imputed, sig, tau=4.0)

# Partición + plan por worker, y comprobación de que un worker iguala al centralizado
plan, m = make_plan(g, "community", k=4, depth=cfg.halo_depth, seed=1)
verify_plan(plan, g)
w = plan.workers[0]                                      # w.owned, w.halo, w.nodes, w.edges
Xs, Ms = sig.X[:, w.nodes], sig.mask[:, w.nodes]
z = spatial_scores(Xs, Ms, adjacency_hops(w.n_local, w.edges, cfg.hops), cfg)[:, :w.n_owned]
print(abs(z - res.scores_spatial[:, w.owned]).max())     # ~1e-4: solo ruido de float32
```

Ejemplo de salida: `worker 0: 1196 propios + 114 de halo | max|dif| vs centralizado = 1.5e-04`.

## 6. Experimentos típicos

### 6.1 Repeticiones (el protocolo pide 20-30 semillas)

```bash
python iot_graph_generator.py --mode hybrid --n 10000 --seeds 1-30 --k-neighbors 8 --out ./data
for s in $(seq 1 30); do
  python signals.py --graph ./data/hybrid_n10000_seed$s --seed $s --anomaly both --missing structured
done
```

En Windows usa Git Bash o WSL para el bucle, o hazlo en Python con `generate(...)` y
`generate_signals(...)` dentro de un `for seed in range(1, 31)`.

Cada semilla da un grafo y unas señales distintos; todos los métodos deben compararse **sobre la
misma pareja (grafo, señales)** para poder usar pruebas pareadas.

### 6.2 Parámetros que se variarán

| Qué quieres variar | Parámetro |
|---|---|
| Tamaño de la red | `--n` (1000, 10000, 100000) |
| Fuerza de las comunidades | `--spread-factor` en `hybrid` (0.1 clara … 0.5 solapada); `--mixing` en `community` |
| Dificultad de las anomalías | `magnitude=(min, max)` en `generate_signals` (3 difícil, 10 fácil) |
| Tipo de anomalía / faltantes | `--anomaly` point/regional/both, `--missing` random/structured |
| Profundidad del halo | `DetectorConfig(hops=…, impute_iters=…)` y `--depth` en `partition.py` |
| Método y número de workers | `--methods`, `--k` |
| Tiempo simulado | `--T` (también reduce la memoria en 100k) |

### 6.3 Protocolo del umbral (importante para que la comparación sea justa)

El umbral **no se ajusta por método**:

1. Se calibra una sola vez con el método **centralizado** sobre **otra semilla** (`best_f1_threshold`).
2. Se congela.
3. Se aplica igual a centralizado, hash, comunidades, etc.

`example_pipeline.py` muestra este flujo completo. Sin `--tau`, `detect.py` usa el mejor umbral sobre
los mismos datos, lo cual es optimista y solo sirve para explorar. AUC y AP no dependen del umbral.

## 7. Cómo leer las métricas

**Calidad de detección** (`metrics.py`): AUC y AP (sin umbral); F1, precisión y recall (con umbral).
Un score no calculable cuenta como falso negativo, para que perder información de frontera penalice.
**Imputación:** MAE y RMSE solo sobre lecturas faltantes, contra la lectura normal real.

**Partición** (`partition.py`):

| Métrica | Significado |
|---|---|
| `cut_ratio` | Fracción de aristas cortadas entre workers |
| `imbalance` | Nodos del worker más grande ÷ promedio (1.0 es perfecto) |
| `boundary_fraction` | Fracción de nodos con algún vecino en otro worker |
| `replication_factor` | (propios + halo) sumado ÷ n. 1.0 = sin replicación; k = todo en todos |
| `effective_imbalance` | Igual que `imbalance`, pero contando propios + halo (la carga real) |
| `halo_bytes_est` | Datos de halo × T × 5 bytes. **Estimación**, no medición |
| `comm_links`, `peers_*` | Con cuántos otros workers habla cada uno |

**Regla del halo:** la profundidad necesaria es `DetectorConfig.halo_depth = max(hops, impute_iters)`.
Con menos, el resultado del worker deja de coincidir con el centralizado; con más, se paga
comunicación y cómputo redundante sin ganar exactitud.

## 8. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `ModuleNotFoundError: iot_graph_generator` | Ejecuta desde la carpeta donde están los `.py` |
| `ModuleNotFoundError: networkx` | `pip install -r requirements.txt` (Louvain lo necesita desde el paso 3) |
| Un test falla por tolerancia numérica | Pega el mensaje; pueden aparecer diferencias de float32 entre versiones de numpy. Los tests de halo usan `rtol=1e-3` |
| `partition.py` con `spatial` falla | Ese método necesita coordenadas: usa los modos `hybrid` o `spatial` del generador |
| `ImportError: pymetis` | `metis` es opcional (`pip install pymetis`) y **no está verificado** |
| 100k nodos tarda o falta memoria | Necesita ~1.2 GB de pico. Baja `--T` (p. ej. 48). Louvain tarda ~20 s a 100k y es lo más lento |
| Resultados distintos a los de esta guía | Revisa que `signals.py` sea v1.1.0 (incluye el componente `weather`) |

## 9. Qué sigue

| Paso | Qué | Estado |
|---|---|---|
| 1 | Generador de grafos y señales | ✅ |
| 2 | Detector centralizado y métricas | ✅ |
| 3 | Particionadores, halos y plan de comunicación | ✅ |
| 4 | Ejecución distribuida simulada en un proceso y test de equivalencia global | pendiente |
| 5 | Workers reales (Ray) con medición de bytes y tiempos | pendiente |
| 6 | Orquestador de experimentos y análisis estadístico | pendiente |
| 7 | Frontera adaptativa | pendiente |

## 10. Limitaciones que debes declarar en el artículo

- **Los datos son sintéticos.** El componente `weather` (fluctuación suave en el espacio e impredecible
  en el tiempo) es una decisión de modelado; documéntala y, si es posible, valida con un dataset real.
- **Los bytes son estimaciones** (intercambio único de datos crudos). Los reales se medirán en el paso 5.
- **Las anomalías regionales casi no se detectan con un detector de 1 salto:** el desvío común oculta a
  los nodos interiores. Es una propiedad del escenario, no un fallo del código.
- **`community` y `spatial` se parecen** cuando las comunidades se solapan: incluye `spatial` como
  baseline para no sobrestimar la ventaja de la partición por comunidades.
- **El refinamiento de frontera mejora el corte de forma modesta** sobre Louvain+empaquetado: entre
  ~3 % y ~19 % en términos relativos según el caso (n=10k, promedio de 5 semillas: 0.0035 → 0.0034 con
  k=4 y 0.0106 → 0.0086 con k=8 en comunidades fáciles; 0.0239 → 0.0215 con k=8 en comunidades
  solapadas), pero en valores absolutos es menos de un punto porcentual de aristas. Es lo que ya hacen
  METIS y otros particionadores, así que como contribución propia es débil. Si "frontera adaptativa"
  va a ser el aporte del artículo, conviene definirla sobre el halo (profundidad variable por nodo) y no
  sobre la partición, o medir si esa ganancia relativa se traduce en menos bytes y mejor F1.
- **Todo se verificó con un ejecutor de tests propio** (sin red para instalar pytest). Tu primera
  corrida con pytest real es la confirmación definitiva.
- El método `metis` está escrito pero **no se ha ejecutado**.

## 11. Reproducibilidad

Cada grafo y cada conjunto de señales guarda en su `manifest`/`.json`: parámetros resueltos, versiones
de Python/numpy/scipy/networkx, hash de git (si hay repositorio) y un SHA-256 del contenido. Con las
mismas versiones y semilla, el SHA-256 debe coincidir. Cada componente aleatorio usa un flujo
independiente (`SeedSequence.spawn`), de modo que cambiar un parámetro no altera los demás.
