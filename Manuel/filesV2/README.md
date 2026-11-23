# Analítica distribuida de grafos IoT — Equipo T800
## Guía de ejecución (pasos 1 a 6)

Estado del proyecto: **datos sintéticos, detector, particionado con halos, ejecución distribuida simulada y
real (procesos del SO), orquestador de campañas y análisis estadístico con figuras: listos.** Falta lanzar la
campaña completa en una máquina con varios núcleos y redactar el artículo (ver [Qué sigue](#9-qué-sigue)).
El plan de pruebas está en [`PLAN_DE_PRUEBAS.md`](PLAN_DE_PRUEBAS.md).

---

## 1. Qué contiene el paquete

| Paso | Archivo | Qué hace |
|---|---|---|
| 1 | `iot_graph_generator.py` | Genera el grafo (nodos, aristas, coordenadas, comunidades de referencia) |
| 1 | `signals.py` | Genera lecturas `X[t,n]`, anomalías, datos faltantes y la verdad de referencia |
| 2 | `detect.py` | Detector de anomalías e imputación, centralizado y diseñado para distribuirse |
| 2 | `metrics.py` | AUC, AP, F1, umbral, MAE/RMSE de imputación |
| 3 | `partition.py` | Particionadores, halos por worker, plan de comunicación, métricas y verificación |
| 4 | `distributed.py` | Ejecución distribuida **simulada**: red con bytes contabilizados, equivalencia global, modelo de costos, filas de resultados |
| 5 | `runtime.py` | Workers **reales** (procesos del SO): halo por colas entre workers, bytes serializados, tiempos de pared y de CPU, calibración del transporte |
| 6 | `orchestrator.py` | Campañas reanudables y paralelas por semilla, tau calibrado por escenario, ids por contenido, escritura atómica |
| 6 | `analysis.py` | Bootstrap, Wilcoxon + Holm, TOST, δ de Cliff, Hodges-Lehmann, ajuste log-log, Pareto, figuras F2-F11 e informe |
| — | `configs/*.json` | `demo.json` (≈ 1 min) y `campaign_full.json` (plan completo E0-E4b: 12 696 corridas) |
| — | `run_pilot.py` | Piloto del plan de pruebas (rejilla pequeña, umbral calibrado, resumen estadístico) |
| — | `run_e4b.py` | Piloto de la **prueba adversa E4b**: anomalías en la frontera de la partición (H8) |
| — | `PLAN_DE_PRUEBAS.md` | Hipótesis, diseño experimental, análisis estadístico, figuras y presupuesto para el artículo |
| — | `results/` | Resultados de los pilotos (`pilot_*`, `e4b_*`), del runtime real (`runtime_*`) y un ejemplo de campaña con su análisis (`demo_campaign_results.csv`, `demo_analysis/`) |
| — | `example_pipeline.py` | Recorrido completo de los pasos 1-2 con calibración de umbral y demo de halo |
| — | `test_*.py` (8 archivos) | 144 tests (143 se ejecutan sin extras; 1 requiere `pymetis`) |
| — | `requirements.txt` | Dependencias con las versiones verificadas |

**Cómo se conectan los módulos**

```
iot_graph_generator ──► signals ──► detect ──► metrics
        │                              ▲
        └──────────► partition ────────┘   (partition usa detect.adjacency_hops/induced_edges;
                                            detect define cuánto halo se necesita)
                          │
                          └──► distributed (simula k workers con mensajes y compara con el centralizado)
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
pytest -q                            # esperado: 143 passed, 1 skipped
```

El test omitido (`test_metis_optional`) es normal si no instalaste `pymetis`. Tarda alrededor de un minuto.
Para ejecutar solo una parte:

```bash
pytest -q test_partition.py                       # solo particionado
pytest -q -k "centralized_result"                 # solo la prueba de equivalencia con halo
pytest -q -x                                      # detenerse en el primer fallo
```

| Archivo de tests | Casos | Qué protege |
|---|---|---|
| `test_iot_graph_generator.py` | 14 | Determinismo, conectividad, IDs permutados, peso como similitud, calibración de grado |
| `test_signals.py` | 16 | Convenciones de la verdad de referencia, tasas objetivo, escenarios pareados, colocación de anomalías (`anomaly_nodes`) |
| `test_detect.py` | 18 | Métricas con casos calculados a mano, detectores, equivalencia con halo |
| `test_partition.py` | 30 + 1 omitido | Validez y balance, teoría del hash, ausencia de fuga, invariantes del plan, equivalencia por worker |
| `test_distributed.py` | 18 | Equivalencia global, degradación por halo insuficiente, red contabilizada, modelo de costos, esquema de filas |
| `test_runtime.py` | 15 | Equivalencia con procesos reales, bytes serializados, reutilización del pool, propagación de errores, tiempos de espera, apagado limpio |
| `test_orchestrator.py` | 17 | Plan de la rejilla, ids estables, reanudación, escritura atómica, determinismo, paralelismo, colocación adversa |
| `test_analysis.py` | 15 | Cada procedimiento estadístico contra valores calculados a mano; integración orquestador → análisis |

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

```bash
# 5. Ejecución distribuida simulada: compara con el centralizado variando método, k y profundidad de halo
python distributed.py --graph ./data/hybrid_n10000_seed1 \
    --signals ./data/hybrid_n10000_seed1/signals_seed1_both_structured.npz \
    --methods hash spatial community --k 4 8 --depth 0 1 2 --hops 1 --impute-iters 2 --tau 4.0
```

Salida esperada del paso 5 (extracto): con halo 2 (el requerido) todo es **equivalente** al centralizado;
sin halo, `hash` pierde calidad y las particiones estructuradas casi nada:

```
método              k halo  corte  MB halo speedup*  efic*  infl.    ΔAUC     ΔF1    ΔMAE   max|Δ| equiv
hash                8    0  0.876     0.00     2.07   0.26   0.76  -0.162  -0.233   0.395  1.7e+01 no
hash                8    2  0.876    32.68     1.17   0.15   2.41  -0.000   0.000   0.000  3.2e-04 SI
community           8    0  0.010     0.00     2.12   0.26   0.72  -0.001  -0.001   0.001  5.6e+00 no
community           8    2  0.010     0.62     1.94   0.24   0.86   0.000   0.000  -0.000  1.7e-04 SI
```

`speedup*`, `efic*` son **modelados** (cota ideal con red de 1 Gbit/s); los bytes, el cómputo por worker y las
métricas de calidad son **medidos**. `infl.` = cómputo total de los workers ÷ cómputo centralizado.
Los valores absolutos de tiempo varían con tu máquina; las columnas de calidad y `equiv` no.

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

### 6.3 Piloto del plan de pruebas

```bash
python run_pilot.py --n 10000 --seeds 5 --out ./results/pilot_n10000.csv     # ~3 min
```

Ejecuta 4 métodos × k ∈ {4, 8} × profundidad ∈ {0, 1, 2} sobre 5 semillas, con el umbral calibrado en
semillas aparte, y resume ΔF1, bytes, speedup modelado, el contraste de equivalencia (TOST ±2.5 pp) y las
repeticiones necesarias. Es una plantilla de lo que hará el orquestador (paso 6), no su reemplazo
(no es reanudable). Resultados e interpretación en `PLAN_DE_PRUEBAS.md`.

### 6.3b Prueba adversa E4b (anomalías en la frontera)

```bash
python run_e4b.py --n 10000 --seeds 4 --k 8 --out ./results/e4b.csv                        # toda la frontera (~3 min)
python run_e4b.py --n 10000 --seeds 4 --k 8 --depths 0 1 --min-cut-fraction 0.5 --rate 0.2 \
                  --out ./results/e4b_strong.csv                                             # adversario fuerte
```

Coloca las anomalías en nodos de frontera de una partición de referencia (`community`, k=8) y evalúa todos los
métodos sobre esos mismos datos. `--min-cut-fraction 0.5` limita a los nodos que pierden la mitad o más de sus
vecinos al cortar. Desde Python:

```python
from partition import partition_graph, boundary_nodes
part, _ = partition_graph(g, "community", 8, seed=1)
sig = generate_signals(g, seed=1, anomaly="both", anomaly_rate=0.2,
                       anomaly_nodes=boundary_nodes(g, part, min_cut_fraction=0.5),
                       anomaly_placement="boundary_community_k8")
```

La tasa se aplica a las lecturas **elegibles** (las de esos nodos). `anomaly_nodes=None` (por defecto) conserva el
comportamiento anterior, y con todos los nodos el resultado es idéntico.

### 6.4a Workers reales con procesos (paso 5)

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python runtime.py --graph ./data/hybrid_n10000_seed1 \
    --signals ./data/hybrid_n10000_seed1/signals_seed1_both_structured.npz \
    --methods hash community --k 2 4 8 --depth 0 2 --tau 4.0 --calibrate-ipc --csv results/runtime.csv
```

Cada worker es un proceso del sistema operativo; el halo viaja **directamente entre workers** (una cola de entrada
por worker) ya serializado. Se miden los bytes reales, el tiempo de **pared** y el tiempo de **CPU** de cada worker.

- **Si tu máquina tiene ≥ k núcleos**, `sp.pared` es el speedup real.
- **Si k > núcleos** el programa avisa (`rt_oversubscribed=True`): los procesos se reparten el CPU y el tiempo de pared
  **no es un speedup**. Entonces es válida la estimación por CPU (`sp.CPU`): ruta crítica con el CPU real de cada worker
  (incluye serialización), con reparto ideal.
- `--calibrate-ipc` mide el transporte local (ancho de banda efectivo y latencia por mensaje) para el modelo de tiempo.
  El resultado varía entre ejecuciones (≈ 650-1 600 MB/s en las pruebas): repítelo y reporta el rango.
- Exporta `OMP_NUM_THREADS=1` y `OPENBLAS_NUM_THREADS=1` en el shell para que workers y centralizado no usen hilos de BLAS
  distintos (si no, la comparación no es justa). Se registra en `rt_blas_threads`.

Resultados medidos (1 núcleo, ver `results/runtime_*.csv`): en las 14 corridas, todas las de profundidad 2 son equivalentes al
centralizado; los bytes serializados difieren de la estimación en 0.00-0.87 %; y el modelo calibrado queda a −1.8 % a −13.5 %
(media −5.9 %) de la estimación por CPU. **Tu speedup de pared real, en una máquina multinúcleo, es el dato que falta.**

### 6.4b Campañas con el orquestador (paso 6)

```bash
python orchestrator.py run configs/demo.json --out ./campaigns --dry-run     # qué se va a ejecutar y tiempo aprox.
python orchestrator.py run configs/demo.json --out ./campaigns               # ≈ 1 min
python orchestrator.py status configs/demo.json --out ./campaigns            # avance por experimento
python orchestrator.py run configs/campaign_full.json --out ./campaigns --workers 4   # campaña completa del plan
```

- **Reanudable:** si se interrumpe (Ctrl-C, corte de luz), vuelve a lanzar el mismo comando: salta los trabajos terminados.
  Cada trabajo se guarda en su archivo con escritura atómica.
- **Por contenido:** si cambias la configuración, los resultados viejos no se mezclan con los nuevos (se ignoran como "huérfanos").
- `--limit N` ejecuta solo N trabajos; `--workers N` reparte las semillas entre N procesos.
- `"executor": "real"` en el JSON usa los workers reales (fuerza `--workers 1`).
- Escribe `campaigns/<nombre>/{manifest.json, tau_cache.json, rows/, results.csv}`. El manifest guarda la configuración, versiones,
  hardware y sesiones.
- Una campaña = rejilla de **experimentos** con barridos (`sweep`) y cuadrícula (`grid`); ver `configs/campaign_full.json`
  (reproduce exactamente las 12 696 corridas del plan; ≈ 6-8 h de un núcleo).

### 6.4c Análisis y figuras

```bash
python analysis.py --results campaigns/campaign_full/results.csv --out ./analysis
```

Genera tablas (`summary`, `comparisons` con Wilcoxon + Holm, `equivalence` con TOST ±2.5 y ±0.5 pp), el ajuste log-log,
las figuras F2-F11 y `report.md` con las cifras de H1, H2, H3 y H8. Ejemplo ya generado: `results/demo_analysis/`.
Con pocas semillas (n < 10) los intervalos son anchos y los veredictos no son definitivos.

### 6.4 Protocolo del umbral (importante para que la comparación sea justa)

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
| Resultados distintos a los de esta guía | Revisa que `signals.py` sea v1.2.0 (componente `weather` y `anomaly_nodes`) |
| `ModuleNotFoundError: pandas` / `matplotlib` | Solo `analysis.py` los necesita: `pip install -r requirements.txt` |
| `AVISO: k máximo > núcleos` en `runtime.py` | Es correcto: el tiempo de pared no es un speedup con menos núcleos que workers; usa la estimación por CPU o una máquina con más núcleos |
| La campaña se interrumpió | Vuelve a lanzar el mismo comando de `orchestrator.py run`: reanuda sola |
| `results.csv` no aparece | Solo se crea cuando todos los trabajos del plan están terminados; `orchestrator.py collect` lo fuerza con lo que haya |

## 9. Qué sigue

| Paso | Qué | Estado |
|---|---|---|
| 1 | Generador de grafos y señales | ✅ |
| 2 | Detector centralizado y métricas | ✅ |
| 3 | Particionadores, halos y plan de comunicación | ✅ |
| 4 | Ejecución distribuida simulada en un proceso y test de equivalencia global | ✅ |
| 5 | Workers reales (procesos del SO) con bytes y tiempos medidos; calibra el modelo de tiempo | ✅ (Ray no estaba disponible; backend de Ray opcional) |
| 6 | Orquestador (reanudable, paralelo) y scripts de análisis y figuras | ✅ |
| — | **Lanzar la campaña completa y medir el speedup de pared en una máquina multinúcleo** | pendiente (requiere tu equipo) |
| 7 | Frontera adaptativa | **reevaluado**: ver `PLAN_DE_PRUEBAS.md`, sección 13 |

## 10. Limitaciones que debes declarar en el artículo

- **Los datos son sintéticos.** El componente `weather` (fluctuación suave en el espacio e impredecible
  en el tiempo) es una decisión de modelado; documéntala y, si es posible, valida con un dataset real.
- **Distingue tres tipos de cifra.** (1) *Medidas*: bytes, cómputo por worker y métricas de calidad. (2) *Modeladas*
  (`speedup_model*`, de `distributed.py`): cota ideal con una red supuesta. (3) *Estimadas con CPU real*
  (`speedup_cpu_est`, de `runtime.py`): ruta crítica con el CPU real de cada worker, incluida la serialización. El speedup de
  **pared** (`speedup_real_wall`) solo es válido con ≥ k núcleos. Las pruebas de desarrollo tuvieron **1 núcleo**: ahí solo
  valen (1) y (3). No presentes (2) ni (3) como mediciones de speedup en paralelo.
- **La recolección de resultados es un término constante** (~n·T·12 bytes) que domina cuando la partición es buena:
  reporta `speedup_model` y `speedup_model_nogather` y declara cuál aplica.
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
