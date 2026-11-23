# Plan de pruebas para el artículo

**Proyecto:** analítica distribuida basada en grafos para calidad de datos en redes IoT (Equipo T800)
**Base:** pasos 1-6 implementados y verificados (incluye workers reales con procesos del SO, orquestador y análisis);
piloto con 120 corridas (n=10 000, 5 semillas) y pilotos de la
prueba adversa E4b (96 corridas con n=10 000; 80 con n=3 000 y k=16; 64 con el adversario fuerte) y 14 corridas del
runtime real (n=10 000 y 100 000).
**Estado de cada afirmación:** lo marcado **[piloto]** ya tiene evidencia preliminar; **[por probar]** no.

---

## 0. Resumen

El piloto cambió el planteamiento. Con halo suficiente el resultado distribuido es **idéntico** al
centralizado, así que la pregunta de la propuesta ("¿cuánta precisión se pierde al distribuir?") se
convierte en otra más precisa: **¿cuánta comunicación de halo hace falta, y para qué?**

Lo que muestra el piloto, con las advertencias de la sección 9:

| Hallazgo | Evidencia del piloto |
|---|---|
| Equivalencia exacta con halo ≥ requerido | 40 de 40 corridas con profundidad 2 (la requerida) son equivalentes; con profundidad 0 ninguna, con profundidad 1 solo 6 de 40 |
| Con partición estructurada, **sin halo** el F1 queda a ≤ 0.11 pp del centralizado | spatial/community: −0.01 a −0.11 pp |
| Con partición por hash, sin halo se pierde mucho | −11.1 pp (k=4), −22.2 pp (k=8) |
| El halo de hash es caro | 32.7 MB y replicación 7.8× (k=8, 10k nodos) frente a 0.7 MB y 1.14× en comunidades |
| Detección e imputación necesitan halos distintos | hash k=8, profundidad 1: ΔF1 = 0, pero ΔMAE = +0.021 |
| **La prueba adversa (E4b) depende de la fuerza del adversario** | Anomalías en toda la frontera: `community` sin halo pierde 1.46 pp (dentro del margen). Solo en nodos que pierden ≥ 50 % de sus vecinos: **−4.74 pp (IC90 [−7.8, −1.7]), fuera del margen** |
| Una profundidad de halo de 1 elimina esa pérdida | ΔF1 = 0.00 con solo 0.23 MB (frente a 24.4 MB de hash) |
| Un halo **selectivo** ahorraría mucho en términos relativos, pero poco en absoluto | −63 % a −76 % de los bytes del halo de profundidad 1 en `community`, que ya son solo 1.3-4.6 % del volumen de datos (sección 13) |

La hipótesis original (F1 dentro de 2-3 pp con menos comunicación) se cumple **de forma holgada** en el
escenario piloto, pero eso exige prudencia: puede deberse a que el escenario es "fácil" para la partición
estructurada. Por eso se construyó la prueba adversa E4b (sección 3.3). Resultado del piloto: con anomalías en toda la
frontera el margen se mantiene; solo con un adversario fuerte (nodos que pierden la mitad o más de sus vecinos)
sale del margen, y un halo de 1 salto lo recupera por completo.

---

## 1. Reencuadre de la hipótesis

| Propuesta original | Versión que se puede defender |
|---|---|
| "Una partición consciente de comunidades mantiene F1 dentro de 2-3 pp del centralizado y reduce comunicación" | Se confirma, pero hay que separar **dos** efectos: (a) la estructura reduce los bytes de halo, y (b) además hace que el halo sea casi prescindible para la detección |
| "Frontera adaptativa" | Los datos sugieren definirla sobre el **halo por tarea** (profundidad 1 para detección, 2 para imputación) y no sobre mover nodos de frontera (ganancia de 3 % a 19 % en corte, sin efecto visible en F1) |

La contribución científica pasa a ser una **caracterización cuantitativa**: cuánto cuesta el halo, cuánta
exactitud compra, y bajo qué condiciones deja de ser prescindible.

---

## 2. Hipótesis y criterios de decisión

Unidad experimental: **la semilla** (un grafo + sus señales). Todos los métodos se evalúan sobre la misma
pareja (grafo, señales), de modo que las comparaciones son pareadas.

| ID | Hipótesis | Prueba | Se acepta si… | Se rechaza si… | Piloto |
|---|---|---|---|---|---|
| **H1** | Con profundidad de halo ≥ requerida, distribuido ≡ centralizado | E0, E1 | 0 contraejemplos (`equivalent=True`, diferencias < 1e-3 relativo) | Cualquier contraejemplo reproducible | ✅ 40/40 con prof. 2; control negativo: 0/40 con prof. 0 |
| **H2** | Con partición estructurada y **sin halo**, ΔF1 está dentro de ±2.5 pp | E1, E3, E4 | TOST: IC 90 % dentro de ±2.5 pp, en cada escenario | IC fuera del margen en algún escenario no adverso | ✅ IC90 de −0.15 a +0.01 pp |
| **H3** | La pérdida sin halo crece con el corte de aristas según una ley de potencia | E1 + E3 | Ajuste log-log con R² ≥ 0.8 y exponente con IC que excluya 0 | Sin relación monótona (Spearman no significativo) | ⚠ exploratorio: \|ΔF1\| ≈ 15.8·corte^1.17, R²=0.92 |
| **H4** | El speedup (modelado) crece con k para particiones estructuradas y no para hash | E2 | Pendiente positiva con IC; eficiencia de hash < 0.5 de la de comunidades | Hash iguala a comunidades | ⚠ 100k, k=8: 3.0-3.4× vs 1.7× (hash) |
| **H5** | La ventaja de comunidades sobre `spatial` desaparece cuando las comunidades se solapan | E3 | Diferencia de corte y de bytes → 0 al subir `spread_factor` (IC incluye 0 desde ~0.4) | Diferencia constante | ⚠ brecha de corte spatial − community (k=8): 0.0148 con `spread_factor` 0.2 → 0.0066 con 0.4; se reduce a menos de la mitad pero no desaparece |
| **H6** | El resultado de H2 **no** depende del tipo de anomalía ni de su magnitud | E4 | TOST dentro del margen en los 12 escenarios | Escenario fuera del margen | [por probar] |
| **H7** | Detección e imputación requieren halos distintos: `hops` y `impute_iters` | E1 | ΔF1 = 0 con profundidad = `hops`; ΔMAE > 0 si profundidad < `impute_iters` | Misma profundidad óptima para ambas | ✅ hash k=8, prof. 1 |
| **H8** | Si las anomalías se concentran en nodos que pierden ≥ 50 % de sus vecinos al cortar, `community` sin halo sale del margen | E4b | TOST: IC 90 % **no** contenido en ±2.5 pp, y se recupera con profundidad = `hops` | Dentro del margen aun con el adversario fuerte | ✅ −4.74 pp, IC90 [−7.8, −1.7] (4 semillas; con toda la frontera: −1.46 pp, dentro del margen) |
| **H9** | Las conclusiones de H2 y H4 son robustas al ancho de banda y la latencia | E5 | Orden entre métodos estable en las 9 combinaciones | Cambia el orden | [por probar] |

**H8 es la más importante para la credibilidad.** Si solo se prueba donde el halo es prescindible, un
revisor dirá que el escenario favorece el resultado.

---

## 3. Diseño experimental

### 3.1 Factores y niveles

| Factor | Niveles | Parámetro en el código |
|---|---|---|
| Tamaño `n` | 1 000 · 10 000 · 100 000 | `generate(n=...)` |
| Método de partición | `hash`, `spatial`, `community`, `community_refined` (+ `metis` si se instala) | `partition_graph(method=...)` |
| Workers `k` | 2 · 4 · 8 · 16 | `k` |
| Profundidad de halo | 0 · 1 · 2 · 3 | `depth` |
| Detector (`hops`, `impute_iters`) | (1,1) · (1,2) · (2,2) | `DetectorConfig` |
| Estructura de comunidades | `spread_factor` ∈ {0.1, 0.2, 0.3, 0.4, 0.5} | `generate(spread_factor=...)` |
| Anomalía | `point`, `regional`, `both` | `generate_signals(anomaly=...)` |
| Faltantes | `random`, `structured` | `generate_signals(missing=...)` |
| Magnitud | (3, 6) difícil · (5, 10) fácil | `generate_signals(magnitude=...)` |
| Red (solo modelo) | 100 Mbit/s · 1 · 10 Gbit/s × 0.1 · 0.5 · 5 ms | `CostModel` |

**Valores fijos** salvo que el experimento los varíe: grafo `hybrid` con `k_neighbors=8`; `T=96`;
anomalías 1 %, faltantes 5 %; `spread_factor=0.2`; detector (1, 2).

### 3.2 Experimentos

| Exp. | Objetivo | Rejilla | Semillas | Corridas | Costo estimado |
|---|---|---|---|---|---|
| **E0** | Validación determinista de H1 | 4 métodos × k{2,4,8,16} × prof. requerida × 2 escenarios · n=10k | 3 | 96 | ~2.5 min |
| **E1** | Ablación del halo (H1, H2, H3, H7) | 4 métodos × k{4,8,16} × prof.{0,1,2,3} × 3 detectores · n=10k | 30 | 4 320 | ~1.8 h |
| **E2** | Escalabilidad (H4) | 4 métodos × k{2,4,8,16} × prof.{0, req.} · n=1k y 10k (30 sem.) y n=100k (10 sem., k{4,8,16}) | 30 / 10 | 960 + 960 + 240 | ~2 min + 24 min + 1.7 h |
| **E3** | Estructura de comunidades (H2, H3, H5) | 5 `spread_factor` × 4 métodos × k=8 × prof.{0,1,2} · n=10k | 30 | 1 800 | ~45 min |
| **E4** | Tipo de datos y dificultad (H6) | 12 escenarios × 4 métodos × k=8 × prof.{0,2} · n=10k | 30 | 2 880 | ~1.2 h |
| **E4b** | **Prueba adversa**: anomalías en la frontera (H8) | 2 adversarios (frontera completa, ≥ 50 % de vecinos cortados) × 4 métodos × k{8,16} × prof.{0,1,2} · n=10k | 30 | 1 440 | ~45 min |
| **E5** | Sensibilidad de red (H9) | Postproceso de E1/E2 con 9 combinaciones de red | — | 0 | segundos |
| **E6** | Validar el modelo de tiempo con procesos reales (`runtime.py`) y medir el speedup de **pared** en una máquina con ≥ k núcleos | 3 configuraciones (p. ej. `community` k=8, `spatial` k=8, `hash` k=4, n=100k) × 10 repeticiones | — | 30 | ≈ 15 min a 1 h (estimación) |

**Total E0-E4b:** ≈ 6.6 h de CPU de un núcleo (≈ 2 h en un equipo de 4 núcleos paralelizando por semilla).

Costos medidos: n=1k ≈ 0.13 s y n=10k ≈ 1.5 s por corrida; n=100k ≈ 21-30 s por corrida más ~25-40 s por semilla
(grafo+señales ≈ 4 s, referencia centralizada ≈ 5 s, Louvain ≈ 13-20 s). Memoria de pico a 100k: 1.7 GB.

### 3.3 Prueba adversa E4b (implementada)

Se colocan las anomalías (puntuales y centros de eventos regionales) en nodos de frontera de una partición de
referencia (`community`, k=8) y se evalúan **todos** los métodos sobre esos mismos datos, con lo que se conserva la
comparación pareada. La referencia sale perjudicada a propósito (es su peor caso).

| Adversario | Nodos elegibles (n=10k, k=8) | Tasa sobre lecturas elegibles | Cómo se genera |
|---|---|---|---|
| Control uniforme | todos | 1 % | `generate_signals(...)` |
| **Frontera completa** | 4.2 % (≥ 1 vecino cortado) | 5 % | `anomaly_nodes=boundary_nodes(g, part)` |
| **Frontera fuerte** | 0.43 % (≥ 50 % de vecinos cortados) | 20 % | `anomaly_nodes=boundary_nodes(g, part, 0.5)` |

Implementación: `anomaly_nodes` en `signals.py` (v1.2.0; con todos los nodos el resultado es idéntico al valor
por defecto y hay un test que lo exige), `boundary_nodes(graph, part, min_cut_fraction)` en `partition.py` y
`run_e4b.py`. La tasa se aplica a las lecturas elegibles, por lo que el número de positivos es menor que en el
control (≈ 2 400 y ≈ 1 300 frente a ≈ 9 500).

**Interpretación:** las anomalías puntuales caen solo en los nodos elegibles; los eventos regionales se *centran* en
ellos pero se extienden alrededor (en una prueba, 1 414 de 2 107 lecturas regionales cayeron fuera del conjunto). "Anomalías en la
frontera" significa entonces "centradas en la frontera"; `anomalies_outside_nodes` en el reporte de señales lo cuantifica.

Variantes que quedan por correr: adversario **propio de cada método** (cada método con anomalías en su propia
frontera; rompe el pareado pero da el peor caso por método) y `--ref-method spatial`.

---

## 4. Métricas

| Tipo | Métrica | Fuente | ¿Medida o modelada? |
|---|---|---|---|
| Calidad | AUC, AP, F1 (con τ congelado), ΔAUC, ΔAP, ΔF1 respecto al centralizado | `metrics.py` | Medida |
| Calidad | MAE y ΔMAE de imputación (solo lecturas faltantes) | `metrics.py` | Medida |
| Calidad | ΔAUC y ΔMAE en **nodos de frontera** y **nodos interiores** por separado | `d_*_boundary`, `d_*_interior` | Medida |
| Equivalencia | `max_abs_diff_spatial`, `max_abs_diff_imputed`, `equivalent` | `distributed.py` | Medida |
| Partición | corte de aristas, desbalance, fracción de frontera, replicación, carga efectiva | `partition.py` | Medida |
| Comunicación | `halo_bytes`, `halo_msgs`, `halo_bytes_max_worker`, `gather_bytes`, `topology_bytes` | `distributed.py` | **Medida** (sobre los arreglos enviados) |
| Tiempo | `t_central`, `t_compute_max`, `t_compute_sum`, `work_inflation` | `distributed.py` | Medida (secuencial, sin contención) |
| Tiempo | `t_exchange_model`, `t_dist_model`, `speedup_model`, `efficiency_model`, `speedup_model_nogather` | `distributed.py` | **Modelada** (cota ideal) |
| Tiempo (real) | `rt_t_dist_wall`, `speedup_real_wall`, `rt_oversubscribed` | `runtime.py` | **Medida** (de pared); solo es un speedup si núcleos ≥ k (`rt_oversubscribed=False`) |
| Tiempo (CPU real) | `rt_t_dist_cpu_est`, `speedup_cpu_est`, `efficiency_cpu_est`, `model_error_rel_cpu` | `runtime.py` | **Estimada** con el CPU real de cada worker (incluye serialización); válida con cualquier número de núcleos |
| Transporte | bytes serializados reales (`halo_bytes`), `rt_halo_bytes_est`, `rt_scatter_bytes` | `runtime.py` | Medida |

Regla de redacción: el speedup de `distributed.py` se llama siempre "modelado"; el de `runtime.py` por CPU, "estimado con
CPU real"; solo `speedup_real_wall` con núcleos ≥ k es "medido". Los bytes de `distributed.py` no incluyen serialización;
los de `runtime.py` sí (pickle).

---

## 5. Protocolo de ejecución

1. **Congelar versiones**: `pip freeze`, hash de git, hardware (`lscpu`, memoria) en el manifest de la campaña.
2. **Umbral τ**: calibrar con el **centralizado** sobre semillas de calibración **1001-1010**, mezclando sus scores
   (`best_f1_threshold`), **un τ por escenario** (cambia con tipo y magnitud de anomalía). Congelar y aplicar igual a todos
   los métodos. Análisis de sensibilidad de τ: repetir E1 con τ ± 20 %.
3. **Semillas de evaluación**: 1-30 (1-10 para n=100k), **disjuntas** de las de calibración.
4. **Orden aleatorio** de las corridas dentro de cada semilla; no ejecutar otros procesos pesados; fijar el gobernador
   de CPU; descartar una corrida de calentamiento.
5. **Cronometraje**: mínimo de 3 repeticiones por fase de cómputo (`repeats=3`); referencia centralizada con el mismo criterio.
6. **Registro**: una fila por corrida (esquema de `run_distributed`), más el manifest del grafo y las señales
   (SHA-256) para poder auditar. Conservar los logs.
7. **Pruebas de humo antes de cada campaña**: `pytest -q` y E0.

Comando de referencia (el orquestador del paso 6 automatizará esto): `python run_pilot.py` es la plantilla.

---

## 6. Análisis estadístico

| Pregunta | Método | Reporte |
|---|---|---|
| Descripción | Media, mediana, desviación, IQR; **IC 95 % bootstrap** (10 000 remuestreos **sobre semillas**) | Tablas y barras de error |
| Comparar métodos (pareado) | **Wilcoxon de rangos con signo** por (k, profundidad, escenario); **corrección de Holm** por familia | p corregido |
| Tamaño de efecto | **δ de Cliff** y diferencia mediana (Hodges-Lehmann) con IC | Siempre junto al p (la propuesta lo exige) |
| Hipótesis de ±2.5 pp (H2, H6) | **TOST** (equivalencia), α=0.05, es decir IC 90 % dentro de ±2.5 pp. Margen estricto adicional de ±0.5 pp | "Equivalente / no concluyente" |
| Relación cuantitativa (H3) | Regresión log-log de \|ΔF1\| contra corte de aristas; IC bootstrap del exponente; validación dejando una semilla fuera; Spearman | Exponente, R², figura |
| Escalabilidad (H4) | Regresión de speedup contra k por método; punto de cruce `n*` en que distribuir supera al centralizado | Curvas con bandas |
| Frente de Pareto (bytes vs ΔF1) | Puntos no dominados por (método, profundidad) | Figura principal |
| Red (H9) | Recalcular `t_exchange_model` con 9 combinaciones y comparar órdenes | Mapa de calor |

**Pseudo-replicación:** k, profundidad y escenario no son réplicas independientes; la réplica es la semilla.
Si se quiere un análisis conjunto, usar un modelo mixto con la semilla como efecto aleatorio.

**Repeticiones (piloto).** La desviación de las diferencias pareadas entre `community` y `hash` fue de
0.68 pp; con potencia 80 % y α=0.05 bastan 4 semillas para detectar 1 pp. Aun así se recomiendan **30**
porque (a) las colas y la heterogeneidad entre escenarios no se ven con 5, (b) las diferencias **entre** particiones
estructuradas son del orden de 0.03-0.1 pp, y (c) la propuesta pide 20-30. Distinguir `spatial` de `community`
en F1 no tiene relevancia práctica; ahí lo informativo son los bytes y el corte.

---

## 7. Figuras y tablas del artículo

| Id | Contenido | Experimento | Apoya |
|---|---|---|---|
| F1 | Arquitectura: sensores → grafo → partición → workers con halo → agregador | — | Diseño |
| F2 | Diferencia máxima respecto al centralizado frente a la profundidad de halo, por método (escalón en la profundidad requerida) | E0, E1 | H1 |
| **F3** | **Frontera de Pareto: ΔF1 (pp) frente a bytes de halo (MB)**, banda ±2.5 pp | E1 | H2, H7 |
| F4 | Pérdida sin halo frente al corte de aristas (log-log) con ajuste e IC | E1, E3 | H3 |
| F5 | Speedup modelado y eficiencia frente a k, n=100k, con y sin recolección | E2 | H4 |
| F6 | Speedup frente a n (1k, 10k, 100k): punto de cruce | E2 | H4 |
| F7 | Corte, bytes y ΔF1 frente a `spread_factor` (solape de comunidades) | E3 | H5 |
| F8 | Mapa de calor de ΔAUC por escenario × método, y su versión adversa | E4, E4b | H6, H8 |
| F9 | Desglose frontera / interior de ΔAUC y ΔMAE | E1, E4b | Mecanismo |
| F10 | Sensibilidad a ancho de banda y latencia | E5 | H9 |
| T1 | Parámetros de los datos sintéticos y del detector | — | Reproducibilidad |
| T2 | Propiedades de las particiones (corte, desbalance, replicación, bytes) | E2 | H2, H4 |
| T3 | Resumen de hipótesis: p corregido, δ de Cliff, IC, veredicto | todos | Todas |
| T4 | Presupuesto de cómputo y artefactos | — | Reproducibilidad |

---

## 8. Amenazas a la validez y mitigación

| Tipo | Amenaza | Mitigación |
|---|---|---|
| **Constructo** | Datos 100 % sintéticos; el componente `weather` y el ciclo diario son decisiones de modelado | Documentar el generador completo; validar al menos con **un dataset real** (p. ej. Intel Berkeley Lab) aunque sea pequeño; análisis de sensibilidad de `weather_std` y de la autocorrelación |
| Constructo | Un solo detector (vecinos + normalización por nodo): los resultados de halo dependen de que sea local | Añadir, si hay tiempo, un segundo detector (p. ej. GCN/autoencoder) como prueba de generalidad |
| Constructo | El escenario puede favorecer a las particiones estructuradas | **E4b** (anomalías en la frontera), múltiples magnitudes, grafos con comunidades solapadas (E3) |
| Interna | Tiempos medidos en secuencia (efectos de caché: `work_inflation` puede ser < 1); sin contención real | Etiquetar como modelo; **E6** con Ray para calibrar; reportar `repeats` y mínimo |
| Interna | Louvain depende de la versión de networkx y de la semilla | Fijar versión; registrar `louvain_n_communities`, modularidad y NMI; comparar con otra semilla |
| Interna | Umbral τ elegido con datos de otro conjunto de semillas | Calibración disjunta; sensibilidad ±20 % |
| Externa | Un tipo de grafo (`hybrid`), k ≤ 16, n ≤ 100k | Incluir `community` (sin coordenadas) y `powerlaw` como extras; declarar los límites en el artículo |
| Conclusión | Muchas comparaciones | Corrección de Holm; hipótesis fijadas **antes** de la campaña; informar también los resultados nulos |
| Conclusión | Pseudo-replicación | La semilla es la unidad; modelo mixto opcional |

---

## 9. Resultados del piloto (n=10 000, 5 semillas, 120 corridas)

Calidad respecto al centralizado y costo de halo. ΔF1 en puntos porcentuales, media ± desviación entre semillas;
speedup **modelado** (1 Gbit/s, 0.5 ms, con recolección de resultados).

| Método | k | Prof. | ΔF1 (pp) | ΔAUC | ΔMAE | MB halo | Replic. | Speedup* |
|---|---|---|---|---|---|---|---|---|
| hash | 4 | 0 | −11.06 ± 0.65 | −0.058 | +0.293 | 0.0 | 1.00 | 1.66 |
| hash | 4 | 2 | 0.00 ± 0.00 | 0.000 | 0.000 | 14.4 | 4.00 | 1.03 |
| hash | 8 | 0 | −22.20 ± 0.66 | −0.173 | +0.443 | 0.0 | 1.00 | 1.92 |
| hash | 8 | 1 | 0.00 ± 0.00 | 0.000 | +0.021 | 24.4 | 6.07 | 1.23 |
| hash | 8 | 2 | 0.00 ± 0.00 | 0.000 | 0.000 | 32.7 | 7.81 | 1.10 |
| spatial | 8 | 0 | −0.11 ± 0.05 | −0.001 | +0.005 | 0.0 | 1.00 | 1.92 |
| spatial | 8 | 2 | 0.00 ± 0.00 | 0.000 | 0.000 | 0.9 | 1.18 | 1.90 |
| community | 8 | 0 | −0.08 ± 0.03 | −0.001 | +0.001 | 0.0 | 1.00 | 1.98 |
| community | 8 | 2 | 0.00 ± 0.00 | 0.000 | 0.000 | 0.7 | 1.14 | 1.91 |
| community_refined | 8 | 0 | −0.08 ± 0.05 | −0.001 | +0.001 | 0.0 | 1.00 | 1.96 |

Contraste de equivalencia sin halo (IC 90 %, margen ±2.5 pp): `spatial`, `community` y `community_refined`
**dentro del margen** con k=4 y 8 (por ejemplo `community` k=8: [−0.11, −0.04] pp). `hash` fuera (−11 y −22 pp).

Relación con el corte de aristas (40 corridas con profundidad 0): Spearman(corte, ΔF1) = −0.70; ajuste
\|ΔF1\| ≈ 15.8 · corte^1.17, R² = 0.92. **Es exploratorio:** los puntos forman dos grupos (hash con corte
0.75-0.88 y estructuradas con 0.004-0.025), y casi todo el ajuste lo explica esa separación. E3 es lo que dará
valores intermedios de corte.

**Escala 100k (una semilla, k=8, profundidad 2):** speedup modelado de 3.4× (`community`, sin halo), 3.0×
(`community`), 3.2× (`spatial`) y 1.7× (`hash`, con 326 MB de halo y replicación 7.8×); sin contar la recolección de
resultados, 7.4-10.5× para las estructuradas y 2.6× para hash. Con n=10k el speedup modelado de las estructuradas
es solo de ~1.9× porque el centralizado tarda 0.26 s y la recolección domina.

**Piloto E4b (n=10 000, k=8, 4 semillas, ΔF1 sin halo en pp; IC 90 % entre corchetes):**

| Adversario | `hash` | `spatial` | `community` | `community_refined` | `community`, prof. 1 |
|---|---|---|---|---|---|
| Control uniforme | −22.28 | −0.11 | −0.09 | −0.09 | 0.00 |
| Frontera completa | −24.51 | −0.28 | **−1.46** [−1.52, −1.40] | −1.23 | 0.00 (0.23 MB) |
| Frontera fuerte | −19.40 ± 8.30 | +0.33 | **−4.74** [−7.79, −1.69] | −2.24 [−3.79, −0.69] | 0.00 (0.23 MB) |

Variante extrema con toda la frontera (n=3 000, k=16, 5 semillas): `community` −0.92 pp [−1.34, −0.50], `spatial`
−0.41, `hash` −31.7; sigue dentro del margen. En el adversario fuerte `spatial` apenas se ve afectada porque sus
fronteras son otras (solo el 17.5 % de las anomalías caen en su frontera frente al 68 % en la de `community`).

Lectura: colocar las anomalías en la frontera multiplica la pérdida de `community` por 16 (de −0.09 a −1.46 pp) y
hasta por 50 con el adversario fuerte, pero **solo ese último sale del margen de ±2.5 pp**. Con pocos eventos
regionales por semilla (1-2) la varianza es alta (±2.6 pp), así que el IC90 es amplio; con 30 semillas se afinará.
Un halo de 1 salto cuesta 0.23 MB y elimina la pérdida en todos los casos.

**Lo que el piloto NO permite afirmar:**
- No hay evidencia por separado en anomalías **regionales** ni con dificultad alta (E4); E4b mezcla puntuales y regionales.
- Con una sola semilla a 100k no hay inferencia; solo orden de magnitud.
- El speedup es una cota ideal. A n=10k el centralizado tarda ~0.26 s, así que **es esperable que el sistema real no
  gane nada a esa escala**; eso es un resultado válido, no un fallo.

---

## 9b. Resultados del paso 5 (procesos reales; entorno de desarrollo de **1 núcleo**)

Profundidad de halo 2 (la requerida). `speedup CPU` = estimación con el CPU real de cada worker; `modelo` = tiempo modelado con el
transporte local calibrado; `error` = (modelo − CPU est.) / CPU est. Todas las corridas con profundidad 2 son **equivalentes** al
centralizado; las 6 con profundidad 0 (solo a n=10k) no lo son, como se espera.

| n | Método | k | MB halo | Bytes reales / estimados | Speedup CPU | Speedup modelo | Error del modelo |
|---|---|---|---|---|---|---|---|
| 10 000 | hash | 4 | 14.40 | 1.0000 | 2.06 | 2.22 | −7.3 % |
| 10 000 | hash | 8 | 32.69 | 1.0000 | 2.23 | 2.48 | −10.1 % |
| 10 000 | community | 2 | 0.18 | 1.0020 | 2.19 | 2.26 | −3.0 % |
| 10 000 | community | 4 | 0.37 | 1.0060 | 4.71 | 4.80 | −1.8 % |
| 10 000 | community | 8 | 0.68 | 1.0090 | 6.12 | 6.52 | −6.2 % |
| 100 000 | community | 8 | 2.34 | 1.0030 | 5.70 | 6.59 | −13.5 % |
| 100 000 | hash | 4 | 143.92 | 1.0000 | 1.80 | 1.96 | −8.4 % |

Lectura:
- **Los bytes reales coinciden con la estimación** (0.00-0.87 % de diferencia: el encabezado de la serialización).
- **El modelo, calibrado con el transporte local, es optimista en promedio un 5.9 %** en las 14 corridas medidas (de 1.8 % a 13.5 %): omite el CPU de
  ensamblado y de serialización por corrida. Es una validación **de la estructura** del modelo, no de un speedup en paralelo.
- **La partición estructurada da ≈ 2.7× más speedup que hash con el mismo k** (6.1× frente a 2.2× con k=8 y n=10k), porque el
  halo de hash obliga a cada worker a procesar casi todo el grafo.
- **Sin halo se gana poco más**: `community` k=8 pasa de 6.12× a 6.78× al quitar el halo (+11 %).
- **El transporte local varió entre ejecuciones** (≈ 650-1 600 MB/s): repetir la calibración y reportar el rango.
- **Las cifras de speedup modeladas con 1 Gbit/s (≈ 1.9×, secciones 0 y 9) eran pesimistas para el transporte local**; el resultado
  depende de la red supuesta, que es justo lo que mide E5.
- **El tiempo de pared medido (0.46-1.86 s a n=10k) NO es un speedup**: con 1 núcleo los procesos se reparten el CPU
  (`rt_oversubscribed=True`). **Falta ese dato en una máquina con ≥ k núcleos.**

---

## 10. Desarrollo pendiente para ejecutar el plan

| Prioridad | Tarea | Necesaria para |
|---|---|---|
| ✅ Hecho | `anomaly_nodes` en `signals.py` (v1.2.0), `boundary_nodes` en `partition.py`, `run_e4b.py` | E4b (H8) |
| ✅ Hecho | Orquestador (`orchestrator.py`): JSON/YAML, semillas, orden aleatorio, **reanudable**, paralelo por semilla | Toda la campaña |
| ✅ Hecho | Análisis y figuras (`analysis.py`): bootstrap, Wilcoxon + Holm, TOST, δ de Cliff, Hodges-Lehmann, Pareto, ajuste log-log | Resultados |
| ✅ Hecho | Workers reales con procesos del SO (`runtime.py`); un backend de Ray queda como opción **sin probar** (Ray no disponible aquí) | E6, calibrar el modelo |
| **Alta** | **Lanzar `configs/campaign_full.json` y E6 con tiempo de pared en una máquina con varios núcleos** | Resultados finales (requiere tu equipo) |
| Media | Cargador de un dataset real | Validez de constructo |
| Baja | Segundo detector (GCN/autoencoder) | Generalidad |
| Baja | Louvain más rápido (`igraph`/`leidenalg`) | n=100k con muchas semillas |
| Baja | Halo por tarea o selectivo (sección 13): reevaluado, beneficio absoluto pequeño | Paso 7 |

---

## 11. Riesgos y plan B

| Riesgo | Probabilidad | Respuesta |
|---|---|---|
| E4b: el halo sigue siendo prescindible con el adversario de frontera completa | **Observado en el piloto** (−1.46 pp) | Resultado positivo: "la detección basada en vecinos tolera particiones estructuradas sin halo"; se publica con los límites declarados |
| E4b: el halo sí importa con el adversario fuerte | **Observado en el piloto** (−4.74 pp) | Se obtiene el criterio de cuándo hace falta (nodos que pierden ≥ 50 % de vecinos) y se motiva la frontera adaptativa: un halo de 1 salto solo en esos nodos |
| E6 muestra que el modelo de tiempo es muy optimista | Alta a n=10k | Se reporta el factor de error del modelo y se limita la afirmación de speedup a los puntos validados |
| `spatial` ≈ `community` en todo | Media | Se informa como hallazgo: para redes con coordenadas, la partición espacial basta y evita el costo de Louvain |
| Resultados nulos en speedup | Media | Es publicable como análisis de **cuándo conviene distribuir**, con el punto de cruce `n*` |

---

## 12. Lista de verificación antes de enviar

- [ ] `pytest -q` pasa en la máquina de la campaña; versiones congeladas.
- [ ] Hipótesis y márgenes (±2.5 pp, ±0.5 pp) registrados **antes** de ejecutar la campaña.
- [ ] τ calibrado con semillas disjuntas; sensibilidad ±20 % reportada.
- [ ] Cada figura y tabla se genera con un comando desde los CSV de resultados.
- [ ] Tiempos y speedup rotulados como "modelados" o validados con E6.
- [ ] E4b ejecutado y reportado, sea cual sea su resultado.
- [ ] Un dataset real (aunque pequeño) como validación de sentido.
- [ ] Paquete reproducible: código, manifests con SHA-256, CSV de resultados, `requirements.txt`, README.


---

## 13. Reevaluación del paso 7 (frontera adaptativa), con datos

Antes de construirla se midió el margen máximo de ahorro. Regla evaluada: **halo selectivo de 1 salto**: solo los nodos propios
cuya fracción de vecinos en otra partición es ≥ θ reciben a sus vecinos externos (θ=0 equivale a toda la frontera con
profundidad 1). Se contaron los nodos de halo necesarios (T=96; 5 bytes por valor). Es solo bytes: **no se ha medido la calidad**.

| n | Método | Datos completos | Prof. 1 | Prof. 2 | θ=0 (% nodos · MB) | θ=0.25 | θ=0.5 | θ=0.75 |
|---|---|---|---|---|---|---|---|---|
| 10 000 | community | 4.8 MB | 0.220 | 0.673 | 4.55 % · 0.220 | 2.04 % · 0.195 | 0.47 % · 0.081 | 0.02 % · 0.006 |
| 10 000 | spatial | 4.8 MB | 0.333 | 0.686 | 6.71 % · 0.333 | 3.92 % · 0.326 | 1.29 % · 0.221 | 0.15 % · 0.044 |
| 10 000 | hash | 4.8 MB | 24.236 | 32.681 | 100 % · 24.236 | 100 % · 24.236 | 99.94 % · 24.236 | 90.05 % · 23.922 |
| 100 000 | community | 48 MB | 0.626 | 2.335 | 1.30 % · 0.626 | 0.45 % · 0.494 | 0.07 % · 0.150 | 0.00 % · 0.000 |
| 100 000 | spatial | 48 MB | 1.597 | 3.268 | 3.30 % · 1.597 | 1.82 % · 1.548 | 0.54 % · 0.978 | 0.04 % · 0.122 |

Lectura:
- **Ahorro relativo grande en particiones estructuradas.** Con θ=0.5 respecto al halo de profundidad 1: `community` −63 % (n=10k) y
  −76 % (n=100k); `spatial` −34 % y −39 %. Ese θ protege exactamente a los nodos que el adversario fuerte de E4b ataca (43 nodos, 0.43 %).
- **Ahorro nulo con `hash`**: casi todos sus nodos pierden más de la mitad de sus vecinos, así que no hay a quién omitir.
- **Pero el ahorro absoluto es pequeño**: 0.14 MB a n=10k (3 % del volumen de datos) y 0.48 MB a n=100k (1 %). El halo de
  profundidad 1 de `community` ya cuesta solo el 4.6 % (10k) y el 1.3 % (100k) de los datos.
- **Halo por tarea** (detección con profundidad = `hops`, imputación con `impute_iters`): `community` pasa de 0.673 a 0.220 MB
  (−67 %) y `hash` de 32.7 a 24.2 MB (−26 %), con un costo de calidad solo en la imputación (ΔMAE +0.021 en `hash`, profundidad 1).

**Conclusión.** La "frontera adaptativa" como contribución central tiene un techo bajo: ahorra mucho en proporción, pero sobre
cantidades que ya son pequeñas, y no ayuda donde los bytes sí pesan (`hash`, cuyo problema es la partición). Como **ablación** sí es
defendible y barata de probar. Lo que falta para afirmar algo es medir si el halo selectivo con θ=0.5 recupera la pérdida de
−4.74 pp de E4b (con 0.081 MB en lugar de 0.220 MB).

| Opción | Valor científico | Esfuerzo | Recomendación |
|---|---|---|---|
| A. Lanzar la campaña completa + E6 con tiempo de pared; analizar y redactar | **Alto**: es lo que sustenta el artículo | Tiempo de cómputo (≈ 6-8 h de un núcleo) | **Hacerlo ya** |
| B. Validar con un dataset real (p. ej. Intel Berkeley Lab) | **Alto** en validez de constructo | Moderado; requiere descargar los datos (sin red aquí) | Hacerlo en paralelo con A |
| C. Prototipo de halo selectivo / por tarea como ablación | Medio-bajo | Pequeño-moderado | Solo si sobra tiempo; presentarlo como ablación, no como contribución |
| D. Backend de Ray | Bajo (el modelo ya se valida con procesos) | Moderado; no se puede probar aquí | Descartar salvo que lo exija la rúbrica |
