# Informe de análisis

Filas: 42 | experimentos: E1_demo_halo, E3_demo_overlap, E4b_demo_adversarial

**H1 (equivalencia con halo suficiente):** 14 de 14 corridas con profundidad ≥ requerida son equivalentes al centralizado.

**H2 (±2.5 pp sin halo, anomalías no adversas, sin `hash`):** 4 de 5 combinaciones (caso × método × k) dentro del margen; 1 dentro del margen estricto ±0.5 pp.

**H8 (prueba adversa E4b, sin `hash`):** 1 de 1 combinaciones salen del margen ±2.5 pp sin halo.

*Nota:* con pocas semillas (n < 10) los IC son amplios; los veredictos solo son definitivos con las 30 semillas del plan.

**H3 (ley de potencia):** |ΔF1| ≈ 12.65·corte^1.03 (R²=0.56, IC95 exponente [0.77, 1.26], n=18). Exploratorio si los puntos forman solo dos grupos.


## Archivos
- `summary.csv`
- `comparisons.csv`
- `equivalence.csv`
- `loglog_fit.json`
- `figures/F2_equivalencia_vs_profundidad.png`
- `figures/F3_pareto_bytes_vs_F1.png`
- `figures/F5_speedup_vs_k.png`
- `figures/F7_solape_de_comunidades.png`
- `figures/F8b_mapa_calor_adversario.png`
- `figures/F9_frontera_vs_interior.png`
- `figures/F11_prueba_adversa.png`
- `figures/F4_perdida_vs_corte.png`