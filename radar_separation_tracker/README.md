# Radar Separation Tracker (Stone Soup)

Suivi multi-cibles radar 3D (distance, azimut, élévation @ 1 Hz) avec détection de **séparation probable** via MHT local à fenêtre glissante.

**Stone Soup figé :** `stonesoup==1.9.1` (voir `requirements.txt`).

**Hypothèses de démonstration :** fichier `config/demo.yaml` — à remplacer par les caractéristiques radar réelles.

Documentation technique : [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Installation

```bash
cd radar_separation_tracker
pip install -r requirements.txt
pip install -e .
```

## Commandes

### 1. Générer un scénario

```bash
python scripts/generate_scenario.py --scenario unresolved_group_disperses --output-dir output
```

### 2. Exécuter le suivi

```bash
python scripts/run_tracking.py --detections output/unresolved_group_disperses_detections.csv --output-dir output
```

### 3. Visualiser

```bash
python scripts/visualize.py --detections output/probable_detach_opposite_detections.csv --output output/plot.png
```

### Film (séparation vérité vs détection)

```bash
python scripts/render_film.py --scenario probable_detach_opposite --output output/simulation_film.mp4 --fps 6
```

### 4. Tests

```bash
cd radar_separation_tracker && pytest -q
```

### 5. Benchmark

```bash
python scripts/run_benchmark.py --config config/demo.yaml
```

Le benchmark affiche temps médian/p95/max par scan, débit, mémoire max (RSS), pistes et hypothèses — sur **la machine où il est exécuté** (pas de promesse temps réel sans ce rapport).

## Format CSV

```
timestamp, radar_id, scan_id, detection_id, range_m, azimuth_rad, elevation_rad
```

Réordonnancement interne vers Stone Soup : `[élévation, azimut, distance]`.

## Scénarios simulés

| Nom | Description |
|-----|-------------|
| `probable_detach_opposite` / `aircraft_asm` | **Démo par défaut** : **avion** (virage lent, pas de 180° instantané) + **missile air-mer** au lâcher (t = 60 s) |
| `single_target` | Pas de séparation |
| `unresolved_group_disperses` | Scénario A : deux cibles toujours présentes mais **fusionnées** en mesure jusqu’à la résolution (séparation à 40 s dans la démo) |
| `probable_detach` | Détachement avec poursuite (écart de vitesse modéré, séparation à 50 s) |
| `crossing` | Croisement sans origine commune |
| `independent_birth_near` | Naissance indépendante proche |
| `dense_clutter` | Clutter dense |
| `missed_detections` | Rafale de Pd manquante |
| `azimuth_wrap` | Passage ±π |
| `stress_1000_det` | Charge ~1000 détections/scan |

Le tracker **ne reçoit jamais** la vérité terrain ni les étiquettes de scénario.

## Limites v1

- Une détection max par cible résolue et par scan.
- Au plus **deux branches** par événement de séparation (configurable).
- Scores = **scores** heuristiques, pas probabilités calibrées.
