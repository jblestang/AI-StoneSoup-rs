# Choix techniques et hypothèses

## Mesures et Stone Soup 1.9.1

- État cartésien : \(\mathbf{x} = [x, v_x, y, v_y, z, v_z]^\top\) (m, m/s), repère radar local (Est, Nord, Haut).
- Modèle de transition : `CombinedLinearGaussianTransitionModel` avec trois `ConstantVelocity(noise_diff_coeff)` (Stone Soup).
- Modèle de mesure : `CartesianToElevationBearingRange`, `ndim_state=6`, `mapping=(0, 2, 4)`.
- **Ordre Stone Soup** : `[élévation, azimut (bearing), distance]` (rad, rad, m).
- **CSV d'entrée** : `range_m, azimuth_rad, elevation_rad` → réordonné à l'import.
- `rotation_offset` du modèle aligne l'orientation du radar (rad, axes x,y,z Stone Soup).

### Conventions angulaires

- Azimut : `atan2(y, x)` (bearing Stone Soup), rad, wrap via `Bearing` / `mod_2pi` pour les résidus ±π.
- Élévation : `asin(z / r)`.

## Filtre : EKF vs UKF

**Choix retenu : EKF** (`ExtendedKalmanPredictor` / `ExtendedKalmanUpdater`).

- Transition CV : linéaire → pas de gain UKF sur la prédiction.
- Mesure CBR : non linéaire mais lisse hors singularités (r > 0) ; linéarisation suffisante à 1 Hz pour des cibles aériennes typiques.
- UKF disponible via `filter: ukf` dans la config (même modèles, `UnscentedKalman*`).

## Scans et temps

- Chaque `(timestamp, scan_id)` = une acquisition ; les détections d'un scan ne sont pas des pas temporels successifs.
- Δt de prédiction = différence d'horodatages entre scans **reçus** (scan sans détection ≠ scan manquant).

## Résolution vs précision

- **Précision** : diagonal de `noise_covar` (σ_θ, σ_φ, σ_r).
- **Résolution** : deux cibles dont les mesures sphériques diffèrent de moins que `(Δθ_res, Δφ_res, Δr_res)` produisent **une** détection fusionnée.

### Mesure fusionnée (approximation)

Positions cartésiennes \(\mathbf{p}_i\), poids \(w_i \propto 1/\sigma_{r,i}^2\) (proxy SNR) :

\[
\mathbf{p}_\mathrm{fused} = \frac{\sum_i w_i \mathbf{p}_i}{\sum_i w_i}, \quad
\mathbf{R}_\mathrm{fused} = \mathbf{R}_\mathrm{prec} + \mathrm{diag}(\Delta\theta_\mathrm{res}^2, \Delta\phi_\mathrm{res}^2, \Delta r_\mathrm{res}^2)
\]

La détection fusionnée compte **une fois** dans la vraisemblance (une mise à jour / une branche d'association).

## Score (pas une probabilité calibrée)

Pour une hypothèse \(H\) sur une fenêtre de scans sans double comptage (chaque scan compté une fois par branche active) :

\[
S(H) = \sum_k \log L(z_k \mid H) + \sum_k \log P_d^{n_\mathrm{det}} (1-P_d)^{n_\mathrm{miss}} + n_\mathrm{clutter} \log \lambda - \eta \cdot n_\mathrm{branches}
\]

- \(L\) : densité gaussienne du gating / mise à jour EKF (innovation avec covariance).
- Résidus d'azimut : différence wrapée dans \([-\pi,\pi)\).
- **Origine commune** (score additif) : rétropropagation des états estimés ; test si les tubes d'incertitude (σ ellipsoïde) intersectent un point de séparation plausible avant le premier scan résolu observable.

## Architecture MHT locale

1. Association NN/gating pour pistes non ambiguës.
2. Groupes locaux : pistes + détections mutuellement proches dans le gate.
3. Hypothèses à 2 branches max : (1) cible seule, (2) naissance indépendante, (3) groupe non résolu → multi, (4) détachement (poursuite + fille).
4. Élagage par score + budget `max_hypotheses`.
5. Événements : confirmation retardée (`separation_confirmation_scans`).

Une bifurcation dans l'arbre **n'est pas** un événement ; export seulement après confirmation.

## Performance

- Présélection : grille 3D grossière (cell size ∝ gate range-dependent).
- Gating χ² (3 dof, `gate_probability`).
- Pas d'énumération globale jointe.

## Avion + missile air-mer (`probable_detach_opposite`, alias `aircraft_asm`)

Hypothèses **simulation** :

- **AC (avion)** : ~155 m/s, altitude ~8,5 km, lacet **15°/s** (demi-tour en ~12 s, charge ~5 g à cette vitesse — hyp. simulation) — **pas de flip instantané**.
- **MS (missile)** : lâcher à t = 60 s avec l’état du porteur ; boost ~**40 m/s²** pendant ~**7 s** ; cap de tir incliné ~**25°** vers la mer ; descente ensuite (gravité réduite en phase terminale simplifiée).
- Événement : `probable_detach` (porteur + fille missile), pas une preuve balistique.

## Limites v1

- Au plus **une** détection par cible résolue et par scan (pas de multi-échos / filiation automatique).
- Dépendances entre branches issues d'un même historique : approximation **conservatrice** — score partagé du tronc commun compté une fois par hypothèse globale, pas par branche dupliquée.
