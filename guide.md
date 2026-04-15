# Guide IA — PPO

## Contexte

Le projet simule un jeu type football top-down : 1 à N joueurs par équipe sur un terrain rectangulaire, une balle, des buts à gauche/droite. L'IA est entraînée par **PPO (Proximal Policy Optimization)** en self-play. Le moteur physique tourne en Numba vectorisé sur 512 environnements en parallèle (`Settings.N_WORKERS`).

Trois modes d'entraînement sont disponibles dans `main.py` :
- **Solo (1v0)** — un seul joueur contre l'environnement (Utilisé pour tester de nouveaux algorithmes et vérifier que l'environnement est correct).
- **1v1** — un joueur contre un opposant en self-play.
- **Team (NvN)** — chaque équipe partage une seule policy, self-play.

---

## Espace d'observation (entrée du réseau)

La vision est calculée par joueur, **dans son référentiel** (les coordonnées sont mirrorées pour l'équipe droite, donc la policy partagée traite les deux côtés de manière symétrique).

| Index |          Contenu           |              Description                |
|-------|----------------------------|-----------------------------------------|
|  0–1  |  `sin(angle), cos(angle)`  |         Orientation du joueur           |
|  2–3  |     `dx_ball, dy_ball`     | Position relative - balle / dim_terrain |
|  4–5  | `dx_own_goal, dy_own_goal` |      Position relative - but allié      |
|  6–7  | `dx_opp_goal, dy_opp_goal` |     Position relative - but adverse     |
|  8–9  | `rel_vx_ball, rel_vy_ball` |        Vitesse relative - balle         |
|  10+  | `dx_ball, dy_ball [, team_flag]` | Pour chaque autre joueur (coéquipiers d'abord) |
|  −2   |        `my_score / 10`     |   Score de l'équipe du joueur (max=10)  |
|  −1   |        `opp_score / 10`    |  Score de l'équipe adverse (max=10)     |

**Tailles concrètes** :
- 1v0 : `ENTRY_NEURONS = 10 + 2 = 12`
- 1v1 : `10 + 1×2 + 2 = 14`
- 2v2 : `10 + 3×3 + 2 = 21`

**L'ordre coéquipiers-d'abord** (`Vision.py:121-148`, `numba_env.py:218-235`) garantit que le slot du coéquipier est identique pour tous les membres de l'équipe — sinon la policy partagée ne serait plus symmétrique.

**Critic centralisé (team uniquement)** : `getGlobalVision()` produit un état global de dimension `6 + n_players × 5` (ball : pos+vel = 4, par joueur : pos+vel+team = 5, plus `score_left` et `score_right` normalisés). Le critic team voit tout, l'actor voit uniquement la vision locale. C'est l'architecture **CTDE** (Centralized Training, Decentralized Execution).

Les deux scores normalisés (`score / 10`) sont nécessaires pour que toute reward dépendant du score (ex. `static_lead_reward`) soit prédictible par le critic — sinon ce signal devient du bruit pur dans la fonction de valeur.

---

## Espace d'action

Espace **discret de 4 actions** (`AIActions.py:31-38`) :

| Action | Effet |
|--------|-------|
| 0 | Avancer dans la direction de l'orientation |
| 1 | Tourner à gauche (vitesse angulaire négative) |
| 2 | Tourner à droite (vitesse angulaire positive) |
| 3 | Tirer (propulse la balle si à portée, sinon no-op) |

Le tir est conditionnel à `canShoot()` (distance avant-balle < `PLAYER_SHOOTING_RANGE`).

---

## Fonction de récompense

Combinaison linéaire de termes shaped (`AI/Rewards/Reward.py`, coefficients dans `main.py`). Chaque step renvoie une reward par joueur :

| Coefficient | Sémantique |
|-------------|------------|
| `static_lead_reward` | Bonus constant si l'équipe mène |
| `static_draw_reward` | Malus constant si égalité |
| `delta_ball_player_coeff` | Récompense quand le joueur s'approche de la balle (bootstrap, induit le ball-chasing) |
| `delta_ball_goal_coeff` | Récompense quand la balle s'approche du but adverse |
| `has_ball_coeff` | Récompense de possession |
| `can_shoot_coeff` | Récompense de positionnement de tir |
| `goal_coeff` | Reward sparse à chaque but marqué |
| `wrong_goal_coeff` | Pénalité sur but encaissé |

En **team**, la reward effective d'un joueur est `(1 - team_reward_ratio) × indiv + team_reward_ratio × mean(équipe)` (`PPO.py:1119-1128`). Un `team_reward_ratio` élevé (~0.9) limite le ball-chasing collectif.

---

## Architecture des réseaux

Networks fully-connected (`AI/Networks/DeepRLNetwork.py`) :
- **Actor** : `(ENTRY_NEURONS, 128, 64, 64, 32, 4)` → softmax sur 4 actions, sample dans une `Categorical`.
- **Critic** : `(critic_dim, 128, 128, 64, 64, 1)` → scalaire `V(s)`.

Optimizer Adam, `lr_actor=1e-4`, `lr_critic=3e-4`, ReLU entre couches.

---

## PPO — algorithme et hyperparamètres

Implémentation standard (`PPO.py:243`, `replay()`) :
1. **Collecte** : `rollout_size = N_WORKERS × 64` transitions, exécutées en parallèle sur 512 envs. (Assure de récupérer 64 steps de chaque partie à chaque rollout)
2. **GAE** (`compute_gae`, `lmbda=0.95`, `gamma=0.99~0.995`) — réordonné pour respecter la séparation par-env (`reorder_memory_for_gae`).
3. **Backprop** : `n_epoch=4` passages sur le rollout, mini-batches, loss clipée :
   `L = -min(rπ·A, clip(rπ, 1±ε)·A) + 0.5·MSE(V) - 0.01·H(π)`
   avec `clip_eps=0.2`, `entropy_loss_coeff=0.01`, `critic_loss_coeff=0.5`, `max_grad_norm=1.0`.
4. **`normalize_advantage=True`** stabilise le gradient quand la magnitude de la reward varie.

Un *replay* dans les logs = un cycle collecte+backprop, **pas** une partie complète.

---

## Self-play et pool d'adversaires

Le pool est une rolling window de `max_pool_size` clones gelés. À chaque `opponent_save_interval` replays : un clone du modèle courant est ajouté, le plus ancien est éjecté, et un nouveau snapshot `actor_{N}.pt` est écrit sur disque.

**Sélection** : à un instant donné, **seulement 2 opposants distincts** sont actifs sur les 512 environnements (`PPO.py:365-378`) :
- 85 % des envs jouent contre `opponent_pool[-1]` (le clone le plus récent).
- 15 % jouent contre un `random_pick` unique, rafraîchi au prochain `opponent_save`.

Chaque environnement garde son opposant assigné jusqu'à la **fin de sa partie** (`done_timeout`), puis re-tire.
J'aurai préféré utiliser plus d'adversaires, mais chaque ajout d'adversaire augmente fortement le temps d'un forward par le GPU (plus de batchs, et plus petits).

---

## Checkpoint / reprise

Format de save (un dossier par mode, ex. `Networks/1v1/`) :

| Fichier | Contenu | Politique |
|---------|---------|-----------|
| `actor_{N}.pt` | weights actor, snapshots numérotés | tous conservés, alimentent le pool |
| `critic.pt` | weights critic | écrasé à chaque save |
| `optimizer.pt` | états Adam (actor + critic) | écrasé à chaque save |
| `checkpoint.json` | `total_replays`, `total_games`, `total_models`, `mode` | écrasé |

Pour reprendre un entraînement : passer `resume_from=save_folder+"1v1/"` à la fonction. Elle charge automatiquement le `actor_{N}.pt` avec le N max trouvé sur disque, ainsi que le critic, l'optimizer, les métadonnées, et reconstitue le pool d'adversaires depuis les `max_pool_size` derniers actors. L'optimizer Adam garde ses moments (pas de spike post-reprise).

---

## Tentative d'entraînement 2v2

### Configuration de l'environnement

Le terrain est réduit pour densifier les interactions :
- `Settings.SIZE_MODIFIER = 0.7` — joueurs, balle et range de tir multipliés par 0.7. La friction du sol et la vitesse de tir sont également scalées (`Settings.py:apply_size_modifier`) pour que la balle ne traverse pas le terrain en un seul tir.
- `players_number = (2, 2)` → `n_players = 4`, `ENTRY_NEURONS = 21`.
- Critic centralisé dim = `6 + 4×5 = 26`.

### Coefficients de reward

| Coefficient | Valeur | Rôle |
|-------------|--------|------|
| `static_lead_reward` | `+0.003` | Bonus constant si l'équipe mène |
| `static_draw_reward` | `−0.001` | Léger malus en cas d'égalité |
| `delta_ball_player_coeff` | `+0.002` | Bootstrap : récompense l'approche de la balle |
| `delta_ball_goal_coeff` | `+0.02` | Récompense quand la balle progresse vers le but adverse |
| `has_ball_coeff` | `+0.01` | Possession |
| `can_shoot_coeff` | `+0.2` | Positionnement de tir valide |
| `goal_coeff` | `+1.0` | But marqué |
| `wrong_goal_coeff` | `−1.0` | But encaissé |

`team_reward_ratio = 0.9` — chaque joueur reçoit 10 % de sa reward individuelle et 90 % de la moyenne d'équipe, pour limiter le ball-chasing collectif.

### Hyperparamètres PPO

- Architecture : actor `(21, 128, 64, 64, 32, 4)`, critic `(26, 128, 128, 64, 64, 1)`.
- `lr_actor = 1e-4`, `lr_critic = 3e-4`, Adam, `n_epoch = 4`, pas de lr_decay.
- `gamma = 0.995`, `lmbda = 0.95`, `clip_eps = 0.2`.
- `entropy_loss_coeff = 0.01`, `critic_loss_coeff = 0.5`, `normalize_advantage = True`, `max_grad_norm = 1`.
- `rollout_size = N_WORKERS × n_left × 64 = 65 536` transitions par cycle.
- `max_steps_per_game = 2048`, `draw_penalty = 0`.
- Self-play : `opponent_save_interval = 1024`, `max_pool_size = 20`.

### Observations

Le réseau a atteint sans difficulté les 95 %+ de réussite face à deux réseaux random, mais lors du test au bout de plusieurs heures d'entraînement, il ne semble pas y avoir de réelle coopération. Malgré le passage du `team_reward_ratio` à 0.9, les réseaux semblent tous les deux chasser la balle.

Enlever la reward pour se rapprocher de la balle (`delta_ball_player_coeff`) rend le début d'entraînement catastrophique.

Je me demande si je devrais l'enlever une fois le réseau efficace en test contre deux randoms. J'ai quand même peur que ça casse la policy actuelle, et surtout le critic, qui aura été entraîné sur une distribution de reward différente.
