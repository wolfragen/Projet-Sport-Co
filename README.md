## Projet Sport Co

Simulateur de foot en 2D, avec entraînement d'agents par renforcement (PPO principalement, DQN et une tentative de NEAT qui a misérablement échoué pour l'instant). La version actuelle repose sur un moteur physique vectorisé compilé avec Numba pour pouvoir entraîner sur plusieurs parties en parallèle.

### Description rapide

- Un terrain, des joueurs (carré), une balle (cercle), deux buts. Collisions gérées maison.
- Des agents : `PPO`, `DQN`, `NEAT`, `Random`. PPO est celui qui marche le mieux, autant en solo que 1v1.
- Un mode solo (1v0), 1v1, et XvX (jusqu'à 2v2 pour l'instant — au-delà ça n'a pas vraiment été testé).
- Un mode "humain" pour jouer au clavier contre un agent entraîné.
- Du multi-threading pour faire tourner les évaluations / matchs en parallèle, et un système de pool d'adversaires (self-play) pour le 1v1 et le XvX.
- Un classement Elo pour départager les modèles sauvegardés en cours d'entraînement.

### Lancer un Entraînement

Tout passe par `main.py`. La variable `MODE` en haut du fichier choisit ce qu'on fait :

- `MODE = 0` → entraînement solo (1v0)
- `MODE = 1` → entraînement 1v1
- `MODE = 2` → entraînement 2v2

Pour rejouer rapidement contre un modèle sauvegardé :

```
python demo.py
```
=> Il suffit juste de changer agent.load(save_folder + "1v1/actor_199.pt") avec le bon chemin d'accès.

### Structure

```
Engine/        moteur de jeu (entités, actions, vision, environnement)
Engine/Physics/    moteur physique vectorisé (Numba) — c'est lui qui fait tourner le batch
AI/Algorithms/ implémentations des algos (PPO, DQN, NEAT, Random)
AI/Rewards/    fonctions de récompense
Graphics/      rendu pygame
Play.py        modes de jeu humain / debug
Settings.py    constantes globales (taille terrain, vitesses, etc.)
```

Les modèles entraînés sont sauvegardés dans le dossier pointé par `Settings.SAVE_FOLDER` (pas versionné... Oui, il aurait dû être en gitignore et à l'intérieur du projet...).

### Dépendances

- Python 3.11
- PyTorch
- Numba
- Pygame
- numpy

### Notes

- Les anciens tests `Tests/` ont été supprimés après la dernière refonte, ils ne correspondaient plus au reste du code.
