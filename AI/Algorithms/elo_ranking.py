# -*- coding: utf-8 -*-
"""
Elo ranking system for saved actors.
Loads all actor_*.pt from a folder, runs a tournament, computes Elo ratings.
Three phases: Swiss (all), round-robin top 10 (500 games), round-robin top 3 (1000 games).
"""

import os
import re
import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from AI.AgentProxy import AgentProxy
from AI.Algorithms.parallel_testing import batched_run_tests
from Settings import Settings


def _load_actors(save_path, dimensions_actor):
    """Load all actor_*.pt files from a folder as AgentProxy objects."""
    pattern = re.compile(r"actor_(\d+|[\w]+)\.pt")
    actors = []
    for f in os.listdir(save_path):
        if pattern.match(f):
            path = os.path.join(save_path, f)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            proxy = AgentProxy(
                state_dict=state_dict,
                dimensions=list(dimensions_actor),
                agent_type="ppo",
                device="cpu"
            )
            actors.append({"file": f, "proxy": proxy, "elo": 1500.0, "games": 0, "wins": 0})
    actors.sort(key=lambda a: a["file"])
    print(f"Loaded {len(actors)} actors from {save_path}")
    return actors


def _elo_update(elo_a, elo_b, score_a, k=32):
    """Standard Elo update. score_a in [0, 1] (win rate of A)."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_elo_a = elo_a + k * (score_a - expected_a)
    new_elo_b = elo_b + k * (score_b - expected_b)
    return new_elo_a, new_elo_b


def _play_match(actor_a, actor_b, nb_games, players_number, reward_coeff_dict, max_steps):
    """Play nb_games between two actors. Returns (win_rate_a, win_rate_b, draw_rate)."""
    result = batched_run_tests(
        agents=[actor_a["proxy"], actor_b["proxy"]],
        nb_tests=nb_games,
        max_steps=max_steps,
        players_number=players_number,
        reward_coeff_dict=reward_coeff_dict,
        should_print=False,
    )
    win_rate_a = result[2]   # score_left / nb_tests
    win_rate_b = result[3]   # score_right / nb_tests
    draw_rate = 1.0 - win_rate_a - win_rate_b
    return win_rate_a, win_rate_b, draw_rate


def _swiss_pairing(actors):
    """Pair actors by closest Elo. Returns list of (idx_a, idx_b) tuples."""
    sorted_indices = sorted(range(len(actors)), key=lambda i: actors[i]["elo"], reverse=True)
    pairs = []
    used = set()
    for i in range(len(sorted_indices)):
        if sorted_indices[i] in used:
            continue
        for j in range(i + 1, len(sorted_indices)):
            if sorted_indices[j] in used:
                continue
            pairs.append((sorted_indices[i], sorted_indices[j]))
            used.add(sorted_indices[i])
            used.add(sorted_indices[j])
            break
    return pairs


def _run_round_robin(actors, indices, nb_games, players_number, reward_coeff_dict, max_steps, elo_k, phase_name):
    """Run round-robin between actors at given indices. Updates Elo in-place."""
    n = len(indices)
    total_matches = n * (n - 1) // 2
    print(f"\n{'='*60}")
    print(f"  {phase_name}: {n} actors, {total_matches} matchups x {nb_games} games")
    print(f"{'='*60}")

    match_count = 0
    start = time.time()

    for i in range(n):
        for j in range(i + 1, n):
            idx_a, idx_b = indices[i], indices[j]
            a, b = actors[idx_a], actors[idx_b]

            win_a, win_b, draw = _play_match(a, b, nb_games, players_number, reward_coeff_dict, max_steps)

            # Score for Elo: win=1, draw=0.5, loss=0
            score_a = win_a + draw * 0.5
            a["elo"], b["elo"] = _elo_update(a["elo"], b["elo"], score_a, k=elo_k)
            a["games"] += nb_games
            b["games"] += nb_games
            a["wins"] += int(round(win_a * nb_games))
            b["wins"] += int(round(win_b * nb_games))

            match_count += 1
            elapsed = time.time() - start
            eta = (elapsed / match_count) * (total_matches - match_count)
            print(f"  [{match_count}/{total_matches}] {a['file']} vs {b['file']} | "
                  f"W/D/L: {win_a:.0%}/{draw:.0%}/{win_b:.0%} | "
                  f"Elo: {a['elo']:.0f} vs {b['elo']:.0f} | ETA: {int(eta)}s")


def rank_actors(
    save_path,
    players_number=(1, 1),
    reward_coeff_dict=None,
    dimensions_actor=None,
    nb_games_phase1=100,
    nb_games_phase2=1000,
    nb_games_phase3=10000,
    max_steps=2048,
    swiss_rounds=15,
    elo_k=320,
    top_n_phase2=10,
    top_n_phase3=3,
):
    """
    Rank all actors in a folder using a 3-phase tournament.
    Phase 1: Swiss tournament (all actors, fast)
    Phase 2: Round-robin top 10 (500 games each)
    Phase 3: Round-robin top 3 (1000 games each)

    Returns: filename of the best actor.
    Saves elo_ranking.csv in save_path.
    """
    assert reward_coeff_dict is not None, "reward_coeff_dict required"
    assert dimensions_actor is not None, "dimensions_actor required"

    actors = _load_actors(save_path, dimensions_actor)
    if len(actors) < 2:
        print("Need at least 2 actors to rank.")
        return actors[0]["file"] if actors else None

    # ================================================================
    # Phase 1: Swiss tournament
    # ================================================================
    n_rounds = min(swiss_rounds, len(actors) - 1)
    print(f"\n{'='*60}")
    print(f"  Phase 1: Swiss tournament — {len(actors)} actors, {n_rounds} rounds x {nb_games_phase1} games")
    print(f"{'='*60}")

    start = time.time()
    for rnd in range(1, n_rounds + 1):
        pairs = _swiss_pairing(actors)
        for idx_a, idx_b in pairs:
            a, b = actors[idx_a], actors[idx_b]
            win_a, win_b, draw = _play_match(a, b, nb_games_phase1, players_number, reward_coeff_dict, max_steps)
            score_a = win_a + draw * 0.5
            a["elo"], b["elo"] = _elo_update(a["elo"], b["elo"], score_a, k=elo_k)
            a["games"] += nb_games_phase1
            b["games"] += nb_games_phase1
            a["wins"] += int(round(win_a * nb_games_phase1))
            b["wins"] += int(round(win_b * nb_games_phase1))

        # Print round summary
        sorted_actors = sorted(actors, key=lambda a: a["elo"], reverse=True)
        top3 = ", ".join(f"{a['file']}({a['elo']:.0f})" for a in sorted_actors[:3])
        elapsed = int(time.time() - start)
        print(f"  Round {rnd}/{n_rounds} done [{elapsed}s] | Top 3: {top3}")

    # ================================================================
    # Phase 2: Round-robin top N (selected from Swiss ranking)
    # ================================================================
    sorted_by_elo = sorted(range(len(actors)), key=lambda i: actors[i]["elo"], reverse=True)
    n_phase2 = min(top_n_phase2, len(actors))
    top_indices = sorted_by_elo[:n_phase2]

    print(f"\n  Phase 2 candidates: {[actors[i]['file'] for i in top_indices]}")

    # Reset Elo for top actors to avoid Swiss bias
    for idx in top_indices:
        actors[idx]["elo"] = 1500.0

    _run_round_robin(actors, top_indices, nb_games_phase2, players_number, reward_coeff_dict, max_steps, elo_k,
                     f"Phase 2: Top {n_phase2} round-robin")

    # ================================================================
    # Phase 3: Round-robin top 3 (selected from Phase 2 results ONLY)
    # ================================================================
    # Sort only the phase 2 participants by their updated Elo
    phase2_sorted = sorted(top_indices, key=lambda i: actors[i]["elo"], reverse=True)
    n_phase3 = min(top_n_phase3, len(phase2_sorted))
    top3_indices = phase2_sorted[:n_phase3]

    print(f"\n  Phase 3 candidates: {[actors[i]['file'] for i in top3_indices]}")

    # Reset Elo for final precision
    for idx in top3_indices:
        actors[idx]["elo"] = 1500.0

    _run_round_robin(actors, top3_indices, nb_games_phase3, players_number, reward_coeff_dict, max_steps, elo_k,
                     f"Phase 3: Top {n_phase3} final precision")

    # ================================================================
    # Results — hierarchical ranking: top3 (Phase 3) > top10 (Phase 2) > rest (Swiss)
    # Elo is rescaled so tiers don't overlap.
    # ================================================================

    # Tier 1: top 3 sorted by Phase 3 Elo
    tier1 = sorted(top3_indices, key=lambda i: actors[i]["elo"], reverse=True)
    # Tier 2: top 10 minus top 3, sorted by Phase 2 Elo
    tier2 = sorted([i for i in top_indices if i not in top3_indices],
                   key=lambda i: actors[i]["elo"], reverse=True)
    # Tier 3: everyone else, sorted by Swiss Elo
    tier3 = sorted([i for i in range(len(actors)) if i not in top_indices],
                   key=lambda i: actors[i]["elo"], reverse=True)

    # Rescale Elo so tiers are coherent:
    # Tier 3 keeps its Swiss Elo as-is (base range)
    # Tier 2 is shifted so its minimum is above tier 3's maximum
    # Tier 1 is shifted so its minimum is above tier 2's maximum
    if tier3:
        tier3_max = max(actors[i]["elo"] for i in tier3)
    else:
        tier3_max = 1500.0

    if tier2:
        tier2_min = min(actors[i]["elo"] for i in tier2)
        tier2_offset = tier3_max + 50.0 - tier2_min  # 50 points gap between tiers
        for i in tier2:
            actors[i]["elo"] += tier2_offset
        tier2_max = max(actors[i]["elo"] for i in tier2)
    else:
        tier2_max = tier3_max

    if tier1:
        tier1_min = min(actors[i]["elo"] for i in tier1)
        tier1_offset = tier2_max + 50.0 - tier1_min  # 50 points gap between tiers
        for i in tier1:
            actors[i]["elo"] += tier1_offset

    final_ranking = tier1 + tier2 + tier3

    print(f"\n{'='*60}")
    print(f"  FINAL RANKING")
    print(f"{'='*60}")
    for rank, idx in enumerate(final_ranking, 1):
        a = actors[idx]
        wr = a["wins"] / max(a["games"], 1)
        tier = "***" if idx in top3_indices else "** " if idx in top_indices else "   "
        print(f"  #{rank:3d} {tier} | {a['file']:25s} | Elo: {a['elo']:7.1f} | Games: {a['games']:5d} | Win%: {wr:.1%}")

    # Save CSV
    rows = []
    for rank, idx in enumerate(final_ranking, 1):
        a = actors[idx]
        rows.append({
            "rank": rank,
            "actor_file": a["file"],
            "elo": round(a["elo"], 1),
            "games_played": a["games"],
            "win_rate": round(a["wins"] / max(a["games"], 1), 4),
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_path, "elo_ranking.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRanking saved to {csv_path}")

    best_idx = final_ranking[0]
    best = actors[best_idx]["file"]
    print(f"Best actor: {best} (Elo: {actors[best_idx]['elo']:.1f})")
    return best
