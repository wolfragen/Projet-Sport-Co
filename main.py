# -*- coding: utf-8 -*-
"""
PPO training entrypoint.
"""

import argparse
import os

import Settings
from AI.Algorithms.PPO import (
    PPOAgent,
    pretrain_actor_imitation,
    train_PPO_competitive,
    train_PPO_model,
)
from AI.Rewards.Reward import computeReward


def configure_vision(competitive: bool) -> None:
    Settings.COMPETITIVE_VISION = competitive
    # Use a fixed input size to avoid shape mismatches between stages.
    # 11 = sin/cos + ball(2) + goals(4) + canShoot(1) + extras(2)
    Settings.ENTRY_NEURONS = 11
    Settings.RANDOM_BALL_POSITION = True


def build_ppo_agent(action_dim: int = 4, cuda: bool = False) -> tuple[PPOAgent, dict]:
    reward_coeff_dict = {
        "static_reward": -0.002,
        "delta_ball_player_coeff": 0.03,
        "delta_ball_goal_coeff": 0.08,
        "can_shoot_coeff": 0.3,
        "has_ball_coeff": 0.0,
        "goal_coeff": 25.0,
        "wrong_goal_coeff": -10.0,
        "align_ball_coeff": 0.05,
        "align_goal_coeff": 0.2,
        "shoot_when_can_coeff": 1.5,
        "shoot_without_ball_coeff": 0.12,
        "shoot_center_coeff": 1.0,
    }

    def scoring_fn(player, action, ball, left_goal_position, right_goal_position, score, training_progression=0.0):
        return computeReward(
            reward_coeff_dict,
            player,
            action,
            ball,
            left_goal_position,
            right_goal_position,
            score,
            training_progression,
        )

    actor_dims = [Settings.ENTRY_NEURONS, 256, 128, 64, action_dim]
    critic_dims = [Settings.ENTRY_NEURONS, 256, 128, 64, 1]

    agent = PPOAgent(
        dimensions=(actor_dims, critic_dims),
        scoring_function=scoring_fn,
        reward_coeff_dict=reward_coeff_dict,
        rollout_size=1024,
        lr_actor=3e-4,
        lr_critic=1e-3,
        n_epoch=10,
        lr_decay=True,
        clip_eps=0.2,
        gamma=0.99,
        lmbda=0.95,
        critic_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        normalize_advantage=True,
        max_grad_norm=1.0,
        cuda=cuda,
    )

    return agent, reward_coeff_dict


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    if not path.endswith(os.sep):
        path += os.sep
    return path


def unique_save_dir(base_path: str) -> str:
    """
    If base_path/model.pt exists, create an incremented folder: base_path_001, base_path_002, ...
    """
    base_path = base_path.rstrip("\\/")
    candidate = base_path
    if not os.path.exists(os.path.join(candidate, "model.pt")):
        return ensure_dir(candidate)
    idx = 1
    while True:
        candidate = f"{base_path}_{idx:03d}"
        if not os.path.exists(os.path.join(candidate, "model.pt")):
            return ensure_dir(candidate)
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent for SportCo.")
    parser.add_argument("--mode", choices=["1v0", "1v1"], default="1v1")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--save-dir", type=str, default=os.path.join("AI", "Models", "PPO"))
    parser.add_argument("--auto-save-dir", action="store_true")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--goal-len", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--no-imitation", action="store_true")
    parser.add_argument("--pretrain-episodes", type=int, default=8000)
    parser.add_argument("--pretrain-seconds", type=int, default=2400)
    parser.add_argument("--pretrain-max-steps", type=int, default=256)
    parser.add_argument("--warmup-random-episodes", type=int, default=500)
    parser.add_argument("--random-opponent-prob", type=float, default=0.25)
    parser.add_argument("--cuda", action="store_true", default=True)
    args = parser.parse_args()

    if args.seconds == 0:
        args.seconds = 10**9

    if args.goal_len is not None:
        Settings.GOAL_LEN = int(args.goal_len)

    competitive = args.mode == "1v1"
    configure_vision(competitive=competitive)

    agent, _ = build_ppo_agent(action_dim=4, cuda=args.cuda)
    if args.load_model:
        agent.load(args.load_model)
    save_dir = unique_save_dir(args.save_dir) if args.auto_save_dir else ensure_dir(args.save_dir)

    print(
        f"mode={args.mode} | episodes={args.episodes} | seconds={args.seconds} | "
        f"max_steps={args.max_steps} | log_interval={args.log_interval} | "
        f"save_dir={save_dir} | pretrain={args.pretrain} | imitation={not args.no_imitation}"
    )

    if args.pretrain:
        print(
            f"pretrain_episodes={args.pretrain_episodes} | pretrain_seconds={args.pretrain_seconds} | "
            f"pretrain_max_steps={args.pretrain_max_steps}"
        )

    try:
        if args.mode == "1v0":
            train_PPO_model(
                model=agent,
                max_duration=args.seconds,
                num_episodes=args.episodes,
                save_path=save_dir,
                interval_notify=args.log_interval,
                max_steps_per_game=args.max_steps,
                draw_penalty=-0.5,
            )
            return

        if args.pretrain:
            # Stage 1: 1v0 pretraining (learn to chase ball and shoot)
            configure_vision(competitive=False)
            if not args.no_imitation:
                pretrain_actor_imitation(
                    model=agent,
                    max_duration=args.pretrain_seconds,
                    num_episodes=max(1, args.pretrain_episodes // 3),
                    max_steps_per_game=args.pretrain_max_steps,
                    interval_notify=args.log_interval,
                )
            train_PPO_model(
                model=agent,
                max_duration=args.pretrain_seconds,
                num_episodes=args.pretrain_episodes,
                save_path=save_dir,
                interval_notify=args.log_interval,
                max_steps_per_game=args.pretrain_max_steps,
                draw_penalty=-0.5,
            )
            configure_vision(competitive=True)

        train_PPO_competitive(
            model=agent,
            max_duration=args.seconds,
            num_episodes=args.episodes,
            max_steps_per_game=args.max_steps,
            save_path=save_dir,
            interval_notify=args.log_interval,
            opponent_save_interval=50,
            max_pool_size=10,
            draw_penalty=-1.0,
            eval_interval=500,
            warmup_random_episodes=args.warmup_random_episodes,
            random_opponent_prob=args.random_opponent_prob,
        )
    except KeyboardInterrupt:
        print("Training interrupted, saving model...")
        agent.save(save_dir + "model.pt")
        print("Model saved.")


if __name__ == "__main__":
    main()
