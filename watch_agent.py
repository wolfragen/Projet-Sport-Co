import argparse

import pygame

import Settings
from AI.Algorithms.PPO import PPOAgent
from AI.Algorithms.RANDOM import RandomAgent
from AI.Rewards.Reward import computeReward
from Engine.Environment import LearningEnvironment
from Graphics.GraphicEngine import startDisplay


def configure_vision(competitive: bool) -> None:
    Settings.COMPETITIVE_VISION = competitive
    Settings.ENTRY_NEURONS = 11 if competitive else 9


def build_ppo_agent(action_dim: int = 4, cuda: bool = False) -> PPOAgent:
    actor_dims = [Settings.ENTRY_NEURONS, 256, 128, 64, action_dim]
    critic_dims = [Settings.ENTRY_NEURONS, 256, 128, 64, 1]
    reward_coeff_dict = {
        "static_reward": -0.002,
        "delta_ball_player_coeff": 0.01,
        "delta_ball_goal_coeff": 0.03,
        "can_shoot_coeff": 0.1,
        "has_ball_coeff": 0.0,
        "goal_coeff": 8.0,
        "wrong_goal_coeff": -2.0,
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

    return PPOAgent(
        dimensions=(actor_dims, critic_dims),
        scoring_function=scoring_fn,
        reward_coeff_dict=reward_coeff_dict,
        rollout_size=256,
        lr_actor=3e-4,
        lr_critic=1e-3,
        n_epoch=4,
        cuda=cuda,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a PPO agent play.")
    parser.add_argument("--model", type=str, required=True, help="Path to model.pt")
    parser.add_argument("--opponent", choices=["random", "self", "human"], default="random")
    parser.add_argument("--human-side", choices=["left", "right"], default="left")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    configure_vision(competitive=True)

    agent = build_ppo_agent(action_dim=4, cuda=args.cuda)
    agent.load(args.model)

    if args.opponent == "self":
        opponent = agent
    else:
        opponent = RandomAgent(action_dim=4)

    screen, draw_options = startDisplay()
    env = LearningEnvironment(
        players_number=(1, 1),
        scoring_function=agent.scoring_function,
        display=True,
        simulation_speed=args.speed,
        screen=screen,
        draw_options=draw_options,
        human=(args.opponent == "human"),
    )

    if args.opponent == "human":
        if args.human_side == "right":
            env.selected_player = 1
        else:
            env.selected_player = 0

    games = 0
    step = 0
    running = True
    while running and env.display and games < args.episodes:
        if args.opponent == "human":
            should_stop, action = env._processHumanEvents()
            if should_stop:
                running = False
                break

            human_id = env.selected_player
            agent_id = 1 if human_id == 0 else 0

            if action != -1:
                env.playerAct(human_id, action)

            action_agent = agent.act(env.getState(agent_id), train=False)
            env.playerAct(agent_id, action_agent)
            env.step(human_events=False)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            action_0 = agent.act(env.getState(0), train=False)
            action_1 = opponent.act(env.getState(1), train=False)

            env.playerAct(0, action_0)
            env.playerAct(1, action_1)
            env.step()

        step += 1
        if env.isDone() or step >= args.max_steps:
            env.reset()
            step = 0
            games += 1

    env._endDisplay()


if __name__ == "__main__":
    main()
