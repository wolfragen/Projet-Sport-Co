from Engine.Environment import LearningEnvironment
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import time
import torch

_executor = None

def get_executor(n_workers, initargs):
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=n_workers, 
                                        initializer=init_worker, initargs=initargs)
    return _executor

def init_worker(players_number, scoring_function, reward_coeff_dict):
    global runTest_env
    from Engine.Environment import LearningEnvironment
    runTest_env = LearningEnvironment(players_number=players_number, scoring_function=scoring_function, reward_coeff_dict=reward_coeff_dict, 
                              training_progression=1.0, display=False, human=False)


def _worker_runTest(n_game, n_players, agents, max_steps):
    torch.set_num_threads(1)
    env: LearningEnvironment = runTest_env
    
    states = [env.getState(player_id) for player_id in range(n_players)]

    sum_rewards = 0
    total_steps = 0
    score_left = 0
    score_right = 0

    sum_reward_dict = {
        "static_reward": 0,
        "delta_ball_goal_reward": 0,
        "delta_ball_player_reward": 0,
        "can_shoot_reward": 0,
        "has_ball_reward": 0,
        "goal_reward": 0,
        "total_reward": 0,
    }
    
    for _ in range(n_game):

        step = 1
        done = False

        total_reward = 0
        total_reward_dict = {
            "static_reward": 0,
            "delta_ball_goal_reward": 0,
            "delta_ball_player_reward": 0,
            "can_shoot_reward": 0,
            "has_ball_reward": 0,
            "goal_reward": 0,
            "total_reward": 0,
        }

        actions = [[] for _ in range(n_players)]

        while(not done):
            
            for player_id in range(n_players):
                agent = agents[player_id]
                state = states[player_id]
                with torch.no_grad():
                    action = agent.act(state, train=False)
                env.playerAct(player_id, action)
                    
                states[player_id] = state
                actions[player_id].append(action)
            
            rewards = env.step(debug=True)
            done = (env.isDone() or step>=max_steps)
            
            for player_id in range(n_players):
                state = states[player_id]
                next_state = env.getState(player_id)
                action = actions[player_id][step-1]
                reward = rewards[player_id][0]
                
                states[player_id] = next_state
                total_reward += reward
                
                reward_dict = rewards[player_id][1]
                for key,value in reward_dict.items():
                    total_reward_dict[key] += value
                
            step += 1

        if(env.score[0]+env.score[1]!=0):
            sum_rewards += total_reward
            total_steps += step-1
            score_left += env.score[0]
            score_right += env.score[1]

            for key,value in total_reward_dict.items():
                sum_reward_dict[key] += value

            total_reward = 0
            step = 1
        env.reset()

    return sum_rewards, total_steps, int(score_left), int(score_right), sum_reward_dict

def get_worker_distribution(n_workers: int, n_iter: int) -> list[int]:
    new_n_workers = cpu_count() if n_workers <= 0 else n_workers
    val: int = n_iter//new_n_workers
    worker_distribution: list[int] = [val]*new_n_workers
    remains = n_iter%new_n_workers
    for i in range(remains):
        worker_distribution[i] +=1
    return worker_distribution

def runTests_multithread(players_number, agents, scoring_function, reward_coeff_dict, max_steps, nb_tests=10_000, n_workers = 1):
    multithread_executor = get_executor(n_workers=n_workers, initargs=(players_number,scoring_function, reward_coeff_dict))
    total_rewards = 0
    total_rewards_dict = {
        "static_reward": 0,
        "delta_ball_goal_reward": 0,
        "delta_ball_player_reward": 0,
        "can_shoot_reward": 0,
        "has_ball_reward": 0,
        "goal_reward": 0,
        "total_reward": 0,
        }
    total_steps = 0
    total_score_left = 0
    total_score_right = 0

    worker_distribution = get_worker_distribution(n_workers = n_workers, n_iter=nb_tests)
    
    n_players=players_number[0] + players_number[1]

    start_time = time.time()

    results = list(multithread_executor.map(_worker_runTest, worker_distribution, [n_players]*n_workers, [agents]*n_workers, [max_steps]*n_workers))

    for i in range(n_workers):
        
        rewards, n_steps, score_left, score_right, reward_dict= results[i]
        
        total_rewards += rewards
        total_steps += n_steps
        total_score_left += score_left
        total_score_right += score_right

        for key,value in reward_dict.items():
            total_rewards_dict[key] += value
    
    total_not_draw = total_score_left + total_score_right
    total_nb_fail = nb_tests - total_score_right

    print(f"[{int(time.time()-start_time)}s] {nb_tests} tests | Reward: {total_rewards/total_not_draw:.2f} | Steps: {total_steps/total_not_draw:.1f} | W/D/L: {total_score_left}/{nb_tests-total_not_draw}/{total_score_right} |Score: {total_score_left/nb_tests:.2f} / {total_score_right/nb_tests:.2f} | failed: {total_nb_fail/nb_tests:.3f}")
    print("Reward breakdown: ")
    for key,value in total_rewards_dict.items():
        new_val = value/total_not_draw
        total_rewards_dict[key] = new_val
        print(f"{key}: {new_val:.2f} | ", end="")
    print("")
    return total_rewards/total_not_draw, total_steps/total_not_draw, total_score_left/nb_tests, total_score_right/nb_tests, total_nb_fail/nb_tests, total_rewards_dict