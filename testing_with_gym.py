import gym
import ppo
import numpy as np

np.random.seed(42)

#hyperparameters
learning_rate = 2.5e-4
gamma = 0.99
lambd = 0.95
epsilon = 0.1
num_epochs = 3
num_actors = 8
T_horizon = 128
batch_size = 32
num_iters = 10000
print_interval = 20

if __name__ == "__main__":
    env=gym.make("CartPole-v1")
    s=env.reset()
    print(env.action_space)
    model=ppo.PPO(s.shape[0], 2,  learning_rate=learning_rate, gamma=gamma, lambd=lambd, epsilon=epsilon, num_epochs=num_epochs, batch_size=batch_size)

    #copy global variables to local
    iters = num_iters
    _print_interval = print_interval
    horizon = T_horizon
    score = 0
    # run the algorithm
    for i in range(1, iters+1):
        start_state = env.reset()
        avg_score = 0
        # collect data with N actors (N = num_actors)
        for actors in range(num_actors):
            actor_history, actor_score, start_state, done = ppo.generate_trajectory(env, model, start_state, horizon)
            model.add_advantage_targets(actor_history)
            avg_score += actor_score
            if done:
                start_state = env.reset()
        score += (avg_score / num_actors)
        # now train for K epochs (K = num_epochs)
        model.train()
        # print results
        if i % print_interval== 0:
            print("# of iters:{},avg_score:{}".format(i, score / print_interval))
            score = 0
