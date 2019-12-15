import gym
from gym import wrappers
import ppo
import numpy as np

np.random.seed(42)

#hyperparameters
learning_rate = 5e-4
gamma = 0.99
lambd = 0.95
epsilon = 0.1
num_epochs = 3
num_actors = 8
T_horizon = 128
batch_size = 128
num_iters = 500
print_interval = 20

def train(env, model, num_iters):
    #copy global variables to local
    iters = num_iters
    _print_interval = print_interval
    horizon = T_horizon
    score = 0
    render = False
    # run the algorithm
    for i in range(1, iters+1):
        start_state = env.reset()
        avg_score = 0
        # collect data with N actors (N = num_actors)
        for actors in range(num_actors):
            actor_history, actor_score, start_state, done = ppo.generate_trajectory(env, model, start_state, horizon, render)
            model.add_advantage_targets(actor_history)
            avg_score += actor_score
            if done:
                start_state = env.reset()
        score += (avg_score / num_actors)
        # now train for K epochs (K = num_epochs)
        model.train()
        # print results
        if i % print_interval== 0:
            score /= print_interval
            print("# of iters:{},avg_score:{}".format(i, score))
            score = 0

def store_output(model, folder_name):
    # render and store final output
    env_to_wrap = gym.make("CartPole-v1")
    env = wrappers.Monitor(env_to_wrap, './videos/{}/'.format(folder_name), force=True)
    render = True
    start_state = env.reset()
    actor_history, actor_score, start_state, done = ppo.generate_trajectory(env, model, start_state, 500, render)
    env.close()
    env_to_wrap.close()

if __name__ == "__main__":
    env=gym.make("CartPole-v1")
    s=env.reset()
    model=ppo.PPO(s.shape[0], 2,  learning_rate=learning_rate, gamma=gamma, lambd=lambd, epsilon=epsilon, num_epochs=num_epochs, batch_size=batch_size)

    store_output(model, '0')

    train(env, model, 250)
    env.close()

    store_output(model, '250')

    env=gym.make("CartPole-v1")
    train(env, model, 250)
    env.close()

    store_output(model, '500')

    env=gym.make("CartPole-v1")
    train(env, model, 500)
    env.close()

    store_output(model, '1000')
