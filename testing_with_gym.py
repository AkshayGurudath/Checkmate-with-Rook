import gym
import ppo
import numpy as np

np.random.seed(42)

if __name__ == "__main__":
    env=gym.make("CartPole-v1")
    s=env.reset()
    print(env.action_space)
    model=ppo.PPO(s.shape[0],2, learning_rate=5e-4, gamma=0.98, lambd=0.95, epsilon=0.1, num_epochs=3, batch_size=64)
    iters=10000
    for i in range(iters):
        s=env.reset()
        history, reward=ppo.collect_data(env, 20, 30, model, 64)
        model.populate_history(history)
        print("# of iters:{},avg_score:{}".format(i,reward))
        model.train()







