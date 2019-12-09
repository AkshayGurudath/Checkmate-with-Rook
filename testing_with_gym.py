import gym
import ppo
import numpy as np

if __name__ == "__main__":
    env=gym.make("CartPole-v1")
    s=env.reset()
    model=ppo.PPO(s.shape[0],2)
    iters=1000
    for i in range(iters):
        s=env.reset()
        history=ppo.collect_data(env, 20, 20, model)
        reward = [ r for (_, _, _, r, _) in history]
        reward=np.average(reward)
        model.populate_history(history)
        print("# of iters:{},avg_score:{}".format(i,reward))
        model.train()







