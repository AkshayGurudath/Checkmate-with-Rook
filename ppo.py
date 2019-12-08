#use python 3

import tensorflow as tf
import numpy as np

def add_advantage_targets(history, value, gamma, lambd):
    advantage = 0.0
    for i in range(len(history)-1, 0, -1):
        s, a, s_prime, r, prob_action = history[i]
        #TODO: value
        target = r + gamma * value(s_prime)
        td_error = target - value(s)
        advantage_list[i] = td_error + gamma * lambd * advantage
        history[i] = (s, a, prob_action, target, advantage)
    return history

def generate_trajectory(env, pi, value, gamma, lambd, start_state, horizon):
    s = start_state
    done = False
    history = []
    for _ in range(horizon):
        #TODO: pi
        prob_action = pi(s)
        a = np.argmax(prob_action)
        #TODO: env.step
        s_prime, reward, done, info = env.step(a)
        history.append((s, a, s_prime, reward, prob_action))
        s = s_prime
        if not done:
            break
    #TODO: get value function, gamma, lambd
    history = add_advantage_targets(history, value, gamma, lambd)
    return history

def collect_data(env, num_actors, horizon, gamma, lambd):
    experience_buffer = []
    for actor in range(num_actors):
        #TODO: env.get_start_state
        start_state = env.get_start_state()
        history = generate_trajectory(env, pi, value, gamma, lambd, start_state, horizon)
        experience_buffer.extend(history)
    #experience_buffer = np.asarray(experience_buffer)
    #TODO: shuffle experience buffer
    return experience_buffer

def dense(inp, num_units):
    #TODO
    kernel1 = tf.Variable(tf.random.normal(..))
    bias1 = tf.Variable(tf.random.normal(...))
    return tf.nn.relu(tf.matmul(inp, kernel1) + bias1)

@tf.function
def pi(inp, num_actions):
    fc_pi = dense(inp, num_actions)
    sigmoid_pi = tf.nn.sigmoid(fc_pi)
    return sigmoid_pi

@tf.function
def value(inp):
    fc_v = dense(inp, 1)
    return fc_v

# def permissible_action(inp):



def ppo(env, num_actions, num_iters, num_actors, batch_size, num_epochs, epsilon, learning_rate):
    
    feature_layer = dense(inp, 64)
    for i in range(num_iters):
        experience_buffer = collect_data(num_actors, horizon)
        for j in range(num_epochs):
            train_batch = np.random.choice(experience_buffer, batch_size)
            s_list, a_list, prob_action_list, target_list, advantage_list = zip(*train_batch)
            s_tensor = tf.Variable(s_list)
            a_tensor = tf.Variable(a_list)
            prob_action_tensor = tf.Variable(prob_action_list)
            target_tensor = tf.Variable(target_list)
            advantage_tensor = tf.Variable(advantage_list)
            
            difference_v = target_tensor - value(s_tensor)
            loss_v = tf.nn.l2_loss(difference_v)

            prob_distr = pi(s_tensor)
            numerator = tf.gather_nd(prob_distr, a_tensor)
            denominator = tf.gather_nd(prob_action_tensor, a_tensor)
            prob_ratio = tf.divide(numerator,denominator)
            loss_term_1 = tf.multiply (prob_ratio,advantage_tensor)
            clip_min_value = 1 - epsilon
            clip_max_value = 1 + epsilon
            prob_ratio_clipped = tf.clip_by_value(prob_ratio, clip_min_value, clip_max_value)
            loss_term_2 = tf.multiply(prob_ratio_clipped, advantage_tensor)
            loss_pi = tf.math.minimum(loss_term_1, loss_term_2)

            loss_total = -loss_pi + loss_v

            adam_optimiser = tf.keras.optimizers.Adam(learning_rate)
            




