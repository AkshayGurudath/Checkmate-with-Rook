#use python 3, tensorflow 2.0

import tensorflow as tf
import numpy as np

class PPO:

    def __init__(self, state_dimension, action_dimension, learning_rate=1e-3, gamma=0.99, lambd=0.95, epsilon=0.2, batch_size=32, num_epochs=10):
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

        print("PPO.__init__() : Set hyperparameters")

        self.weights = []
        create_variable = lambda shape : tf.Variable(tf.keras.initializers.GlorotUniform()(shape, dtype=tf.float32))
        # # feature_layer
        self.common_kernel = create_variable((state_dimension, 256))
        self.common_bias = create_variable((256,))
        # pi
        self.pi_kernel = [create_variable((256, action_dimension))]
        # value
        self.value_kernel = [create_variable((256, 1))]
        self.weights = []
        self.weights.extend([self.common_kernel, self.common_bias])
        self.weights.extend(self.pi_kernel)
        self.weights.extend(self.value_kernel)
        print("PPO.__init__() : Set weights - ", len(self.weights))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.experience_buffer = [] # (s, a, prob_a, TD_target, advantage)
        return

    @tf.function
    def pi(self, inp):
        feature_layer = tf.matmul(inp, self.common_kernel)
        bias_layer = tf.nn.bias_add(feature_layer, self.common_bias)
        activation_layer = tf.nn.relu(bias_layer)

        feature_layer_3 = tf.matmul(activation_layer, self.pi_kernel[0])
        softmax_pi = tf.nn.softmax(feature_layer_3)
        return softmax_pi

    @tf.function
    def value(self, inp):
        feature_layer = tf.matmul(inp, self.common_kernel)
        bias_layer = tf.nn.bias_add(feature_layer, self.common_bias)
        activation_layer = tf.nn.relu(bias_layer)

        fc_v = tf.matmul(activation_layer, self.value_kernel[0])
        fc_v_squeeze = tf.squeeze(fc_v, axis=-1)
        return fc_v_squeeze

    @tf.function
    def train_single_step(self, s_list, a_list, prob_action_list, target_list, advantage_list):
        with tf.GradientTape() as tape:
            difference_v = target_list - self.value(s_list)
            loss_v = tf.math.reduce_mean(tf.square(difference_v))

            prob_distr = self.pi(s_list)
            numerator = tf.gather_nd(prob_distr, a_list)
            denominator = tf.gather_nd(prob_action_list, a_list)
            prob_ratio = tf.divide(numerator,denominator)
            loss_term_1 = tf.multiply(prob_ratio, advantage_list)
            prob_ratio_clipped = tf.clip_by_value(prob_ratio, 1 - self.epsilon, 1 + self.epsilon)
            loss_term_2 = tf.multiply(prob_ratio_clipped, advantage_list)
            loss_pi = tf.math.reduce_mean(-tf.math.minimum(loss_term_1, loss_term_2))
            loss_total = loss_pi + loss_v
        gradients = tape.gradient(loss_total, self.weights)
        self.optimizer.apply_gradients(zip(gradients, self.weights))

    def add_advantage_targets(self, actor_history): # for one actor only
        advantage = 0.0
        actor_experience_buffer = [0] * len(actor_history)
        for i in range(len(actor_history)-1, -1, -1):
            s, a, s_prime, r, prob_action, done = actor_history[i]
            target = r + self.gamma * self.value(tf.constant(np.expand_dims(s_prime, 0),dtype=tf.float32)).numpy()[0] * (not done)
            td_error = target - self.value(tf.constant(np.expand_dims(s, 0),dtype=tf.float32)).numpy()[0]
            advantage = td_error + self.gamma * self.lambd * advantage * (not done)
            actor_experience_buffer[i] = (s, a, prob_action, target, advantage)
        self.experience_buffer.extend(actor_experience_buffer)
        return

    def train(self):
        for k in range(self.num_epochs):
            train_batch_indices = np.random.choice(range(len(self.experience_buffer)), self.batch_size)
            train_batch = [self.experience_buffer[i] for i in train_batch_indices]
            nth_list = lambda i : [el[i] for el in train_batch]
            s_list, a_list, prob_action_list, target_list, advantage_list = map(nth_list, range(5))
            a_list = [ [i, x] for i,x in enumerate(a_list)]
            s_tensor = tf.constant(s_list, dtype=tf.float32)
            a_tensor = tf.constant(a_list, dtype=tf.int32)
            prob_action_tensor = tf.constant(prob_action_list, dtype=tf.float32)
            target_tensor = tf.constant(target_list, dtype = tf.float32)
            advantage_tensor = tf.constant(advantage_list, dtype=tf.float32)

            self.train_single_step(s_tensor, a_tensor, prob_action_tensor, target_tensor, advantage_tensor)

        # delete all experience to accomodate new ones for next train
        self.experience_buffer = []
        return


def generate_trajectory(env, model, start_state, horizon, render=False, renormalize=False):
    s = start_state
    done = False
    history = []
    score = 0
    for _ in range(horizon):
        prob_action = model.pi(tf.constant(np.expand_dims(s,0), dtype=tf.float32)).numpy()[0]
        if renormalize:
            prob_action = renormalize_prob_dist(env, prob_action)
        a = np.random.choice(list(range(model.action_dimension)), p=prob_action)
        s_prime, reward, done, _ = env.step(a)
        if render:
            env.render()
        history.append((s, a, s_prime, reward, prob_action, done))
        s = s_prime
        score += reward
        if done:
            break
    return history, score, s_prime, done


def renormalize_prob_dist(env, prob_distr):
    prob_dist = prob_distr[:]
    boolean_action_list = env.legal_moves()
    for i in range(len(prob_dist)):
        if boolean_action_list[i] == 0:
            prob_dist[i] = 0
    sum_prob = np.sum(prob_dist)
    prob_dist_modified = prob_dist / sum_prob # for some reason np broadcasting ensures better precision
    return prob_dist_modified
