#use python 3, tensorflow 2.0

import tensorflow as tf
import numpy as np

class PPO:

    def __init__(self, input_shape, num_actions, learning_rate=1e-3, gamma=0.99, lambd=0.95, epsilon=0.2, batch_size=32, num_epochs=10):
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.weights = []
        create_variable = lambda shape : tf.Variable(tf.random.normal(shape=shape, dtype=tf.float32))
        # # feature_layer
        # self.common_kernel = create_variable((input_shape, 64))
        # self.common_bias = create_variable((64,))
        # # pi
        # self.pi_kernel = [create_variable((64, 64)), create_variable((64, num_actions))]
        # self.pi_bias = [create_variable((64,))]
        # # value
        # self.value_kernel = [create_variable((64, 64)), create_variable((64, 1))]
        # self.value_bias = [create_variable((64,))]

        self.common_kernel = create_variable((input_shape, 256))
        self.common_bias = create_variable((256,))
        # pi
        self.pi_kernel = [create_variable((256, num_actions))]
        # self.pi_bias = [create_variable((,))]
        # value
        self.value_kernel = [create_variable((256, 1))]
        # self.value_bias = [create_variable((64,))]
        self.weights = []
        self.weights.extend([self.common_kernel, self.common_bias])
        self.weights.extend(self.pi_kernel)
        # self.weights.extend(self.pi_bias)
        self.weights.extend(self.value_kernel)
        # self.weights.extend(self.value_bias)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        self.history = [] # (s,a,s_prime,r,prob_a)
        self.experience_buffer = [] # (s, a, prob_a, TD_target, advantage)
        return

    @tf.function
    def pi(self, inp):
        feature_layer = tf.matmul(inp, self.common_kernel)
        bias_layer = tf.nn.bias_add(feature_layer, self.common_bias)
        activation_layer = tf.nn.relu(bias_layer)

        # feature_layer_2 = tf.matmul(activation_layer, self.pi_kernel[0])
        # bias_layer_2 = tf.nn.bias_add(feature_layer_2, self.pi_bias[0])
        # activation_layer_2 = tf.nn.relu(bias_layer_2)

        feature_layer_3 = tf.matmul(activation_layer, self.pi_kernel[0])
        softmax_pi = tf.nn.softmax(feature_layer_3)
        return softmax_pi

    @tf.function
    def value(self, inp):
        feature_layer = tf.matmul(inp, self.common_kernel)
        bias_layer = tf.nn.bias_add(feature_layer, self.common_bias)
        activation_layer = tf.nn.relu(bias_layer)

        # feature_layer_2 = tf.matmul(activation_layer, self.value_kernel[0])
        # bias_layer_2 = tf.nn.bias_add(feature_layer_2, self.value_bias[0])
        # activation_layer_2 = tf.nn.relu(bias_layer_2)

        fc_v = tf.matmul(activation_layer, self.value_kernel[0])
        return fc_v
    
    @tf.function
    def train_single_step(self, s_list, a_list, prob_action_list, target_list, advantage_list):
        with tf.GradientTape() as tape:
            difference_v = target_list - self.value(s_list)
            loss_v = tf.nn.l2_loss(difference_v)

            prob_distr = self.pi(s_list)
            numerator = tf.gather(prob_distr, a_list, axis = 1)
            denominator = tf.gather(prob_action_list, a_list, axis = 1)
            prob_ratio = tf.divide(numerator,denominator)
            loss_term_1 = tf.multiply(prob_ratio, advantage_list)
            prob_ratio_clipped = tf.clip_by_value(prob_ratio, 1 - self.epsilon, 1 + self.epsilon)
            loss_term_2 = tf.multiply(prob_ratio_clipped, advantage_list)
            loss_pi = tf.math.minimum(loss_term_1, loss_term_2)

            loss_total = -loss_pi + loss_v
        
        gradients = tape.gradient(loss_total, self.weights)
        self.optimizer.apply_gradients(zip(gradients, self.weights))

    def add_advantage_targets(self):
        advantage = 0.0
        self.experience_buffer = [0] * len(self.history)
        for i in range(len(self.history)-1, -1, -1):
            s, a, s_prime, r, prob_action = self.history[i]
            target = r + self.gamma * self.value(tf.constant(np.expand_dims(s_prime, 0),dtype=tf.float32)).numpy()[0]
            td_error = target - self.value(tf.constant(np.expand_dims(s, 0),dtype=tf.float32)).numpy()[0]
            advantage = td_error + self.gamma * self.lambd * advantage
            self.experience_buffer[i] = (s, a, prob_action, target, advantage)
        return
    
    def populate_history(self, data):
        self.history = data
        return

    def train(self):
        self.add_advantage_targets()
        for k in range(self.num_epochs):
            train_batch_indices = np.random.choice(range(len(self.experience_buffer)), self.batch_size)
            train_batch = [self.experience_buffer[i] for i in train_batch_indices]
            nth_list = lambda i : [el[i] for el in train_batch]
            # create_tensor = lambda x : tf.constant(x, dtype=tf.float32)
            s_list, a_list, prob_action_list, target_list, advantage_list = map(nth_list, range(5))
            s_tensor = tf.constant(s_list, dtype=tf.float32)
            a_tensor = tf.constant(a_list, dtype=tf.int32)
            prob_action_tensor = tf.constant(prob_action_list, dtype=tf.float32)
            target_tensor = tf.constant(target_list, dtype = tf.float32)
            advantage_tensor = tf.constant(advantage_list, dtype=tf.float32)

            self.train_single_step(s_tensor, a_tensor, prob_action_tensor, target_tensor, advantage_tensor)


def generate_trajectory(env, model, start_state, horizon):
    s = start_state
    done = False
    history = []
    score = 0
    for _ in range(horizon):
        prob_action = model.pi(tf.constant(np.expand_dims(s,0),dtype=tf.float32)).numpy()[0]
        a = np.argmax(prob_action)
        #TODO: env.step
        s_prime, reward, done, _ = env.step(a)
        # env.render()
        history.append((s, a, s_prime, reward, prob_action))
        s = s_prime
        score += reward
        if done:
            break
    return history, score

def collect_data(env, num_actors, horizon, model):
    history_buffer = []
    avg_score = 0
    for actor in range(num_actors):
        #TODO: env.get_start_state
        # start_state = env.get_start_state()
        start_state = env.reset()
        history, score = generate_trajectory(env, model, start_state, horizon)
        history_buffer.extend(history)
        avg_score += score
    return history_buffer, avg_score / num_actors

# def permissible_action(inp)







