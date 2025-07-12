# file: dddqn_agent.py
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DDDQNAgent:
    def __init__(self, state_size, action_space_size):
        self.state_size = state_size
        self.action_space = action_space_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.state_size,))
        layer1 = tf.keras.layers.Dense(128, activation="relu")(input_layer)
        layer2 = tf.keras.layers.Dense(128, activation="relu")(layer1)

        value_stream = tf.keras.layers.Dense(1, activation="linear")(layer2)
        advantage_stream = tf.keras.layers.Dense(self.action_space, activation="linear")(layer2)
        
        advantage_mean = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage_stream)
        advantage_processed = tf.keras.layers.Subtract()([advantage_stream, advantage_mean])
        q_values = tf.keras.layers.Add()([value_stream, advantage_processed])

        model = tf.keras.Model(inputs=input_layer, outputs=q_values)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def learn(self, batch_size):
        if len(self.memory) < batch_size: return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        next_q_values_main = self.model.predict(next_states, verbose=0)
        next_q_values_target = self.target_model.predict(next_states, verbose=0)
        target_q_values = self.model.predict(states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                action = np.argmax(next_q_values_main[i])
                q_value = next_q_values_target[i][action]
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * q_value
        
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay