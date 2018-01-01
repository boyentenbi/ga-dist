
from copy import deepcopy
import gym
#import roboschool
import numpy as np
# from OpenGL import GL
import tensorflow as tf

from multiprocessing import Queue, Process, Pool


class Individual(object):

    def __init__(self, number, sess):
        self.sess = sess
        self.number = number
        self.s, self.a, self.params = self.get_model(self.number)
        self.assign_phs, self.assign_ops = self.get_assign_op()

    def get_model(self):
        raise NotImplementedError


    def get_assign_op(self):
        assign_phs = [tf.placeholder(tf.float32, p.shape) for p in self.params]
        assign_ops = [tf.assign(p, ph) for p, ph in zip(self.params, assign_phs)]
        return assign_phs, assign_ops

    def set_new_params(self, new_params):
        self.sess.run(self.assign_ops, feed_dict = dict(zip(self.assign_phs, new_params)))

    def act(self, actual_state):
        return self.sess.run(self.a, feed_dict={self.s: actual_state})


class EvolutionMethod(object):

    def __init__(self, IndClass, evaluator, n_pop, sess):

        self.n_pop = n_pop
        self.sess = sess
        self.population = [IndClass(i, self.sess) for i in range(self.n_pop)]
        self.evaluators = [deepcopy(evaluator) for _ in range(self.n_pop)]

    def get_next_gen(self):
        raise NotImplementedError

    def mutate(self, individual):
        raise NotImplementedError

class SimpleGA(EvolutionMethod):

    def __init__(self, IndClass, evaluator, n_elite, n_pop, n_parents, std, sess):
        super().__init__(IndClass, evaluator, n_pop, sess)
        self.n_elite = n_elite
        self.n_parents = n_parents
        self.std = std
        self.params, self.noised_params = self.get_new_params_op([p.shape for p in IndClass(-1, tf.Session()).params]) # TODO replace instance
        self.n_die = self.n_pop - self.n_parents

    def get_new_params_op(self, param_shapes):

        params = [tf.placeholder(tf.float32, s) for s in param_shapes]
        noises = [tf.random_normal(s, mean=0., stddev=self.std) for s in param_shapes]

        noised_params = [p + n for p, n in zip(params, noises)]
        return params, noised_params

    def get_noised_params(self, parent):
        parent_param_values = self.sess.run(parent.params)
        return self.sess.run(self.noised_params, feed_dict=dict(zip(self.params, parent_param_values)))

    def next_gen(self):

        # Evaluate each member of the population
        # pool = Pool(8)
        # args_list = zip(self.evaluators, self.population)
        scores = [e.evaluate(p) for e, p in zip(self.evaluators, self.population)]
        #scores = [self.evaluators[i].evaluate(self.population[i]) for i in range(self.n_pop)]

        # Sort based on the scores
        order = np.argsort(scores)
        self.population = [self.population[i] for i in order]

        # Don't actually make new Individuals, but replace the params of dead ones with children
        parents = self.population[self.n_die:]
        children = self.population[:self.n_die]
        for birth_num in range(len(children)):
            child = children[birth_num]
            parent = np.random.choice(parents)
            child_params = self.get_noised_params(parent)
            child.set_new_params(child_params)

        return [scores[i] for i in order] # return the sorted scores

class CartPoleIndividual(Individual):

    def __init__(self, number, sess):
        super().__init__(number, sess)

    @classmethod
    def get_model(self, number):
        scope = "model_{}".format(number)
        with tf.variable_scope(scope):
            s = tf.placeholder("float32", [None, 4])
            r = s
            for i in range(N_LAYERS):
                x = tf.layers.dense(r, 8, activation=tf.nn.relu, name="dense_{}".format(i))
                r = tf.concat([x, r], axis=1)
            a = tf.layers.dense(r, 2, activation=tf.nn.softmax, name = "dense_{}".format(N_LAYERS))

            params = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        return s, a, params

class CartPoleEvaluator(object):

    def __init__(self, n_steps):
        self.n_steps = 1000
        self.env = gym.make('CartPole-v0')
        self.a_shape = self.env.action_space.shape
        self.s_shape = self.env.observation_space.shape

    def evaluate(self, individual):
        all_eps_reward = 0
        n_eps = 0

        observation = self.env.reset()
        ep_reward = 0
        done = False
        for t in range(self.n_steps):
            if done:
                all_eps_reward += ep_reward
                n_eps += 1
                ep_reward = 0
                observation = self.env.reset()
                done = False

            action_probs = individual.act(np.expand_dims(observation, axis=0))[0]
            action = np.random.choice([0,1], p = action_probs)
            observation, reward, done, info = self.env.step(action)
            #reward = np.exp(reward)
            all_eps_reward += reward

        return all_eps_reward / (n_eps +1) # add one to eps to account for the unfinished ep


POP = 100
N_PARENTS = 40
N_ELITE = 1
N_GENS = 100
N_LAYERS = 1
STD = 0.2




with tf.Session() as sess:
    # Initialize
    method = SimpleGA(IndClass=CartPoleIndividual, evaluator = CartPoleEvaluator(1000), n_elite=N_ELITE, n_pop=POP, n_parents=N_PARENTS, std=STD, sess=sess)

    sess.run(tf.global_variables_initializer())

    order_stats = []

    # Iterate for some generations
    for i in range(N_GENS):
        sorted_scores = method.next_gen()
        order_stats.append(sorted_scores)
        if i % 10 ==0:
            print("Finished generation {}".format(i))