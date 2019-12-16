import numpy as np
from layer import Layer
from layer_il import Layer_IL
from environment import Environment
import pickle as cpickle
import tensorflow as tf
import os
import pickle as cpickle
from worker import Worker


# Below class instantiates an agent
class Agent():
    def __init__(self, FLAGS, env, agent_params):

        self.FLAGS = FLAGS
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                                   intra_op_parallelism_threads=1,
                                   device_count={'GPU': 0})
        # tf_config.gpu_options.allow_growth = True # may need if using GPU
        self.sess = tf.Session(config=tf_config)

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        # Create agent with number of levels specified by user
        self.layers = [
            Layer(i, FLAGS, env, self.sess, agent_params)
            for i in range(FLAGS.layers)
        ]

        # Below attributes will be used help save network parameters
        self.saver = None
        self.model_dir = None
        self.model_names = None
        self.attempts = 0

        # Initialize actor/critic networks.  Load saved parameters if not retraining

        layer_save = self.layers[FLAGS.layers - 1]
        self.layers[FLAGS.layers - 1] = Layer_IL(FLAGS.layers - 1, FLAGS, env,
                                                 self.sess, agent_params)
        self.initialize_networks()
        # self.save_model(20)

        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(FLAGS.layers)]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = 400

        # Below parameters will be used to store performance results
        self.performance_log = []

        self.other_params = agent_params
        self.worker = None

    def set_worker(self, worker):
        self.worker = worker

    def set_enc(self, enc):
        self.enc = enc

    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self, env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.FLAGS.layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim,
                                                    self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim,
                                                      self.current_state)

        for i in range(self.FLAGS.layers):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.FLAGS.layers - 1:

                # Check dimensions are appropriate
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(
                    env.end_goal_thresholds
                ), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_end_goal)):
                    if np.absolute(self.goal_array[i][j] - proj_end_goal[j]
                                   ) > env.end_goal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(
                    env.subgoal_thresholds
                ), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]
                                   ) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    def initialize_networks(self, round = 0):

        var_low = []
        var_high = []
        for l in range(self.FLAGS.layers - 1):
            var_low.extend(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='actor_%d' % l))
            var_low.extend(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope='critic_%d' % l))
        var_high.extend(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='layer_hi'))
        self.saver_low = tf.train.Saver(var_low, save_relative_paths=True,
                                        max_to_keep=200)
        self.saver_high = tf.train.Saver(var_high, save_relative_paths=True,
                                         max_to_keep=200)

        # Set up directory for saving models
        prefix = "exp%d_"%(self.FLAGS.expnum)
        if self.FLAGS.active_learning == 1:
            prefix = prefix+"al_noise_"
        if self.FLAGS.active_learning == 2:
            prefix = prefix+"al_bag_"
        self.model_dir = os.getcwd() + '/models'
        self.model_names = {'low': 'low.ckpt', 'high': 'high.ckpt'}
        self.model_demo = os.path.join(os.getcwd(), 'models',
                                       prefix+'layer_il_buf.pkl')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Initialize actor/critic networks
        self.sess.run(tf.global_variables_initializer())

        # Always load low level model
        self.saver_low.restore(
            self.sess,
            tf.train.latest_checkpoint(self.model_dir,
                                       latest_filename='low_level'))

        # If not retraining, restore weights
        # if we are not retraining from scratch, just restore weights
        if self.FLAGS.retrain == False and self.FLAGS.results == False:
            self.saver_high.restore(
                self.sess,
                tf.train.latest_checkpoint(self.model_dir,
                                           latest_filename='high_level'))

        if self.FLAGS.test and self.FLAGS.results:
            ckpt=tf.train.get_checkpoint_state(self.model_dir,
                                               latest_filename='high_level')
            self.saver_high.restore(
                self.sess,
                ckpt.all_model_checkpoint_paths[round])

        # Load demonstrations
        if self.FLAGS.randomize or self.FLAGS.reset_maze or self.FLAGS.retrain:
            if os.path.exists(self.model_demo):
                os.remove(self.model_demo)
        if os.path.exists(self.model_demo):
            with open(self.model_demo, 'rb') as f:
                self.layers[self.FLAGS.layers -
                            1].replay_buffer = cpickle.load(f)

    # Save neural network parameters
    def save_model(self, episode):
        print('Agent saving')
        # Currently do not save low level model
        prefix = "exp%d_"%(self.FLAGS.expnum)
        if self.FLAGS.active_learning == 1:
            prefix = prefix+"al_noise_"
        if self.FLAGS.active_learning == 2:
            prefix = prefix+"al_bag_"
        self.saver_high.save(self.sess,
                             os.path.join(self.model_dir,
                                          prefix+self.model_names['high']),
                             global_step=episode,
                             latest_filename='high_level')
        with open(self.model_demo, 'wb') as f:
            cpickle.dump(self.layers[self.FLAGS.layers - 1].replay_buffer, f)

    # Update actor and critic networks for each layer
    def learn(self):

        for i in range(len(self.layers)):
            self.layers[i].learn(self.num_updates)

    # Train agent for an episode
    def train(self, env, episode_num, total_episodes):

        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal_array[self.FLAGS.layers - 1] = env.get_next_goal(
            self.FLAGS.test)
        print("Next End Goal: ", self.goal_array[self.FLAGS.layers - 1])

        # Select initial state from in initial state space, defined in environment.py
        self.current_state = env.reset_sim(self.goal_array[self.FLAGS.layers - 1], self.FLAGS.active_learning, self.layers[self.FLAGS.layers-1], episode_num)

        # Reset step counter
        self.steps_taken = 0
        if self.FLAGS.show:
            env.viewer.render()

        # Train for an episode
        if self.FLAGS.show:
            self.worker.agentMove(self.current_state[:2])
            self.worker.goalChange(self.goal_array[self.FLAGS.layers - 1][:2],
                                   self.FLAGS.layers - 1)
            for i in range(self.FLAGS.layers - 1):
                self.worker.goalChange(np.array([300., 300.]), i)
            self.worker.posChange()
        goal_status, max_lay_achieved = self.layers[
            self.FLAGS.layers - 1].train(self, env, episode_num=episode_num)
        
        if self.FLAGS.show:
            if self.worker.reset:
                return None

        # Update networks if not testing
        if not self.FLAGS.test and total_episodes >= 10 and total_episodes % 5 == 0:
            self.learn()

        # Return whether end goal was achieved
        return goal_status[self.FLAGS.layers - 1]

    # Save performance evaluations
    def log_performance(self, success_rate):

        # Add latest success_rate to list
        self.performance_log.append(success_rate)

        # Save log
        cpickle.dump(self.performance_log, open("performance_log.p", "wb"))
