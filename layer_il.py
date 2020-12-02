import numpy as np
from experience_buffer import ExperienceBuffer
import tensorflow as tf
from utils import layer
from time import sleep

class Layer_IL():
    def __init__(self, layer_number, FLAGS, env, sess, agent_params):
        self.layer_number = layer_number
        self.FLAGS = FLAGS
        self.sess = sess

        # Set time limit for each layer.  If agent uses only 1 layer, time limit is the max number of low-level actions allowed in the episode (i.e, env.max_actions).
        if FLAGS.layers > 1:
            self.time_limit = FLAGS.time_scale
        else:
            self.time_limit = env.max_actions
        self.time_limit = 20

        ##@@
        self.current_state = None
        self.goal = None
        self.beta = 1.0
        self.state_dim = env.state_dim
        self.goal_dim = env.end_goal_dim
        self.enc_dim = 32
        self.action_space_size = env.subgoal_dim
        self.action_space_bounds = env.subgoal_bounds_symmetric
        print(self.action_space_bounds)
        self.action_offset = env.subgoal_bounds_offset
        self.action_ph = tf.placeholder(tf.float32, shape=(None, self.action_space_size))
        ##@@

        # Initialize Replay Buffer.  Below variables determine size of replay buffer.

        # Ceiling on buffer size
        self.buffer_size_ceiling = 10**7

        # Number of full episodes stored in replay buffer
        self.episodes_to_store = agent_params["episodes_to_store"]

        # Set number of transitions to serve as replay goals during goal replay
        self.num_replay_goals = 2

        # Number of the transitions created for each attempt (i.e, action replay + goal replay + subgoal testing)
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)

        # Buffer size = transitions per attempt * # attempts per episode * num of episodes stored
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.FLAGS.layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)

        # self.buffer_size = 10000000
        self.batch_size = 1024
        self.replay_buffer = ExperienceBuffer(self.buffer_size, self.batch_size)

        # Create buffer to store not yet finalized goal replay transitions
        self.temp_goal_replay_storage = []

        ##@@
        self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_dim))
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))
        # self.enc_ph = tf.placeholder(tf.float32, shape=(None, self.enc_dim))
        self.features_ph = tf.concat([self.state_ph, self.goal_ph], axis=1)
        self.batch_size = tf.placeholder(tf.float32)
        self.infer = self.create_nn(self.features_ph, name="layer_hi")
        self.loss = tf.losses.mean_squared_error(self.infer, self.action_ph)
        self.train_fn = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        ##@@

        if self.FLAGS.active_learning == 2:
            self.infer1 = self.create_nn1(self.features_ph, name="layer_hi1")
            self.infer2= self.create_nn2(self.features_ph, name="layer_hi2")
            self.infer3 = self.create_nn3(self.features_ph, name="layer_hi3")
            self.infer4 = self.create_nn4(self.features_ph, name="layer_hi4")
            self.loss1 = tf.losses.mean_squared_error(self.infer1, self.action_ph)
            self.loss2 = tf.losses.mean_squared_error(self.infer2, self.action_ph)
            self.loss3 = tf.losses.mean_squared_error(self.infer3, self.action_ph)
            self.loss4 = tf.losses.mean_squared_error(self.infer4, self.action_ph)
            self.train_fn1 = tf.train.AdamOptimizer(0.001).minimize(self.loss1)
            self.train_fn2 = tf.train.AdamOptimizer(0.001).minimize(self.loss2)
            self.train_fn3 = tf.train.AdamOptimizer(0.001).minimize(self.loss3)
            self.train_fn4 = tf.train.AdamOptimizer(0.001).minimize(self.loss4)

        # Parameter determines degree of noise added to actions during training
        # self.noise_perc = noise_perc
        if self.layer_number == 0:
            self.noise_perc = agent_params["atomic_noise"]
        else:
            self.noise_perc = agent_params["subgoal_noise"]

        # Create flag to indicate when layer has ran out of attempts to achieve goal.  This will be important for subgoal testing
        self.maxed_out = False

        self.subgoal_penalty = agent_params["subgoal_penalty"]



    # Add noise to provided action
    def add_noise(self,action, env):

        # Noise added will be percentage of range
        if self.layer_number == 0:
            action_bounds = env.action_bounds
            action_offset = env.action_offset
        else:
            action_bounds = env.subgoal_bounds_symmetric
            action_offset = env.subgoal_bounds_offset

        assert len(action) == len(action_bounds), "Action bounds must have same dimension as action"
        assert len(action) == len(self.noise_perc), "Noise percentage vector must have same dimension as action"

        # Add noise to action and ensure remains within bounds
        for i in range(len(action)):
            action[i] += np.random.normal(0,self.noise_perc[i] * action_bounds[i])

            action[i] = max(min(action[i], action_bounds[i]+action_offset[i]), -action_bounds[i]+action_offset[i])

        return action


    # Select random action
    def get_random_action(self, env):

        if self.layer_number == 0:
            action = np.zeros((env.action_dim))
        else:
            action = np.zeros((env.subgoal_dim))

        # Each dimension of random action should take some value in the dimension's range
        for i in range(len(action)):
            if self.layer_number == 0:
                action[i] = np.random.uniform(-env.action_bounds[i] + env.action_offset[i], env.action_bounds[i] + env.action_offset[i])
            else:
                action[i] = np.random.uniform(env.subgoal_bounds[i][0],env.subgoal_bounds[i][1])

        return action


    # Function selects action using an epsilon-greedy policy
    def choose_action(self,agent, env, subgoal_test, first=False):
        # User interface
        data_list = []
        action = None
        policy_type = None
        prev = self.current_state[:2]
        expert_action = None
        if not self.FLAGS.test:
            pos = np.array([0.53]*3)
            if first:
                agent.worker.myWait()
            pos[:2] = 17*(np.array(agent.worker.in_pos,dtype=np.float32))/340 - 8.5
            ##@@
            vel = pos[:2]-prev
            vel = 2*vel/np.linalg.norm(vel)
            expert_action = np.concatenate((pos,vel))
        if not self.FLAGS.test and np.random.random_sample() < self.beta:
            assert(not self.FLAGS.test)
            action = expert_action
            policy_type = "Human"
        else:
            _cur_state = np.reshape(self.current_state, (1,len(self.current_state)))
            _goal = np.reshape(self.goal, (1,len(self.goal)))
            feed_dict = {
                self.state_ph: _cur_state,
                self.goal_ph: _goal
            }
            action = self.sess.run(self.infer, feed_dict=feed_dict)
            action = action[0]
            policy_type = "Policy"

        return action, policy_type, subgoal_test, expert_action


    # Determine whether layer is finished training
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):

        # Return to higher level if (i) a higher level goal has been reached, (ii) maxed out episode time steps (env.max_actions), (iii) not testing and layer is out of attempts, and (iv) testing, layer is not the highest level, and layer is out of attempts.  NOTE: during testing, highest level will continue to ouput subgoals until either (i) the maximum number of episdoe time steps or (ii) the end goal has been achieved.

        # Return to previous level when any higher level goal achieved.  NOTE: if not testing and agent achieves end goal, training will continue until out of time (i.e., out of time steps or highest level runs out of attempts).  This will allow agent to experience being around the end goal.
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True

        else:
            return False


    # Learn to achieve goals with actions belonging to appropriate time scale.  "goal_array" contains the goal states for the current layer and all higher layers
    def train(self, agent, env, subgoal_test = False, episode_num = None):

        # print("\nTraining Layer %d" % self.layer_number)

        # Set layer's current state and new goal state
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state

        # Reset flag indicating whether layer has ran out of attempts.  This will be used for subgoal testing.
        self.maxed_out = False

        # Display all subgoals if visualizing training and current layer is bottom layer
        if self.layer_number == 0 and agent.FLAGS.show and agent.FLAGS.layers > 1:
            env.display_subgoals(agent.goal_array)
            # env.sim.data.mocap_pos[3] = env.project_state_to_end_goal(env.sim,self.current_state)
            # print("Subgoal Pos: ", env.sim.data.mocap_pos[1])

        # Current layer has self.time_limit attempts to each its goal state.
        attempts_made = 0
        agent.attempts = 0

        while True:

            # Select action to achieve goal state using epsilon-greedy policy or greedy policy if in test mode
            action, action_type, next_subgoal_test, expert_action = self.choose_action(agent, env, subgoal_test, attempts_made==0)

            # If next layer is not bottom level, propose subgoal for next layer to achieve and determine whether that subgoal should be tested
            if self.layer_number > 0:

                agent.goal_array[self.layer_number - 1] = action

                goal_status, max_lay_achieved = agent.layers[self.layer_number - 1].train(agent, env, next_subgoal_test, episode_num)

            if goal_status[self.layer_number]:
                reward = 1
                finished = True
            else:
                reward = 0
                finished = False

            if self.FLAGS.show:
                if agent.worker.reset:
                    return None, None
            attempts_made += 1
            agent.attempts += 1

            # Print if goal from current layer as been achieved
            if goal_status[self.layer_number]:
                if self.layer_number < agent.FLAGS.layers - 1:
                    print("SUBGOAL ACHIEVED")
                print("\nEpisode %d, Layer %d, Attempt %d Goal Achieved" % (episode_num, self.layer_number, attempts_made))
                print("Goal: ", self.goal)
                if self.layer_number == agent.FLAGS.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(env.sim, agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(env.sim, agent.current_state))

            # Perform hindsight learning using action actually executed (low-level action or hindsight subgoal)
            if self.layer_number == 0:
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                # Otherwise, use subgoal that was achieved in hindsight
                else:
                    hindsight_action = env.project_state_to_subgoal(env.sim, agent.current_state)


            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:
                info = 0
            else:
                info = 1

            transition = [self.current_state, expert_action, reward, None, self.goal, finished, info]
            self.replay_buffer.add(np.copy(transition))

            if not self.FLAGS.test:
                if info == 0 and reward == 0:
                    self.replay_buffer.clear(length=attempts_made)


            # Update state of current layer
            self.current_state = agent.current_state

            # Return to previous level to receive next subgoal if applicable
            # if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:
                if self.layer_number == agent.FLAGS.layers-1:
                    print("HL Attempts Made: ", attempts_made)

                # If goal was not achieved after max number of attempts, set maxed out flag to true
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True
                    # print("Layer %d Out of Attempts" % self.layer_number)

                # If not testing, finish goal replay by filling in missing goal and reward values before returning to prior level.

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
                    return goal_status, max_lay_achieved


    # Update networks
    def learn(self, num_updates):
        loss = None
        if self.FLAGS.active_learning == 2:
            loss1 = None
            loss2 = None
            loss3 = None
            loss4 = None

        if self.replay_buffer.size > 32:
            for _ in range(num_updates):
            # Update weights of non-target networks
                old_states, actions, rewards, new_states, goals, is_terminals, enc = self.replay_buffer.get_batch_with_enc()

                next_batch_size = min(self.replay_buffer.size, self.replay_buffer.batch_size)

                feed_dict = {
                    self.state_ph: old_states,
                    self.goal_ph: goals,
                    # self.enc_ph: enc,
                    self.action_ph: actions,
                    self.batch_size: next_batch_size # ??
                }
                _,loss = self.sess.run([self.train_fn,self.loss],feed_dict=feed_dict)
                if self.FLAGS.active_learning == 2:
                    _,loss1 = self.sess.run([self.train_fn1,self.loss1],feed_dict=feed_dict)
                    _,loss2 = self.sess.run([self.train_fn2,self.loss2],feed_dict=feed_dict)
                    _,loss3 = self.sess.run([self.train_fn3,self.loss3],feed_dict=feed_dict)
                    _,loss4 = self.sess.run([self.train_fn4,self.loss4],feed_dict=feed_dict)
            print('Loss: ', loss)
            self.beta /= 1.025
            print("New beta: ", self.beta)

    def create_nn(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset

        return output

    def create_nn1(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 32)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 32)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 32)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset

        return output

    def create_nn2(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 128)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 128)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 128)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.tanh(fc4) * self.action_space_bounds + self.action_offset

        return output

    def create_nn3(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.sigmoid(fc4) * self.action_space_bounds + self.action_offset

        return output            

    def create_nn4(self, features, name=None):

        if name is None:
            name = self.actor_name

        with tf.variable_scope(name + '_fc_1'):
            fc1 = layer(features, 128)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = layer(fc1, 128)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = layer(fc2, 128)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = layer(fc3, self.action_space_size, is_output=True)

        output = tf.sigmoid(fc4) * self.action_space_bounds + self.action_offset

        return output
