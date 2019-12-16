from tkinter import *
from tkinter import ttk
import time
import numpy as np
from numpy.linalg import norm
from mujoco_py import load_model_from_path, MjSim, MjViewer
from xmlwrapper import XMLWrapper
# from mjviewer import MjViewer

class Environment():

    def __init__(self, model_name, goal_space_train, goal_space_test, project_state_to_end_goal, end_goal_thresholds, initial_state_space, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, max_actions = 1200, num_frames_skip = 10, show = False):

        self.name = model_name

        # Create Mujoco Simulation
        self.model = load_model_from_path("./mujoco_files/" + model_name)
        self.sim = MjSim(self.model)

        # Set dimensions and ranges of states, actions, and goals in order to configure actor/critic networks
        if model_name == "ant_maze.xml":
            self.state_dim = len(self.sim.data.qpos) + len(self.sim.data.qvel) # State will include (i) joint angles and (ii) joint velocities
        
        if model_name=="ant_maze.xml":
            xw = XMLWrapper("./mujoco_files/"+model_name)
            self.vpos, self.vsize = xw.get_walls()
            self.vpos = np.array(self.vpos)
            self.vsize = np.array(self.vsize)
        else:
            self.xw = None

        self.action_dim = len(self.sim.model.actuator_ctrlrange) # low-level action dim
        self.action_bounds = self.sim.model.actuator_ctrlrange[:,1] # low-level action bounds
        self.action_offset = np.zeros((len(self.action_bounds))) # Assumes symmetric low-level action ranges
        self.end_goal_dim = len(goal_space_test)
        self.subgoal_dim = len(subgoal_bounds)
        self.subgoal_bounds = subgoal_bounds

        # Projection functions
        self.project_state_to_end_goal = project_state_to_end_goal
        self.project_state_to_subgoal = project_state_to_subgoal


        # Convert subgoal bounds to symmetric bounds and offset.  Need these to properly configure subgoal actor networks
        self.subgoal_bounds_symmetric = np.zeros((len(self.subgoal_bounds)))
        self.subgoal_bounds_offset = np.zeros((len(self.subgoal_bounds)))

        for i in range(len(self.subgoal_bounds)):
            self.subgoal_bounds_symmetric[i] = (self.subgoal_bounds[i][1] - self.subgoal_bounds[i][0])/2
            self.subgoal_bounds_offset[i] = self.subgoal_bounds[i][1] - self.subgoal_bounds_symmetric[i]


        # End goal/subgoal thresholds
        self.end_goal_thresholds = end_goal_thresholds
        self.subgoal_thresholds = subgoal_thresholds

        # Set inital state and goal state spaces
        self.initial_state_space = initial_state_space
        self.goal_space_train = goal_space_train
        self.goal_space_test = goal_space_test
        self.subgoal_colors = ["Magenta","Green","Red","Blue","Cyan","Orange","Maroon","Gray","White","Black"]

        self.max_actions = max_actions

        # Implement visualization 
        self.visualize = show  
        if self.visualize:
            self.viewer = MjViewer(self.sim)
            self.viewer._run_speed *= 4
        self.num_frames_skip = num_frames_skip

    def check_valid(self, pos, margin):
        if self.vpos is None or self.vsize is None:
            return True
        _pos = np.array(pos)
        for i in range(self.vpos.shape[0]):
            diff = np.absolute(_pos - self.vpos[i])
            _cmp = self.vsize[i]+margin <= diff
            if np.sum(_cmp) == 0:
                return False
        return True

    # Get state, which concatenates joint positions and velocities
    def get_state(self):
        return np.concatenate((self.sim.data.qpos, self.sim.data.qvel))

    # Reset simulation to state within initial state specified by user
    def reset_sim(self, next_goal = None, active_learning = 0, layer_IL = None, episode_num = 0):

        # Reset controls
        self.sim.data.ctrl[:] = 0

        if self.name == "ant_maze.xml":

            if active_learning == 0:
                # Set initial joint positions and velocities
                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])
                for i in range(len(self.sim.data.qvel)):
                    self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])

                # Move ant to random postion
                new_pos = np.random.uniform(-8.25,8.25,(2))
                while not self.check_valid(new_pos, 0.8) \
                      or norm(new_pos-next_goal[:2])<2.3:
                    new_pos = np.random.uniform(-8.25,8.25,(2))
                self.sim.data.qpos[:2] = new_pos
                self.sim.step()
                
                # Return state
                return self.get_state()

            # Noise-based active learning
            if active_learning == 1:

                positions = []
                variance = []
                init_pos = []
                init_vel = []

                if episode_num < 50:
                    position_number = 1
                else:
                    position_number = episode_num - 39
                state_number = 30

                for i in range(position_number):
                    actions = []
                    action_sum = np.zeros(self.subgoal_dim)
                    
                    # Move ant to correct room
                    new_pos = np.random.uniform(-8.25,8.25,(2))
                    while not self.check_valid(new_pos, 0.8) \
                          or norm(new_pos-next_goal[:2])<2.3:
                        new_pos = np.random.uniform(-8.25,8.25,(2))
                    self.sim.data.qpos[:2] = new_pos

                    for j in range(state_number): 

                        if i == 0:
                            # Set initial joint positions and velocities
                            for k in range(2, len(self.sim.data.qpos)):
                                self.sim.data.qpos[k] = np.random.uniform(self.initial_state_space[k][0],self.initial_state_space[k][1])
                            init_pos.append(self.sim.data.qpos.copy())

                            for k in range(len(self.sim.data.qvel)):
                                self.sim.data.qvel[k] = np.random.uniform(-0.5, 0.5)
                            init_vel.append(self.sim.data.qvel.copy())

                        if i > 0:
                            # print(init_vel)
                            # Set initial joint positions and velocities
                            for k in range(2, len(self.sim.data.qpos)):
                                self.sim.data.qpos[k]  = init_pos[j][k]
                            for k in range(len(self.sim.data.qvel)):
                                self.sim.data.qvel[k] = init_vel[j][k]

                        _cur_state =np.reshape(self.get_state(), (1,len(self.get_state())))
                        _goal = np.reshape(next_goal, (1,len(next_goal)))
                        feed_dict = {
                            layer_IL.state_ph: _cur_state,
                            layer_IL.goal_ph: _goal
                        }
                        action = layer_IL.sess.run(layer_IL.infer, feed_dict=feed_dict)
                        action_sum += action[0]
                        actions.append(action[0])

                    action_avg = action_sum / state_number

                    var = 0
                    for i in range(state_number):
                        var += np.sum(np.square(actions[i] - action_avg))

                    variance.append(var)
                    positions.append([self.sim.data.qpos.copy()[0], self.sim.data.qpos.copy()[1]])

                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

                for i in range(len(self.sim.data.qvel)):
                    self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])
                index = variance.index(max(variance))
                self.sim.data.qpos[0] = positions[index][0]
                self.sim.data.qpos[1] = positions[index][1]

                print("Start Position Number:", index)
                self.sim.step()
                # Return state
                return self.get_state()

            # Multi-policy active learning
            if active_learning == 2:
                if episode_num < 50:
                    position_number = 1
                else:
                    position_number = episode_num - 39
                
                variance = []
                positions = []

                # Set initial joint positions and velocities
                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

                for i in range(len(self.sim.data.qvel)):
                    self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])                
                
                for i in range(position_number):
                    actions = []
                    action_sum = np.zeros(self.subgoal_dim)                    

                    # Move ant to correct room
                    new_pos = np.random.uniform(-8.25,8.25,(2))
                    while not self.check_valid(new_pos, 0.8) \
                          or norm(new_pos-next_goal[:2])<2.3:
                        new_pos = np.random.uniform(-8.25,8.25,(2))
                    self.sim.data.qpos[:2] = new_pos

                    _cur_state =np.reshape(self.get_state(), (1,len(self.get_state())))
                    _goal = np.reshape(next_goal, (1,len(next_goal)))
                    feed_dict = {
                        layer_IL.state_ph: _cur_state,
                        layer_IL.goal_ph: _goal
                    }

                    action = layer_IL.sess.run(layer_IL.infer, feed_dict=feed_dict)
                    action1 = layer_IL.sess.run(layer_IL.infer1, feed_dict=feed_dict)
                    action2 = layer_IL.sess.run(layer_IL.infer2, feed_dict=feed_dict)
                    action3 = layer_IL.sess.run(layer_IL.infer3, feed_dict=feed_dict)
                    action4 = layer_IL.sess.run(layer_IL.infer4, feed_dict=feed_dict)

                    action_sum = action[0] + action1[0] + action2[0] + action3[0] + action4[0]

                    actions.append(action[0])
                    actions.append(action1[0])
                    actions.append(action2[0])
                    actions.append(action3[0])
                    actions.append(action4[0])

                    action_avg = action_sum / 5

                    var = 0
                    for i in range(5):
                        var += np.sum(np.square(actions[i] - action_avg))
                    variance.append(var)
                    positions.append([self.sim.data.qpos.copy()[0], self.sim.data.qpos.copy()[1]])

                for i in range(len(self.sim.data.qpos)):
                    self.sim.data.qpos[i] = np.random.uniform(self.initial_state_space[i][0],self.initial_state_space[i][1])

                for i in range(len(self.sim.data.qvel)):
                    self.sim.data.qvel[i] = np.random.uniform(self.initial_state_space[len(self.sim.data.qpos) + i][0],self.initial_state_space[len(self.sim.data.qpos) + i][1])
                index = variance.index(max(variance))

                self.sim.data.qpos[0] = positions[index][0]
                self.sim.data.qpos[1] = positions[index][1]

                print("Start Position Number:", index)

                self.sim.step()
                # Return state
                return self.get_state()

    # Execute low-level action for number of frames specified by num_frames_skip
    def execute_action(self, action):

        self.sim.data.ctrl[:] = action
        for _ in range(self.num_frames_skip):
            self.sim.step()
            if self.visualize:
                self.viewer.render()

        return self.get_state()


    # Visualize end goal.  This function may need to be adjusted for new environments.
    def display_end_goal(self,end_goal):

        if self.name ==  "ant_maze.xml":
            self.sim.data.mocap_pos[0][:3] = np.copy(end_goal[:3])

        else:
            assert False, "Provide display end goal function in environment.py file"


    # Function returns an end goal
    def get_next_goal(self,test):

        end_goal = np.zeros((len(self.goal_space_test)))

        # Pick exact goal location
        end_goal[2] = np.random.uniform(0.45,0.55)
        end_goal[:2] = np.random.uniform(-8.25,8.25,(2))
        while not self.check_valid(end_goal[:2],0.8):
            end_goal[:2] = np.random.uniform(-8.25,8.25,(2))
        # Visualize End Goal
        self.display_end_goal(end_goal)

        return end_goal


    # Visualize all subgoals
    def display_subgoals(self,subgoals):

        # Display up to 10 subgoals and end goal
        if len(subgoals) <= 11:
            subgoal_ind = 0
        else:
            subgoal_ind = len(subgoals) - 11


        for i in range(1,min(len(subgoals),11)):

            if self.name == "ant_maze.xml":
                self.sim.data.mocap_pos[i][:3] = np.copy(subgoals[subgoal_ind][:3])
                self.sim.model.site_rgba[i][3] = 1

                subgoal_ind += 1

            else:
                # Visualize desired gripper position, which is elements 18-21 in subgoal vector
                self.sim.data.mocap_pos[i] = subgoals[subgoal_ind]
                # Visualize subgoal
                self.sim.model.site_rgba[i][3] = 1
                subgoal_ind += 1
