"""
This is the file to initialize the Active Hierarchical Imitation and Reinforcement Learning (AHIRL) algorithm. 
"""

from design_agent_and_env import design_agent_and_env
from options import parse_options
from agent import Agent
from worker import Worker
from run_AHIRL import run_AHIRL
from window import MainWindow
from xmlwrapper import XMLWrapper
from maze import Maze
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
import threading
import os
import sys
import time

def initial_AHIRL(FLAGS,worker):
    if FLAGS.show:
        # time.sleep(0.5)
        # worker.screen.emit()
        # while not os.path.exists('maze.png'):
        #     time.sleep(0.1)
        # # Compute embedding for picture
        # if os.path.exists('maze.png'):
        #     os.remove('maze.png')
        agent, env = design_agent_and_env(FLAGS)
        agent.set_worker(worker)
        agent.set_enc(None)
        run_AHIRL(FLAGS, env, agent)
    else:
        agent, env = design_agent_and_env(FLAGS)
        run_AHIRL(FLAGS, env, agent)

# Determine training options specified by user. 
FLAGS = parse_options()

# Instantiate the agent and Mujoco environment. 
model_name = "ant_maze.xml"
model_path = "./mujoco_files/" + model_name
xw = XMLWrapper(model_path)
if FLAGS.randomize:
    maze = Maze(3, 3)
    maze.randomize()
    # Only get the inner walls
    inner_vpos, inner_vsize = maze.to_walls()
    assert(len(inner_vpos) == len(inner_vsize))
    walls = xw.get_inner_walls()
    for i in range(len(walls[0])):
        xw.del_wall("wall_%d"%i)
    for i,(pos,size) in enumerate(zip(inner_vpos, inner_vsize)):
        xw.set_wall("wall_%d"%i, pos, size)
    xw.write()
elif FLAGS.reset_maze:
    xw = XMLWrapper("./mujoco_files/ant_maze_orig.xml")
    xw.write(model_path)
else:
    xw = XMLWrapper(model_path)

vpos, vsize = xw.get_walls()

# Display the user interface
if FLAGS.show:
    app = QApplication(sys.argv)
    window = MainWindow(vpos, vsize)
    worker = Worker()
    worker.pos_sig.connect(window.onPosChange)
    worker.exit_sig.connect(window.myexit)
    worker.screen.connect(window.take_screenshot)
    worker.setCond(window.mtx, window.cond)
    window.mouse_sig.connect(worker.setPos)
    window.reset_sig.connect(worker.setReset)
else:
    worker = None
    
# Begin training
t = threading.Thread(name='agent_loop', target=initial_AHIRL, args=(FLAGS,worker))
t.start()
if FLAGS.show:
    sys.exit(app.exec_())
