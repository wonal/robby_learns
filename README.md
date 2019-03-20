## Robby Learns

#### A robot that learns to pick up cans in a grid using Q-learning

This is a command-line application that trains a robot to navigate a 10x10 grid, avoid crashing into walls, and pick up randomly-placed cans.  Picking up a can yields a reward of 10, crashing into a wall yields a penalty of -5, and picking up a non-existent can yields a penalty of -1.  The robot is trained over 5000 epochs consisting of 200 moves.  A test phase occurs afterwards consisting of 5000 epochs, where the robot's total reward per epoch is averaged and displayed along with the standard deviation.  

#### Setup and How to Run:

- Install numpy using your preferred installer
- Clone the repository
- Run the program: `python -m src.main`
- Tests can be run with the command: `python -m unittest discover -s Tests -p "*_tests.py`

#### Implementation Details:

 - Update equation:
 Q(s<sub>t,</sub>a<sub>t</sub>) = Q(s<sub>t,</sub>a<sub>t</sub>) + eta(r<sub>t</sub> + gamma*max<sub>a'</sub>Q(s<sub>t+1,</sub>a') - Q(s<sub>t,</sub>a<sub>t</sub>))
 - Epsilon value during training started at 1.0, then decreased by 0.01 every 50 epochs until it remained fixed at 0.1.  During the test phase, it remained fixed at 0.1.  

