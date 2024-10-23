# MIND MELD

Mutual Information-driven Meta-Learning from Demonstration (MIND MELD) is a Learning from Demonstration algorithm that learns from suboptimal and heterogenous human demonstrators.  MIND MELD learns an embedding that encapsulates a person's suboptimal teaching style and utilizes this embedding to shift the person's suboptimal labels to labels closer to optimal.

This code includes the MIND MELD algorithm and the code for the experiment in our paper: [MIND MELD: Personalized Meta-Learning for Robot-Centric Imitation Learning](https://ieeexplore.ieee.org/abstract/document/9889616)

Authors: Mariah L. Schrum, Erin Hedlund-Botti, Nina Moorman, and Matthew C. Gombolay

If you use our code, please cite us:

`M. L. Schrum, E. Hedlund-Botti, N. Moorman and M. C. Gombolay, "MIND MELD: Personalized Meta-Learning for Robot-Centric Imitation Learning," 2022 17th ACM/IEEE International Conference on Human-Robot Interaction (HRI), Sapporo, Japan, 2022, pp. 157-165, doi: 10.1109/HRI53351.2022.9889616.`


## Codebase Overview

### Car

We implement and test MIND MELD using a driving simulator environment (AirSim) and a steering wheel (Logitech G920).  To run the experiment, run `study_main.py`

##### BaseAgent.py
- Base class for MIND MELD algorithm and DAgger and Behavioral Cloning (BC) baselines.

##### Baselines
- Includes code for DAgger baseline agent
- Includes code for BC baseline agent

##### CarEnv.py
- Includes code for controlling and recording data from the AirSim car.

##### config.py
- Edit this file to update file paths and experiment parameters
  
##### Data
- Includes calibration trajectories and ground truth labels for calibration trajectories.
- New data will be saved here. Participant data from our user studies is not included due to privacy and Institutional Review Board policies.

##### DrivingSim/utils
- Includes AirSim utils and data loaders

##### FlaskApp
- Code for communicating between SteeringWheel UWP App and DrivingSim Python code

##### intial_demo.py
- Records initial demonstration for all algorithms (MIND MELD, DAgger, and BC)

##### MeLD
- Includes MIND MELD algorithm
- Pretest runs the calibration to determine people's embedding 
- Also includes code for generating ground truth labels (adapted from the following references)
  - [MPC](https://github.com/matssteinweg/Multi-Purpose-MPC)
  - [RRT and Stanley](https://github.com/AtsushiSakai/PythonRobotics)
 
##### Model.py
- Base class for the learning models

##### practice.py
- Script for letting participants practice driving the car

##### Record.py
- Code for recording study data (car and steering wheel states)

##### SteeringWheel
To record data from the steering wheel, we edited the RawGameControllerUWP app from https://github.com/microsoft/Xbox-ATG-Samples
- Edited files are included in SteeringWheel/UWPSamples/System/RawGameControllerUWP/

##### stop_listener.py
- Script for stopping a demonstration on keypress

##### study_main.py
Code for running study:
1. Calibration
2. Initial Demonstration
3. Teach MIND MELD, DAgger, and BC in a specified order

##### utils
- Data loaders and scripts for plotting data


### MeLD_ToyExperiment
Synthetic data for testing MIND MELD
- To train using synthetic data, run `train_synthetic.py`

## Installation Instructions

*Note: If using Logitech G920 Steering wheel, you must use Windows*

1. Install [AirSim](https://microsoft.github.io/AirSim/build_windows/) for the driving simulation environment
   - Note: we use the Blocks environment
3. Install [Steering Wheel UWP](https://github.com/microsoft/Xbox-ATG-Samples)
   - Update RawGameControllerUWP code with our changes
4. Install necessary python packages using requirements.txt
5. Update file paths in config
6. Run study_main.py

### Customizing
The code currently will run preset calibration trajectories.  If you want to create new calibration trajectories based on your map or setup, use Car/MeLD/create_pretest_rollouts.py and generate new ground truth labels. The config file has all of the starting environment locations, which can be customized.
