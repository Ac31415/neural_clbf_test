import os
import sys
import inspect
import random

import torch

from math import pi


import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt

show_animation = True

# Add the parent directory to the path to load the trainer module
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))  # type: ignore
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from trainer import Trainer  # noqa
from dynamics import (  # noqa
    f_damped_integrator,
    AB_damped_integrator,
    f_turtlebot,
    AB_turtlebot,
)
from nonlinear_mpc_controller import turtlebot_mpc_casadi  # noqa

from simulation import (
    simulate,
    generate_random_reference,
    DynamicsCallable,
)


def test_trainer_init():
    """Test initializing the trainer object; also returns a trainer object for
    use in other tests."""

    # Create a new trainer object, and make sure it works
    hyperparameters = {
        "n_state_dims": 2,
        "n_control_dims": 1,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 10,
        "n_trajs": 10,
        "controller_dt": 0.1,
        "sim_dt": 1e-2,
        "demonstration_noise": 0.3,
    }
    state_space = [
        (-5.0, 5.0),  # px
        (-5.0, 5.0),  # vx
    ]
    error_bounds = [
        5.0,  # px
        5.0,  # vx
    ]
    control_bounds = [
        3.0,  # u
    ]

    expert_horizon = 1.0

    def dummy_expert(x, x_ref, u_ref):
        return u_ref[0, :]

    my_trainer = Trainer(
        "test_network",
        hyperparameters,
        f_damped_integrator,
        AB_damped_integrator,
        dummy_expert,
        expert_horizon,
        state_space,
        error_bounds,
        control_bounds,
        0.1,  # validation_split
    )
    assert my_trainer

    return my_trainer


def test_trainer_normalize_state():
    """Test state normalization"""
    my_trainer = test_trainer_init()

    test_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    expected_x_norm = test_x / 5.0

    x_norm = my_trainer.normalize_state(test_x)
    assert torch.allclose(x_norm, expected_x_norm)


def test_trainer_normalize_error():
    """Test state normalization"""
    # Set a random seed for repeatability
    random.seed(0)
    torch.manual_seed(0)

    my_trainer = test_trainer_init()

    test_x_err = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    expected_x_err_norm = test_x_err / 5.0

    x_err_norm = my_trainer.normalize_error(test_x_err)
    assert torch.allclose(x_err_norm, expected_x_err_norm)


def test_trainer_positive_definite_loss():
    """Test loss"""
    my_trainer = test_trainer_init()

    # Define a generic matrix known to be positive definite
    test_M = torch.tensor(
        [
            [1.0, 0.1, 0.0],
            [0.1, 1.0, -0.1],
            [0.0, -0.1, 1.0],
        ]
    ).reshape(-1, 3, 3)

    pd_loss = my_trainer.positive_definite_loss(test_M)

    assert (pd_loss <= torch.tensor(0.0)).all()

    # Now repeat for a matrix known to be negative definite
    pd_loss = my_trainer.positive_definite_loss(-1.0 * test_M)

    assert (pd_loss > torch.tensor(0.0)).all()


def test_add_data_turtlebot():
    hyperparameters = {
        "n_state_dims": 3,
        "n_control_dims": 2,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 10,
        "n_trajs": 5,
        "controller_dt": 0.1,
        "sim_dt": 1e-2,
        "demonstration_noise": 0.3,
    }
    state_space = [
        (-5.0, 5.0),  # px
        (-5.0, 5.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]
    error_bounds = [
        0.5,  # px
        0.5,  # py
        1.0,  # theta
    ]
    control_bounds = [
        3.0,  # v
        pi,  # omega
    ]

    expert_horizon = 1.0
    # expert_horizon = hyperparameters["controller_dt"]

    def expert(x, x_ref, u_ref):
        return turtlebot_mpc_casadi(
            x, x_ref, u_ref, hyperparameters["controller_dt"], control_bounds
        )

    my_trainer = Trainer(
        "test_trainer",
        hyperparameters,
        f_turtlebot,
        AB_turtlebot,
        expert,
        expert_horizon,
        state_space,
        error_bounds,
        control_bounds,
        0.2,  # validation_split
    )

    # Get initial amounts of data
    n_training_points_initial = my_trainer.x_training.shape[0]
    n_validation_points_initial = my_trainer.x_validation.shape[0]

    # Try adding data
    counterexample_x = torch.zeros((1, hyperparameters["n_state_dims"]))
    counterexample_x_ref = torch.zeros((1, hyperparameters["n_state_dims"]))
    counterexample_u_ref = torch.zeros((1, hyperparameters["n_control_dims"]))
    my_trainer.add_data(
        counterexample_x, counterexample_x_ref, counterexample_u_ref, 10, 5, 0.2
    )

    # Make sure we added some data
    assert my_trainer.x_training.shape[0] > n_training_points_initial
    assert my_trainer.x_validation.shape[0] > n_validation_points_initial


# Define a function to run training and save the results
# (don't run when we run pytest, only when we run this file)
def do_training_turtlebot():
    hyperparameters = {
        "n_state_dims": 3,
        "n_control_dims": 2,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 100,
        "n_trajs": 100,
        "controller_dt": 0.1,
        "sim_dt": 1e-2,
        "demonstration_noise": 0.3,
    }
    state_space = [
        (-5.0, 5.0),  # px
        (-5.0, 5.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]

    # cartesian_state_space = [
    #     (-10.0, 10.0),  # x
    #     (-10.0, 10.0),  # y
    # ]

    error_bounds = [
        0.5,  # px
        0.5,  # py
        1.0,  # theta
    ]
    control_bounds = [
        3.0,  # v
        pi,  # omega
    ]

    expert_horizon = 1.0
    # expert_horizon = hyperparameters["controller_dt"]

    def expert(x, x_ref, u_ref):
        return turtlebot_mpc_casadi(
            x, x_ref, u_ref, hyperparameters["controller_dt"], control_bounds
        )

    my_trainer = Trainer(
        (
            "clone_M_cond_2x32_policy_2x32_metric_1e4_noisy_examples_100x0.1"
            "_no_L_lr1e-3_1s_horizon"
        ),
        hyperparameters,
        f_turtlebot,
        AB_turtlebot,
        expert,
        expert_horizon,
        state_space,
        # cartesian_state_space,
        error_bounds,
        control_bounds,
        0.3,  # validation_split
    )

    n_steps = 502
    # my_trainer.run_training(n_steps, debug=True, sim_every_n_steps=100)




class RobotType(Enum):
    circle = 0
    rectangle = 1



class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        # self.dt = 0.1  # [s] Time tick for motion prediction
        self.dt = 1e-2  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def check_if_dw(u, dw):

    if dw[0] <= u[0] <= dw[1] and dw[2] <= u[1] <= dw[3]:
        return True
    else:
        return False
    

def find_min_cost_u(x_init, x, v, y, config, ob, goal, min_cost, best_u, best_trajectory):

    trajectory = predict_trajectory(x_init, v, y, config)
    # calc cost
    to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
    speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
    ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

    final_cost = to_goal_cost + speed_cost + ob_cost

    # search minimum trajectory
    if min_cost >= final_cost:
        min_cost = final_cost
        best_u = [v, y]
        best_trajectory = trajectory
        if abs(best_u[0]) < config.robot_stuck_flag_cons \
                and abs(x[3]) < config.robot_stuck_flag_cons:
            # to ensure the robot do not get stuck in
            # best v=0 m/s (in front of an obstacle) and
            # best omega=0 rad/s (heading to the goal with
            # angle difference of 0)
            best_u[1] = -config.max_delta_yaw_rate

    return best_u, best_trajectory, min_cost


def average_best_u_cm_u(x_init, cm_u, best_u, config):

    cm_u_v = cm_u[0]
    cm_u_omega = cm_u[1]

    best_u_v = best_u[0]
    best_u_omega = best_u[1]

    average_u_v = (cm_u_v + best_u_v) / 2
    average_u_omega = (cm_u_omega + best_u_omega) / 2

    ave_u = np.array([average_u_v, average_u_omega])

    trajectory = predict_trajectory(x_init, average_u_v, average_u_omega, config)
    

    

    return ave_u, trajectory


def weighted_average_best_u_cm_u(x_init, cm_u, best_u, config):

    cm_u_v = cm_u[0]
    cm_u_omega = cm_u[1]

    best_u_v = best_u[0]
    best_u_omega = best_u[1]

    weighted_average_u_v = 0.1 * cm_u_v + 0.9 * best_u_v
    weighted_average_u_omega = 0.1 * cm_u_omega + 0.9 * best_u_omega

    weighted_ave_u = np.array([weighted_average_u_v, weighted_average_u_omega])

    trajectory = predict_trajectory(x_init, weighted_average_u_v, weighted_average_u_omega, config)
    

    

    return weighted_ave_u, trajectory




# def calc_control_and_trajectory(x, dw, config, goal, ob):
#     """
#     calculation final input with dynamic window
#     """

#     x_init = x[:]
#     min_cost = float("inf")
#     best_u = [0.0, 0.0]
#     best_trajectory = np.array([x])

#     # evaluate all trajectory with sampled input in dynamic window
#     for v in np.arange(dw[0], dw[1], config.v_resolution):
#         for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

#             trajectory = predict_trajectory(x_init, v, y, config)
#             # calc cost
#             to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
#             speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
#             ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

#             final_cost = to_goal_cost + speed_cost + ob_cost

#             # search minimum trajectory
#             if min_cost >= final_cost:
#                 min_cost = final_cost
#                 best_u = [v, y]
#                 best_trajectory = trajectory
#                 if abs(best_u[0]) < config.robot_stuck_flag_cons \
#                         and abs(x[3]) < config.robot_stuck_flag_cons:
#                     # to ensure the robot do not get stuck in
#                     # best v=0 m/s (in front of an obstacle) and
#                     # best omega=0 rad/s (heading to the goal with
#                     # angle difference of 0)
#                     best_u[1] = -config.max_delta_yaw_rate
#     return best_u, best_trajectory



def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory



def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    print("dw: ", dw)

    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    return u, trajectory


def calc_control_and_trajectory_dwa_cm(x, x_current, x_ref_current, u_ref_current, trained_controller, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])


    


    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            best_u, best_trajectory, min_cost = find_min_cost_u(x_init, x, v, y, config, ob, goal, min_cost, best_u, best_trajectory)

            # # ave_u, trajectory = average_best_u_cm_u(x_init, u_current, best_u, config)
            # ave_u, trajectory = weighted_average_best_u_cm_u(x_init, u_current, best_u, config)

            # best_u = ave_u
            # best_trajectory = trajectory

            # trajectory = predict_trajectory(x_init, v, y, config)
            # # calc cost
            # to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            # speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            # ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            # final_cost = to_goal_cost + speed_cost + ob_cost

            # # search minimum trajectory
            # if min_cost >= final_cost:
            #     min_cost = final_cost
            #     best_u = [v, y]
            #     best_trajectory = trajectory
            #     if abs(best_u[0]) < config.robot_stuck_flag_cons \
            #             and abs(x[3]) < config.robot_stuck_flag_cons:
            #         # to ensure the robot do not get stuck in
            #         # best v=0 m/s (in front of an obstacle) and
            #         # best omega=0 rad/s (heading to the goal with
            #         # angle difference of 0)
            #         best_u[1] = -config.max_delta_yaw_rate


    


    


    # v = u_current[0]
    # y = u_current[1]

    # best_u, best_trajectory, min_cost = find_min_cost_u(x_init, x, v, y, config, ob, goal, min_cost, best_u, best_trajectory)


    # average this cm control with the dwa control within the for loop below or after the for loop
    u_current = trained_controller(x_current, x_ref_current, u_ref_current) # need to convert this to numpy array later
    # u_current_tensor = u_current

    u_current = u_current.T.cpu().detach().numpy()
    u_current = np.array([u_current[0][0], u_current[1][0]])
    # u_current = np.array(u_current)

    # current_x_tensor = torch.from_numpy(x[:3]).float()

    # u_current = trained_controller(current_x_tensor, x_ref_current, u_ref_current)

    print('u_current from trained controller: ', u_current)



    ave_u, trajectory = weighted_average_best_u_cm_u(x_init, u_current, best_u, config)

    best_u = ave_u
    best_trajectory = trajectory



    u_current_tensor = torch.from_numpy(np.array([best_u])).float()


    # if check_if_dw(u_current, dw):

    #     v = u_current[0]
    #     y = u_current[1]

    #     best_u, best_trajectory, min_cost = find_min_cost_u(x_init, x, v, y, config, ob, goal, min_cost, best_u, best_trajectory)

    #     u_current_tensor = torch.from_numpy(np.array([best_u])).float()

    #     # trajectory = predict_trajectory(x_init, v, y, config)
    #     # # calc cost
    #     # to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
    #     # speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
    #     # ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

    #     # final_cost = to_goal_cost + speed_cost + ob_cost

    #     # # search minimum trajectory
    #     # if min_cost >= final_cost:
    #     #     min_cost = final_cost
    #     #     best_u = [v, y]
    #     #     best_trajectory = trajectory
    #     #     if abs(best_u[0]) < config.robot_stuck_flag_cons \
    #     #             and abs(x[3]) < config.robot_stuck_flag_cons:
    #     #         # to ensure the robot do not get stuck in
    #     #         # best v=0 m/s (in front of an obstacle) and
    #     #         # best omega=0 rad/s (heading to the goal with
    #     #         # angle difference of 0)
    #     #         best_u[1] = -config.max_delta_yaw_rate

    # else:

    #     u_current_tensor = torch.from_numpy(np.array([best_u])).float()




    return best_u, u_current_tensor, best_trajectory



# def dwa_control(x, config, goal, ob):
#     """
#     Dynamic Window Approach control
#     """
#     dw = calc_dynamic_window(x, config)

#     u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

#     return u, trajectory


def dwa_cm_control(x, x_current, x_ref_current, u_ref_current, trained_controller, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    # u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)

    u, u_current_tensor, trajectory = calc_control_and_trajectory_dwa_cm(x, x_current, x_ref_current, u_ref_current, trained_controller, dw, config, goal, ob)

    return u, u_current_tensor, trajectory



def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")



def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)



def calc_path_length(trajectory):

    # Initialize the path length
    path_length = 0

    # Loop over the coordinates in the trajectory
    for i in range(1, len(trajectory)):
        # Calculate the Euclidean distance between the current point and the previous point
        distance = math.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i][1] - trajectory[i-1][1])**2)
        
        # Add the distance to the path length
        path_length += distance

    # # Display the path length
    # print(path_length)

    return path_length



def DWA_PathPlanner(x_init, gx=10.0, gy=10.0, robot_type=RobotType.circle):
    config = Config()
    print(" start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    x = np.array(x_init)
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)

        print("u", u)
        print("predicted_trajectory", predicted_trajectory)

        x = motion(x, u, config.dt)  # simulate robot

        print("x", x)

        trajectory = np.vstack((trajectory, x))  # store state history

        print("trajectory", trajectory)

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")

    print(trajectory[0])
    path_length = calc_path_length(trajectory)
    print("DWA Planner Path length: ", path_length)

    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


    return path_length




def DWA_cm_PathPlanner(x_list, x_init, x_ref_sim, u_ref_sim, N_batch, trained_cm_controller, dt, gx=10.0, gy=10.0, robot_type=RobotType.circle):
    config = Config()

    N_steps = x_ref_sim.shape[1]
    n_state_dims = x_ref_sim.shape[-1]
    n_control_dims = u_ref_sim.shape[-1]
    x_init = x_init.reshape(N_batch, n_state_dims)

    x_sim = torch.zeros(N_batch, N_steps, n_state_dims).type_as(x_ref_sim)
    x_sim[:, 0, :] = x_init

    print(" start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # x = np.array(x_init)
    x = np.array(x_list)
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob

    step = 0

    while True:

        x_current = x_sim[:, step].reshape(N_batch, n_state_dims)
        x_ref_current = x_ref_sim[:, step].reshape(N_batch, n_state_dims)
        u_ref_current = u_ref_sim[:, step].reshape(N_batch, n_control_dims)

        u, u_current_tensor, predicted_trajectory = dwa_cm_control(x, x_current, x_ref_current, u_ref_current, trained_cm_controller, config, goal, ob)

        # u, predicted_trajectory = dwa_control(x, config, goal, ob)

        print("u: ", u)
        print("u_current_tensor: ", u_current_tensor)
        print("predicted_trajectory", predicted_trajectory)

        x = motion(x, u, config.dt)  # simulate robot

        print("x: ", x)

        trajectory = np.vstack((trajectory, x))  # store state history

        print("trajectory: ", trajectory)








        # Get the derivatives and update the state
        x_dot = f_turtlebot(x_current, u_current_tensor).detach()
        x_current = x_current.detach()
        x_sim[:, step + 1, :] = x_current + dt * x_dot

        print('x_sim: ', x_sim)


        step += 1



        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")

    print(trajectory[0])
    path_length = calc_path_length(trajectory)
    print("DWA+CM Planner Path length: ", path_length)

    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


    return path_length



def do_training_dwa_testing_turtlebot():
    hyperparameters = {
        "n_state_dims": 3,
        "n_control_dims": 2,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 100,
        "n_trajs": 100,
        "controller_dt": 0.1,
        # "controller_dt": 0.01,
        "sim_dt": 1e-2,
        # "sim_dt": 0.1,
        "demonstration_noise": 0.3,
    }
    # state_space = [
    #     (-5.0, 5.0),  # px
    #     (-5.0, 5.0),  # py
    #     (-2 * pi, 2 * pi),  # theta
    # ]


    state_space = [
        (-30.0, 30.0),  # px
        (-30.0, 30.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]

    # cartesian_state_space = [
    #     (-10.0, 10.0),  # x
    #     (-10.0, 10.0),  # y
    # ]

    error_bounds = [
        0.5,  # px
        0.5,  # py
        1.0,  # theta
    ]
    control_bounds = [
        3.0,  # v
        pi,  # omega
    ]

    expert_horizon = 1.0
    # expert_horizon = hyperparameters["controller_dt"]

    def expert(x, x_ref, u_ref):
        return turtlebot_mpc_casadi(
            x, x_ref, u_ref, hyperparameters["controller_dt"], control_bounds
        )

    my_trainer = Trainer(
        (
            "clone_M_cond_2x32_policy_2x32_metric_1e4_noisy_examples_100x0.1"
            "_no_L_lr1e-3_1s_horizon"
        ),
        hyperparameters,
        f_turtlebot,
        AB_turtlebot,
        expert,
        expert_horizon,
        state_space,
        # cartesian_state_space,
        error_bounds,
        control_bounds,
        0.3,  # validation_split
    )





    # n_steps = 502
    n_steps = 100
    # my_trainer.run_training(n_steps, debug=True, sim_every_n_steps=100)
    my_trainer.run_training(n_steps, debug=True, sim_every_n_steps=20)


    # Generate a random reference trajectory
    N_batch = 1  # number of test trajectories
    T = 20.0  # length of trajectory
    x_init, x_ref_sim, u_ref_sim = generate_random_reference(
        N_batch,
        T,
        my_trainer.sim_dt,
        my_trainer.n_state_dims,
        my_trainer.n_control_dims,
        my_trainer.state_space,
        my_trainer.control_bounds,
        my_trainer.error_bounds,
        my_trainer.dynamics,
    )


    # # Simulate
    # x_sim, u_sim, M_sim, dMdt_sim = simulate(
    #     x_init,
    #     x_ref_sim,
    #     u_ref_sim,
    #     my_trainer.sim_dt,
    #     my_trainer.controller_dt,
    #     my_trainer.dynamics,
    #     my_trainer.u,
    #     my_trainer.metric_value,
    #     my_trainer.metric_derivative_t,
    #     my_trainer.control_bounds,
    # )


    print(x_init[0].T.cpu().detach().numpy())
    # print(x_init.T.cpu().detach().numpy())


    x_ref_sim[0, :, 0].T.cpu().detach().numpy()
    x_ref_sim[0, :, 1].T.cpu().detach().numpy()
    u_ref_sim[0, 1:, 0].T.cpu().detach().numpy()
    u_ref_sim[0, 1:, 1].T.cpu().detach().numpy()


    print(x_ref_sim[0, :, 0].T.cpu().detach().numpy())
    print(x_ref_sim[0, :, 1].T.cpu().detach().numpy())
    print(u_ref_sim[0, :, 0].T.cpu().detach().numpy())
    print(u_ref_sim[0, :, 1].T.cpu().detach().numpy())

    print(x_ref_sim[0, 0].T.cpu().detach().numpy())
    print(u_ref_sim[0, 0].T.cpu().detach().numpy())

    # x_init[0].T.cpu().detach().numpy()
    # x_ref_sim_0 = x_init[0].T.cpu().detach().numpy()


    x_ref_sim_0 = x_ref_sim[0, 0].T.cpu().detach().numpy()
    u_ref_sim_0 = u_ref_sim[0, 0].T.cpu().detach().numpy()

    init_x = x_init[0].T.cpu().detach().numpy()
    init_u = np.array([0.0, 0.0])


    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    # x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # x = np.append(x_ref_sim_0, u_ref_sim_0)
    x = np.append(init_x, init_u)

    # x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])

    print(x)


    # DWA_PathPlanner(x_init, gx=10.0, gy=10.0, robot_type=RobotType.circle)
    dwa_path_length = DWA_PathPlanner(x, gx=10.0, gy=10.0, robot_type=RobotType.rectangle)



    dwa_cm_path_length = DWA_cm_PathPlanner(x, x_init, x_ref_sim, u_ref_sim, N_batch, my_trainer.u, my_trainer.sim_dt, gx=10.0, gy=10.0, robot_type=RobotType.rectangle)


    print("DWA Path length: ", dwa_path_length)
    print("DWA+CM Path length: ", dwa_cm_path_length)





    # x_ref_current = x_ref_sim[0, step].T.cpu().detach().numpy()
    # u_ref_current = u_ref_sim[0, step].T.cpu().detach().numpy()


    


    

    

    



    # while True:

    #     x_ref_current = x_ref_sim[0, step].T.cpu().detach().numpy()
    #     u_ref_current = u_ref_sim[0, step].T.cpu().detach().numpy()

    #     # u, predicted_trajectory = dwa_control(x, config, goal, ob)
    #     u, predicted_trajectory = dwa_cm_control(x, x_ref_current, u_ref_current, my_trainer.u, config, goal, ob)
    #     # u_current = my_trainer.u(x_current, x_ref_current, u_ref_current)
    #     # u_current = my_trainer.u(x, x_ref_current, u_ref_current)



    #     step += 1



    # my_trainer.u(x, x_ref, u_ref)


    # # Simulate
    # x_sim, u_sim, M_sim, dMdt_sim = simulate(
    #     x_init,
    #     x_ref_sim,
    #     u_ref_sim,
    #     my_trainer.sim_dt,
    #     my_trainer.controller_dt,
    #     my_trainer.dynamics,
    #     my_trainer.u,
    #     my_trainer.metric_value,
    #     my_trainer.metric_derivative_t,
    #     my_trainer.control_bounds,
    # )




def do_training_tracking_DWA_reference_trajectory_turtlebot():


    hyperparameters = {
        "n_state_dims": 3,
        "n_control_dims": 2,
        "lambda_M": 0.1,
        "metric_hidden_layers": 2,
        "metric_hidden_units": 32,
        "policy_hidden_layers": 2,
        "policy_hidden_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 100,
        "n_trajs": 100,
        "controller_dt": 0.1,
        "sim_dt": 1e-2,
        "demonstration_noise": 0.3,
    }
    state_space = [
        (-30.0, 30.0),  # px
        (-30.0, 30.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]

    state_space = [
        (-10.0, 20.0),  # px
        (-2.5, 15.0),  # py
        (-2 * pi, 2 * pi),  # theta
    ]

    # cartesian_state_space = [
    #     (-10.0, 10.0),  # x
    #     (-10.0, 10.0),  # y
    # ]

    error_bounds = [
        0.5,  # px
        0.5,  # py
        1.0,  # theta
    ]
    control_bounds = [
        3.0,  # v
        pi,  # omega
    ]

    expert_horizon = 1.0
    # expert_horizon = hyperparameters["controller_dt"]

    def expert(x, x_ref, u_ref):
        return turtlebot_mpc_casadi(
            x, x_ref, u_ref, hyperparameters["controller_dt"], control_bounds
        )
    

    # config = Config()
    # ob = config.ob

    # gx=10.0
    # gy=10.0

    # goal = np.array([gx, gy])

    my_trainer = Trainer(
        (
            "clone_M_cond_2x32_policy_2x32_metric_1e4_noisy_examples_100x0.1"
            "_no_L_lr1e-3_1s_horizon"
        ),
        hyperparameters,
        f_turtlebot,
        AB_turtlebot,
        expert,
        expert_horizon,
        state_space,
        # cartesian_state_space,
        error_bounds,
        control_bounds,
        0.3,  # validation_split
        # ob,
    )


    # n_steps = 502
    # # my_trainer.run_training(n_steps, debug=True, sim_every_n_steps=100)

    n_steps = 100
    # my_trainer.run_training(n_steps, debug=True, sim_every_n_steps=100)
    my_trainer.run_training(n_steps, debug=True, sim_every_n_steps=20)


if __name__ == "__main__":
    # do_training_turtlebot()

    # do_training_dwa_testing_turtlebot()

    do_training_tracking_DWA_reference_trajectory_turtlebot()
