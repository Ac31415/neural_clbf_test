"""A Python class for training the contraction metric and controller networks"""
from collections import OrderedDict
from itertools import product
# from typing import Any, cast, Callable, Dict, List, Tuple, Union
from typing import cast, Callable, Dict, List, Tuple, Union
import subprocess
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import math

import random


from simulation_mod_case_studies import (
    simulate,
    simulate_follower,
    simulate_multi_agent,
    generate_random_reference,
    generate_random_reference_dwa,
    # generate_random_reference_with_Cartesian_trajectory,
    DynamicsCallable,

    Follower,
)

from nonlinear_mpc_controller_mod_case_studies import turtlebot_mpc_casadi_torch  # noqa






class Trainer(nn.Module):
    """
    Run batches of training in between searching for counterexamples.
    """

    required_hparams = [
        "n_state_dims",
        "n_control_dims",
        "lambda_M",
        "metric_hidden_layers",
        "metric_hidden_units",
        "policy_hidden_layers",
        "policy_hidden_units",
        "learning_rate",
        "batch_size",
        "n_trajs",
        "controller_dt",
        "sim_dt",
        "demonstration_noise",
    ]
    n_state_dims: int
    n_control_dims: int
    lambda_M: float
    metric_hidden_layers: int
    metric_hidden_units: int
    policy_hidden_layers: int
    policy_hidden_units: int
    learning_rate: float
    batch_size: int
    n_trajs: int
    controller_dt: float
    sim_dt: float
    demonstration_noise: float

    def __init__(
        self,
        network_name: str,
        hyperparameters: Dict[str, Union[int, float]],
        dynamics: DynamicsCallable,
        A_B_fcn: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ],
        expert_controller: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        expert_horizon: float,
        state_space: List[Tuple[float, float]],
        # cartesian_state_space: List[Tuple[float, float]],
        error_bounds: List[float],
        control_bounds: List[float],
        validation_split: float,
        # ob: np.ndarray,
    ):
        super(Trainer, self).__init__()
        """
        Initialize a pair of networks, dataset, and optimizer.

        args:
            network_name: name of network, used to name log directory
            hyperparameters: layers and nodes. Dictionary of named hyperparameters.
                "n_state_dims": number of state dimensions
                "n_control_dims": number of control dimensions
                "lambda_M": desired contraction rate
                "metric_hidden_layers": number of hidden layers in metric net
                "metric_hidden_units": number of neurons per hidden layer in metric net
                "policy_hidden_layers": number of hidden layers in policy net
                "policy_hidden_units": number of neurons per hidden layer in policy net
                "learning_rate": learning rate
                "batch_size": for batched gradient descent
                ""n_trajs": number of reference trajectories over which we gather
                    expert demonstrations.
                "controller_dt": period at which controller should be run (seconds)
                "sim_dt": period at which dynamics are simulated (seconds)
                "demonstration_noise": proportion of control bounds added as noise to
                                       demonstrations.

            dynamics: to be used in M_dot, this should be f(x, u)
            A_B_fcn: we need linearization of dynamics around training points,
                     e.g. a function that takes a state and reference control and
                     returns a linearization about that point.

                     More specifically: the dynamics are a function f(x, u) that maps
                     from state/action pairs to state derivatives
                     (R^n_dims x R^n_controls -> R^n_dims). In contraction analysis,
                     we work with linearizations of the dynamics (I guess it would be
                     more accurate to say contraction analysis considers the dynamics on
                     the tangent space, which is the same to the best of my knowledge
                     except you drop the constant term).

                     So, the linearization about (x_ref, u_ref) looks like

                     f(x, u) \\approx  f(x_ref, u_ref) + df/dx(x_ref, u_ref) (x - x_ref)
                                                       + df/du(x_ref, u_ref) (u - u_ref)

                     For the tangent dynamics, we don't care about the constant term (we
                     assume the reference trajectory is consistent with the dynamics),
                     so the linearization we care about is

                     d/dt(x - x_ref) \\approx  df/dx(x_ref, u_ref) (x - x_ref)
                                               + df/du(x_ref, u_ref) (u - u_ref)

                     We define A = df/dx(x_ref, u_ref) (an n_dims x n_dims matrix) and
                     B = df/du(x_ref, u_ref) (an n_dims x n_controls matrix).
            expert_controller: expert control policy to supervise the learned policy.
            expert_horizon: time in seconds the expert requires to anticipate reference.
            state_space: the region we want our contraction metric to apply over.
                         This is just used for normalizing the input and generating
                         data, it can be a hyperrectangle for simplicity.
                         Defined as a list of tuples of min and max value for each state
            error_bounds: The amount of tracking error we want to train and test on.
                          Used for normalizing inputs and for generating training data.
                          Defined as a list of max error magnitudes for each state
            control_bounds: Defines symmetric bounds on reference control inputs. Used
                            to generate training data.
            validation_split: fraction of points to reserve for validation set.
        """
        # =================================================================
        # Validate hyperparameters and save them.
        # =================================================================

        self.check_hyperparameters(hyperparameters)

        # for hparam_name in Trainer.required_hparams:
        #     error_str = f"Required hyper-parameter {hparam_name} missing"
        #     assert hparam_name in hyperparameters, error_str

        #     setattr(self, hparam_name, hyperparameters[hparam_name])

        # self.save_hyperparameters(hyperparameters, network_name, dynamics, A_B_fcn, expert_controller, expert_horizon, control_bounds)

        self.network_name = network_name
        self.hparams = hyperparameters
        self.dynamics = dynamics
        self.A_B_fcn = A_B_fcn
        self.expert_controller = expert_controller
        self.expert_horizon = expert_horizon
        self.control_bounds = control_bounds

        # =================================================================
        # Initialize the tensorboard logger
        # =================================================================
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
        path_base = f"logs/{network_name}/commit_{commit_hash}"
        version_num = 0
        while os.path.isdir(path_base + f"/version_{version_num}"):
            version_num += 1
        self.writer = SummaryWriter(path_base + f"/version_{version_num}")
        self.global_steps = 0

        # =================================================================
        # Initialize the contraction and policy networks
        # =================================================================

        # -----------------------------------------------------------------
        # Define a normalization layer for states. This is just a weight
        # and bias to shift and scale each input dimension to the range [-1, 1]
        # -----------------------------------------------------------------
        self.state_space = state_space
        self.state_normalization_bias = torch.zeros(self.n_state_dims)
        self.state_normalization_weights = torch.eye(self.n_state_dims)
        for state_dim, state_limits in enumerate(self.state_space):
            # Pull out the min and max value and compute the center and range
            state_min, state_max = state_limits
            state_semi_range = (state_max - state_min) / 2.0
            state_center = (state_max + state_min) / 2.0

            # Save these as an bias and weights so that
            # x_normalized = W * x + b
            self.state_normalization_bias[state_dim] = -state_center
            self.state_normalization_weights[state_dim, state_dim] = (
                1 / state_semi_range
            )

        # Scale the bias by the weights so we can do W * x + b instead of W*(x+b)
        self.state_normalization_bias = (
            self.state_normalization_weights @ self.state_normalization_bias
        )

        # -----------------------------------------------------------------
        # Define a normalization layer for state errors. This is just a weight
        # to scale each input dimension to the range [-1, 1]
        # -----------------------------------------------------------------
        self.error_bounds = error_bounds
        self.error_normalization_weights = torch.eye(self.n_state_dims)
        for state_dim, error_limit in enumerate(self.error_bounds):
            # Save this bound as a weight. Errors are assumed to be centered about 0
            self.error_normalization_weights[state_dim, state_dim] = 1 / error_limit

        # -----------------------------------------------------------------
        # Define a normalization layer for control bounds. This is just a weight
        # to scale each output dimension to an appropriate range
        # -----------------------------------------------------------------
        self.control_normalization_weights = torch.eye(self.n_control_dims)
        for control_dim, control_limit in enumerate(self.control_bounds):
            # Save this bound as a weight. Errors are assumed to be centered about 0
            self.control_normalization_weights[control_dim, control_dim] = control_limit

        # -----------------------------------------------------------------
        # Define the contraction network
        #
        # The contraction network is structured as M = A.T + A.T for some
        # learned matrix A. A is represented as a vector (i.e. "unwrapped"),
        # so we can represent the mapping from M to A as a fixed linear layer.
        # -----------------------------------------------------------------

        # Define some fully-connected layers for the unwrapped A matrix
        # n_state_dims -> hidden -> ... -> n_state_dims * n_state_dims
        self.metric_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.metric_layers["input_linear"] = nn.Linear(
            self.n_state_dims,
            self.metric_hidden_units,
        )
        self.metric_layers["input_activation"] = nn.ReLU()
        for i in range(self.metric_hidden_layers):
            self.metric_layers[f"layer_{i}_linear"] = nn.Linear(
                self.metric_hidden_units, self.metric_hidden_units
            )
            self.metric_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.metric_layers["output_linear"] = nn.Linear(
            self.metric_hidden_units, self.n_state_dims ** 2
        )
        # self.metric_layers["output_activation"] = nn.ReLU()
        self.A = nn.Sequential(self.metric_layers)


        # # -----------------------------------------------------------------
        # # Define the follower contraction network
        # #
        # # The contraction network is structured as M = A.T + A.T for some
        # # learned matrix A. A is represented as a vector (i.e. "unwrapped"),
        # # so we can represent the mapping from M to A as a fixed linear layer.
        # # -----------------------------------------------------------------

        # # Define some fully-connected layers for the unwrapped A matrix
        # # n_state_dims -> hidden -> ... -> n_state_dims * n_state_dims
        # self.follower_metric_layers: OrderedDict[str, nn.Module] = OrderedDict()
        # self.follower_metric_layers["input_linear"] = nn.Linear(
        #     self.n_state_dims,
        #     self.metric_hidden_units,
        # )
        # self.follower_metric_layers["input_activation"] = nn.ReLU()
        # for i in range(self.metric_hidden_layers):
        #     self.follower_metric_layers[f"layer_{i}_linear"] = nn.Linear(
        #         self.metric_hidden_units, self.metric_hidden_units
        #     )
        #     self.follower_metric_layers[f"layer_{i}_activation"] = nn.ReLU()
        # self.follower_metric_layers["output_linear"] = nn.Linear(
        #     self.metric_hidden_units, self.n_state_dims ** 2
        # )
        # # self.metric_layers["output_activation"] = nn.ReLU()
        # self.follower_A = nn.Sequential(self.follower_metric_layers)


        # Define the linear map for M = A.T + A (operating on unwrapped matrices).
        # This is some dense code. Let's explain some:
        #
        # We want M[i, j] = A[i, j] + A[j, i]. In unwrapped form, [i, j] = [i*n + j],
        # so this means M[i*n + j] = A[i*n + j] + A[j*n + i].
        # This means there is a linear map M = (I + W) A, where W is all zero except
        # that W[i*n + j, j*n + i] = 1.0
        n = self.n_state_dims
        self.A_to_M = torch.eye(n * n)
        for i, j in product(range(n), range(n)):
            self.A_to_M[i * n + j, j * n + i] += 1.0


        # # Define the follower linear map for M = A.T + A (operating on unwrapped matrices).
        # # This is some dense code. Let's explain some:
        # #
        # # We want M[i, j] = A[i, j] + A[j, i]. In unwrapped form, [i, j] = [i*n + j],
        # # so this means M[i*n + j] = A[i*n + j] + A[j*n + i].
        # # This means there is a linear map M = (I + W) A, where W is all zero except
        # # that W[i*n + j, j*n + i] = 1.0
        # n = self.n_state_dims
        # self.follower_A_to_M = torch.eye(n * n)
        # for i, j in product(range(n), range(n)):
        #     self.follower_A_to_M[i * n + j, j * n + i] += 1.0

        # We'll use these components to construct M in the function defined below.

        # -----------------------------------------------------------------
        # Now define the policy network, which is a function of both the
        # current state and the reference state. We'll also add the reference
        # control signal to the output of this policy network later.
        # -----------------------------------------------------------------

        # We're going to build the network up layer by layer, starting with the input
        self.policy_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.policy_layers["input_linear"] = nn.Linear(
            2 * self.n_state_dims,
            self.policy_hidden_units,
        )
        self.policy_layers["input_activation"] = nn.ReLU()
        for i in range(self.policy_hidden_layers):
            self.policy_layers[f"layer_{i}_linear"] = nn.Linear(
                self.policy_hidden_units, self.policy_hidden_units
            )
            self.policy_layers[f"layer_{i}_activation"] = nn.ReLU()
        self.policy_layers["output_linear"] = nn.Linear(
            self.policy_hidden_units, self.n_control_dims
        )
        self.policy_nn = nn.Sequential(self.policy_layers)


        # # -----------------------------------------------------------------
        # # Now define the follower policy network, which is a function of both the
        # # current state and the reference state. We'll also add the reference
        # # control signal to the output of this policy network later.
        # # -----------------------------------------------------------------

        # # We're going to build the network up layer by layer, starting with the input
        # self.follower_policy_layers: OrderedDict[str, nn.Module] = OrderedDict()
        # self.follower_policy_layers["input_linear"] = nn.Linear(
        #     2 * self.n_state_dims,
        #     self.policy_hidden_units,
        # )
        # self.follower_policy_layers["input_activation"] = nn.ReLU()
        # for i in range(self.policy_hidden_layers):
        #     self.follower_policy_layers[f"layer_{i}_linear"] = nn.Linear(
        #         self.policy_hidden_units, self.policy_hidden_units
        #     )
        #     self.follower_policy_layers[f"layer_{i}_activation"] = nn.ReLU()
        # self.follower_policy_layers["output_linear"] = nn.Linear(
        #     self.policy_hidden_units, self.n_control_dims
        # )
        # self.follower_policy_nn = nn.Sequential(self.follower_policy_layers)


        # We'll combine the output of this policy network with the reference control
        # in a function defined below.

        # =================================================================
        # Initialize the optimizer
        # =================================================================
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )


        # self.follower = Follower(
        #     network_name,
        #     hyperparameters,
        #     dynamics,
        #     A_B_fcn,
        #     expert_controller,
        #     expert_horizon,
        #     state_space,
        #     # cartesian_state_space,
        #     error_bounds,
        #     control_bounds,
        #     validation_split,  # validation_split
        # )

        num_agents = 2

        self.list_of_followers = []

        for i in range(num_agents):

            follower = Follower(
                network_name,
                hyperparameters,
                dynamics,
                A_B_fcn,
                expert_controller,
                expert_horizon,
                state_space,
                # cartesian_state_space,
                error_bounds,
                control_bounds,
                validation_split,  # validation_split
            )

            self.list_of_followers.append(follower)


        # self.leader = self.list_of_followers[0]
        # self.follower = self.list_of_followers[1]
        # self.follower = self.list_of_followers[0]

        self.follower = Follower(
            network_name,
            hyperparameters,
            dynamics,
            A_B_fcn,
            expert_controller,
            expert_horizon,
            state_space,
            # cartesian_state_space,
            error_bounds,
            control_bounds,
            validation_split,  # validation_split
        )

        self.follower_1 = Follower(
            network_name,
            hyperparameters,
            dynamics,
            A_B_fcn,
            expert_controller,
            expert_horizon,
            state_space,
            # cartesian_state_space,
            error_bounds,
            control_bounds,
            validation_split,  # validation_split
        )
        
        self.leader = Follower(
            network_name,
            hyperparameters,
            dynamics,
            A_B_fcn,
            expert_controller,
            expert_horizon,
            state_space,
            # cartesian_state_space,
            error_bounds,
            control_bounds,
            validation_split,  # validation_split
        )
        
        # self.list_of_followers.insert(0, self.leader)  # Prepend 1 to the list

        self.list_of_followers = []
        self.list_of_followers.append(self.leader)
        self.list_of_followers.append(self.follower)
        self.list_of_followers.append(self.follower_1)


        # =================================================================
        # Set up initial dataset with training/validation split
        # =================================================================
        self.validation_split = validation_split


        # =================================================================
        # Set up initial dataset with training/validation split
        # =================================================================
        # self.ob = ob

        # Generate data using self.n_trajs trajectories of length batch_size
        T = self.batch_size * self.controller_dt + self.expert_horizon
        print("Constructing initial dataset...")
        # get these trajectories from a larger range of errors than we expect in testing
        error_bounds_demonstrations = [1.5 * bound for bound in self.error_bounds]
        x_init, x_ref, u_ref = generate_random_reference(
            self.n_trajs,
            T,
            self.controller_dt,
            self.n_state_dims,
            self.n_control_dims,
            state_space,
            control_bounds,
            error_bounds_demonstrations,
            self.dynamics,
        )

        # x_init, x_ref, u_ref = generate_random_reference_dwa(
        #     self.n_trajs,
        #     T,
        #     self.controller_dt,
        #     self.n_state_dims,
        #     self.n_control_dims,
        #     state_space,
        #     control_bounds,
        #     error_bounds_demonstrations,
        #     self.dynamics,
        # )

        # x_init, x_ref, cart_x_ref, u_ref = generate_random_reference_with_Cartesian_trajectory(
        #     self.n_trajs,
        #     T,
        #     self.controller_dt,
        #     self.n_state_dims,
        #     self.n_control_dims,
        #     state_space,
        #     cartesian_state_space,
        #     control_bounds,
        #     error_bounds_demonstrations,
        #     self.dynamics,
        # )

        traj_length = x_ref.shape[1]

        # print(x_init)
        # print(x_init.shape[0])
        # print(x_init.shape[1])

        # print(x_ref)
        # print(x_ref.shape[0])
        # print(x_ref.shape[1])

        # # print(cart_x_ref)
        # # print(cart_x_ref.shape[0])
        # # print(cart_x_ref.shape[1])

        # # # print(x_ref[0, :, 3])
        # # # print(x_ref[1, :, 3])
        # # # print(x_ref[2, :, 3])

        # print(u_ref)
        # print(u_ref.shape[0])
        # print(u_ref.shape[1])


        # # print(x_ref.shape[1])
        # # print(u_ref.shape[1])

        # # print(x_init[traj_idx, :])
        # # print(x_ref[traj_idx, :, :])
        # # print(u_ref[traj_idx, :, :])


        # # Simulate
        # # ref_x_sim, ref_u_sim, M_sim, dMdt_sim = simulate(
        # #     x_init[traj_idx, :],
        # #     x_ref[traj_idx, :, :],
        # #     u_ref[traj_idx, :, :],
        # #     self.sim_dt,
        # #     self.controller_dt,
        # #     self.dynamics,
        # #     self.u,
        # #     self.metric_value,
        # #     self.metric_derivative_t,
        # #     self.control_bounds,
        # # )

        # # ref_x_sim, ref_u_sim, M_sim, dMdt_sim = simulate(
        # #     x_init,
        # #     x_ref,
        # #     u_ref,
        # #     self.sim_dt,
        # #     self.controller_dt,
        # #     self.dynamics,
        # #     self.u,
        # #     None,
        # #     None,
        # #     self.control_bounds,
        # # )

        # # # ref_x_sim, ref_u_sim, M_sim, dMdt_sim = simulate(
        # # #     x_init,
        # # #     x_ref,
        # # #     u_ref,
        # # #     self.sim_dt,
        # # #     self.controller_dt,
        # # #     self.dynamics,
        # # #     self.u,
        # # #     self.metric_value,
        # # #     self.metric_derivative_t,
        # # #     self.control_bounds,
        # # # )
        # # ref_x_sim = ref_x_sim.detach()
        # # ref_u_sim = ref_u_sim.detach()

        # Make a plot for state error
        t_range = np.arange(0, self.controller_dt + self.expert_horizon, self.sim_dt)




        self.plot_ref_trajectories(x_init, x_ref)

        self.plot_ref_controls(t_range, u_ref)
        

        

        # for traj_idx in range(self.n_trajs):


        #     print(traj_idx)

        #     fig, ax = plt.subplots()
        #     fig.set_size_inches(8, 8)


        #     # Plot the reference and actual trajectories
        #     ax.plot([], [], linestyle=":", color="k", label="Reference")
        #     ax.plot([], [], marker="o", color="k", label="Start")

        #     ax.plot(
        #         x_ref[traj_idx, :, 0].T.cpu().detach().numpy(),
        #         x_ref[traj_idx, :, 1].T.cpu().detach().numpy(),
        #         linestyle=":",
        #     )
        #     ax.plot(
        #         x_ref[traj_idx, 0, 0].T.cpu().detach().numpy(),
        #         x_ref[traj_idx, 0, 1].T.cpu().detach().numpy(),
        #         marker="o",
        #         color="k",
        #     )

        #     # ax.set_xlabel("px (m * vx)")
        #     # ax.set_ylabel("py (m * vy)")
        #     ax.set_xlabel("x (m)")
        #     ax.set_ylabel("y (m)")

        #     ax.legend()

        #     # ax.set_title(f"Reference Trajectory {traj_idx} in Phase Plane")
        #     ax.set_title(f"Reference Trajectory {traj_idx}")


        #     # Save the figure
        #     self.writer.add_figure(
        #         f"Reference Trajectory {traj_idx}",
        #         fig,
        #     )



        #     # fig, ax = plt.subplots()
        #     # fig.set_size_inches(8, 8)


        #     # # Plot the reference and actual trajectories
        #     # ax.plot([], [], linestyle=":", color="k", label="Reference")
        #     # ax.plot([], [], marker="o", color="k", label="Start")

        #     # ax.plot(
        #     #     cart_x_ref[traj_idx, :, 0].T.cpu().detach().numpy(),
        #     #     cart_x_ref[traj_idx, :, 1].T.cpu().detach().numpy(),
        #     #     linestyle=":",
        #     # )
        #     # ax.plot(
        #     #     cart_x_ref[traj_idx, 0, 0].T.cpu().detach().numpy(),
        #     #     cart_x_ref[traj_idx, 0, 1].T.cpu().detach().numpy(),
        #     #     marker="o",
        #     #     color="k",
        #     # )

        #     # ax.set_xlabel("x (m)")
        #     # ax.set_ylabel("y (m)")

        #     # ax.legend()

        #     # ax.set_title(f"Reference Trajectory {traj_idx} in Cartesian Coordinates")
            

        #     # # Save the figure
        #     # self.writer.add_figure(
        #     #     f"Cartesian Reference Trajectory {traj_idx}",
        #     #     fig,
        #     # )



        #     # Make a plot for each control
        #     for control_idx in range(self.n_control_dims):
        #         fig, ax = plt.subplots()
        #         fig.set_size_inches(8, 4)

        #         # Plot the reference and actual trajectories
        #         ax.plot([], [], linestyle=":", color="k", label="Reference")
        #         # ax.plot(
        #         #     t_range[1:],
        #         #     ref_u_sim[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
        #         #     linestyle=":",
        #         # )
        #         ax.plot(
        #             t_range[1:],
        #             u_ref[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
        #             linestyle=":",
        #         )
        #         ax.set_xlabel("time (s)")
        #         ax.set_ylabel(f"Control {control_idx} for Reference Trajectory {traj_idx}")
        #         ax.legend()

        #         # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

        #         # Save the figure
        #         self.writer.add_figure(
        #             f"Simulated Control Trajectory/Control {control_idx} for Reference Trajectory {traj_idx}",
        #             fig,
        #         )


        

        # Create some places to store the simulation results
        x = torch.zeros((self.n_trajs, traj_length, self.n_state_dims))
        x[:, 0, :] = x_init
        u_expert = torch.zeros((self.n_trajs, traj_length, self.n_control_dims))
        u_current = torch.zeros((self.n_control_dims,))


        # this is where the actual expert is

        # The expert policy requires a sliding window over the trajectory, so we need
        # to iterate through that trajectory.
        # Make sure we don't overrun the end of the reference while planning
        n_steps = traj_length - int(self.expert_horizon / self.controller_dt)
        dynamics_updates_per_control_update = int(self.controller_dt / self.sim_dt)
        for traj_idx in tqdm(range(self.n_trajs)):
            traj_range = range(n_steps - 1)
            for tstep in traj_range:
                # Get the current states and references
                x_current = x[traj_idx, tstep].reshape(-1, self.n_state_dims).clone()

                # Pick out sliding window into references for use with the expert
                x_ref_expert = (
                    x_ref[
                        traj_idx,
                        tstep : tstep + int(self.expert_horizon // self.controller_dt),
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )
                u_ref_expert = (
                    u_ref[
                        traj_idx,
                        tstep : tstep + int(self.expert_horizon // self.controller_dt),
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )

                # Run the expert
                u_current = torch.tensor(
                    self.expert_controller(
                        x_current.detach().cpu().numpy().squeeze(),
                        x_ref_expert,
                        u_ref_expert,
                    )
                )

                u_expert[traj_idx, tstep, :] = u_current

                # Add a bit of noise to widen the distribution of states
                u_current += torch.normal(
                    0, self.demonstration_noise * torch.tensor(self.control_bounds)
                )

                # Update state
                for _ in range(dynamics_updates_per_control_update):
                    x_dot = self.dynamics(
                        x_current,
                        u_current.reshape(-1, self.n_control_dims),
                    )
                    x_current += self.sim_dt * x_dot
                x[traj_idx, tstep + 1, :] = x_current

            # plt.plot(x[traj_idx, :n_steps, 0], x[traj_idx, :n_steps, 1], "-")
            # plt.plot(x_ref[traj_idx, :n_steps, 0], x_ref[traj_idx, :n_steps, 1], ":")
            # plt.plot(x[traj_idx, 0, 0], x[traj_idx, 0, 1], "ko")
            # plt.plot(x_ref[traj_idx, 0, 0], x_ref[traj_idx, 0, 1], "ko")
            # plt.show()

            # plt.plot(u_expert[traj_idx, :, 0], "r:")
            # plt.plot(u_expert[traj_idx, :, 1], "r--")
            # plt.plot(u_ref[traj_idx, :, 0], "k:")
            # plt.plot(u_ref[traj_idx, :, 1], "k--")
            # plt.show()

        print(" Done!")

        # print('x: ', x)
        # print('x_dim_0: ', x.shape[0])
        # print('x_dim_1: ', x.shape[1])

        # print('x_init: ', x_init)
        # print('x_init_dim_0: ', x_init.shape[0])
        # print('x_init_dim_1: ', x_init.shape[1])

        # print('x_ref: ', x_ref)
        # print('x_ref_dim_0: ', x_ref.shape[0])
        # print('x_ref_dim_1: ', x_ref.shape[1])

        # print('u_ref: ', u_ref)
        # print('u_ref_dim_0: ', u_ref.shape[0])
        # print('u_ref_dim_1: ', u_ref.shape[1])

        # print('u_expert: ', u_expert)
        # print('u_expert_dim_0: ', u_expert.shape[0])
        # print('u_expert_dim_1: ', u_expert.shape[1])

        # self.plot_expert_trajectories(x)

        # self.plot_expert_vs_ref_trajectories(x_init, x_ref[:, : tstep + 1, :], x[:, : tstep + 1, :])
        self.plot_expert_vs_ref_trajectories(x_ref[:, : tstep + 1, :], x[:, : tstep + 1, :])
        self.plot_expert_vs_ref_controls(traj_range, u_ref[:, : tstep + 1, :], u_expert[:, : tstep + 1, :])

        # self.plot_expert_controls(traj_range, u_expert[:, : tstep + 1, :])

        # Reshape
        x = x[:, : tstep + 1, :].reshape(-1, self.n_state_dims)
        x_ref = x_ref[:, : tstep + 1, :].reshape(-1, self.n_state_dims)
        u_ref = u_ref[:, : tstep + 1, :].reshape(-1, self.n_control_dims)
        u_expert = u_expert[:, : tstep + 1, :].reshape(-1, self.n_control_dims)

        # Split data into training and validation and save it
        random_indices = torch.randperm(x.shape[0])
        val_points = int(x.shape[0] * self.validation_split)
        validation_indices = random_indices[:val_points]
        training_indices = random_indices[val_points:]

        self.x_ref_training = x_ref[training_indices]
        self.x_ref_validation = x_ref[validation_indices]

        self.u_ref_training = u_ref[training_indices]
        self.u_ref_validation = u_ref[validation_indices]

        self.x_training = x[training_indices]
        self.x_validation = x[validation_indices]

        self.u_expert_training = u_expert[training_indices]
        self.u_expert_validation = u_expert[validation_indices]


    # =================================================================
    # Define some utility functions for plotting trajectories
    # =================================================================

    def plot_ref_trajectories(self, x_init, x_ref):

        print('x_init: ', x_init)
        print('Number of initial states: ', x_init.shape[0])
        print('Size of each state: ', x_init.shape[1])

        print('x_ref: ', x_ref)
        print('Number of x_ref: ', x_ref.shape[0])
        print('Length of each trajectory: ', x_ref.shape[1])

        for traj_idx in range(self.n_trajs):


            print(traj_idx)

            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)


            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot([], [], marker="o", color="k", label="Start")

            ax.plot(
                x_ref[traj_idx, :, 0].T.cpu().detach().numpy(),
                x_ref[traj_idx, :, 1].T.cpu().detach().numpy(),
                linestyle=":",
            )
            ax.plot(
                x_ref[traj_idx, 0, 0].T.cpu().detach().numpy(),
                x_ref[traj_idx, 0, 1].T.cpu().detach().numpy(),
                marker="o",
                color="k",
            )

            # ax.plot(
            #     x_init[traj_idx, 0].T.cpu().detach().numpy(),
            #     x_init[traj_idx, 1].T.cpu().detach().numpy(),
            #     marker="o",
            #     color="k",
            # )

            
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            ax.legend()

            ax.set_title(f"Reference Trajectory {traj_idx}")


            # Save the figure
            self.writer.add_figure(
                f"Reference Trajectory {traj_idx}",
                fig,
            )


    def plot_expert_vs_ref_trajectories(self, x_ref, x):

        for traj_idx in range(self.n_trajs):


            print(traj_idx)

            fig, ax = plt.subplots()
            fig.set_size_inches(8, 8)


            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot([], [], marker="o", color="k", label="Start")

            ax.plot(
                x_ref[traj_idx, :, 0].T.cpu().detach().numpy(),
                x_ref[traj_idx, :, 1].T.cpu().detach().numpy(),
                linestyle=":",
            )
            ax.plot(
                x_ref[traj_idx, 0, 0].T.cpu().detach().numpy(),
                x_ref[traj_idx, 0, 1].T.cpu().detach().numpy(),
                marker="o",
                color="k",
            )


            ax.set_prop_cycle(None)  # Re-use colors for the reference
            ax.plot([], [], linestyle="-", color="k", label="Expert")
            ax.plot(
                x[traj_idx, :, 0].T.cpu().detach().numpy(),
                x[traj_idx, :, 1].T.cpu().detach().numpy(),
                linestyle="-",
            )
            ax.plot(
                x[traj_idx, 0, 0].T.cpu().detach().numpy(),
                x[traj_idx, 0, 1].T.cpu().detach().numpy(),
                marker="o",
                color="k",
            )
            ax.legend()

            
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

            ax.legend()

            ax.set_title(f"Expert vs. Reference Trajectory {traj_idx}")


            # Save the figure
            self.writer.add_figure(
                f"Expert vs. Reference Trajectory {traj_idx}",
                fig,
            )

    # def plot_expert_trajectories(self, x):

    #     for traj_idx in range(self.n_trajs):


    #         print(traj_idx)

    #         fig, ax = plt.subplots()
    #         fig.set_size_inches(8, 8)


    #         # Plot the reference and actual trajectories
    #         ax.plot([], [], linestyle=":", color="k", label="Expert")
    #         ax.plot([], [], marker="o", color="k", label="Start")

    #         ax.plot(
    #             x[traj_idx, :, 0].T.cpu().detach().numpy(),
    #             x[traj_idx, :, 1].T.cpu().detach().numpy(),
    #             linestyle=":",
    #         )
    #         ax.plot(
    #             x[traj_idx, 0, 0].T.cpu().detach().numpy(),
    #             x[traj_idx, 0, 1].T.cpu().detach().numpy(),
    #             marker="o",
    #             color="k",
    #         )

            
    #         ax.set_xlabel("x (m)")
    #         ax.set_ylabel("y (m)")

    #         ax.legend()

    #         ax.set_title(f"Expert Trajectory {traj_idx}")


    #         # Save the figure
    #         self.writer.add_figure(
    #             f"Expert Trajectory {traj_idx}",
    #             fig,
    #         )

    def plot_ref_controls(self, t_range, u_ref):


        print('u_ref: ', u_ref)
        print('Number of u_ref: ', u_ref.shape[0])
        print('Length of each u_ref trajectory: ', u_ref.shape[1])


        for traj_idx in range(self.n_trajs):

            print(traj_idx)

            # Make a plot for each control
            for control_idx in range(self.n_control_dims):
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 4)

                # Plot the reference and actual trajectories
                ax.plot([], [], linestyle=":", color="k", label="Reference")
                
                ax.plot(
                    t_range[1:],
                    u_ref[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
                    linestyle=":",
                )
                ax.set_xlabel("time (s)")
                ax.set_ylabel(f"Control {control_idx} for Reference Trajectory {traj_idx}")
                ax.legend()


                # Save the figure
                self.writer.add_figure(
                    f"Simulated Control Trajectory/Control {control_idx} for Reference Trajectory {traj_idx}",
                    fig,
                )


    def plot_expert_vs_ref_controls(self, t_range, u_ref, u_expert):


        for traj_idx in range(self.n_trajs):

            print(traj_idx)

            # Make a plot for each control
            for control_idx in range(self.n_control_dims):
                fig, ax = plt.subplots()
                fig.set_size_inches(8, 4)

                # Plot the reference and actual trajectories
                ax.plot([], [], linestyle=":", color="k", label="Reference")
                
                ax.plot(
                    t_range[1:],
                    u_ref[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
                    linestyle=":",
                )

                ax.set_prop_cycle(None)  # Re-use colors for the reference
                ax.plot([], [], linestyle="-", color="k", label="Expert")
                ax.plot(
                    t_range[1:],
                    u_expert[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
                    linestyle="-",
                )

                ax.set_xlabel("time (s)")
                ax.set_ylabel(f"Control {control_idx} for Expert and Reference Trajectory {traj_idx}")
                ax.legend()


                # Save the figure
                self.writer.add_figure(
                    f"Simulated Control Trajectory/Control {control_idx} for Expert and Reference Trajectory {traj_idx}",
                    fig,
                )




    # def plot_expert_controls(self, t_range, u_expert):

    #     for traj_idx in range(self.n_trajs):

    #         print(traj_idx)

    #         # Make a plot for each control
    #         for control_idx in range(self.n_control_dims):
    #             fig, ax = plt.subplots()
    #             fig.set_size_inches(8, 4)

    #             # Plot the reference and actual trajectories
    #             ax.plot([], [], linestyle=":", color="k", label="Expert")
                
    #             ax.plot(
    #                 t_range[1:],
    #                 u_expert[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
    #                 linestyle=":",
    #             )
    #             ax.set_xlabel("time (s)")
    #             ax.set_ylabel(f"Control {control_idx} for Expert Trajectory {traj_idx}")
    #             ax.legend()


    #             # Save the figure
    #             self.writer.add_figure(
    #                 f"Simulated Control Trajectory/Control {control_idx} for Expert Trajectory {traj_idx}",
    #                 fig,
    #             )




        

    # =================================================================
    # Define some utility functions for getting the control input and
    # metric at states (and for getting Jacobians of those) quantities
    # =================================================================

    def check_hyperparameters(self, hyperparameters: Dict[str, Union[int, float]]) -> None:
        """Check that the hyperparameters are valid"""
        for hparam_name in Trainer.required_hparams:
            error_str = f"Required hyper-parameter {hparam_name} missing"
            assert hparam_name in hyperparameters, error_str

            setattr(self, hparam_name, hyperparameters[hparam_name])

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the given state values"""
        x = torch.matmul(self.state_normalization_weights, x.T).T
        x = x + self.state_normalization_bias
        return x

    def normalize_error(self, x_err: torch.Tensor) -> torch.Tensor:
        """Normalize the given state error values"""
        x = torch.matmul(self.error_normalization_weights, x_err.T).T
        return x

    def normalize_control(self, u: torch.Tensor) -> torch.Tensor:
        """Normalize the given control values (to be used on outputs)"""
        u = torch.matmul(self.control_normalization_weights, u.T).T
        return u

    def jacobian(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the jacobian of function outputs f wrt x. Based on Dawei's code from
        https://github.com/sundw2014/C3M

        args:
            f - B x m x 1 function outputs (computed with requires_grad=True)
            x - B x n x 1 function inputs. Assumed to be independent of each other
        returns
            df/dx - B x m x n Jacobian
        """
        f = f + 0.0 * x.sum()  # to avoid the case that f is independent of x

        bs = x.shape[0]
        m = f.size(1)
        n = x.size(1)
        J = torch.zeros(bs, m, n).type(x.type())

        # Compute the gradient for each output dimension, then accumulate
        for i in range(m):
            J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
        return J

    def jacobian_matrix(self, M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the jacobian of a matrix function f wrt x. Based on Dawei's code from
        https://github.com/sundw2014/C3M

        args:
            f - B x m x m function outputs (computed with requires_grad=True)
            x - B x n x 1 function inputs. Assumed to be independent of each other
        returns
            df/dx - B x m x m x n Jacobian
        """
        bs = x.shape[0]
        m = M.size(-1)
        n = x.size(1)
        J = torch.zeros(bs, m, m, n).type(x.type())
        for i in range(m):
            for j in range(m):
                J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(
                    -1
                )
        return J

    def weighted_gradients(
        self, W: torch.Tensor, v: torch.Tensor, x: torch.Tensor, detach: bool = False
    ) -> torch.Tensor:
        """
        Return the Jacobian-vector product of the Jacobian of W (wrt x) and v.
        Based on Dawei's code from
        https://github.com/sundw2014/C3M

        args:
            f - B x m x m function outputs (computed with requires_grad=True)
            v - B x n x 1 function inputs. Assumed to be independent of each other
            x - B x n x 1 function inputs. Assumed to be independent of each other
        returns
            df/dx - B x m x m
        """
        assert v.size() == x.size()
        bs = x.shape[0]
        if detach:
            return (self.jacobian_matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(
                dim=3
            )
        else:
            return (self.jacobian_matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

    def Bbot_fcn(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a default annihilator matrix"""
        bs = x.shape[0]
        Bbot = torch.cat(
            (
                torch.eye(
                    self.n_state_dims - self.n_control_dims,
                    self.n_state_dims - self.n_control_dims,
                ),
                torch.zeros(
                    self.n_control_dims, self.n_state_dims - self.n_control_dims
                ),
            ),
            dim=0,
        )
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

    def M_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the metric matrix M at a given point x (in unwrapped form)

        args:
            x - batch_size x self.n_state_dims tensor
        returns:
            M - batch_size x self.n_state_dims ^ 2 tensor
        """
        x = self.normalize_state(x)

        # We define M as A.T + A, so start by getting A in unwrapped form
        A = self.A(x)  # batch_size x self.n_state_dims ^ 2

        # Map from A to M (unwrapped) via the saved linear map
        M = torch.matmul(self.A_to_M, A.T).T  # batch_size x self.n_state_dims ^ 2

        return M
    
    # def follower_M_flat(self, x: torch.Tensor) -> torch.Tensor:
    #     """Compute the metric matrix M at a given point x (in unwrapped form)

    #     args:
    #         x - batch_size x self.n_state_dims tensor
    #     returns:
    #         M - batch_size x self.n_state_dims ^ 2 tensor
    #     """
    #     x = self.normalize_state(x)

    #     # We define M as A.T + A, so start by getting A in unwrapped form
    #     A = self.follower_A(x)  # batch_size x self.n_state_dims ^ 2

    #     # Map from A to M (unwrapped) via the saved linear map
    #     M = torch.matmul(self.follower_A_to_M, A.T).T  # batch_size x self.n_state_dims ^ 2

    #     return M

    def M(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the metric matrix M at a given point x

        args:
            x - batch_size x self.n_state_dims tensor
        returns:
            M - batch_size x self.n_state_dims x self.n_state_dims tensor
        """
        # Get flattened M and reshape it into a matrix
        M = self.M_flat(x)
        M = M.reshape(-1, self.n_state_dims, self.n_state_dims)
        M = M + 1.0 * torch.eye(M.shape[-1])

        return M
    
    # def follower_M(self, x: torch.Tensor) -> torch.Tensor:
    #     """Compute the metric matrix M at a given point x

    #     args:
    #         x - batch_size x self.n_state_dims tensor
    #     returns:
    #         M - batch_size x self.n_state_dims x self.n_state_dims tensor
    #     """
    #     # Get flattened M and reshape it into a matrix
    #     M = self.follower_M_flat(x)
    #     M = M.reshape(-1, self.n_state_dims, self.n_state_dims)
    #     M = M + 1.0 * torch.eye(M.shape[-1])

    #     return M

    def metric_value(self, x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """Compute the metric x^T M x at a given point x

        args:
            x - batch_size x self.n_state_dims tensor
            x_ref - batch_size x self.n_state_dims tensor of reference state
        returns:
            M - batch_size x self.n_state_dims x self.n_state_dims tensor
        """
        # Get the metric matrix
        M = self.M(x)

        # Compute the metric value
        metric = torch.bilinear(x - x_ref, x - x_ref, M, bias=None)

        return metric
    
    # def follower_metric_value(self, x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    #     """Compute the metric x^T M x at a given point x

    #     args:
    #         x - batch_size x self.n_state_dims tensor
    #         x_ref - batch_size x self.n_state_dims tensor of reference state
    #     returns:
    #         M - batch_size x self.n_state_dims x self.n_state_dims tensor
    #     """
    #     # Get the metric matrix
    #     M = self.follower_M(x)

    #     # Compute the metric value
    #     metric = torch.bilinear(x - x_ref, x - x_ref, M, bias=None)

    #     return metric

    def metric_derivative_t(
        self,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        u_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the time derivative of the metrix at x

        args:
            x - batch_size x self.n_state_dims tensor of state
            x_ref - batch_size x self.n_state_dims tensor of reference state
            u_ref - batch_size x self.n_state_dims tensor of reference control
        returns:
            d/dt metrix
        """
        # We need to enable grad on x to enable computing gradients wrt x
        x = x.requires_grad_()

        # Get the metric
        M = self.M(x)

        # Get control and jacobian
        u = self.u(x, x_ref, u_ref)

        # Get rate of change of metric
        xdot = self.dynamics(x, u)
        Mdot = self.weighted_gradients(M, xdot, x, detach=False)

        # Get dynamics Jacobians. Only requires one call to grad, since this will
        # also compute the gradient through u
        closed_loop_jacobian = self.jacobian(xdot.reshape(-1, self.n_state_dims, 1), x)

        MABK = M.matmul(closed_loop_jacobian)

        dmetric_dt = torch.bilinear(
            x - x_ref, x - x_ref, Mdot + MABK.transpose(1, 2) + MABK, bias=None
        )

        # Disable tracking gradients when we're done
        x = x.detach()

        return dmetric_dt
    
    # def follower_metric_derivative_t(
    #     self,
    #     x: torch.Tensor,
    #     x_ref: torch.Tensor,
    #     u_ref: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Compute the time derivative of the metrix at x

    #     args:
    #         x - batch_size x self.n_state_dims tensor of state
    #         x_ref - batch_size x self.n_state_dims tensor of reference state
    #         u_ref - batch_size x self.n_state_dims tensor of reference control
    #     returns:
    #         d/dt metrix
    #     """
    #     # We need to enable grad on x to enable computing gradients wrt x
    #     x = x.requires_grad_()

    #     # Get the metric
    #     M = self.follower_M(x)

    #     # Get control and jacobian
    #     u = self.follower_u(x, x_ref, u_ref)

    #     # Get rate of change of metric
    #     xdot = self.dynamics(x, u)
    #     Mdot = self.weighted_gradients(M, xdot, x, detach=False)

    #     # Get dynamics Jacobians. Only requires one call to grad, since this will
    #     # also compute the gradient through u
    #     closed_loop_jacobian = self.jacobian(xdot.reshape(-1, self.n_state_dims, 1), x)

    #     MABK = M.matmul(closed_loop_jacobian)

    #     dmetric_dt = torch.bilinear(
    #         x - x_ref, x - x_ref, Mdot + MABK.transpose(1, 2) + MABK, bias=None
    #     )

    #     # Disable tracking gradients when we're done
    #     x = x.detach()

    #     return dmetric_dt

    def u(
        self, x: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor
    ) -> torch.Tensor:
        """Compute the control input at a given point x given reference state and
        controls

        args:
            x - batch_size x self.n_state_dims tensor of state
            x_ref - batch_size x self.n_state_dims tensor of reference state
            u_ref - batch_size x self.n_control_dims tensor of reference control
        """
        # The control policy will be based on the current state, reference state, and
        # reference input. The policy network will take as input the current state and
        # the tracking error (desired_state - current_state), and we'll add the output
        # of the policy network to the reference input
        x_error = x - x_ref
        x_norm = self.normalize_state(x)
        x_error_norm = self.normalize_error(x_error)

        # Ensure that policy = u_ref whenever x = x_ref
        policy_input = torch.cat([x_norm, x_error_norm], dim=1)
        baseline_input = torch.cat([x_norm, 0.0 * x_error_norm], dim=1)

        tracking_policy = self.policy_nn(policy_input) - self.policy_nn(baseline_input)
        tracking_policy = self.normalize_control(tracking_policy)

        control_input = tracking_policy + u_ref

        return control_input
    
    # def follower_u(
    #     self, x: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor
    # ) -> torch.Tensor:
    #     """Compute the control input at a given point x given reference state and
    #     controls

    #     args:
    #         x - batch_size x self.n_state_dims tensor of state
    #         x_ref - batch_size x self.n_state_dims tensor of reference state
    #         u_ref - batch_size x self.n_control_dims tensor of reference control
    #     """
    #     # The control policy will be based on the current state, reference state, and
    #     # reference input. The policy network will take as input the current state and
    #     # the tracking error (desired_state - current_state), and we'll add the output
    #     # of the policy network to the reference input
    #     x_error = x - x_ref
    #     x_norm = self.normalize_state(x)
    #     x_error_norm = self.normalize_error(x_error)

    #     # Ensure that policy = u_ref whenever x = x_ref
    #     policy_input = torch.cat([x_norm, x_error_norm], dim=1)
    #     baseline_input = torch.cat([x_norm, 0.0 * x_error_norm], dim=1)

    #     tracking_policy = self.follower_policy_nn(policy_input) - self.follower_policy_nn(baseline_input)
    #     tracking_policy = self.normalize_control(tracking_policy)

    #     control_input = tracking_policy + u_ref

    #     return control_input

    #     # mpc_output = torch.zeros_like(control_input)
    #     # for t in range(x.shape[0]):
    #     #     mpc_output[t] = turtlebot_mpc_casadi_torch(
    #     #         x[t].reshape(1, 3),
    #     #         x_ref[t].reshape(1, 1, 3),
    #     #         u_ref[t].reshape(1, 1, 2),
    #     #         self.controller_dt,
    #     #         self.control_bounds
    #     #     )

    #     # # return control_input
    #     # return mpc_output + 1e-5 * control_input

    # =================================================================
    # Define some losses
    # =================================================================

    def compute_losses(
        self,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        u_ref: torch.Tensor,
        u_expert: torch.Tensor,
        i: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the loss

        args:
            x - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of state
            x_ref - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of reference state
            u_ref - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of reference control
                u_ref - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of expert control
            i - epoch number (if <= 1, apply only conditioning losses)
        """
        losses = {}

        # Metric-related losses
        losses["conditioning"] = self.contraction_loss_conditioning(x, x_ref, u_ref)
        losses["M"] = self.contraction_loss_M(x, x_ref, u_ref)
        # if i > 1:
        #     losses["W"] = self.contraction_loss_W(x, x_ref, u_ref)

        # Expert behavior cloning loss
        losses["u"] = self.policy_cloning_loss(x, x_ref, u_ref, u_expert)

        return losses

    def contraction_loss_conditioning(
        self,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        u_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Construct a loss based on the contraction metric positive definite property

        args:
            x - batch_size x self.n_state_dims tensor of state
            x_ref - batch_size x self.n_state_dims tensor of reference state
            u_ref - batch_size x self.n_state_dims tensor of reference control
        """
        loss = torch.tensor(0.0)
        M = self.M(x)

        # Penalize M if it is not positive definite
        loss += self.positive_definite_loss(M, eps=0.1)

        # Make sure that M has bounded eigenvalues
        m_lb = 1e0
        m_ub = 5e2
        loss += self.positive_definite_loss(M - m_lb * torch.eye(M.shape[-1]), eps=0.1)
        loss += self.positive_definite_loss(-M + m_ub * torch.eye(M.shape[-1]), eps=0.1)

        return loss

    @torch.enable_grad()
    def contraction_loss_M(
        self,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        u_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Construct a loss based on the contraction metric change property

        args:
            x - batch_size x self.n_state_dims tensor of state
            x_ref - batch_size x self.n_state_dims tensor of reference state
            u_ref - batch_size x self.n_state_dims tensor of reference control
        """
        loss = torch.tensor(0.0)

        # We need to enable grad on x to enable computing gradients wrt x
        x = x.requires_grad_()

        # Get the metric
        M = self.M(x)

        # Get control and jacobian
        u = self.u(x, x_ref, u_ref)

        # Get rate of change of metric
        xdot = self.dynamics(x, u)
        Mdot = self.weighted_gradients(M, xdot, x, detach=False)

        # Get dynamics Jacobians. Only requires one call to grad, since this will
        # also compute the gradient through u
        closed_loop_jacobian = self.jacobian(xdot.reshape(-1, self.n_state_dims, 1), x)

        MABK = M.matmul(closed_loop_jacobian)

        # This is the simple loss (which Dawei describes as "hard") from eq(5)
        # in the neural contraction paper.

        contraction_cond = Mdot + MABK.transpose(1, 2) + MABK + 2 * self.lambda_M * M
        loss += self.positive_definite_loss(-contraction_cond, eps=0.1)

        # Disable tracking gradients when we're done
        x = x.requires_grad_(False)

        return loss

    @torch.enable_grad()
    def contraction_loss_W(
        self,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        u_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Construct a loss based on the dual contraction metric change property

        args:
            x - batch_size x self.n_state_dims tensor of state
            x_ref - batch_size x self.n_state_dims tensor of reference state
            u_ref - batch_size x self.n_state_dims tensor of reference control
        """
        loss = torch.tensor(0.0)

        # We need to enable grad on x to enable computing gradients wrt x
        x = x.requires_grad_()

        bs = x.shape[0]

        # Get the metric and dual
        M = self.M(x)
        W = torch.linalg.inv(M)

        # Only use this loss when M is invertible
        M_invertible = torch.linalg.cond(M) < 1e3

        # Make sure that W has bounded eigenvalues
        w_lb = 1e-1
        w_ub = 1e2
        loss += self.positive_definite_loss(
            W[M_invertible] - w_lb * torch.eye(W.shape[-1]), eps=0.1
        )
        loss += self.positive_definite_loss(
            -W[M_invertible] + w_ub * torch.eye(W.shape[-1]), eps=0.1
        )

        # Get the control
        u = self.u(x, x_ref, u_ref)

        # Get the dynamics info
        _, B = self.A_B_fcn(x, u)
        Bbot = self.Bbot_fcn(x)
        f = self.dynamics(x, torch.zeros_like(u))
        DfDx = self.jacobian(f.reshape(-1, self.n_state_dims, 1), x)

        # Compute the C1 term, based on Dawei's C3M code
        C1_inner = (
            -self.weighted_gradients(W, f, x, detach=False)
            + DfDx.matmul(W)
            + W.matmul(DfDx.transpose(1, 2))
            + 2 * self.lambda_M * W
        )
        C1 = Bbot.transpose(1, 2).matmul(C1_inner).matmul(Bbot)
        # C1 has to be a negative definite matrix
        loss += self.positive_definite_loss(C1[M_invertible], eps=0.1)

        # Compute the C2 terms, based on Dawei's C3M code
        DBDx = torch.zeros(
            bs, self.n_state_dims, self.n_state_dims, self.n_control_dims
        ).type_as(x)
        for i in range(self.n_control_dims):
            DBDx[:, :, :, i] = self.jacobian(B[:, :, i].unsqueeze(-1), x)

        for j in range(self.n_control_dims):
            C2_inner = self.weighted_gradients(W, B[:, :, j], x, detach=False)
            C2_inner -= DBDx[:, :, :, j].matmul(W) + W.matmul(
                DBDx[:, :, :, j].transpose(1, 2)
            )
            C2 = Bbot.transpose(1, 2).matmul(C2_inner).matmul(Bbot)

            # C2 should be zero, so penalize the norm of it
            loss += (C2 ** 2).reshape(bs, -1).sum(dim=-1)[M_invertible].mean()

        # Disable tracking gradients when we're done
        x = x.requires_grad_(False)

        return loss

    def policy_cloning_loss(
        self,
        x: torch.Tensor,
        x_ref: torch.Tensor,
        u_ref: torch.Tensor,
        u_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Construct a loss based on the control (prefer small-magnitude controls)

        args:
            x - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of state
            x_ref - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of reference state
            u_ref - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of reference control
            u_expert - (batch_size + self.expert_horizon // self.controller_dt) x
                self.n_state_dims tensor of expert control
        """
        # Compare the learned and expert policies
        u = self.u(x, x_ref, u_ref)
        u_err = u - u_expert
        loss = (u_err ** 2).mean()

        return loss

    def positive_definite_loss(self, A, eps=0.1, gershgorin=False) -> torch.Tensor:
        """Return a loss that is 0 if A is positive definite and 0 otherwise.

        Optionally uses the Gershgorin Circle Theorem, so if loss is zero then A is PD,
        but not every PD matrix will yield a loss of zero.
        """
        if gershgorin:
            # let lambda_min_i be the i-th diagonal of A minus the sum of the absolute
            # values of the other elements in that row. Then min_i (lambda_min_i) > 0
            # implies A is positive definite

            diagonal_entries = torch.diagonal(A, dim1=-2, dim2=-1)
            off_diagonal_sum = torch.abs(A).sum(dim=-1) - torch.abs(diagonal_entries)
            min_gershgorin_eig_A = diagonal_entries - off_diagonal_sum

            # Use an offset relu
            loss = F.relu(eps - min_gershgorin_eig_A)
        else:
            # Otherwise, actually compute the eigenvalues
            A_eigval = torch.linalg.eigvals(A)  # type: ignore
            min_eig_A, _ = A_eigval.real.min(dim=-1)
            loss = F.relu(eps - min_eig_A)

        return loss.mean()
    
    def calculate_path_length(self, x_coords, y_coords):
        # Ensure the arrays are numpy arrays
        x_coords = np.array(x_coords).flatten()
        y_coords = np.array(y_coords).flatten()

        # print(x_coords, y_coords)

        # Calculate the differences between consecutive points
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # print(dx, dy)

        # Calculate the distance between each point
        distances = np.sqrt(dx**2 + dy**2)

        # Sum the distances to get the total path length
        path_length = np.sum(distances)

        return path_length
    
    def copy_network_weights(self, source, target):
        target.load_state_dict(source.state_dict())

    def train_networks(self, ):
        pass


    def plot_agents_vs_ref_trajectories(self, x_ref, x):

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")

        ax.plot(
            x_ref[:, :, 0].T.cpu().detach().numpy(),
            x_ref[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref[:, 0, 0].T.cpu().detach().numpy(),
            x_ref[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )



        for traj_idx in range(len(x)):


            print(traj_idx)


            # Generate a random color
            color = (random.random(), random.random(), random.random())


            ax.set_prop_cycle(None)  # Re-use colors for the reference
            ax.plot([], [], linestyle="-", color=color, label=f"Agent {traj_idx}")
            ax.plot(
                x[traj_idx, :, 0].T.cpu().detach().numpy(),
                x[traj_idx, :, 1].T.cpu().detach().numpy(),
                linestyle="-",
            )
            ax.plot(
                x[traj_idx, 0, 0].T.cpu().detach().numpy(),
                x[traj_idx, 0, 1].T.cpu().detach().numpy(),
                marker="o",
                color=color,
            )

        ax.legend()

            
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        ax.legend()

        ax.set_title(f"Agents vs. Reference Trajectory")


        # Save the figure
        self.writer.add_figure(
            f"Agent vs. Reference Trajectory",
            fig,
        )


    # def plot_expert_vs_ref_controls(self, t_range, u_ref, u_expert):


    #     for traj_idx in range(self.n_trajs):

    #         print(traj_idx)

    #         # Make a plot for each control
    #         for control_idx in range(self.n_control_dims):
    #             fig, ax = plt.subplots()
    #             fig.set_size_inches(8, 4)

    #             # Plot the reference and actual trajectories
    #             ax.plot([], [], linestyle=":", color="k", label="Reference")
                
    #             ax.plot(
    #                 t_range[1:],
    #                 u_ref[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
    #                 linestyle=":",
    #             )

    #             ax.set_prop_cycle(None)  # Re-use colors for the reference
    #             ax.plot([], [], linestyle="-", color="k", label="Expert")
    #             ax.plot(
    #                 t_range[1:],
    #                 u_expert[traj_idx, 1:, control_idx].T.cpu().detach().numpy(),
    #                 linestyle="-",
    #             )

    #             ax.set_xlabel("time (s)")
    #             ax.set_ylabel(f"Control {control_idx} for Expert and Reference Trajectory {traj_idx}")
    #             ax.legend()


    #             # Save the figure
    #             self.writer.add_figure(
    #                 f"Simulated Control Trajectory/Control {control_idx} for Expert and Reference Trajectory {traj_idx}",
    #                 fig,
    #             )



    def plot_leader_follower_policy_nn_weights(self, policy_nn, follower_policy_nn, epoch):

        # Store output weights
        policy_nn_params = list(policy_nn.parameters())
        policy_nn_params = policy_nn_params[-2].detach().cpu().numpy().flatten()

        follower_policy_nn_params = list(follower_policy_nn.parameters())
        follower_policy_nn_params = follower_policy_nn_params[-2].detach().cpu().numpy().flatten()

        # Assuming output_weights_history is a list of flattened weight arrays
        num_weights_policy_network = len(policy_nn_params)

        # episodes = range(1, len(output_weights_history) + 1)

        for i in range(num_weights_policy_network):

            # title = f"Policy Output Weights/Weight {i + 1}"

            self.writer.add_scalar(f"Leader Policy NN Output Weights/Weight {i + 1}", policy_nn_params[i], epoch)

            self.writer.add_scalar(f"Follower Policy NN Output Weights/Weight {i + 1}", follower_policy_nn_params[i], epoch)


    def plot_leader_follower_contraction_nn_weights(self, A, follower_A, epoch):

        # Store output weights
        contraction_nn_params = list(A.parameters())
        contraction_nn_params = contraction_nn_params[-2].detach().cpu().numpy().flatten()



        follower_contraction_nn_params = list(follower_A.parameters())
        follower_contraction_nn_params = follower_contraction_nn_params[-2].detach().cpu().numpy().flatten()

        # output_weights_history.append(params[-2].detach().cpu().numpy().flatten())

        # Assuming output_weights_history is a list of flattened weight arrays
        num_weights_contraction_network = len(contraction_nn_params)

        # episodes = range(1, len(output_weights_history) + 1)


        for i in range(num_weights_contraction_network):

            # title = f"Policy Output Weights/Weight {i + 1}"

            self.writer.add_scalar(f"Leader Contraction NN Output Weights/Weight {i + 1}", contraction_nn_params[i], epoch)

            self.writer.add_scalar(f"Follower Contraction NN Output Weights/Weight {i + 1}", follower_contraction_nn_params[i], epoch)



    def plot_leader_MA_follower_policy_nn_weights(self, policy_nn, follower_policy_nn, epoch):

        # Store output weights
        policy_nn_params = list(policy_nn.parameters())
        policy_nn_params = policy_nn_params[-2].detach().cpu().numpy().flatten()

        follower_policy_nn_params = list(follower_policy_nn.parameters())
        follower_policy_nn_params = follower_policy_nn_params[-2].detach().cpu().numpy().flatten()

        # Assuming output_weights_history is a list of flattened weight arrays
        num_weights_policy_network = len(policy_nn_params)

        # episodes = range(1, len(output_weights_history) + 1)

        for i in range(num_weights_policy_network):

            # title = f"Policy Output Weights/Weight {i + 1}"

            self.writer.add_scalar(f"Leader Policy NN Output Weights/Weight {i + 1}", policy_nn_params[i], epoch)

            self.writer.add_scalar(f"Follower MA Policy NN Output Weights/Weight {i + 1}", follower_policy_nn_params[i], epoch)



    def plot_leader_MA_follower_contraction_nn_weights(self, A, follower_A, epoch):

        # Store output weights
        contraction_nn_params = list(A.parameters())
        contraction_nn_params = contraction_nn_params[-2].detach().cpu().numpy().flatten()



        follower_contraction_nn_params = list(follower_A.parameters())
        follower_contraction_nn_params = follower_contraction_nn_params[-2].detach().cpu().numpy().flatten()

        # output_weights_history.append(params[-2].detach().cpu().numpy().flatten())

        # Assuming output_weights_history is a list of flattened weight arrays
        num_weights_contraction_network = len(contraction_nn_params)

        # episodes = range(1, len(output_weights_history) + 1)


        for i in range(num_weights_contraction_network):

            # title = f"Policy Output Weights/Weight {i + 1}"

            self.writer.add_scalar(f"Leader Contraction NN Output Weights/Weight {i + 1}", contraction_nn_params[i], epoch)

            self.writer.add_scalar(f"Follower MA Contraction NN Output Weights/Weight {i + 1}", follower_contraction_nn_params[i], epoch)



    def plot_state_error(self, t_range, x_ref_sim, x_sim):

        # Make a plot for state error
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

        # Plot the reference and actual trajectories
        ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
        ax.plot(
            t_range,
            (x_ref_sim - x_sim).norm(dim=-1).cpu().detach().numpy().squeeze(),
            linestyle=":",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("State Error")

        # Save the figure
        self.writer.add_figure(
            "Simulated State Trajectory/Error",
            fig,
            self.global_steps,
        )


    def plot_state_error_follower(self, t_range, x_ref_sim, follower_x_sim):

        # Make a plot for state error
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

        # Plot the reference and actual trajectories
        ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
        ax.plot(
            t_range,
            (x_ref_sim - follower_x_sim).norm(dim=-1).cpu().detach().numpy().squeeze(),
            linestyle=":",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Follower State Error")

        # Save the figure
        self.writer.add_figure(
            "Simulated Follower State Trajectory/Error",
            fig,
            self.global_steps,
        )

    def plot_state_error_follower_1(self, t_range, x_ref_sim, follower_x_sim):

        # Make a plot for state error
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

        # Plot the reference and actual trajectories
        ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
        ax.plot(
            t_range,
            (x_ref_sim - follower_x_sim).norm(dim=-1).cpu().detach().numpy().squeeze(),
            linestyle=":",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Follower State Error")

        # Save the figure
        self.writer.add_figure(
            "Simulated Follower 1 State Trajectory/Error",
            fig,
            self.global_steps,
        )


    def plot_state_error_follower_MA(self, t_range, x_ref_sim, x_sim_multi):

        # Make a plot for state error
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

        for i in range(len(x_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())

            # Plot the reference and actual trajectories
            ax.plot(t_range, 0 * t_range, linestyle=":", color=color, label=f"Agent {i}")
            ax.plot(
                t_range,
                (x_ref_sim - x_sim_multi[i]).norm(dim=-1).cpu().detach().numpy().squeeze(),
                linestyle=":",
            )


        ax.set_xlabel("time (s)")
        ax.set_ylabel("State Error")

        # Save the figure
        self.writer.add_figure(
            "Multi Simulated State Trajectory/Error",
            fig,
            self.global_steps,
        )
        
        
        
        
    def plot_state_error_follower_MA_leader(self, t_range, x_ref_sim, x_sim_multi):

        # Make a plot for state error
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

        for i in range(len(x_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())
            
            if i == 0:
                
                # Plot the reference and actual trajectories
                ax.plot(t_range, 0 * t_range, linestyle=":", color=color, label=f"Leader")
                ax.plot(
                    t_range,
                    (x_ref_sim - x_sim_multi[i]).norm(dim=-1).cpu().detach().numpy().squeeze(),
                    linestyle=":",
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.plot(t_range, 0 * t_range, linestyle=":", color=color, label=f"Agent {i}")
                ax.plot(
                    t_range,
                    (x_ref_sim - x_sim_multi[i]).norm(dim=-1).cpu().detach().numpy().squeeze(),
                    linestyle=":",
                )

            


        ax.set_xlabel("time (s)")
        ax.set_ylabel("State Error")

        # Save the figure
        self.writer.add_figure(
            "Multi Simulated State Trajectory/Error",
            fig,
            self.global_steps,
        )
        
        
        
        
    def plot_state_error_follower_MA_leader_RGB(self, t_range, x_ref_sim, x_sim_multi):

        # Make a plot for state error
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        
        # Define a list of colors
        colors = ['red', 'green', 'blue']

        for i in range(len(x_sim_multi)):
            
            # Use predefined colors for the first three lines, then random colors
            if i < len(colors):
                color = colors[i]
            else:
                color = (random.random(), random.random(), random.random())
            
            if i == 0:
                
                # Plot the reference and actual trajectories
                ax.plot(t_range, 0 * t_range, linestyle=":", color=color, label=f"Leader")
                ax.plot(
                    t_range,
                    (x_ref_sim - x_sim_multi[i]).norm(dim=-1).cpu().detach().numpy().squeeze(),
                    linestyle=":",
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.plot(t_range, 0 * t_range, linestyle=":", color=color, label=f"Agent {i}")
                ax.plot(
                    t_range,
                    (x_ref_sim - x_sim_multi[i]).norm(dim=-1).cpu().detach().numpy().squeeze(),
                    linestyle=":",
                )

            


        ax.set_xlabel("time (s)")
        ax.set_ylabel("State Error")

        # Save the figure
        self.writer.add_figure(
            "Multi Simulated State Trajectory/Error",
            fig,
            self.global_steps,
        )




    def plot_controls(self, t_range, u_ref_sim, u_sim):

        # Make a plot for each control
        for control_idx in range(self.n_control_dims):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 4)

            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot(
                t_range[1:],
                u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle=":",
            )
            ax.set_prop_cycle(None)  # Re-use colors for the reference
            ax.plot([], [], linestyle="-", color="k", label="Actual")
            ax.plot(
                t_range[1:],
                u_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle="-",
            )
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"Control {control_idx}")
            ax.legend()

            # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

            # Save the figure
            self.writer.add_figure(
                f"Simulated Control Trajectory/Control {control_idx}",
                fig,
                self.global_steps,
            )


    def plot_controls_follower(self, t_range, u_ref_sim, follower_u_sim):

        # Make a plot for each control
        for control_idx in range(self.n_control_dims):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 4)

            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot(
                t_range[1:],
                u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle=":",
            )
            ax.set_prop_cycle(None)  # Re-use colors for the reference
            ax.plot([], [], linestyle="-", color="k", label="Follower Actual")
            ax.plot(
                t_range[1:],
                follower_u_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle="-",
            )
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"Follower Control {control_idx}")
            ax.legend()

            # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

            # Save the figure
            self.writer.add_figure(
                f"Simulated Follower Control Trajectory/Control {control_idx}",
                fig,
                self.global_steps,
            )


    def plot_controls_follower_1(self, t_range, u_ref_sim, follower_u_sim):

        # Make a plot for each control
        for control_idx in range(self.n_control_dims):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 4)

            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot(
                t_range[1:],
                u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle=":",
            )
            ax.set_prop_cycle(None)  # Re-use colors for the reference
            ax.plot([], [], linestyle="-", color="k", label="Follower Actual")
            ax.plot(
                t_range[1:],
                follower_u_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle="-",
            )
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"Follower Control {control_idx}")
            ax.legend()

            # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

            # Save the figure
            self.writer.add_figure(
                f"Simulated Follower 1 Control Trajectory/Control {control_idx}",
                fig,
                self.global_steps,
            )


    def plot_controls_follower_MA(self, t_range, u_ref_sim, u_sim_multi):

        # Make a plot for each control
        for control_idx in range(self.n_control_dims):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 4)

            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot(
                t_range[1:],
                u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle=":",
            )


            for i in range(len(u_sim_multi)):

                # Generate a random color
                color = (random.random(), random.random(), random.random())

                # Plot the reference and actual trajectories
                ax.set_prop_cycle(None)
                ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
                ax.plot(
                    t_range[1:],
                    u_sim_multi[i, 1:, control_idx].T.cpu().detach().numpy(),
                    linestyle="-",
                )


            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"Control {control_idx}")
            ax.legend()

            # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

            # Save the figure
            self.writer.add_figure(
                f"Multi Simulated Control Trajectory/Control {control_idx}",
                fig,
                self.global_steps,
            )
            
            
            
    def plot_controls_follower_MA_leader(self, t_range, u_ref_sim, u_sim_multi):

        # Make a plot for each control
        for control_idx in range(self.n_control_dims):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 4)

            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot(
                t_range[1:],
                u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle=":",
            )


            for i in range(len(u_sim_multi)):

                # Generate a random color
                color = (random.random(), random.random(), random.random())
                
                if i == 0:
                    # Plot the reference and actual trajectories
                    ax.set_prop_cycle(None)
                    ax.plot([], [], linestyle="-", color=color, label=f"Leader Actual")
                    ax.plot(
                        t_range[1:],
                        u_sim_multi[i, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle="-",
                    )
                    
                else:
                    # Plot the reference and actual trajectories
                    ax.set_prop_cycle(None)
                    ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
                    ax.plot(
                        t_range[1:],
                        u_sim_multi[i, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle="-",
                    )

                


            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"Control {control_idx}")
            ax.legend()

            # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

            # Save the figure
            self.writer.add_figure(
                f"Multi Simulated Control Trajectory/Control {control_idx}",
                fig,
                self.global_steps,
            )
            
            
            
            
    def plot_controls_follower_MA_leader_all(self, t_range, u_ref_sim, u_sim_multi):

        # Make a plot for each control
        for control_idx in range(self.n_control_dims):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 4)

            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot(
                t_range[1:],
                u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle=":",
            )


            for i in range(len(u_sim_multi)):

                # Generate a random color
                color = (random.random(), random.random(), random.random())
                
                if i == 0:
                    # Plot the reference and actual trajectories
                    ax.set_prop_cycle(None)
                    ax.plot([], [], linestyle="-", color=color, label=f"Leader Actual")
                    ax.plot(
                        t_range[1:],
                        u_sim_multi[i][:, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle="-",
                    )
                    
                else:
                    # Plot the reference and actual trajectories
                    ax.set_prop_cycle(None)
                    ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
                    ax.plot(
                        t_range[1:],
                        u_sim_multi[i][:, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle="-",
                    )

                


            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"Control {control_idx}")
            ax.legend()

            # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

            # Save the figure
            self.writer.add_figure(
                f"Multi Simulated Control Trajectory/Control {control_idx}",
                fig,
                self.global_steps,
            )
            
            
            
            
            
    def plot_controls_follower_MA_leader_all_RGB(self, t_range, u_ref_sim, u_sim_multi):
        
        # Define a list of colors
        colors = ['red', 'green', 'blue']

        # Make a plot for each control
        for control_idx in range(self.n_control_dims):
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 4)

            # Plot the reference and actual trajectories
            ax.plot([], [], linestyle=":", color="k", label="Reference")
            ax.plot(
                t_range[1:],
                u_ref_sim[:, 1:, control_idx].T.cpu().detach().numpy(),
                linestyle=":",
            )


            for i in range(len(u_sim_multi)):
                
                # Use predefined colors for the first three lines, then random colors
                if i < len(colors):
                    color = colors[i]
                else:
                    color = (random.random(), random.random(), random.random())
                
                if i == 0:
                    # Plot the reference and actual trajectories
                    ax.set_prop_cycle(None)
                    ax.plot([], [], linestyle="-", color=color, label=f"Leader Actual")
                    ax.plot(
                        t_range[1:],
                        u_sim_multi[i][:, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle="-",
                    )
                    
                else:
                    # Plot the reference and actual trajectories
                    ax.set_prop_cycle(None)
                    ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
                    ax.plot(
                        t_range[1:],
                        u_sim_multi[i][:, 1:, control_idx].T.cpu().detach().numpy(),
                        linestyle="-",
                    )

                


            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"Control {control_idx}")
            ax.legend()

            # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

            # Save the figure
            self.writer.add_figure(
                f"Multi Simulated Control Trajectory/Control {control_idx}",
                fig,
                self.global_steps,
            )




    def plot_phase_plane(self, x_ref_sim, x_sim):

        # Also make a phase plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )
        ax.set_prop_cycle(None)  # Re-use colors for the reference
        ax.plot([], [], linestyle="-", color="k", label="Actual")
        ax.plot(
            x_sim[:, :, 0].T.cpu().detach().numpy(),
            x_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle="-",
        )
        ax.plot(
            x_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Phase Plane",
            fig,
            self.global_steps,
        )



    def plot_phase_plane_follower(self, x_ref_sim, follower_x_sim):

        # Also make a phase plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )
        ax.set_prop_cycle(None)  # Re-use colors for the reference
        ax.plot([], [], linestyle="-", color="k", label="Follower Actual")
        ax.plot(
            follower_x_sim[:, :, 0].T.cpu().detach().numpy(),
            follower_x_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle="-",
        )
        ax.plot(
            follower_x_sim[:, 0, 0].T.cpu().detach().numpy(),
            follower_x_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Follower Phase Plane",
            fig,
            self.global_steps,
        )


    def plot_phase_plane_follower_1(self, x_ref_sim, follower_x_sim):

        # Also make a phase plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )
        ax.set_prop_cycle(None)  # Re-use colors for the reference
        ax.plot([], [], linestyle="-", color="k", label="Follower Actual")
        ax.plot(
            follower_x_sim[:, :, 0].T.cpu().detach().numpy(),
            follower_x_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle="-",
        )
        ax.plot(
            follower_x_sim[:, 0, 0].T.cpu().detach().numpy(),
            follower_x_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Follower 1 Phase Plane",
            fig,
            self.global_steps,
        )


    def plot_phase_plane_follower_MA(self, x_ref_sim, x_sim_multi):

        # Also make a phase plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )

        for i in range(len(x_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())

            # Plot the reference and actual trajectories
            ax.set_prop_cycle(None)
            ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
            ax.plot(
                x_sim_multi[i, :, 0].T.cpu().detach().numpy(),
                x_sim_multi[i, :, 1].T.cpu().detach().numpy(),
                linestyle="-",
                color=color,
            )
            ax.plot(
                x_sim_multi[i, 0, 0].T.cpu().detach().numpy(),
                x_sim_multi[i, 0, 1].T.cpu().detach().numpy(),
                marker="o",
                color=color,
            )


        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Agents Phase Plane",
            fig,
            self.global_steps,
        )
        
        
        
        
    def plot_phase_plane_follower_MA_leader(self, x_ref_sim, x_sim_multi):

        # Also make a phase plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )

        for i in range(len(x_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())
            
            if i == 0:
                # Plot the reference and actual trajectories
                ax.set_prop_cycle(None)
                ax.plot([], [], linestyle="-", color=color, label=f"Leader Actual")
                ax.plot(
                    x_sim_multi[i, :, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i, :, 1].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                )
                ax.plot(
                    x_sim_multi[i, 0, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color=color,
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.set_prop_cycle(None)
                ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
                ax.plot(
                    x_sim_multi[i, :, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i, :, 1].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                )
                ax.plot(
                    x_sim_multi[i, 0, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color=color,
                )

            


        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Agents Phase Plane",
            fig,
            self.global_steps,
        )
        
        
        
        
    def plot_phase_plane_follower_MA_leader_all(self, x_ref_sim, x_sim_multi):

        # Also make a phase plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )

        for i in range(len(x_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())
            
            if i == 0:
                # Plot the reference and actual trajectories
                ax.set_prop_cycle(None)
                ax.plot([], [], linestyle="-", color=color, label=f"Leader Actual")
                ax.plot(
                    x_sim_multi[i][:, :, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, :, 1].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                )
                ax.plot(
                    x_sim_multi[i][:, 0, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color=color,
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.set_prop_cycle(None)
                ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
                ax.plot(
                    x_sim_multi[i][:, :, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, :, 1].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                )
                ax.plot(
                    x_sim_multi[i][:, 0, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color=color,
                )

            


        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Agents Phase Plane",
            fig,
            self.global_steps,
        )
        
        
        
        
    def plot_phase_plane_follower_MA_leader_all_RGB(self, x_ref_sim, x_sim_multi):

        # Also make a phase plane plot
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)


        # Plot the reference and actual trajectories
        ax.plot([], [], linestyle=":", color="k", label="Reference")
        ax.plot([], [], marker="o", color="k", label="Start")
        ax.plot(
            x_ref_sim[:, :, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, :, 1].T.cpu().detach().numpy(),
            linestyle=":",
        )
        ax.plot(
            x_ref_sim[:, 0, 0].T.cpu().detach().numpy(),
            x_ref_sim[:, 0, 1].T.cpu().detach().numpy(),
            marker="o",
            color="k",
        )
        
        # Define a list of colors
        colors = ['red', 'green', 'blue']

        for i in range(len(x_sim_multi)):
            
            # Use predefined colors for the first three lines, then random colors
            if i < len(colors):
                color = colors[i]
            else:
                color = (random.random(), random.random(), random.random())
            
            if i == 0:
                # Plot the reference and actual trajectories
                ax.set_prop_cycle(None)
                ax.plot([], [], linestyle="-", color=color, label=f"Leader Actual")
                ax.plot(
                    x_sim_multi[i][:, :, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, :, 1].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                )
                ax.plot(
                    x_sim_multi[i][:, 0, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color=color,
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.set_prop_cycle(None)
                ax.plot([], [], linestyle="-", color=color, label=f"Agent {i} Actual")
                ax.plot(
                    x_sim_multi[i][:, :, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, :, 1].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                )
                ax.plot(
                    x_sim_multi[i][:, 0, 0].T.cpu().detach().numpy(),
                    x_sim_multi[i][:, 0, 1].T.cpu().detach().numpy(),
                    marker="o",
                    color=color,
                )

            


        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Agents Phase Plane",
            fig,
            self.global_steps,
        )




    def plot_metric(self, t_range, M_sim):

        # Also plot the metric
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.plot([], [], linestyle="-", color="k", label="Metric")
        ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
        ax.plot(
            t_range[:-1],
            M_sim[:, :-1, 0].T.cpu().detach().numpy(),
            linestyle="-",
        )
        # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
        # ax.plot(
        #     t_range[:-1],
        #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
        #     linestyle=":",
        # )
        # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
        # ax.plot(
        #     t_range[:-2],
        #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
        #     linestyle="--",
        # )
        ax.set_xlabel("time (s)")
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Simulated Metric",
            fig,
            self.global_steps,
        )


    def plot_metric_follower(self, t_range, follower_M_sim):

        # Also plot the metric
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.plot([], [], linestyle="-", color="k", label="Follower Metric")
        ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
        ax.plot(
            t_range[:-1],
            follower_M_sim[:, :-1, 0].T.cpu().detach().numpy(),
            linestyle="-",
        )
        # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
        # ax.plot(
        #     t_range[:-1],
        #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
        #     linestyle=":",
        # )
        # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
        # ax.plot(
        #     t_range[:-2],
        #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
        #     linestyle="--",
        # )
        ax.set_xlabel("time (s)")
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Simulated Follower Metric",
            fig,
            self.global_steps,
        )


    def plot_metric_follower_1(self, t_range, follower_M_sim):

        # Also plot the metric
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.plot([], [], linestyle="-", color="k", label="Follower Metric")
        ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
        ax.plot(
            t_range[:-1],
            follower_M_sim[:, :-1, 0].T.cpu().detach().numpy(),
            linestyle="-",
        )
        # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
        # ax.plot(
        #     t_range[:-1],
        #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
        #     linestyle=":",
        # )
        # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
        # ax.plot(
        #     t_range[:-2],
        #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
        #     linestyle="--",
        # )
        ax.set_xlabel("time (s)")
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Simulated Follower 1 Metric",
            fig,
            self.global_steps,
        )


    def plot_metric_follower_MA(self, t_range, M_sim_multi):

        # Also plot the metric
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.plot([], [], linestyle="-", color="k", label="Metric")
        ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")

        for i in range(len(M_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())

            # Plot the reference and actual trajectories
            ax.plot(
                t_range[:-1],
                M_sim_multi[i, :-1, 0].T.cpu().detach().numpy(),
                linestyle="-",
                color=color,
                label=f"Agent {i}"
            )
            

        
        # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
        # ax.plot(
        #     t_range[:-1],
        #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
        #     linestyle=":",
        # )
        # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
        # ax.plot(
        #     t_range[:-2],
        #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
        #     linestyle="--",
        # )
        ax.set_xlabel("time (s)")
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Agents Simulated Metric",
            fig,
            self.global_steps,
        )
        
        
        
        
    def plot_metric_follower_MA_leader(self, t_range, M_sim_multi):

        # Also plot the metric
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.plot([], [], linestyle="-", color="k", label="Metric")
        ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")

        for i in range(len(M_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())
            
            if i == 0:
                # Plot the reference and actual trajectories
                ax.plot(
                    t_range[:-1],
                    M_sim_multi[i, :-1, 0].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                    label=f"Leader"
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.plot(
                    t_range[:-1],
                    M_sim_multi[i, :-1, 0].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                    label=f"Agent {i}"
                )
                
                
                
                
                
                
                
    def plot_metric_follower_MA_leader_all(self, t_range, M_sim_multi):

        # Also plot the metric
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.plot([], [], linestyle="-", color="k", label="Metric")
        ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")

        for i in range(len(M_sim_multi)):

            # Generate a random color
            color = (random.random(), random.random(), random.random())
            
            if i == 0:
                # Plot the reference and actual trajectories
                ax.plot(
                    t_range[:-1],
                    M_sim_multi[i][:, :-1, 0].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                    label=f"Leader"
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.plot(
                    t_range[:-1],
                    M_sim_multi[i][:, :-1, 0].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                    label=f"Agent {i}"
                )
                
                
                
                
                
        
                

            
            

        
        # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
        # ax.plot(
        #     t_range[:-1],
        #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
        #     linestyle=":",
        # )
        # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
        # ax.plot(
        #     t_range[:-2],
        #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
        #     linestyle="--",
        # )
        ax.set_xlabel("time (s)")
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Agents Simulated Metric",
            fig,
            self.global_steps,
        )
        
        
        
        
    def plot_metric_follower_MA_leader_all_RGB(self, t_range, M_sim_multi):

        # Also plot the metric
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)
        ax.plot([], [], linestyle="-", color="k", label="Metric")
        ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
        
        # Define a list of colors
        colors = ['red', 'green', 'blue']

        for i in range(len(M_sim_multi)):
            
            # Use predefined colors for the first three lines, then random colors
            if i < len(colors):
                color = colors[i]
            else:
                color = (random.random(), random.random(), random.random())
            
            if i == 0:
                # Plot the reference and actual trajectories
                ax.plot(
                    t_range[:-1],
                    M_sim_multi[i][:, :-1, 0].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                    label=f"Leader"
                )
                
            else:
                # Plot the reference and actual trajectories
                ax.plot(
                    t_range[:-1],
                    M_sim_multi[i][:, :-1, 0].T.cpu().detach().numpy(),
                    linestyle="-",
                    color=color,
                    label=f"Agent {i}"
                )
                
                
                
                
                
        
                

            
            

        
        # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
        # ax.plot(
        #     t_range[:-1],
        #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
        #     linestyle=":",
        # )
        # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
        # ax.plot(
        #     t_range[:-2],
        #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
        #     linestyle="--",
        # )
        ax.set_xlabel("time (s)")
        ax.legend()

        # Save the figure
        self.writer.add_figure(
            "Agents Simulated Metric",
            fig,
            self.global_steps,
        )




    def run_training(
        self,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # Generate a random reference trajectory
        N_batch = 1  # number of test trajectories
        T = 20.0  # length of trajectory
        # T = 50.0  # length of trajectory
        x_init, x_ref_sim, u_ref_sim = generate_random_reference(
            N_batch,
            T,
            self.sim_dt,
            self.n_state_dims,
            self.n_control_dims,
            self.state_space,
            self.control_bounds,
            self.error_bounds,
            self.dynamics,
        )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # if epoch % 12 == 1 and epoch >= 12:
            #     print('---------------------Weights transfered---------------------------')
            #     # # add weigts transfer here
            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)


            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)

            #     self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            #     self.copy_network_weights(self.A, self.list_of_followers[1].A)

            #     # print('---------------------------')

            #     # print(id(self.policy_nn))
            #     # print(id(self.follower_policy_nn))
            #     # print(id(self.A))
            #     # print(id(self.follower_A))

            #     # raise

            print('---------------------Weights transfered---------------------------')

            # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # self.copy_network_weights(self.A, self.follower.A)

            # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
        
        
        
        
    def run_training_case_1(
        self,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # Generate a random reference trajectory
        N_batch = 1  # number of test trajectories
        T = 20.0  # length of trajectory
        # T = 50.0  # length of trajectory
        x_init, x_ref_sim, u_ref_sim = generate_random_reference(
            N_batch,
            T,
            self.sim_dt,
            self.n_state_dims,
            self.n_control_dims,
            self.state_space,
            self.control_bounds,
            self.error_bounds,
            self.dynamics,
        )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            if epoch % 50 == 1 and epoch >= 50:
                print('---------------------Weights transfered---------------------------')
                # # add weigts transfer here
                # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
                # self.copy_network_weights(self.A, self.follower.A)


                # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
                # self.copy_network_weights(self.A, self.follower.A)

                # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
                # self.copy_network_weights(self.A, self.list_of_followers[1].A)
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[1].A)
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[2].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[2].A)

                # print('---------------------------')

                # print(id(self.policy_nn))
                # print(id(self.follower_policy_nn))
                # print(id(self.A))
                # print(id(self.follower_A))

                # raise

            # print('---------------------Weights transfered---------------------------')

            # # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # # self.copy_network_weights(self.A, self.follower.A)

            # # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            # # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            # # self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            # # self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            # # self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[2].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
        
        
        
        
    def run_training_case_1_diff_train_set_same_eval_traj(
        self,
        x_init: torch.Tensor,
        x_ref_sim: torch.Tensor,
        u_ref_sim: torch.Tensor,
        T: float,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # # Generate a random reference trajectory
        # N_batch = 1  # number of test trajectories
        # T = 20.0  # length of trajectory
        # # T = 50.0  # length of trajectory
        # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
        #     N_batch,
        #     T,
        #     self.sim_dt,
        #     self.n_state_dims,
        #     self.n_control_dims,
        #     self.state_space,
        #     self.control_bounds,
        #     self.error_bounds,
        #     self.dynamics,
        # )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            if epoch % 50 == 1 and epoch >= 50:
                print('---------------------Weights transfered---------------------------')
                # # add weigts transfer here
                # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
                # self.copy_network_weights(self.A, self.follower.A)


                # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
                # self.copy_network_weights(self.A, self.follower.A)

                # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
                # self.copy_network_weights(self.A, self.list_of_followers[1].A)
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[1].A)
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[2].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[2].A)

                # print('---------------------------')

                # print(id(self.policy_nn))
                # print(id(self.follower_policy_nn))
                # print(id(self.A))
                # print(id(self.follower_A))

                # raise

            # print('---------------------Weights transfered---------------------------')

            # # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # # self.copy_network_weights(self.A, self.follower.A)

            # # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            # # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            # # self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            # # self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            # # self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[2].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
        
        
        
        
        
    def run_training_case_2(
        self,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # Generate a random reference trajectory
        N_batch = 1  # number of test trajectories
        T = 20.0  # length of trajectory
        # T = 50.0  # length of trajectory
        x_init, x_ref_sim, u_ref_sim = generate_random_reference(
            N_batch,
            T,
            self.sim_dt,
            self.n_state_dims,
            self.n_control_dims,
            self.state_space,
            self.control_bounds,
            self.error_bounds,
            self.dynamics,
        )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # if epoch % 12 == 1 and epoch >= 12:
            #     print('---------------------Weights transfered---------------------------')
            #     # # add weigts transfer here
            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)


            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)

            #     self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            #     self.copy_network_weights(self.A, self.list_of_followers[1].A)

            #     # print('---------------------------')

            #     # print(id(self.policy_nn))
            #     # print(id(self.follower_policy_nn))
            #     # print(id(self.A))
            #     # print(id(self.follower_A))

            #     # raise

            print('---------------------Weights transfered---------------------------')

            # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # self.copy_network_weights(self.A, self.follower.A)

            # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            # self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            # self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)
            
            self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            self.copy_network_weights(self.policy_nn, self.list_of_followers[2].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
        
        
    def run_training_case_2_diff_train_set_same_eval_traj(
        self,
        x_init: torch.Tensor,
        x_ref_sim: torch.Tensor,
        u_ref_sim: torch.Tensor,
        T: float,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # # Generate a random reference trajectory
        # N_batch = 1  # number of test trajectories
        # T = 20.0  # length of trajectory
        # # T = 50.0  # length of trajectory
        # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
        #     N_batch,
        #     T,
        #     self.sim_dt,
        #     self.n_state_dims,
        #     self.n_control_dims,
        #     self.state_space,
        #     self.control_bounds,
        #     self.error_bounds,
        #     self.dynamics,
        # )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # if epoch % 12 == 1 and epoch >= 12:
            #     print('---------------------Weights transfered---------------------------')
            #     # # add weigts transfer here
            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)


            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)

            #     self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            #     self.copy_network_weights(self.A, self.list_of_followers[1].A)

            #     # print('---------------------------')

            #     # print(id(self.policy_nn))
            #     # print(id(self.follower_policy_nn))
            #     # print(id(self.A))
            #     # print(id(self.follower_A))

            #     # raise

            print('---------------------Weights transfered---------------------------')

            # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # self.copy_network_weights(self.A, self.follower.A)

            # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            # self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            # self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            # self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)
            
            self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            self.copy_network_weights(self.policy_nn, self.list_of_followers[2].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
        
        
    def run_training_case_3(
        self,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # Generate a random reference trajectory
        N_batch = 1  # number of test trajectories
        T = 20.0  # length of trajectory
        # T = 50.0  # length of trajectory
        x_init, x_ref_sim, u_ref_sim = generate_random_reference(
            N_batch,
            T,
            self.sim_dt,
            self.n_state_dims,
            self.n_control_dims,
            self.state_space,
            self.control_bounds,
            self.error_bounds,
            self.dynamics,
        )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # if epoch % 12 == 1 and epoch >= 12:
            #     print('---------------------Weights transfered---------------------------')
            #     # # add weigts transfer here
            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)


            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)

            #     self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            #     self.copy_network_weights(self.A, self.list_of_followers[1].A)

            #     # print('---------------------------')

            #     # print(id(self.policy_nn))
            #     # print(id(self.follower_policy_nn))
            #     # print(id(self.A))
            #     # print(id(self.follower_A))

            #     # raise

            print('---------------------Weights transfered---------------------------')

            # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # self.copy_network_weights(self.A, self.follower.A)

            # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
        
    def run_training_case_3_diff_train_set_same_eval_traj(
        self,
        x_init: torch.Tensor,
        x_ref_sim: torch.Tensor,
        u_ref_sim: torch.Tensor,
        T: float,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # # Generate a random reference trajectory
        # N_batch = 1  # number of test trajectories
        # T = 20.0  # length of trajectory
        # # T = 50.0  # length of trajectory
        # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
        #     N_batch,
        #     T,
        #     self.sim_dt,
        #     self.n_state_dims,
        #     self.n_control_dims,
        #     self.state_space,
        #     self.control_bounds,
        #     self.error_bounds,
        #     self.dynamics,
        # )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # if epoch % 12 == 1 and epoch >= 12:
            #     print('---------------------Weights transfered---------------------------')
            #     # # add weigts transfer here
            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)


            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)

            #     self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            #     self.copy_network_weights(self.A, self.list_of_followers[1].A)

            #     # print('---------------------------')

            #     # print(id(self.policy_nn))
            #     # print(id(self.follower_policy_nn))
            #     # print(id(self.A))
            #     # print(id(self.follower_A))

            #     # raise

            print('---------------------Weights transfered---------------------------')

            # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # self.copy_network_weights(self.A, self.follower.A)

            # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
        
    def run_training_case_4(
        self,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # Generate a random reference trajectory
        N_batch = 1  # number of test trajectories
        T = 20.0  # length of trajectory
        # T = 50.0  # length of trajectory
        x_init, x_ref_sim, u_ref_sim = generate_random_reference(
            N_batch,
            T,
            self.sim_dt,
            self.n_state_dims,
            self.n_control_dims,
            self.state_space,
            self.control_bounds,
            self.error_bounds,
            self.dynamics,
        )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # if epoch % 12 == 1 and epoch >= 12:
            #     print('---------------------Weights transfered---------------------------')
            #     # # add weigts transfer here
            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)


            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)

            #     self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            #     self.copy_network_weights(self.A, self.list_of_followers[1].A)

            #     # print('---------------------------')

            #     # print(id(self.policy_nn))
            #     # print(id(self.follower_policy_nn))
            #     # print(id(self.A))
            #     # print(id(self.follower_A))

            #     # raise

            print('---------------------Weights transfered---------------------------')

            # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # self.copy_network_weights(self.A, self.follower.A)

            # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        
    def run_training_case_4_diff_train_set_same_eval_traj(
        self,
        x_init: torch.Tensor,
        x_ref_sim: torch.Tensor,
        u_ref_sim: torch.Tensor,
        T: float,
        n_steps: int,
        debug: bool = False,
        sim_every_n_steps: int = 1,
    ):
        """
        Train for n_steps

        args:
            n_steps: the number of steps to train for
            debug: if True, log losses to tensorboard and display a progress bar
            sim_every_n_steps: run a simulation and save it to tensorboard every n steps

        returns:
            A pair of lists, each of which contains tuples defining the metric and
            policy network, respectively. Each tuple represents a layer, either:

                ("linear", numpy matrix of weights, numpy matrix of biases)
                ("relu", empty matrix, empty matrix)

            as well as two additional lists of training and test loss
        """
        # Log histories of training and test loss
        training_losses = []
        test_losses = []
        policy_network_output_weights_history = []
        Contraction_network_output_weights_history = []
        follower_policy_network_output_weights_history = []
        follower_Contraction_network_output_weights_history = []

        # Find out how many training and test examples we have
        N_train = self.x_training.shape[0]
        N_test = self.x_validation.shape[0]


        # # Generate a random reference trajectory
        # N_batch = 1  # number of test trajectories
        # T = 20.0  # length of trajectory
        # # T = 50.0  # length of trajectory
        # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
        #     N_batch,
        #     T,
        #     self.sim_dt,
        #     self.n_state_dims,
        #     self.n_control_dims,
        #     self.state_space,
        #     self.control_bounds,
        #     self.error_bounds,
        #     self.dynamics,
        # )


        epochs = range(n_steps)
        for epoch in epochs:
            # Randomize the presentation order in each epoch
            permutation = torch.randperm(N_train)

            loss_accumulated = 0.0
            pd_loss_accumulated = 0.0
            M_loss_accumulated = 0.0
            u_loss_accumulated = 0.0
            epoch_range = range(0, N_train, self.batch_size)
            if debug:
                epoch_range = tqdm(epoch_range)
                epoch_range.set_description(f"Epoch {epoch} Training")  # type: ignore

            for i in epoch_range:
                # Get samples from the state space
                batch_indices = permutation[i : i + self.batch_size]
                # These samples will be [traj_length + expert_horizon_length, *]
                x = self.x_training[batch_indices]
                x_ref = self.x_ref_training[batch_indices]
                u_ref = self.u_ref_training[batch_indices]
                u_expert = self.u_expert_training[batch_indices]

                # copy the learned weights to the new nn periodically (per 50 or 100 epochs add the learned weights to the new nn with parameters such as alpha = 0.1)
                # hiearachical structure:
                    #one target, one leader, multiple followers
                    # mpc target for each follower parallel

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Compute loss and backpropagate
                losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                loss = torch.tensor(0.0)
                for loss_element in losses.values():
                    loss += loss_element
                loss.backward()

                loss_accumulated += loss.detach().item()

                if "conditioning" in losses:
                    pd_loss_accumulated += losses["conditioning"].detach().item()
                if "M" in losses:
                    M_loss_accumulated += losses["M"].detach().item()
                if "u" in losses:
                    u_loss_accumulated += losses["u"].detach().item()

                # Clip gradients
                max_norm = 1e2
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm, error_if_nonfinite=False
                )

                # Update the parameters
                self.optimizer.step()

                # Log the gradients
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm", total_norm, self.global_steps)


                total_norm = 0.0
                parameters = [
                    p
                    for p in self.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm", total_norm, self.global_steps)







                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.A.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("M grad norm follower", total_norm, self.global_steps)

                
                total_norm = 0.0
                parameters = [
                    p
                    for p in self.follower.policy_nn.parameters()
                    if p.grad is not None and p.requires_grad
                ]
                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                self.writer.add_scalar("Pi grad norm follower", total_norm, self.global_steps)

                # Track the overall number of gradient descent steps
                self.global_steps += 1

                # Clean up
                x = x.detach()



            self.plot_leader_follower_policy_nn_weights(self.policy_nn, self.follower.policy_nn, epoch)
            self.plot_leader_follower_contraction_nn_weights(self.A, self.follower.A, epoch)



            # policy_network_output_weights_history = []
            # Contraction_network_output_weights_history = []
            # follower_policy_network_output_weights_history = []
            # follower_Contraction_network_output_weights_history = []

            # save progress
            training_losses.append(loss_accumulated / (N_train / self.batch_size))
            # Log the running loss
            self.writer.add_scalar("Loss/train", training_losses[-1], self.global_steps)
            self.writer.add_scalar(
                "PD Loss/train",
                pd_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "M Loss/train",
                M_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            self.writer.add_scalar(
                "u Loss/train",
                u_loss_accumulated / (N_train / self.batch_size),
                self.global_steps,
            )
            # Log the number of training points
            self.writer.add_scalar(
                "# Trajectories", self.x_training.shape[0], self.global_steps
            )

            # if epoch % 12 == 1 and epoch >= 12:
            #     print('---------------------Weights transfered---------------------------')
            #     # # add weigts transfer here
            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)


            #     # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            #     # self.copy_network_weights(self.A, self.follower.A)

            #     self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            #     self.copy_network_weights(self.A, self.list_of_followers[1].A)

            #     # print('---------------------------')

            #     # print(id(self.policy_nn))
            #     # print(id(self.follower_policy_nn))
            #     # print(id(self.A))
            #     # print(id(self.follower_A))

            #     # raise

            print('---------------------Weights transfered---------------------------')

            # self.copy_network_weights(self.policy_nn, self.follower.policy_nn)
            # self.copy_network_weights(self.A, self.follower.A)

            # # self.copy_network_weights(self.policy_nn, self.follower_1.policy_nn)
            # # self.copy_network_weights(self.A, self.follower_1.A)
            
            # self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
            # self.copy_network_weights(self.A, self.list_of_followers[0].A)

            self.copy_network_weights(self.policy_nn, self.list_of_followers[1].policy_nn)
            self.copy_network_weights(self.A, self.list_of_followers[1].A)
            
            self.copy_network_weights(self.list_of_followers[1].policy_nn, self.list_of_followers[2].policy_nn)
            self.copy_network_weights(self.list_of_followers[1].A, self.list_of_followers[2].A)


            # Also log a simulation plot every so often
            if sim_every_n_steps == 1 or epoch % sim_every_n_steps == 1:
                # # Generate a random reference trajectory
                # N_batch = 1  # number of test trajectories
                # # T = 20.0  # length of trajectory
                # T = 50.0  # length of trajectory
                # x_init, x_ref_sim, u_ref_sim = generate_random_reference(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )

                # # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                # #     N_batch,
                # #     T,
                # #     self.sim_dt,
                # #     self.n_state_dims,
                # #     self.n_control_dims,
                # #     self.state_space,
                # #     self.control_bounds,
                # #     self.error_bounds,
                # #     self.dynamics,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # #     torch.tensor([0.0, 0.0]),
                # #     # torch.tensor([10.0, 10.0]),
                # #     # self.ob,
                # #     torch.tensor([0.0, 0.0, math.pi / 8.0]),
                # # )


                # x_init_dwa, x_ref_sim_dwa, u_ref_sim_dwa = generate_random_reference_dwa(
                #     N_batch,
                #     T,
                #     self.sim_dt,
                #     self.n_state_dims,
                #     self.n_control_dims,
                #     self.state_space,
                #     self.control_bounds,
                #     self.error_bounds,
                #     self.dynamics,
                # )



                # print(x_init)
                # print(x_ref_sim)
                # print(u_ref_sim)

                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.u,
                #     self.state_space,
                #     self.metric_value,
                #     self.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate_follower(
                #     x_inits,
                #     x_sim,
                #     u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower.u,
                #     self.state_space,
                #     self.follower.metric_value,
                #     self.follower.metric_derivative_t,
                #     self.control_bounds,
                # )

                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.follower_1.u,
                #     self.state_space,
                #     self.follower_1.metric_value,
                #     self.follower_1.metric_derivative_t,
                #     self.control_bounds,
                # )

                # # # Simulate
                # # x_sim_dwa, u_sim_dwa, M_sim_dwa, dMdt_sim_dwa = simulate(
                # #     x_init_dwa,
                # #     x_ref_sim_dwa,
                # #     u_ref_sim_dwa,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.u,
                # #     self.metric_value,
                # #     self.metric_derivative_t,
                # #     self.control_bounds,
                # # )
                
                
                
                
                self.copy_network_weights(self.policy_nn, self.list_of_followers[0].policy_nn)
                self.copy_network_weights(self.A, self.list_of_followers[0].A)
                
                
                # self.copy_network_weights(self.policy_nn, self.leader.policy_nn)
                # self.copy_network_weights(self.A, self.leader.A)
                
                
                
                
                # # Simulate
                # x_sim, u_sim, M_sim, dMdt_sim, x_inits = simulate(
                #     x_init,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[0].u,
                #     self.state_space,
                #     self.list_of_followers[0].metric_value,
                #     self.list_of_followers[0].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                # follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[1].u,
                #     self.state_space,
                #     self.list_of_followers[1].metric_value,
                #     self.list_of_followers[1].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     x_ref_sim,
                #     u_ref_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )
                
                
                

                # # x_sim_multi, u_sim_multi, M_sim_multi, dMdt_sim_multi = simulate_multi_agent(
                # #     x_init,
                # #     x_ref_sim,
                # #     u_ref_sim,
                # #     self.sim_dt,
                # #     self.controller_dt,
                # #     self.dynamics,
                # #     self.list_of_followers,
                # #     self.state_space,
                # #     self.control_bounds,
                # # )
                
                
                # Simulate
                x_sim, u_sim, M_sim, dMdt_sim, x_init = simulate(
                    x_init,
                    x_ref_sim,
                    u_ref_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[0].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[0].metric_value,
                    self.list_of_followers[0].metric_derivative_t,
                    self.control_bounds,
                )
                
                follower_x_sim, follower_u_sim, follower_M_sim, follower_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[1].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[1].metric_value,
                    self.list_of_followers[1].metric_derivative_t,
                    self.control_bounds,
                )
                
                
                follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_init = simulate(
                    x_init,
                    x_sim,
                    u_sim,
                    self.sim_dt,
                    self.controller_dt,
                    self.dynamics,
                    self.list_of_followers[2].u,
                    self.state_space,
                    self.error_bounds,
                    self.list_of_followers[2].metric_value,
                    self.list_of_followers[2].metric_derivative_t,
                    self.control_bounds,
                )
                
                # follower_1_x_sim, follower_1_u_sim, follower_1_M_sim, follower_1_dMdt_sim, x_inits = simulate(
                #     x_inits,
                #     follower_x_sim,
                #     follower_u_sim,
                #     self.sim_dt,
                #     self.controller_dt,
                #     self.dynamics,
                #     self.list_of_followers[2].u,
                #     self.state_space,
                #     self.list_of_followers[2].metric_value,
                #     self.list_of_followers[2].metric_derivative_t,
                #     self.control_bounds,
                # )


                x_sim = x_sim.detach()
                u_sim = u_sim.detach()
                M_sim = M_sim.detach()
                dMdt_sim = dMdt_sim.detach()

                # x_sim_multi = x_sim_multi.detach()
                # u_sim_multi = u_sim_multi.detach()
                # M_sim_multi = M_sim_multi.detach()
                # dMdt_sim_multi = dMdt_sim_multi.detach()

                follower_x_sim = follower_x_sim.detach()
                follower_u_sim = follower_u_sim.detach()
                follower_M_sim = follower_M_sim.detach()
                follower_dMdt_sim = follower_dMdt_sim.detach()

                follower_1_x_sim = follower_1_x_sim.detach()
                follower_1_u_sim = follower_1_u_sim.detach()
                follower_1_M_sim = follower_1_M_sim.detach()
                follower_1_dMdt_sim = follower_1_dMdt_sim.detach()

                # # x_sim_dwa = x_sim_dwa.detach()
                # # u_sim_dwa = u_sim_dwa.detach()
                # # M_sim_dwa = M_sim_dwa.detach()
                # # dMdt_sim_dwa = dMdt_sim_dwa.detach()

                t_range = np.arange(0, T, self.sim_dt)


                self.plot_state_error(t_range, x_ref_sim, x_sim)
                self.plot_state_error_follower(t_range, x_ref_sim, follower_x_sim)
                self.plot_state_error_follower_1(t_range, x_ref_sim, follower_1_x_sim)
                # # self.plot_state_error_follower_MA(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, x_sim_multi)
                # self.plot_state_error_follower_MA_leader(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_state_error_follower_MA_leader_RGB(t_range, x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])




                # # Plot the reference and actual trajectories
                # ax.plot(t_range, 0 * t_range, linestyle=":", color="k")
                # ax.plot(
                #     t_range,
                #     (x_ref_sim_dwa - x_sim_dwa).norm(dim=-1).cpu().detach().numpy().squeeze(),
                #     linestyle=":",
                # )
                # ax.set_xlabel("time (s)")
                # ax.set_ylabel("DWA State Error")

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Simulated State Trajectory/Error",
                #     fig,
                #     self.global_steps,
                # )




                self.plot_controls(t_range, u_ref_sim, u_sim)
                self.plot_controls_follower(t_range, u_ref_sim, follower_u_sim)
                self.plot_controls_follower_1(t_range, u_ref_sim, follower_1_u_sim)
                # # self.plot_controls_follower_MA(t_range, u_ref_sim, u_sim_multi)
                # # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, u_sim_multi)
                # self.plot_controls_follower_MA_leader(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                # self.plot_controls_follower_MA_leader_all(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])
                self.plot_controls_follower_MA_leader_all_RGB(t_range, u_ref_sim, [u_sim, follower_u_sim, follower_1_u_sim])





                # # Make a plot for each control
                # for control_idx in range(self.n_control_dims):
                #     fig, ax = plt.subplots()
                #     fig.set_size_inches(8, 4)

                #     # Plot the reference and actual trajectories
                #     ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                #     ax.plot(
                #         t_range[1:],
                #         u_ref_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle=":",
                #     )
                #     ax.set_prop_cycle(None)  # Re-use colors for the reference
                #     ax.plot([], [], linestyle="-", color="k", label="Actual")
                #     ax.plot(
                #         t_range[1:],
                #         u_sim_dwa[:, 1:, control_idx].T.cpu().detach().numpy(),
                #         linestyle="-",
                #     )
                #     ax.set_xlabel("time (s)")
                #     ax.set_ylabel(f"Control {control_idx}")
                #     ax.legend()

                #     # print(u_sim[:, 1:, control_idx].T.cpu().detach().numpy())

                #     # Save the figure
                #     self.writer.add_figure(
                #         f"DWA Simulated Control Trajectory/Control {control_idx}",
                #         fig,
                #         self.global_steps,
                #     )




                self.plot_phase_plane(x_ref_sim, x_sim)
                self.plot_phase_plane_follower(x_ref_sim, follower_x_sim)
                self.plot_phase_plane_follower_1(x_ref_sim, follower_1_x_sim)
                # # self.plot_phase_plane_follower_MA(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader(x_ref_sim, x_sim_multi)
                # self.plot_phase_plane_follower_MA_leader_all(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])
                self.plot_phase_plane_follower_MA_leader_all_RGB(x_ref_sim, [x_sim, follower_x_sim, follower_1_x_sim])


                self.plot_metric(t_range, M_sim)
                self.plot_metric_follower(t_range, follower_M_sim)
                self.plot_metric_follower_1(t_range, follower_1_M_sim)
                # # self.plot_metric_follower_MA(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader(t_range, M_sim_multi)
                # self.plot_metric_follower_MA_leader_all(t_range, [M_sim, follower_M_sim, follower_1_M_sim])
                self.plot_metric_follower_MA_leader_all_RGB(t_range, [M_sim, follower_M_sim, follower_1_M_sim])






                # Also make a phase plane plot
                #fig, ax = plt.subplots()
                #fig.set_size_inches(8, 8)

                # print(x_ref_sim)
                # print(x_ref_sim.shape[0])
                # print(x_ref_sim.shape[1])
                # print(x_sim)
                # print(x_sim.shape[0])
                # print(x_sim.shape[1])


                # # Plot the reference and actual trajectories
                # ax.plot([], [], linestyle=":", color="k", label="DWA Reference")
                # ax.plot([], [], marker="o", color="k", label="Start")
                # ax.plot(
                #     x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle=":",
                # )
                # ax.plot(
                #     x_ref_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_ref_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.set_prop_cycle(None)  # Re-use colors for the reference
                # ax.plot([], [], linestyle="-", color="k", label="Actual")
                # ax.plot(
                #     x_sim_dwa[:, :, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, :, 1].T.cpu().detach().numpy(),
                #     linestyle="-",
                # )
                # ax.plot(
                #     x_sim_dwa[:, 0, 0].T.cpu().detach().numpy(),
                #     x_sim_dwa[:, 0, 1].T.cpu().detach().numpy(),
                #     marker="o",
                #     color="k",
                # )
                # ax.legend()

                # # Save the figure
                # self.writer.add_figure(
                #     "DWA Phase Plane",
                #     fig,
                #     self.global_steps,
                # )





                

                # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach().numpy(), x_ref_sim[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach().numpy(), x_sim[:, :, 1].T.cpu().detach().numpy())
                # follower_path_length = self.calculate_path_length(follower_x_sim[:, :, 0].T.cpu().detach().numpy(), follower_x_sim[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "Follower Actual Path Length",
                #     follower_path_length,
                #     self.global_steps,
                # )





                # path_length_ref = self.calculate_path_length(x_ref_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_ref_sim_dwa[:, :, 1].T.cpu().detach().numpy())
                # path_length = self.calculate_path_length(x_sim_dwa[:, :, 0].T.cpu().detach().numpy(), x_sim_dwa[:, :, 1].T.cpu().detach().numpy())

                # # print(path_length_ref)
                # # print(path_length)

                # # path_length_ref = self.calculate_path_length(x_ref_sim[:, :, 0].T.cpu().detach(), x_ref_sim[:, :, 1].T.cpu().detach())
                # # path_length = self.calculate_path_length(x_sim[:, :, 0].T.cpu().detach(), x_sim[:, :, 1].T.cpu().detach())

                # self.writer.add_scalar(
                #     "DWA Ref Path Length",
                #     path_length_ref,
                #     self.global_steps,
                # )

                # self.writer.add_scalar(
                #     "DWA Actual Path Length",
                #     path_length,
                #     self.global_steps,
                # )




                

            # self.writer.close()



            # # Also plot the metric
            #     fig, ax = plt.subplots()
            #     fig.set_size_inches(8, 4)
            #     ax.plot([], [], linestyle="-", color="k", label="DWA Metric")
            #     ax.plot(t_range[:-1], 0 * t_range[:-1], linestyle=":", color="k")
            #     ax.plot(
            #         t_range[:-1],
            #         M_sim_dwa[:, :-1, 0].T.cpu().detach().numpy(),
            #         linestyle="-",
            #     )
            #     # ax.plot([], [], linestyle=":", color="k", label="dMetric/dt")
            #     # ax.plot(
            #     #     t_range[:-1],
            #     #     dMdt_sim[:, :-1, 0].T.cpu().detach().numpy(),
            #     #     linestyle=":",
            #     # )
            #     # ax.plot([], [], linestyle="--", color="k", label="dMetric/dt approx")
            #     # ax.plot(
            #     #     t_range[:-2],
            #     #     np.diff(M_sim[:, :-1, 0].T.cpu().detach().numpy(), axis=0) / dt,
            #     #     linestyle="--",
            #     # )
            #     ax.set_xlabel("time (s)")
            #     ax.legend()

            #     # Save the figure
            #     self.writer.add_figure(
            #         "DWA Simulated Metric",
            #         fig,
            #         self.global_steps,
            #     )

            # # self.writer.close()



            # Reset accumulated loss and get loss for the test set
            loss_accumulated = 0.0
            permutation = torch.randperm(N_test)
            with torch.no_grad():
                epoch_range = range(0, N_test, self.batch_size)
                if debug:
                    epoch_range = tqdm(epoch_range)
                    epoch_range.set_description(f"Epoch {epoch} Test")  # type: ignore
                for i in epoch_range:
                    # Get samples from the state space
                    indices = permutation[i : i + self.batch_size]
                    x = self.x_validation[indices]
                    x_ref = self.x_ref_validation[indices]
                    u_ref = self.u_ref_validation[indices]
                    u_expert = self.u_expert_validation[indices]

                    # Compute loss and backpropagate
                    losses = self.compute_losses(x, x_ref, u_ref, u_expert, epoch)
                    for loss_element in losses.values():
                        loss_accumulated += loss_element.detach().item()

            test_losses.append(loss_accumulated / (N_test / self.batch_size))
            # And log the running loss
            self.writer.add_scalar("Loss/test", test_losses[-1], self.global_steps)
            self.writer.close()

        # Create the list representations of the metric and policy network

        # The metric network starts with a normalization layer and ends with map from A
        # to M, which we have to manually add
        contraction_network_list = []
        contraction_network_list.append(
            (
                "linear",
                self.state_normalization_weights.detach().cpu().numpy(),
                self.state_normalization_bias.detach().cpu().numpy(),
            )
        )
        for layer_name, layer in self.metric_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                contraction_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                contraction_network_list.append(
                    ("relu", torch.Tensor(), torch.Tensor())
                )
        contraction_network_list.append(
            (
                "linear",
                self.A_to_M.detach().cpu().numpy(),
                torch.zeros(self.n_state_dims ** 2).detach().cpu().numpy(),
            )
        )

        policy_network_list = []
        # From a learning perspective, it helps to have x and (x-x_ref) as inputs to
        # the control policy; however, from a verification perspective it helps to
        # have x and x_ref as inputs. To get around this, we'll put a linear layer in
        # front of the policy network that takes [x; x_ref] to [x, x - x_ref]
        subtraction_weights = torch.block_diag(
            torch.eye(self.n_state_dims), -torch.eye(self.n_state_dims)
        )
        subtraction_weights[self.n_state_dims :, : self.n_state_dims] = torch.eye(
            self.n_state_dims
        )
        policy_network_list.append(
            (
                "linear",
                subtraction_weights.detach().cpu().numpy(),
                torch.zeros(2 * self.n_state_dims).detach().cpu().numpy(),
            )
        )
        # The policy network includes a normalization layer applied to both x
        # and x_error, so we need to construct a larger normalization matrix here
        policy_network_list.append(
            (
                "linear",
                torch.block_diag(
                    self.state_normalization_weights, self.error_normalization_weights
                )
                .detach()
                .cpu()
                .numpy(),
                torch.cat(
                    [self.state_normalization_bias, torch.zeros(self.n_state_dims)]
                )
                .detach()
                .cpu()
                .numpy(),
            )
        )
        for layer_name, layer in self.policy_layers.items():
            if "linear" in layer_name:
                layer = cast(nn.Linear, layer)
                policy_network_list.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy(),
                        layer.bias.detach().cpu().numpy(),
                    )
                )
            elif "activation" in layer_name:
                policy_network_list.append(("relu", np.array([]), np.array([])))
        # And include output normalization
        policy_network_list.append(
            (
                "linear",
                self.control_normalization_weights.detach().cpu().numpy(),
                torch.zeros(self.n_control_dims).detach().cpu().numpy(),
            )
        )

        # Return the network lists and the loss lists
        return (
            contraction_network_list,
            policy_network_list,
            training_losses,
            test_losses,
        )
        
        
        

    def add_data(
        self,
        counterexample_x: torch.Tensor,
        counterexample_x_ref: torch.Tensor,
        counterexample_u_ref: torch.Tensor,
        traj_length: int,
        n_trajs: int,
        validation_split: float,
    ):
        """
        Add new trajectories to the dataset, starting at the given example

        args:
            counterexample_x: a 1 x n_dims point in state space
            counterexample_x_ref: a 1 x n_dims point in state space
            counterexample_u_ref: a 1 x n_controls point in action space
            traj_length: length of the new trajectory
            n_trajs: the number of new trajectories to generate
            validation_split: the fraction of trajectories to add to the validation
                dataset
        """
        # We're going to add new data by creating one trajectory starting from the
        # given counterexample

        # Start by generating a random reference starting at the counterexample
        T = traj_length * self.controller_dt + self.expert_horizon
        print("Adding data!")
        # get these trajectories from a larger range of errors than we expect in testing
        error_bounds_demonstrations = [1.5 * bound for bound in self.error_bounds]
        x_init, x_ref, u_ref = generate_random_reference(
            n_trajs,
            T,
            self.controller_dt,
            self.n_state_dims,
            self.n_control_dims,
            self.state_space,
            self.control_bounds,
            error_bounds_demonstrations,
            self.dynamics,
            x_ref_init_0=counterexample_x_ref,
        )
        traj_length = x_ref.shape[1]

        # Create some places to store the simulation results
        x = torch.zeros((n_trajs, traj_length, self.n_state_dims))
        x[:, 0, :] = counterexample_x
        u_expert = torch.zeros((n_trajs, traj_length, self.n_control_dims))
        u_current = torch.zeros((self.n_control_dims,))

        # The expert policy requires a sliding window over the trajectory, so we need
        # to iterate through that trajectory.
        # Make sure we don't overrun the end of the reference while planning
        n_steps = traj_length - int(self.expert_horizon / self.controller_dt)
        dynamics_updates_per_control_update = int(self.controller_dt / self.sim_dt)
        for traj_idx in tqdm(range(n_trajs)):
            traj_range = range(n_steps - 1)
            for tstep in traj_range:
                # Get the current states and references
                x_current = x[traj_idx, tstep].reshape(-1, self.n_state_dims).clone()

                # Pick out sliding window into references for use with the expert
                x_ref_expert = (
                    x_ref[
                        traj_idx,
                        tstep : tstep + int(self.expert_horizon // self.controller_dt),
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )
                u_ref_expert = (
                    u_ref[
                        traj_idx,
                        tstep : tstep + int(self.expert_horizon // self.controller_dt),
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )

                # Run the expert
                u_current = torch.tensor(
                    self.expert_controller(
                        x_current.detach().cpu().numpy().squeeze(),
                        x_ref_expert,
                        u_ref_expert,
                    )
                )

                u_expert[traj_idx, tstep, :] = torch.clone(u_current)

                # Add a bit of noise to widen the distribution of states
                u_current += torch.normal(
                    0, self.demonstration_noise * torch.tensor(self.control_bounds)
                )

                # Update state
                for _ in range(dynamics_updates_per_control_update):
                    x_dot = self.dynamics(
                        x_current,
                        u_current.reshape(-1, self.n_control_dims),
                    )
                    x_current += self.sim_dt * x_dot
                x[traj_idx, tstep + 1, :] = x_current

            # plt.plot(x[traj_idx, :n_steps, 0], x[traj_idx, :n_steps, 1], "-")
            # plt.plot(x_ref[traj_idx, :n_steps, 0], x_ref[traj_idx, :n_steps, 1], ":")
            # plt.plot(x[traj_idx, 0, 0], x[traj_idx, 0, 1], "ko")
            # plt.plot(x_ref[traj_idx, 0, 0], x_ref[traj_idx, 0, 1], "ko")
            # plt.show()

            # plt.plot(u_expert[traj_idx, :, 0], "r:")
            # plt.plot(u_expert[traj_idx, :, 1], "r--")
            # plt.plot(u_ref[traj_idx, :, 0], "k:")
            # plt.plot(u_ref[traj_idx, :, 1], "k--")
            # plt.show()

        print(" Done!")

        # Reshape
        x = x[:, : tstep + 1, :].reshape(-1, self.n_state_dims)
        x_ref = x_ref[:, : tstep + 1, :].reshape(-1, self.n_state_dims)
        u_ref = u_ref[:, : tstep + 1, :].reshape(-1, self.n_control_dims)
        u_expert = u_expert[:, : tstep + 1, :].reshape(-1, self.n_control_dims)

        # Split into training and validation
        random_indices = torch.randperm(x.shape[0])
        val_pts = int(x.shape[0] * self.validation_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]

        self.x_ref_training = torch.cat([self.x_ref_training, x_ref[training_indices]])
        self.x_ref_validation = torch.cat(
            [self.x_ref_validation, x_ref[validation_indices]]
        )

        self.u_ref_training = torch.cat([self.u_ref_training, u_ref[training_indices]])
        self.u_ref_validation = torch.cat(
            [self.u_ref_validation, u_ref[validation_indices]]
        )

        self.u_ref_training = torch.cat(
            [self.u_ref_training, u_expert[training_indices]]
        )
        self.u_ref_validation = torch.cat(
            [self.u_ref_validation, u_expert[validation_indices]]
        )

        self.x_training = torch.cat([self.x_training, x[training_indices]])
        self.x_validation = torch.cat([self.x_validation, x[validation_indices]])
