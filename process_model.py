"""
This module implements an inverted pendulum process model
and means to simulate it.
"""

import abc
from typing import Callable, Type, TypeVar
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class ModelBaseClass(abc.ABC):

    @abc.abstractmethod
    def __init__(self):

        x = "x1:" + str(self.num_system_states + 1)
        u = "u1:" + str(self.num_system_inputs + 1)

        """Return continuous time dynamics as a 4x1 vector"""
        self.lambdified_symbolic_system = sp.lambdify(
            sp.symbols(x + " " + u),
            self.dynamics_continuous_sym(),
            modules="numpy",
        )

    @abc.abstractproperty
    def num_system_states(self) -> int:
        pass

    @abc.abstractproperty
    def num_system_inputs(self) -> int:
        pass

    @abc.abstractmethod
    def dynamics_continuous_sym(self) -> sp.Matrix:
        pass

    def dynamics_continuous(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """Return output of continuous dynamics as an array"""
        return np.squeeze(self.lambdified_symbolic_system(*x, *u))


class InvertedPendulum(ModelBaseClass):
    """The InvertedPendulum class

    https://www.koreascience.or.kr/article/CFKO200333239336988.pdf
    """
    gravity = 9.81

    def __init__(
        self,
        length: float,
        pendulum_mass: float,
        cart_mass: float,
        friction_coeff_cart_ground: float,
        pendulum_inertia: float,
    ) -> None:
        self.pole_length = length
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.friction_coeff_cart_ground = friction_coeff_cart_ground
        self.pendulum_inertia = pendulum_inertia
        
        super().__init__()

    @property
    def num_system_states(self)-> int:
        return 4

    @property
    def num_system_inputs(self) -> int:
        return 1

    def dynamics_continuous_sym(self) -> sp.Matrix:
        """Symbolic representation of the continuous state space dynamics.

        This function implements $Dot{x} = f(x)$ using sympy.
        """

        x2, x3, x4 = sp.symbols("x2:5")
        u1 = sp.symbols("u1")

        fraction = self.cart_mass + self.pendulum_mass * (sp.sin(x3) ** 2)

        dynamics_vector_sym = sp.Matrix(
            [
                x2,
                (
                    -self.pendulum_mass
                    * self.gravity
                    * sp.sin(x3)
                    * sp.cos(x3)
                    + self.pendulum_mass
                    * self.pole_length
                    * (x4 ** 2)
                    * sp.sin(x3)
                    - self.friction_coeff_cart_ground * x2
                    + u1
                )
                / fraction,
                x4,
                (
                    -self.pendulum_mass
                    * self.pole_length
                    * (x4 ** 2)
                    * sp.sin(x3)
                    * sp.cos(x3)
                    + self.friction_coeff_cart_ground * x2 * sp.cos(x3)
                    + (self.cart_mass + self.pendulum_mass)
                    * self.gravity
                    * sp.sin(x3)
                    - sp.cos(x3) * u1
                )
                / (self.pole_length * fraction),
            ]
        )

        return dynamics_vector_sym


P = TypeVar('P', bound=ModelBaseClass)


class PlantSimulator:
    rk4_coefficients = np.array([1, 2, 2, 1])

    def __init__(
        self,
        plant: Type[P],
        timestep: float,
        simulation_time: float,
    ):
        self.plant = plant
        self.timestep = timestep
        self.simulation_time = simulation_time
        self.iterations = int(self.simulation_time / self.timestep)

    def set_controller(self, controller: Callable[[np.ndarray], np.ndarray]):
        self.control_law = controller

    def set_plant(
        self,
        plant: Type[P],
    ):
        self.plant = plant

    def _integrate_dynamics_from_state(self, state: np.ndarray):
        """RK4 integration of the system from a state for one timestep"""

        # TODO:check correctness of implementation.
        k = np.zeros((5, state.shape[0]))
        for i in range(4):
            k_state = state + \
                self.timestep * k[i, :] / self.rk4_coefficients[i]
            k[i+1, :] = self.plant.dynamics_continuous(k_state, self.control_law(k_state))

        new_state = (
            state + 
            self.timestep * 
            k[1:, :].transpose() @ self.rk4_coefficients / 6
        )

        return new_state

    def simulate_system(
        self,
        initial_state: np.ndarray,
        enable_plot: bool = False):
        """Simulate the system from initial condition for
        a given timespan and resolution """

        state_list = np.zeros(
            (self.iterations, self.plant.num_system_states)
        )
        actuator_list = np.zeros(
            (self.iterations, self.plant.num_system_inputs)
        )
        t = np.zeros(self.iterations)

        state_list[0, :] = initial_state
        actuator_list[0, :] = self.control_law(initial_state)

        for i in range(1, self.iterations):
            t[i] = (i) * self.timestep
            state_list[i, :] = self._integrate_dynamics_from_state(
                state_list[i-1, :],
            )
            actuator_list[i] = self.control_law(state_list[i, :])

        if enable_plot:
            self.plot_state_evolution(state_list, actuator_list, t)

        return (state_list, actuator_list, t)

    @staticmethod
    def plot_state_evolution(state_evolution, actuator_evolution, timesteps):
        """Plot state evolution of the system for a given state evolution"""
        _, axs = plt.subplots(
            2,
            3,
            constrained_layout=True,
            figsize=(15, 7.5),
        )
        axs[0, 0].plot(timesteps, state_evolution[:, 0])
        axs[0, 0].set_title("x1")
        axs[0, 1].plot(timesteps, state_evolution[:, 1], "tab:orange")
        axs[0, 1].set_title("x2")
        axs[1, 0].plot(timesteps, state_evolution[:, 2], "tab:green")
        axs[1, 0].set_title("x3")
        axs[1, 1].plot(timesteps, state_evolution[:, 3], "tab:red")
        axs[1, 1].set_title("x4")
        axs[0, 2].plot(timesteps, actuator_evolution, "tab:purple")
        axs[0, 2].set_title("u")

        plt.plot()
