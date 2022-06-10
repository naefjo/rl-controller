"""
This module implements an inverted pendulum process model
and means to simulate it.
"""

from typing import Callable
import numpy as np
import numpy.typing as npt
import sympy as sp
import matplotlib.pyplot as plt


# https://www.koreascience.or.kr/article/CFKO200333239336988.pdf
class InvertedPendulum:
    """The InvertedPendulum class"""
    gravity = 9.81

    def __init__(
        self,
        length: float,
        pendulum_mass: float,
        cart_mass: float,
        friction_coeff_cart_ground: float,
        pendulum_inertia: float,
        initial_state: npt.NDArray,
        control_law: Callable[[np.array], float],
    ) -> None:
        self.pole_length = length
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.friction_coeff_cart_ground = friction_coeff_cart_ground
        self.pendulum_inertia = pendulum_inertia
        self.state = initial_state
        self.control_law = control_law

        # Crazy lambda to get symbolic dynamics into an evaluable form
        lambdified_sym = sp.lambdify(
            sp.symbols("x1:5 u"),
            self.dynamics_continuous_sym(),
            modules="numpy",
        )
        self.dynamics_continuous = lambda x: np.squeeze(lambdified_sym(*x, self.control_law(x)))
    
    def get_state(self):
        """state getter"""
        return self.state

    def set_state(self, state):
        """state setter"""
        self.state = state

    def get_controller(self):
        """controller getter"""
        return self.control_law

    def set_controller(self, controller):
        """controller setter"""
        self.control_law = controller

    def dynamics_continuous_sym(self):
        """symbolic continuous equations of motion"""
        x2, x3, x4 = sp.symbols("x2:5")
        u = sp.symbols("u")

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
                    + u  # self.control_law(self.state)
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
                    - sp.cos(x3) * u  # self.control_law(self.state)
                )
                / (self.pole_length * fraction),
            ]
        )

        # print(dynamics_vector_sym)

        return dynamics_vector_sym

    def integrate_dynamics_from_state(self, state, timestep):
        """RK4 integration of the system from a state for one timestep"""
        k_1 = self.dynamics_continuous(state)
        k_2 = self.dynamics_continuous(state + timestep * k_1 / 2)
        k_3 = self.dynamics_continuous(state + timestep * k_2 / 2)
        k_4 = self.dynamics_continuous(state + timestep * k_3)

        new_state = state + timestep * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6

        return new_state

    def integrate_dynamics(self, timestep):
        """integrate the dynamics from the internal state for one timestep"""
        self.state = self.integrate_dynamics_from_state(self.state, timestep)
        return self.state

    def simulate_system(self, initial_state, timespan, delta_t):
        """Simulate the system from initial condition for a given timespan and resolution"""
        iterations = int(timespan/delta_t)
        state_list = np.zeros((4, iterations))
        timesteps = np.zeros(iterations)

        self.set_state(initial_state)

        for i in range(iterations):
            timesteps[i] = i * delta_t
            state_list[:,i] = self.integrate_dynamics(delta_t)

        return (state_list, timesteps)

    def plot_state_evolution(self, state_evolution, timesteps):
        """Plot state evolution of the system for a given state evolution"""
        fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(15,7.5))
        axs[0, 0].plot(timesteps, state_evolution[0,:])
        axs[0, 0].set_title("x1")
        axs[0, 1].plot(timesteps, state_evolution[1,:], "tab:orange")
        axs[0, 1].set_title("x2")
        axs[1, 0].plot(timesteps, state_evolution[2,:], "tab:green")
        axs[1, 0].set_title("x3")
        axs[1, 1].plot(timesteps, state_evolution[3,:], "tab:red")
        axs[1, 1].set_title("x4")
        axs[0,2].plot(
            timesteps,
            [self.control_law(state_evolution[:,i]) for i in range(len(timesteps))],
            "tab:purple"
        )
        axs[0,2].set_title("u")

        plt.plot()
