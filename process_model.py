from typing import Any, Callable
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


# https://www.koreascience.or.kr/article/CFKO200333239336988.pdf
class InvertedPendulum:
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

    def dynamics_continuous_sym(self) -> sp.Matrix((4,1)):
        """Symbolic representation of the continuous state space dynamics.
        
        This function implements $\Dot{x} = f(x)$ using sympy.
        """

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
                    + u
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
                    - sp.cos(x3) * u
                )
                / (self.pole_length * fraction),
            ]
        )

        return dynamics_vector_sym

    #TODO: Fix type annotation of return type...
    def dynamics_continuous(
        self,
        x: np.ndarray((1, 4)),
        u: np.ndarray((1, 1))
    ) -> Callable:
        """Converts symbolic representation into a numpy function.
        
        Convert sympy to numpy array and then use argument unpacking to specify
        x and u as vectors instead of every component as a function argument.
        `lambdified_sym` implements a numpy array with `x1:5` and `u` as variables.
        """
        lambdified_sym = sp.lambdify(
            sp.symbols("x1:5 u"),
            self.dynamics_continuous_sym(),
            modules="numpy",
        )
        
        return np.squeeze(lambdified_sym(*x, *u))


class PlantSimulator:
    rk4_coefficients = np.array([1, 2, 2, 1])

    def __init__(
        self, 
        plant: Callable[[np.ndarray, np.ndarray], np.ndarray],
        timestep: float,
        simulation_time: float,
    ):
        self.plant = plant
        self.timestep = timestep
        self.simulation_time = simulation_time
        self.controller = np.zeros(plant.__annotations__.get('u').shape)
        self.iterations = int(self.simulation_time / self.timestep)

    def set_controller(self, controller: Callable[[np.ndarray], np.ndarray]):
        self.control_law = controller

    def set_plant(self, plant: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.plant = plant

    def _integrate_dynamics_from_state(self, state: np.ndarray):
        # TODO:check correctness of implementation.
        k = np.zeros((5,state.shape[1]))
        for i in range(4):
            k_state = state + self.timestep * k[i, :] / self.rk4_coefficients[i]
            k[i+1,:] = self.plant(k_state, self.control_law(k_state))

        new_state = state + self.timestep * k[1:, :] @ self.rk4_coefficients
        print(new_state)

        # simple RK4 integration
        # TODO(@naefjo): is control_action correct or does it need computed for each k_i
        state0 = state
        k1 = self.plant(state0, self.control_law(state0))
        state1 = state + self.timestep * k1 / 2
        k2 = self.plant(state1, self.control_law(state1))
        state2 = state + self.timestep * k2 / 2
        k3 = self.plant(state2, self.control_law(state2))
        state3 = state + self.timestep * k3
        k4 = self.plant(state3, self.control_law(state3))

        new_state = state + self.timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        print(new_state)

        return new_state

    def simulate_system(self, initial_state):
        state_list = np.zeros(
            (self.iterations+1, self.plant.__annotations__.get('x').shape[1])
        )
        actuator_list = np.zeros(
            (self.iterations+1, self.plant.__annotations__.get('u').shape[1])
        )
        t = np.zeros(self.iterations+1)

        state_list[0,:] = initial_state
        actuator_list[0,:] = self.control_law(initial_state)

        for i in range(self.iterations):
            t[i+1] = (i+1) * self.timestep
            state_list[i+1, :] = self._integrate_dynamics_from_state(
                self.timestep,
                state_list[i, :]
            )
            actuator_list[i+1] = self.control_law(state_list[i+1, :])

        return (state_list, actuator_list, t)

    @staticmethod
    def plot_state_evolution(state_evolution, t):
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        axs[0, 0].plot(t, state_evolution[:, 0])
        axs[0, 0].set_title("x1")
        axs[0, 1].plot(t, state_evolution[:, 1], "tab:orange")
        axs[0, 1].set_title("x2")
        axs[1, 0].plot(t, state_evolution[:, 2], "tab:green")
        axs[1, 0].set_title("x3")
        axs[1, 1].plot(t, state_evolution[:, 3], "tab:red")
        axs[1, 1].set_title("x4")

        plt.plot()

    def plot_actuator_commands(self, state_evolution, t):
        actuator_commands = np.zeros(t.shape)
        for i in range(actuator_commands.shape[0]):
            actuator_commands[i] =  self.control_law(state_evolution[i,:])

        plt.plot(t, actuator_commands)
        plt.title("u")