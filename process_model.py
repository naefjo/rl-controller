import numpy as np
import sympy as sp


# https://www.koreascience.or.kr/article/CFKO200333239336988.pdf
class InvertedPendulum:
    gravity = 9.81

    def __init__(
        self,
        length,
        pendulum_mass,
        cart_mass,
        friction_coeff_cart_ground,
        pendulum_inertia,
        initial_state,
        control_law,
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
        return self.state

    def set_state(self, state):
        self.state = state

    def set_controller(self, controller):
        self.control_law = controller

    def dynamics_continuous_sym(self):

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
        # simple RK4 integration
        # print(state)

        k1 = self.dynamics_continuous(state)
        # print("k1\n", k1)
        k2 = self.dynamics_continuous(state + timestep * k1 / 2)
        # print("k2\n", k2)
        k3 = self.dynamics_continuous(state + timestep * k2 / 2)
        # print("k3\n", k3)
        k4 = self.dynamics_continuous(state + timestep * k3)
        # print("k4\n", k4)

        new_state = state + timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return new_state

    def integrate_dynamics(self, timestep):
        self.state = self.integrate_dynamics_from_state(self.state, timestep)
        return self.state
