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

    def get_state(self):
        return self.state

    def dynamics_continuous_sym(self):

        x1, x2, x3, x4 = sp.symbols("x1 x2 x3 x4")

        fraction = self.cart_mass + self.pendulum_mass * (np.sin(x3) ** 2)

        dynamics_vector_sym = sp.Matrix(
            [
                [x2],
                [
                    (
                        -self.pendulum_mass
                        * self.gravity
                        * np.sin(x3)
                        * np.cos(x3)
                        + self.pendulum_mass
                        * self.pole_length
                        * (x4 ** 2)
                        * np.sin(x3)
                        - self.friction_coeff_cart_ground * x2
                        + self.control_law(self.state)
                    )
                    / fraction
                ],
                [x4],
                [
                    (
                        -self.pendulum_mass
                        * self.pole_length
                        * (x4 ** 2)
                        * np.sin(x3)
                        * np.cos(x3)
                        + self.friction_coeff_cart_ground * x2 * np.cos(x3)
                        + (self.cart_mass + self.pendulum_mass)
                        * self.gravity
                        * np.sin(
                            x3 - np.cos(x3 * self.control_law(self.state))
                        )
                    )
                    / (self.pole_length * fraction)
                ],
            ]
        )

        if not (
            hasattr(self.__class__, "dynamics_continuous")
            and callable(getattr(self.__class__, "dynamics_continuous"))
        ):
            self.dynamics_continuous = sp.lambdify(
                (x1, x2, x3, x4), dynamics_vector_sym, modules="numpy"
            )

        return dynamics_vector_sym

    def integrate_dynamics(self, timestep):
        # simple RK4 integration

        k1 = self.dynamics_continuous(self.state)
        k2 = self.dynamics_continuous(self.state + timestep * k1 / 2)
        k3 = self.dynamics_continuous(self.state + timestep * k2 / 2)
        k4 = self.dynamics_continuous(self.state + timestep * k3)

        return self.state + timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6
