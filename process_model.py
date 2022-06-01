import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


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
        lambdify_sym = sp.lambdify(
            sp.symbols("x1:5"),
            self.dynamics_continuous_sym(),
            modules="numpy",
        )

        self.dynamics_continuous = lambda x: np.squeeze(lambdify_sym(*x))

    def get_state(self):
        return self.state

    def dynamics_continuous_sym(self):

        x2, x3, x4 = sp.symbols("x2:5")

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
                    + self.control_law(self.state)
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
                    - sp.cos(x3) * self.control_law(self.state)
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

        self.state = new_state
        return new_state

    def integrate_dynamics(self, timestep):
        return self.integrate_dynamics_from_state(self.state, timestep)


if __name__ == "__main__":
    dyn = InvertedPendulum(
        1.0,
        2.0,
        30.0,
        0.4,
        5.0,
        np.array([0.0, 0.0, 0.01, 0.0]),
        lambda x: x[0],
    )

    # print(dyn.dynamics_continuous([0.1, 0.2, 0.3, 0.4]))
    iterations = 100
    state_list = np.zeros((iterations, 4))
    t = []
    time_step = 0.1
    for i in range(iterations):
        t.append(i * time_step)
        state_list[i, :] = dyn.integrate_dynamics(time_step)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(t, state_list[:, 0])
    axs[0, 0].set_title("x1")
    axs[0, 1].plot(t, state_list[:, 1], "tab:orange")
    axs[0, 1].set_title("x2")
    axs[1, 0].plot(t, -state_list[:, 2], "tab:green")
    axs[1, 0].set_title("x3")
    axs[1, 1].plot(t, -state_list[:, 3], "tab:red")
    axs[1, 1].set_title("x4")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.savefig("out.png")
