import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Shooting Method: Nonlinear Boundary Value Problem",
    layout="wide"
)

# ======================================================
# Title and problem description
# ======================================================

st.title("Shooting Method: Nonlinear Boundary Value Problem")

st.markdown(
    "We illustrate the shooting method for a nonlinear boundary value problem, "
    "where the unknown initial condition must be adjusted iteratively so that "
    "a terminal condition is satisfied."
)

st.latex(r"""
x'(t)=\lambda x(t)+\alpha x(t)^2, \qquad x(T)=x_T
""")

st.markdown(
    "The initial value $x(0)=\\theta$ is not prescribed. "
    "Instead, it is treated as a free parameter and updated iteratively "
    "using the shooting method combined with a root--finding algorithm."
)

st.info(
    "In contrast to the linear illustrative example, the shooting function "
    "$F(\\theta)=x(T;\\theta)-x_T$ is nonlinear and cannot be evaluated "
    "in closed form. Each shooting iteration therefore requires a full "
    "numerical integration of the initial value problem."
)

# ======================================================
# Sidebar: model parameters
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input("λ (lambda < 0)", value=-1.0, step=0.1, format="%.3f")
    alpha = st.number_input("α (nonlinearity)", value=0.05, step=0.01, format="%.3f")
    T = st.number_input("Terminal time T", value=2.0, step=0.1, format="%.2f")
    x_T = st.number_input("Terminal value x(T)", value=1.0, step=0.1, format="%.2f")

    st.markdown("---")
    st.header("Numerical settings")

    dt = st.number_input("Time step Δt", value=0.001, format="%.4f")
    tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
    max_iter = st.number_input("Maximum iterations", value=25, step=1)

    st.markdown("---")
    st.header("Initial bracket for θ")

    theta_L = st.number_input("Lower bound θ_L", value=0.0, format="%.2f")
    theta_U = st.number_input("Upper bound θ_U", value=6.0, format="%.2f")

# ======================================================
# Time grid
# ======================================================

N = int(T / dt)
t = np.linspace(0, T, N + 1)

# ======================================================
# IVP solver: explicit Euler
# ======================================================

def solve_ivp(theta):
    x = np.zeros(N + 1)
    x[0] = theta
    for n in range(N):
        x[n + 1] = x[n] + dt * (lam * x[n] + alpha * x[n]**2)
        if not np.isfinite(x[n + 1]):
            return None
    return x

# ======================================================
# Shooting function
# ======================================================

def F(theta):
    x = solve_ivp(theta)
    if x is None:
        return np.nan
    return x[-1] - x_T

# ======================================================
# Bracket validation
# ======================================================

FL = F(theta_L)
FU = F(theta_U)

if not (np.isfinite(FL) and np.isfinite(FU)):
    st.error(
        "The shooting function is not finite at the bracket endpoints. "
        "Try reducing α, shortening T, or narrowing the initial bracket."
    )
    st.stop()

if FL * FU > 0:
    st.error(
        "The initial bracket does not enclose a root. "
        "Choose θ_L and θ_U such that F(θ_L) and F(θ_U) have opposite signs."
    )
    st.stop()

# ======================================================
# Shooting iterations: bisection
# ======================================================

st.subheader("Shooting iterations (bisection method)")

st.markdown(
    "The unknown initial value $\\theta$ is updated using the **bisection method**. "
    "At each iteration, the current bracket is halved depending on the sign "
    "of the shooting function $F(\\theta)$. "
    "This strategy is robust but typically requires many iterations."
)

thetas = []
trajectories = []

for k in range(int(max_iter)):
    theta_M = 0.5 * (theta_L + theta_U)
    FM = F(theta_M)

    thetas.append(theta_M)
    trajectories.append(solve_ivp(theta_M))

    sign = ">" if FM > 0 else "<"
    st.write(
        f"k = {k:2d}, θ = {theta_M:.6f}, "
        f"F(θ) {sign} 0, |F| = {abs(FM):.3e}"
    )

    if abs(FM) < tol:
        st.success(f"Converged after {k + 1} iterations.")
        break

    if FL * FM < 0:
        theta_U, FU = theta_M, FM
    else:
        theta_L, FL = theta_M, FM

theta_star = theta_M

st.markdown(f"**Approximate solution:**  \n"
            f"$\\theta^* \\approx {theta_star:.6f}$")

# ======================================================
# Exact solution (reference only)
# ======================================================

def exact_solution(theta):
    num = lam * theta * np.exp(lam * t)
    den = lam + alpha * theta * (1 - np.exp(lam * t))
    return num / den

x_exact = exact_solution(theta_star)

# ======================================================
# Plot
# ======================================================

st.subheader("Trajectories generated by successive shots")

st.markdown(
    "Each dashed curve corresponds to a trial trajectory generated by a different "
    "shooting iteration. Early shots either undershoot or overshoot the terminal "
    "target, while successive iterations progressively narrow the admissible "
    "range for the unknown initial condition."
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    t,
    x_exact,
    color="black",
    linewidth=2,
    label="Exact solution (reference only)"
)

for k, x in enumerate(trajectories):
    ax.plot(
        t,
        x,
        linestyle="--",
        alpha=0.6,
        label=f"Shot k={k}"
    )

ax.scatter(T, x_T, color="red", zorder=5, label=r"Target $x(T)$")

ax.set_xlabel("Time t")
ax.set_ylabel("x(t)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ======================================================
# Interpretation
# ======================================================

st.info(
    "In contrast to the linear case, convergence is achieved only after a substantial "
    "number of forward simulations. This reflects the nonlinear dependence of "
    "$x(T;\\theta)$ on the initial value and illustrates both the flexibility "
    "and the potential numerical fragility of shooting methods in nonlinear models."
)