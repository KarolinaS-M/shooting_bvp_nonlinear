import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

# ======================================================
# Page configuration
# ======================================================

st.set_page_config(
    page_title="Shooting Method for a Linear BVP",
    layout="wide"
)

st.title("Shooting Method for a Boundary Value Problem")
st.markdown(
    r"""
We solve the boundary value problem  
\[
x'(t) = \lambda x(t), \qquad x(T) = x_T,
\]
using the **shooting method**, where the unknown initial value
\(\theta = x(0)\) is iteratively adjusted so that the terminal condition is satisfied.
"""
)

# ======================================================
# Sidebar: user inputs
# ======================================================

with st.sidebar:
    st.header("Model parameters")

    lam = st.number_input(
        "λ (lambda < 0)",
        value=-1.0,
        format="%.6f"
    )

    T = st.number_input(
        "Terminal time T",
        value=5.0,
        format="%.6f"
    )

    x_T = st.number_input(
        "Terminal value x_T",
        value=1.0,
        format="%.6f"
    )

    tol = st.number_input(
        "Tolerance",
        value=1e-6,
        format="%.1e"
    )

    max_iter = st.number_input(
        "Maximum number of iterations",
        value=20,
        step=1
    )

    st.markdown("---")
    st.header("Initial guesses")

    theta0 = st.number_input(
        "Initial guess θ₀",
        value=0.2,
        format="%.6f"
    )

    theta1 = st.number_input(
        "Initial guess θ₁",
        value=2.0,
        format="%.6f"
    )

# ======================================================
# Analytical solution
# ======================================================

def exact_solution(t, theta, lam):
    return theta * np.exp(lam * t)

def terminal_mismatch(theta, lam, T, x_T):
    return theta * np.exp(lam * T) - x_T

# ======================================================
# Shooting method (bisection)
# ======================================================

theta_vals = []
F_vals = []

F0 = terminal_mismatch(theta0, lam, T, x_T)
F1 = terminal_mismatch(theta1, lam, T, x_T)

if F0 * F1 > 0:
    st.error(
        "Initial guesses θ₀ and θ₁ do not bracket a solution. "
        "Please choose values with opposite signs of F(θ)."
    )
    st.stop()

theta_vals.extend([theta0, theta1])
F_vals.extend([F0, F1])

theta_left, theta_right = theta0, theta1
F_left, F_right = F0, F1

converged = False

for k in range(max_iter):
    theta_mid = 0.5 * (theta_left + theta_right)
    F_mid = terminal_mismatch(theta_mid, lam, T, x_T)

    theta_vals.append(theta_mid)
    F_vals.append(F_mid)

    if abs(F_mid) < tol:
        converged = True
        theta_star = theta_mid
        break

    if F_left * F_mid < 0:
        theta_right, F_right = theta_mid, F_mid
    else:
        theta_left, F_left = theta_mid, F_mid

if not converged:
    theta_star = theta_mid

# ======================================================
# Display iteration table
# ======================================================

st.subheader("Shooting iterations")

data = {
    "k": list(range(len(theta_vals))),
    "θ_k": theta_vals,
    "F(θ_k)": F_vals,
    "sign(F)": ["+" if f > 0 else "-" for f in F_vals]
}

st.dataframe(data, use_container_width=True)

# ======================================================
# Plot trajectories
# ======================================================

t = np.linspace(0, T, 400)

plt.figure(figsize=(8, 5))

# Exact solution for optimal theta
x_exact = exact_solution(t, theta_star, lam)
plt.plot(t, x_exact, color="black", linewidth=2, label="Exact solution")

# Shooting trajectories
for k, theta in enumerate(theta_vals):
    x_shot = exact_solution(t, theta, lam)
    plt.plot(
        t,
        x_shot,
        linestyle="--",
        alpha=0.6,
        label=f"Shot k={k}"
    )

# Terminal condition
plt.scatter(T, x_T, color="red", zorder=5, label=r"$x(T)=x_T$")

plt.xlabel("Time t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/shooting_linear_bvp.png", dpi=300)

st.pyplot(plt.gcf())

# ======================================================
# Convergence message
# ======================================================

if converged:
    st.success(
        f"Converged after {len(theta_vals)-1} iterations.\n\n"
        f"θ* = {theta_star:.6f},  |F(θ*)| = {abs(F_mid):.2e}"
    )
else:
    st.warning(
        "Maximum number of iterations reached.\n\n"
        f"Last iterate: θ ≈ {theta_star:.6f},  |F(θ)| ≈ {abs(F_mid):.2e}"
    )

st.markdown(
    r"""
**Interpretation.**  
Each iteration corresponds to a *shot* with a guessed initial value $\theta$.
The terminal mismatch $F(\theta)$ measures how far the resulting trajectory is
from satisfying the boundary condition at $t=T$.
The method adjusts $\theta$ until the discrepancy vanishes within the prescribed tolerance.
"""
)