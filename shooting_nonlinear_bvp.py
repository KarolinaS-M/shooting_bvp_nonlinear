import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Nonlinear Shooting Method", layout="centered")

# ======================================================
# Sidebar: parameters
# ======================================================

st.sidebar.header("Model parameters")

lam = st.sidebar.number_input("λ (lambda < 0)", value=-1.0, step=0.1)
alpha = st.sidebar.number_input("α (nonlinearity)", value=0.05, step=0.01)
T = st.sidebar.number_input("Terminal time T", value=2.0, step=0.1)
x_T = st.sidebar.number_input("Terminal value x(T)", value=1.0, step=0.1)

st.sidebar.header("Numerical settings")

dt = st.sidebar.number_input("Time step Δt", value=0.001, format="%.4f")
tol = st.sidebar.number_input("Tolerance", value=1e-6, format="%.1e")
max_iter = st.sidebar.number_input("Max iterations", value=25, step=1)

theta_L = st.sidebar.number_input("Lower bracket θ_L", value=0.0)
theta_U = st.sidebar.number_input("Upper bracket θ_U", value=6.0)

# ======================================================
# Time grid
# ======================================================

N = int(T / dt)
t = np.linspace(0, T, N + 1)

# ======================================================
# IVP solver (explicit Euler)
# ======================================================

def solve_ivp(theta):
    x = np.zeros(N + 1)
    x[0] = theta
    for n in range(N):
        x[n+1] = x[n] + dt * (lam * x[n] + alpha * x[n]**2)
        if not np.isfinite(x[n+1]):
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
# Bracket check
# ======================================================

FL = F(theta_L)
FU = F(theta_U)

if not (np.isfinite(FL) and np.isfinite(FU)):
    st.error("Non-finite shooting function at the bracket endpoints.")
    st.stop()

if FL * FU > 0:
    st.error("Initial bracket does not enclose a root.")
    st.stop()

# ======================================================
# Bisection shooting
# ======================================================

thetas = []
trajectories = []

st.subheader("Shooting iterations")

for k in range(max_iter):
    theta_M = 0.5 * (theta_L + theta_U)
    FM = F(theta_M)

    thetas.append(theta_M)
    trajectories.append(solve_ivp(theta_M))

    sign = ">" if FM > 0 else "<"
    st.write(f"k = {k:2d}, θ = {theta_M:.6f}, F(θ) {sign} 0, |F| = {abs(FM):.3e}")

    if abs(FM) < tol:
        st.success(f"Converged after {k+1} iterations.")
        break

    if FL * FM < 0:
        theta_U, FU = theta_M, FM
    else:
        theta_L, FL = theta_M, FM

theta_star = theta_M

st.write(f"**Approximate θ\*** = {theta_star:.6f}")

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

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(t, x_exact, color="black", linewidth=2, label="Exact solution")

for k, x in enumerate(trajectories):
    ax.plot(t, x, linestyle="--", alpha=0.6, label=f"Shot k={k}")

ax.scatter(T, x_T, color="red", zorder=5, label=r"Target $x(T)$")

ax.set_xlabel("Time t")
ax.set_ylabel("x(t)")
ax.legend()
ax.grid(True)

st.pyplot(fig)