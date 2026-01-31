import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# Page setup
# ======================================================

st.set_page_config(page_title="Shooting method: linear BVP", layout="centered")

st.title("Shooting method for a linear boundary value problem")
st.write(
    r"We consider the boundary value problem $x'(t)=\lambda x(t)$ "
    r"with $x(0)=\theta$ and terminal condition $x(T)=x_T$."
)

# ======================================================
# User inputs (text boxes only)
# ======================================================

lam = st.number_input("λ (lambda < 0)", value=-1.0, format="%.6f")
T = st.number_input("Terminal time T", value=5.0, format="%.6f")
x_T = st.number_input("Terminal value x_T", value=1.0, format="%.6f")
tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
max_iter = st.number_input("Maximum number of iterations", value=20, step=1)

st.markdown("---")

# ======================================================
# Model definitions
# ======================================================

def exact_solution(t, theta):
    return theta * np.exp(lam * t)

def shooting_function(theta):
    return exact_solution(T, theta) - x_T

# exact theta*
theta_star = x_T * np.exp(-lam * T)

# ======================================================
# Initial guesses
# ======================================================

theta0 = st.number_input("Initial guess θ₀", value=0.2, format="%.6f")
theta1 = st.number_input("Initial guess θ₁", value=2.0, format="%.6f")

# ======================================================
# Secant method
# ======================================================

thetas = [theta0, theta1]
Fvals = [shooting_function(theta0), shooting_function(theta1)]

k = 1
while abs(Fvals[-1]) > tol and k < max_iter:
    theta_new = thetas[-1] - Fvals[-1] * (thetas[-1] - thetas[-2]) / (Fvals[-1] - Fvals[-2])
    F_new = shooting_function(theta_new)

    thetas.append(theta_new)
    Fvals.append(F_new)
    k += 1

# ======================================================
# Display iteration table
# ======================================================

st.subheader("Shooting iterations")

data = []
for i, (th, fv) in enumerate(zip(thetas, Fvals)):
    sign = "positive" if fv > 0 else "negative" if fv < 0 else "zero"
    data.append([i, th, fv, sign])

st.table(
    {
        "k": [row[0] for row in data],
        "θ_k": [f"{row[1]:.6e}" for row in data],
        "F(θ_k)": [f"{row[2]:.6e}" for row in data],
        "sign": [row[3] for row in data],
    }
)

# ======================================================
# Plot
# ======================================================

t = np.linspace(0, T, 300)

plt.figure(figsize=(8, 5))

# exact solution
plt.plot(
    t,
    exact_solution(t, theta_star),
    color="black",
    linewidth=1.5,
    label="Exact solution"
)

# shooting trajectories
for i, theta in enumerate(thetas):
    plt.plot(
        t,
        exact_solution(t, theta),
        linestyle="--",
        alpha=0.8,
        label=f"Shot k={i}"
    )

# terminal condition
plt.axhline(x_T, color="gray", linestyle=":", label="x(T) = x_T")

plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# save figure
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/2.5.png", dpi=300)

st.pyplot(plt)
plt.close()

# ======================================================
# Convergence message
# ======================================================

if abs(Fvals[-1]) <= tol:
    st.success(
        rf"Converged: |F(θ)| ≤ tol after {len(thetas)-1} iterations."
    )
else:
    st.warning(
        rf"Did not converge within {max_iter} iterations."
    )

st.markdown(
    rf"Exact initial value: $\theta^* = x_T e^{{-\lambda T}} = {theta_star:.6e}$"
)