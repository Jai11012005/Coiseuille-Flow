import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the Jacobian matrix and calculate eigenvalues
def jacobian_eigenvalues():
    A = np.array([[0, 1], [0, 0]])  # Jacobian matrix for this system
    eigenvalues = np.linalg.eigvals(A)
    return eigenvalues

# Display equations used in the model
def display_equations():
    st.markdown("### 📘 Equations Used")
    st.latex(r"""
        \frac{d^2u}{dy^2} = -P
    """)
    st.markdown("**Converted into first-order differential equations:**")
    st.latex(r"""
        u_1' = u_2
    """)
    st.latex(r"""
        u_2' = -P
    """)
    st.markdown("**Analytical solution:**")
    st.latex(r"""
        u(y) = y + \frac{P}{2} \cdot y \cdot (1 - y)
    """)
    st.markdown("**BVP Finite Difference Method:**")
    st.latex(r"""
        A u = b
    """)

# Define the analytical solution function
def analytical_solution(y, P):
    return y + (P / 2) * y * (1 - y)

# Define the BVP solver function
def solve_couette_poiseuille(P, N):
    dy = 1.0 / (N - 1)
    y = np.linspace(0, 1, N)
    A = np.zeros((N, N))
    b = np.zeros(N)
    for i in range(1, N - 1):
        A[i, i - 1] = 1
        A[i, i] = -2
        A[i, i + 1] = 1
        b[i] = -P * dy**2
    A[0, 0] = 1
    b[0] = 0
    A[-1, -1] = 1
    b[-1] = 1
    u = np.linalg.solve(A, b)
    return y, u

# Define the IVP solvers for Explicit and Implicit Euler methods
def IVP_Explicit(p, h):
    iteration = round(1/h) + 1
    u1 = np.zeros(iteration)
    u2 = np.zeros(iteration)
    u1[0] = 0
    u2[0] = initial_value(p)
    for i in range(iteration - 1):
        u1[i + 1] = u1[i] + u2[i] * h
        u2[i + 1] = u2[i] + (-p) * h
    return u1, u2

def IVP_Implicit(p, h):
    iteration = round(1/h) + 1
    u1 = np.zeros(iteration)
    u2 = np.zeros(iteration)
    u1[0] = 0
    u2[0] = initial_value(p)
    for i in range(iteration - 1):
        u2[i + 1] = u2[i] + (-p) * h
        u1[i + 1] = u1[i] + u2[i + 1] * h
    return u1, u2

# Additional functions to calculate initial conditions for IVP
def shooting_method(u2_initial_guess, p):
    h = 0.01
    iteration = round(1/h)
    u0 = [0, u2_initial_guess]
    u1 = np.zeros(iteration)
    u2 = np.zeros(iteration)
    u1[0] = u0[0]
    u2[0] = u0[1]
    for i in range(iteration - 1):
        u1[i + 1] = u1[i] + u2[i] * h
        u2[i + 1] = u2[i] + (-p) * h
    return u1[-1]

def initial_value(p):
    closest_u2 = None
    closest_diff = float('inf')
    for u2_initial_guess in range(100):
        final_u1 = shooting_method(u2_initial_guess, p)
        diff = abs(final_u1 - 1)
        if diff < closest_diff:
            closest_diff = diff
            closest_u2 = u2_initial_guess
    return closest_u2

# Streamlit app layout
st.title("🌊 Couette-Poiseuille Flow Simulation")
st.write("Simulate Couette-Poiseuille flow using numerical methods.")

display_equations()

# Display eigenvalues of the Jacobian matrix
st.markdown("### 🧮 Eigenvalues of the Jacobian Matrix")
eigenvalues = jacobian_eigenvalues()
st.write(f"**Eigenvalues:** {eigenvalues}")

# User inputs with tooltips
st.sidebar.markdown("## 🔧 Parameters")
P = st.sidebar.slider("Pressure Gradient (P)", min_value=-2.0, max_value=10.0, step=0.5, value=2.0, help="Adjust the pressure gradient for the simulation.")
N = st.sidebar.slider("Number of Grid Points (N)", min_value=10, max_value=200, step=10, value=100, help="Number of grid points for the BVP solver.")
h = st.sidebar.slider("Step Size for IVP (h)", min_value=0.001, max_value=0.1, step=0.001, value=0.01, help="Step size for IVP solver.")

# Calculate solutions
y_bvp, u_numeric_bvp = solve_couette_poiseuille(P, N)
u_analytic = analytical_solution(y_bvp, P)
u_explicit, _ = IVP_Explicit(P, h)
u_implicit, _ = IVP_Implicit(P, h)

# Create y array for IVP solutions
y_ivp = np.linspace(0, 1, len(u_explicit))

# Plotting results with enhanced styles
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(y_bvp, u_numeric_bvp, 'o-', label=f'Numerical (BVP), P={P}', color='teal', markersize=3)
ax[0].plot(y_bvp, u_analytic, '-', label='Analytical', color='coral')
ax[0].set_xlabel("y")
ax[0].set_ylabel("u")
ax[0].set_title("BVP Solution")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(y_ivp, u_explicit, label='Explicit Euler (IVP)', color='blue')
ax[1].plot(y_bvp, u_analytic, '-', label='Analytical', color='coral')
ax[1].set_xlabel("y")
ax[1].set_ylabel("u")
ax[1].set_title("Explicit Euler Solution (IVP)")
ax[1].legend()
ax[1].grid(True)

ax[2].plot(y_ivp, u_implicit, label='Implicit Euler (IVP)', color='green')
ax[2].plot(y_bvp, u_analytic, '-', label='Analytical', color='coral')
ax[2].set_xlabel("y")
ax[2].set_ylabel("u")
ax[2].set_title("Implicit Euler Solution (IVP)")
ax[2].legend()
ax[2].grid(True)

# Display the plot
st.pyplot(fig)

# Display maximum error for BVP solution as a metric
max_error_bvp = np.max(np.abs(u_numeric_bvp - u_analytic))
st.markdown("### 📈 Results Summary")
st.metric(label="Maximum Absolute Error (BVP vs Analytical)", value=f"{max_error_bvp:.2e}")
