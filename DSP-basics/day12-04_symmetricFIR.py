import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------------
# Symmetric FIR coefficients
# ----------------------------------------

b = np.array([0.2, 0.5, 1.0, 0.5, 0.2])

fig, ax = plt.subplots(figsize=(7,7))

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect("equal")
ax.grid(True)

circle = plt.Circle((0,0),1,fill=False,color='gray')
ax.add_patch(circle)

arrows=[]
result_arrow=None
text=ax.text(-2.9,2.6,"",fontsize=10)

# ----------------------------------------

def update(theta):

    global arrows,result_arrow

    for a in arrows:
        a.remove()

    arrows=[]

    if result_arrow is not None:
        result_arrow.remove()

    total=0+0j

    # Draw every FIR term
    for k,c in enumerate(b):

        v = c*np.exp(-1j*k*theta)

        total += v

        arr=ax.arrow(
            0,
            0,
            np.real(v),
            np.imag(v),
            width=0.01,
            alpha=0.7,
            length_includes_head=True,
            label=f"k={k}"
        )

        arrows.append(arr)

    # Sum vector
    result_arrow=ax.arrow(
        0,
        0,
        np.real(total),
        np.imag(total),
        color="red",
        width=0.03,
        length_includes_head=True
    )

    phase=np.angle(total)

    text.set_text(
f"""
ω = {theta:.2f}

Magnitude = {abs(total):.3f}

Phase = {np.degrees(phase):.2f} deg
"""
)

ani=FuncAnimation(
    fig,
    update,
    frames=np.linspace(0,np.pi,250),
    interval=40,
    repeat=True
)

plt.show()