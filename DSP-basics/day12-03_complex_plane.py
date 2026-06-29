import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(6,6))

ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,2.5)
ax.set_aspect('equal')
ax.grid(True)

circle = plt.Circle((0,0),1,fill=False,color='gray')
ax.add_patch(circle)

# 화살표 저장용
arrow1 = None
arrow2 = None
arrow3 = None

text=ax.text(-2.4,2.2,"")

def update(theta):

    global arrow1,arrow2,arrow3

    if arrow1 is not None:
        arrow1.remove()

    if arrow2 is not None:
        arrow2.remove()

    if arrow3 is not None:
        arrow3.remove()

    v1=np.exp(1j*theta)
    v2=np.exp(-1j*theta)

    s=v1+v2

    arrow1=ax.arrow(
        0,0,
        np.real(v1),
        np.imag(v1),
        color='red',
        width=0.01,
        length_includes_head=True
    )

    arrow2=ax.arrow(
        0,0,
        np.real(v2),
        np.imag(v2),
        color='blue',
        width=0.01,
        length_includes_head=True
    )

    arrow3=ax.arrow(
        0,0,
        np.real(s),
        np.imag(s),
        color='green',
        width=0.02,
        length_includes_head=True
    )

    text.set_text(
f"""
θ = {theta:.2f} rad

Red   = e^(jθ)

Blue  = e^(-jθ)

Green = Sum

Real = {np.real(s):.3f}

Imag = {np.imag(s):.6f}
"""
    )

ani=FuncAnimation(
    fig,
    update,
    frames=np.linspace(0,2*np.pi,250),
    interval=40,
    repeat=True
)

plt.show()