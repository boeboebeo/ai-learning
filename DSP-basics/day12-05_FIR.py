import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ----------------------------
# symmetric FIR
# ----------------------------

b = np.array([0.2,0.5,1.0,0.5,0.2])

fig, ax = plt.subplots(figsize=(7,7))
plt.subplots_adjust(bottom=0.22)

ax.set_aspect("equal")
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.grid(True)

circle=plt.Circle((0,0),1,fill=False,color='gray')
ax.add_patch(circle)

slider_ax=plt.axes([0.2,0.08,0.6,0.03])

slider=Slider(
    slider_ax,
    "ω",
    0,
    np.pi,
    valinit=0
)

def draw(omega):

    ax.clear()

    ax.set_aspect("equal")
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.grid(True)

    circle=plt.Circle((0,0),1,fill=False,color='gray')
    ax.add_patch(circle)

    total=0+0j

    # 시작점
    x=0
    y=0

    for k,c in enumerate(b):

        v = c*np.exp(-1j*k*omega)

        # head-to-tail
        ax.arrow(
            x,
            y,
            np.real(v),
            np.imag(v),
            width=0.015,
            color="blue",
            length_includes_head=True
        )

        x += np.real(v)
        y += np.imag(v)

        total += v

    # 최종 벡터(H)
    ax.arrow(
        0,
        0,
        np.real(total),
        np.imag(total),
        color="red",
        width=0.03,
        length_includes_head=True
    )

    ax.set_title(
f"""ω = {omega:.2f}

Magnitude = {abs(total):.3f}

Phase = {np.degrees(np.angle(total)):.2f}°
"""
    )

slider.on_changed(draw)

draw(0)

plt.show()