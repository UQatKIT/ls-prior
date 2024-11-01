# %%
import matplotlib.pyplot as plt
import numpy as np

from prior_fields.tensor.transformer import angles_to_sample, sample_to_angles

# %%
n = 1000

x_sample = np.linspace(-9, 9, n)
y_angles = sample_to_angles(x_sample)

x_angles = np.linspace(-np.pi / 2, np.pi / 2, n)
y_sample = angles_to_sample(x_angles)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9, 3))

ax[0].plot(x_sample, y_angles, color="#009682")
ax[1].plot(x_angles, y_sample, color="#009682")

for axis in ax:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

ax[0].margins(x=0)
ax[1].margins(y=0)

ax[0].set_xlabel("s", loc="right")
ax[1].set_ylabel("s", loc="top", rotation=0)

theta_label = r"$\theta$"
ax[0].set_ylabel(theta_label, loc="top", rotation=0)
ax[1].set_xlabel(theta_label, loc="right")

plt.tight_layout()

s_lim = 8.5
ax[0].set_xlim(-s_lim, s_lim)
ax[1].set_ylim(-s_lim, s_lim)

s_ticks = np.arange(-8, 9, 2)
ax[0].set_xticks(s_ticks)
ax[1].set_yticks(s_ticks)

plt.savefig("figures/other/sigmoid_trafo_and_inverse.png")
plt.show()

# %%
