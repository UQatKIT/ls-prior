# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

from prior_fields.tensor.parameters import Geometry, PriorParameters
from prior_fields.tensor.transformer import angles_to_sample, sample_to_angles

###############################################################
# Sigmoid and inverse transformations
# %%
n = 1000

x_sample = np.linspace(-21, 21, n)
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

s_lim = 20.5
ax[0].set_xlim(-s_lim, s_lim)
ax[1].set_ylim(-s_lim, s_lim)

s_ticks = np.arange(-20, 21, 5)
ax[0].set_xticks(s_ticks)
ax[1].set_yticks(s_ticks)

plt.savefig("figures/other/sigmoid_trafo_and_inverse.png")
plt.show()


###############################################################
# Motivation of the choice of transformation
# %%
geometry = Geometry(6)
params = PriorParameters.load(Path(f"data/parameters/params_{geometry.value}.npy"))

# %%
s = np.random.normal(0, params.sigma.mean(), int(1e7))
tanh_s = (np.pi / 2) * np.tanh(s)
sigmoid_s = sample_to_angles(s)

# %%
kwargs = {
    "bins": np.linspace(-np.pi / 2, np.pi / 2, 100),
    "density": True,
    "histtype": "step",
}

plt.figure(figsize=(8, 5))

plt.hist(tanh_s, label="tanh(s)", **kwargs)
plt.hist(sigmoid_s, label="sigmoid(s)", **kwargs)

plt.vlines([-np.pi / 2, np.pi / 2], 0, 0.6, lw=1, color="black")

plt.legend()
plt.show()

print(f"Max tanh(s): {abs(tanh_s).max():.4f}")
print(f"Max sigmoid(s): {abs(sigmoid_s).max():.4f}")


# %%
plt.figure(figsize=(8, 5))

x = np.linspace(-15, 15, 1000)
plt.plot(x, (np.pi / 2) * np.tanh(x), label="tanh(x)")
plt.plot(x, sample_to_angles(x), label="sigmoid(x)")

plt.legend()
plt.show()

# %%
