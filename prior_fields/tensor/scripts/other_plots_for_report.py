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

trafo_kwargs = {
    "bins": np.linspace(-np.pi / 2, np.pi / 2, 100),
    "density": True,
    "color": "#009682",
}
sample_kwargs = {"bins": np.linspace(-25, 25, 100), "density": True, "color": "#4664aa"}

# %%
fig, ax = plt.subplots(2, 3, figsize=(10, 6))

mean = 0
sigma = params.sigma.mean()
s = np.random.normal(mean, sigma, int(1e7))
ax[0][0].hist(s, label=f"N({mean}, {sigma:.2f})", **sample_kwargs)
ax[1][0].hist(sample_to_angles(s), **trafo_kwargs)

mean = 0
sigma = params.sigma.max()
s = np.random.normal(mean, sigma, int(1e7))
ax[0][1].hist(s, label=f"N({mean}, {sigma:.2f})", **sample_kwargs)
ax[1][1].hist(sample_to_angles(s), **trafo_kwargs)

mean = params.mean.max()
sigma = params.sigma[np.argmax(params.mean)]
s = np.random.normal(mean, sigma, int(1e7))
ax[0][2].hist(s, label=f"N({mean:.2f}, {sigma:.2f})", **sample_kwargs)
ax[1][2].hist(sample_to_angles(s), **trafo_kwargs)

for axis in ax[0]:
    axis.set_xlim(-25, 25)
    axis.legend(loc="upper left")

plt.tight_layout()
plt.savefig("figures/other/distribution_after_trafo.png")

plt.show()

# %%
