# %%
from prior_fields.tensor.fiber_grid import FiberGrid, FiberGridComputer
from prior_fields.tensor.preprocessing import collect_data_from_human_atrial_fiber_meshes

# %%
# takes about 2.5 minutes
uac, fiber_angles, tags = collect_data_from_human_atrial_fiber_meshes()

# %%
# takes about 30 seconds
FiberGridComputer(
    uac=uac,
    fiber_angles=fiber_angles,
    anatomical_structure_tags=tags,
    max_depth=7,
    point_threshold=100,
).get_fiber_grid().save()

# %%
fiber_grid = FiberGrid.read_from_binary_file(
    "data/LGE-MRI-based/fiber_grid_max_depth7_point_threshold100.npy"
)
fiber_grid.plot("tag")
fiber_grid.plot("mean")
fiber_grid.plot("std")

# %%
