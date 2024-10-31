from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from loguru import logger

from prior_fields.prior.dtypes import Array1d


class Geometry(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    ATLAS = "A"
    UNKNOWN = "unknown"


@dataclass
class PriorParameters:
    geometry: Geometry
    mean: Array1d
    sigma: Array1d
    ell: Array1d

    @classmethod
    def load(cls, file: Path) -> PriorParameters:
        """
        Read 'PriorParameters' from .npy file.

        Parameters
        ----------
        file : Path | str, optional
            Path to binary file.

        Returns
        -------
        PriorParameters
        """
        logger.info(f"Load parameters from {file}.")
        data = np.load(file)

        # Extract geometry from file name
        pattern = r"params_([1234567A])\.npy"
        matched = re.search(pattern, str(file))
        geometry: Geometry
        if matched is None:
            geometry = Geometry.UNKNOWN
        else:
            g = matched.group(1)
            if g == "A":
                geometry = Geometry.ATLAS
            elif g.isnumeric():
                geometry = Geometry(int(g))

        return PriorParameters(
            geometry=geometry, mean=data[0], sigma=data[1], ell=data[2]
        )

    def save(self, path: Path = Path("data/parameters/")) -> None:
        """
        Write parameters for `BiLaplacianPriorNumpyWrapper` to binary file.

        Parameters
        ----------
        path : Path, optional
            Path to which the parameters are saved, defaults to 'data/parameters/'.
        """
        Path.mkdir(path, exist_ok=True)
        file = path / f"params_{self.geometry.value}.npy"
        logger.info(f"Saving collected data to {file}.")
        np.save(file, np.vstack([self.mean, self.sigma, self.ell]))
