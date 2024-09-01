from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1d = Annotated[npt.NDArray[DType], Literal["N"]]
Array2d = Annotated[npt.NDArray[DType], Literal["N", "N"]]
ArrayNx2 = Annotated[npt.NDArray[DType], Literal["N", "2"]]
ArrayNx3 = Annotated[npt.NDArray[DType], Literal["N", "3"]]
