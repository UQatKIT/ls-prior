from typing import overload

import numpy as np

from prior_fields.prior.dtypes import Array1d


def get_sigma_from_kappa_and_tau(kappa: float, tau: float) -> float:
    """
    Compute marginal standard deviation of a stationary BiLaplacianPrior parameterized
    with :math:`\\kappa` and :math:`\\tau`.

    Parameters
    ----------
    kappa : float
    tau : float

    Returns
    -------
    float
        Marginal standard deviation.
    """
    return 1 / (2 * np.sqrt(np.pi) * kappa * tau)


@overload
def get_kappa_from_ell(ell: float) -> float: ...


@overload
def get_kappa_from_ell(ell: Array1d) -> Array1d: ...


def get_kappa_from_ell(ell: float | Array1d) -> float | Array1d:
    """Get scaling parameter :math:`\\kappa` from correlation length :math:`\\ell`."""
    return 1 / ell


@overload
def get_tau_from_sigma_and_ell(sigma: float, ell: float) -> float: ...


@overload
def get_tau_from_sigma_and_ell(sigma: Array1d, ell: Array1d) -> Array1d: ...


def get_tau_from_sigma_and_ell(
    sigma: float | Array1d, ell: float | Array1d
) -> float | Array1d:
    """
    Get :math:`\\tau` from marginal standard deviation :math:`\\sigma` and correlation
    length :math:`\\ell`.

    Notes
    -----
    This transformation is valid for the bi-Laplacian case only.
    """
    return ell / (2 * np.sqrt(np.pi) * sigma)
