from pathlib import Path

import numpy as np
import pytest

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.uq import MCMC, EmceeSampler, PtemceeSampler, get_T0

# model
modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
model = KIMModel(modelname)
model.set_opt_params(A=[["default"]])

# training set
path = Path(__file__).parents[1].joinpath("configs_extxyz/Si_4")
data = Dataset(path)
configs = data.get_configs()

# calculator
calc = Calculator(model)
ca = calc.create(configs)

# loss
loss = Loss(calc)
loss.minimize(method="lm")

# dimensionality
ndim = calc.get_num_opt_params()
nwalkers = 2 * np.random.randint(1, 3)
ntemps = np.random.randint(2, 4)
nsteps = np.random.randint(1, 5)

# samplers
prior_bounds = np.tile([0, 10], (ndim, 1))
ptemcee_avail = True
emcee_avail = True
try:
    ptsampler = MCMC(
        loss, ntemps=ntemps, nwalkers=nwalkers, logprior_args=(prior_bounds,)
    )
except ImportError:
    ptemcee_avail = False

try:
    sampler = MCMC(
        loss, nwalkers=nwalkers, logprior_args=(prior_bounds,), use_ptsampler=False
    )
except ImportError:
    emcee_avail = False


def test_T0():
    """Test if the function to compute T0 works properly. This is done by comparing T0
    computed using the internal function and computed manually.
    """
    # Using internal function
    T0_internal = get_T0(loss)
    # Compute manually
    xopt = calc.get_opt_params()
    T0_manual = 2 * loss._get_loss(xopt) / len(xopt)
    assert T0_internal == T0_manual, "Internal function to compute T0 doesn't work"


def test_MCMC_wrapper():
    """Test if the MCMC wrapper class returns the correct sampler instance."""
    if ptemcee_avail:
        assert (
            type(ptsampler) == PtemceeSampler
        ), "MCMC should return ``PtemceeSampler`` instance"
    if emcee_avail:
        assert (
            type(sampler) == EmceeSampler
        ), "MCMC should return ``EmceeSampler`` instance"


def test_dimensionality():
    """Test the number of temperatures, walkers, steps, and parameters. This is done by
    comparing the shape of the resulting MCMC chains and the variables used to set these
    dimensions.
    """

    # Test for ptemcee wrapper
    if ptemcee_avail:
        p0 = np.random.uniform(0, 10, (ntemps, nwalkers, ndim))
        ptsampler.run_mcmc(p0=p0, iterations=nsteps)
        assert ptsampler.chain.shape == (
            ntemps,
            nwalkers,
            nsteps,
            ndim,
        ), "Dimensionality from the ptemcee wrapper is not right"

    # Test for emcee wrapper
    if emcee_avail:
        p0 = np.random.uniform(0, 10, (nwalkers, ndim))
        sampler.run_mcmc(initial_state=p0, nsteps=nsteps)
        assert sampler.chain.shape == (
            nwalkers,
            nsteps,
            ndim,
        ), "Dimensionality from the emcee wrapper is not right"


if __name__ == "__main__":
    test_T0()
    test_MCMC_wrapper()
    test_dimensionality()