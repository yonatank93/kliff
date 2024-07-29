from pathlib import Path

import numpy as np

from kliff.calculators import Calculator
from kliff.dataset import Dataset
from kliff.dataset.weight import Weight
from kliff.loss import Loss
from kliff.models import KIMModel
from kliff.uq.fim import FIM

seed = 1717
np.random.seed(seed)

# model
modelname = "SW_StillingerWeber_1985_Si__MO_405512056662_006"
model = KIMModel(modelname)
model.set_opt_params(A=[["default"]], sigma=[["default"]])

# training set
path = Path(__file__).absolute().parents[1].joinpath("test_data/configs/Si_4")
data = Dataset(path)
configs = data.get_configs()

# calculator
calc = Calculator(model)
ca = calc.create(configs)
nparams = calc.get_num_opt_params()

# loss
loss = Loss(calc)


fim_class = FIM(loss)
params1 = calc.get_opt_params()
fim_array1 = fim_class.run(params1)


def test_FIM_shape():
    """A simple check of the shape of the FIM. It should be a N by N matrix, where N is
    the number of parameters.
    """
    # Check the shape of the FIM
    assert fim_array1.shape == (nparams, nparams), "The FIM has incorrect shape"


def test_FIM_value():
    """Check the computed FIM values.

    The FIM values depends not only on the parameter values, but also on the data points
    used in the calculation. Unfortunately, usually we use a lot of data points. However,
    we know how the energy and forces values depend on parameter A. In particular, we know
    that parameter A participate linearly in the energy, and thus forces, equation. And
    the FIM calculation for least-squares problem only involves taking first-derivative.
    We will use this knowledge to check if the FIM behaves as expected if we scale A.

    The (0, 0) element of the FIM only involves the derivative wrt A, and so the scale of
    A won't affect this factor. The off-diagonal elements only involve 1 derivative wrt A
    and the other derivative will be scaled accordingly with the scale of A. The (1, 1)
    element of the FIM only involves the derivative wrt the other parameter, which each
    term is scaled with A.
    """
    # Scale the parameters
    params2 = params1.copy()
    scale = np.random.randn() ** 2  # non-negative multiplicative scale of A
    params2[0] *= scale  # Scale A
    # Compute the FIM at these parameters
    fim_array2 = fim_class.run(params2)
    # The element-wise ratio of fim2/fim1 should be this array
    target_ratio = np.array([[1, scale], [scale, scale**2]])

    assert np.allclose(
        fim_array2 / fim_array1, target_ratio
    ), "FIM calculation doesn't pass the value check"


def test_FIM_weights():
    """Test the behavior of the FIM wrt weights

    For least-squares case, the FIM involves taking the first-dervative of the residuals,
    which includes the weights, and then taking the dot product of the first-derivative
    matrix with itself. So, if we scale the weights uniformly for all data points, the
    FIM should change as the square of the scale.
    """
    # Change the weight uniformly by scaling all the configurations equally
    scale3 = np.random.randn() ** 2
    weight = Weight(config_weight=scale3)
    data3 = Dataset(path, weight=weight)  # Set the weights
    configs3 = data3.get_configs()
    calc3 = Calculator(model)
    _ = calc3.create(configs3)
    loss3 = Loss(calc3)
    fim_class3 = FIM(loss3)
    fim_array3 = fim_class3.run(params1)

    # The element-wise ratio of fim3/fim1 should be scale^2
    assert np.allclose(
        fim_array3 / fim_array1, scale3**2
    ), "FIM calculation doesn't handle weights correctly"


def test_FIM_parallel():
    """Test parallelization of FIM calculation.

    We only test if the parallelization works, i.e., not throwing an error, and if the
    result is the same as if we do serial calculation.
    """
    # Parallel FIM calculation
    fim_class4 = FIM(loss, nprocs=2)
    fim_array4 = fim_class4.run(params1)

    # Compare
    assert np.all(fim_array4 == fim_array4), "FIM parallel calculation is broken"
