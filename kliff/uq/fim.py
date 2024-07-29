import copy

import numpy as np
from loguru import logger

from kliff import parallel

try:
    import numdifftools as nd
except ImportError as e:
    raise ImportError(
        '{}\nFisher information analyzer needs "numdifftools". Please install '
        "it first.".format(str(e))
    )


class FIM:
    """Fisher information matrix.

    Compute the Fisher information according to

    ..math::
        I_{ij} = \sum_m w_m^2 \frac{\partial f_m}{\partial \theta_i}
        \cdot \frac{\partial f_m}{\partial \theta_j}

    where :math:`f_m` are the m-th potential predictions of energy, forces, or stress,
    with the corresponding weights in the residual :math:`w_m`, and :math:`\theta_i` is
    the i-th model parameter.
    Derivatives are computed numerically using Ridders' algorithm:
    https://en.wikipedia.org/wiki/Ridders%27_method

    Parameters
    ----------
    loss: kliff.Loss
        Loss function object.
    nprocs: int
        Number of parallel processes to use. Currently this parameter is doing nothing.
    nd_kwargs: dict
        Additional keyword arguments for ``numdifftools.Jacobian``.
    """

    def __init__(self, loss, nprocs=1, nd_kwargs={}):
        self.loss = loss
        self.calculator = self.loss.calculator
        self.nprocs = nprocs
        self.nd_kwargs = nd_kwargs

    def _compute_residual_one_config(self, params, ca):
        """
        Compute residual using a specific set of model parameters.

        Parameters
        ----------
        params: list
          the parameter values

        ca: object
            `compute argument` associated with one configuration

        Return
        ------
        residual: 1D array
            the residual of this configuration, reference data - potential predictions
        """
        # Update parameters
        self.calculator.update_model_params(params)
        residual = self.loss._get_residual_single_config(
            ca, self.calculator, self.loss.residual_fn, self.loss.residual_data
        )
        return residual

    def _compute_jacobian_one_config(self, params, ca):
        """
        Compute the Jacobian of the residual w.r.t. parameters for one configuration.

        Parameters
        ----------
        params: list
          the parameter values

        ca: object
            `compute argument` associated with one configuration.
        """

        # compute Jacobian of forces w.r.t. parameters
        Jfunc = nd.Jacobian(self._compute_residual_one_config, **self.nd_kwargs)
        j = Jfunc(copy.deepcopy(params), ca)

        # restore params back
        self.calculator.update_model_params(params)

        return j

    def _compute_fim_one_config(self, params, ca):
        """
        Compute the FIM for one configuration.

        Parameters
        ----------
        params: list
          the parameter values

        ca: object
            `compute argument` associated with one configuration.
        """
        J = self._compute_jacobian_one_config(params, ca)
        return J.T @ J

    def run(self, params=None):
        """
        Compute the Fisher information matrix and the standard deviation.

        Parameters
        ----------
        params: list
          the parameter values

        Returns
        -------
        I: 2D array, shape(N, N)
            Fisher information matrix, where N is the number of parameters.
        """

        logger.info("Start computing Fisher information matrix.")

        if params is None:
            params = self.calculator.get_opt_params()

        cas = self.calculator.get_compute_arguments()

        if self.nprocs > 1:

            def fim_wrapper(ca, params):
                # Swap the order of parameters
                return self._compute_fim_one_config(params, ca)

            fim_all = parallel.parmap2(
                fim_wrapper,
                cas,
                params,
                nprocs=self.nprocs,
                tuple_X=False,
            )
            fim_all = np.array(fim_all)
        else:
            fim_all = []
            for ca in cas:
                fim_one_config = self._compute_fim_one_config(params, ca)
                fim_all.append(fim_one_config)

        fim = np.sum(fim_all, axis=0)
        logger.info("Finish computing Fisher information matrix.")

        return fim

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    # def _get_residual(self, x):
    #     """
    #     Compute the residual in serial or multiprocessing mode.

    #     This is a callable for optimizing method in scipy.optimize.least_squares,
    #     which is passed as the first positional argument.

    #     Args:
    #         x: optimizing parameter values, 1D array
    #     """

    #     # publish params x to predictor
    #     self.calculator.update_model_params(x)

    #     cas = self.calculator.get_compute_arguments()

    #     # TODO the if else could be combined
    #     if isinstance(self.calculator, _WrapperCalculator):
    #         X = zip(cas, self.calc_list, self.residual_fn)
    #         if self.nprocs > 1:
    #             residuals = parallel.parmap2(
    #                 self._get_residual_single_config,
    #                 X,
    #                 self.residual_data,
    #                 nprocs=self.nprocs,
    #                 tuple_X=True,
    #             )
    #             residual = np.concatenate(residuals)
    #         else:
    #             residual = []
    #             for ca, calculator, residual_fn in X:
    #                 current_residual = self._get_residual_single_config(
    #                     ca, calculator, residual_fn, self.residual_data
    #                 )
    #                 residual = np.concatenate((residual, current_residual))

    #     else:
    #         if self.nprocs > 1:
    #             residuals = parallel.parmap2(
    #                 self._get_residual_single_config,
    #                 cas,
    #                 self.calculator,
    #                 self.residual_fn,
    #                 self.residual_data,
    #                 nprocs=self.nprocs,
    #                 tuple_X=False,
    #             )
    #             residual = np.concatenate(residuals)
    #         else:
    #             residual = []
    #             for ca in cas:
    #                 current_residual = self._get_residual_single_config(
    #                     ca, self.calculator, self.residual_fn, self.residual_data
    #                 )
    #                 residual = np.concatenate((residual, current_residual))

    #     return residual
