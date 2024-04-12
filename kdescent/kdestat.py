from functools import partial

import jax.random
import jax.numpy as jnp

from .keygen import KeyGenerator


class KDEStat:
    def __init__(self, training_x, training_y=None, num_kernels=20,
                 bandwidth_factor=1.0, randkey=0, comm=None,
                 _kernel_inds=None):
        self.training_x = jnp.atleast_2d(jnp.asarray(training_x).T).T
        assert self.training_x.ndim == 2, "x must have shape (ndata, ndim)"
        self.training_y = None
        if training_y is not None:
            self.training_y = jnp.asarray(training_y)
            assert self.training_x.ndim == 1, "y must have shape (ndata,)"
            assert self.training_y.shape[0] == self.training_x.shape[0]
        self.comm = comm
        self.num_kernels = num_kernels
        self.ndim = self.training_x.shape[1]
        self.bandwidth_factor = bandwidth_factor
        self.bandwidth = self._set_bandwidth(self.bandwidth_factor)
        self.kernelcov = self._bandwidth_to_kernelcov(self.bandwidth)
        self.keygen = KeyGenerator(randkey)
        self.kernel_inds = _kernel_inds

    def calculate(self, x, y=None, randkey=None):
        """Realize kernel centers and return the KDE statistics

        Parameters
        ----------
        x : array-like
            Model data of shape (n_model_data, n_features)
        y : array-like, optional
            Supplementary model data of shape (n_model_data,). If supplied,
            function will return sum(y * weights) instead of sum(weights)

        Returns
        -------
        prediction : jnp.ndarray
            KDE stat measured on `x`. Has shape (num_kernels,)
        truth : jnp.ndarray
            KDE stat measured on `training_x`. This is always different
            due to the random kernel placements. Has shape (num_kernels,)
        """
        self.realize_kernels(randkey)
        prediction = self.calc_realized_kdestat(x, y)
        truth = self.calc_realized_training_kdestat()
        return prediction, truth

    def initialized(self):
        # return hasattr(self.kernel_inds, "is_ready")
        return self.kernel_inds is not None

    def realize_kernels(self, randkey=None):
        if randkey is None:
            self.keygen = self.keygen.with_newkey()
            randkey = self.keygen.randkey
        if self.comm is None:
            self.kernel_inds = _sample_kernel_inds(
                self.num_kernels, self.training_x, randkey
            )
        else:
            kernel_inds = []
            if not self.comm.rank:
                kernel_inds = _sample_kernel_inds(
                    self.num_kernels, self.training_x, randkey
                )
            self.kernel_inds = self.comm.bcast(kernel_inds, root=0)

    def get_realized_weights(self, x):
        assert self.initialized(), "First call realize_kernels()"
        return _get_weights(
            x, self.training_x, self.kernelcov, self.kernel_inds)

    def calc_realized_kdestat(self, x, y=None):
        assert self.initialized()
        return _predict_kdestat(
            x, y, self.training_x, self.kernelcov, self.kernel_inds)

    def calc_realized_training_kdestat(self):
        return self.calc_realized_kdestat(self.training_x, self.training_y)

    def _set_bandwidth(self, bandwidth_factor):
        """Scott's rule bandwidth... multiplied by any factor you want!"""
        n = self.num_kernels
        d = self.training_x.shape[1]
        return n ** (-1.0 / (d + 4)) * bandwidth_factor

    def _bandwidth_to_kernelcov(self, bandwidth):
        """
        Scale bandwidth by the empirical covariance matrix. This way we
        don't have to perform a PC transform for every single iteration.
        """
        empirical_cov = jnp.cov(self.training_x, rowvar=False)
        return empirical_cov * bandwidth**2


@partial(jax.jit, static_argnums=[0])
def _sample_kernel_inds(num_kernels, training_x, randkey):
    inds = jax.random.randint(
        randkey, (num_kernels,), 0, len(training_x))
    return inds


@jax.jit
def _weights_in_kernel(x, training_x, cov, kernel_ind):
    x0 = training_x[kernel_ind, :]
    return jax.scipy.stats.multivariate_normal.pdf(
        x, mean=x0, cov=cov)


_vmap_weights_in_kernel = jax.jit(jax.vmap(
    _weights_in_kernel, in_axes=(None, None, None, 0)))


@jax.jit
def _get_weights(x, training_x, cov, kernel_inds):
    # ind_weights = [_weights_in_kernel(x, training_x, cov, ind)
    #                for ind in kernel_inds]
    ind_weights = _vmap_weights_in_kernel(x, training_x, cov, kernel_inds)
    return jnp.asarray(ind_weights)


@jax.jit
def _predict_kdestat_from_weights(y, weights):
    if y is None:
        # Predict a weighted histogram
        return jnp.sum(weights, axis=1)
    else:
        # Parallelizable stat but not meaningful on its own
        # To predict average y value, divide by sum of all weights
        # *after* summing over all MPI ranks, which is not automatic
        return jnp.sum(y[:, None] * weights, axis=1)


@jax.jit
def _predict_kdestat(x, y, training_x, cov, kernel_inds):
    weights = _get_weights(x, training_x, cov, kernel_inds)
    return _predict_kdestat_from_weights(y, weights)
