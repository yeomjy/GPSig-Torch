from typing import Optional, Tuple


import numpy as np
import torch
from gpytorch import constraints
from gpytorch import priors
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import Interval
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from gpsig import lags, low_rank_calculations, signature_algs


class SignatureKernel(Kernel):
    def __init__(
        self,
        input_dim,
        num_features,
        num_levels,
        # kwargs
        order=1,
        variances=1,
        normalization=True,
        difference=True,
        num_lags=None,
        jitter: float = 1e-6,
        float_dtype=torch.float64,
        int_dtype=torch.int32,
        # low-rank options
        low_rank=False,
        num_components=50,
        rank_bound=None,
        sparsity="sqrt",
        # Kernel kwargs
        active_dims=None,
        batch_shape: Optional[torch.Size] = torch.Size([]),
        ard_num_dims: Optional[int] = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
    ):
        """
        # Inputs:
        ## Args:
        :input_dim:         the total size of an input sample to the kernel
        :num_features:      the state-space dimension of the input sequebces,
        :num_levels:        the degree of cut-off for the truncated signatures
                            (i.e. len_examples = len(active_dims) / num_features)
        ## Kwargs:
        ### Kernel options
        :active_dims:       if specified, should contain a list of the dimensions in the input that should be sliced out and fed to the kernel.
                            if not specified, defaults to range(input_dim)
        # use scalekernel at gpytorch
        :variances:          multiplicative scaling applied to the Signature kernel,
                            if ARD is True, there is one parameter for each level, i.e. variances is of size (num_levels + 1)
        :ard_num_dims:      lengthscales for scaling the coordinates of the
        input sequences,
                            if lengthscales is None, no scaling is applied to the paths
                            if ARD is True, there is one lengthscale for each path dimension, i.e. lengthscales is of size (num_features)
        :order:             order of the signature kernel minimum is 1 and maximum is num_levels (set to -1 to set to max)
        :normalization:     False - no normalization, True - normalize signature levels
        :difference:        boolean indicating whether to difference paths
                            False corresponds to the signature of the integrated path
        :num_lags:          Nonnegative integer or None, the number of lags added to each sequence. Usually between 0-5.

        ### Low-rank options:
        :low_rank:          boolean indicating whether to use low-rank kernel
        :num_components:    number of components used in Nystrom approximation
        :rank_bound:        max rank of low-rank factor in signature algs, if None, defaults to num_components.
        :sparsity:          controls the sparsity of the random projection matrix used in the low-rank algorithm
                            possible values are:
                            - 'sqrt' - approximately O(n * sqrt(n)) non-zero entries;
                            - 'log' - approximately O(n * log(n)) non-zero entries;
                            - 'lin' - approximately O(n) non-zero entries;
        """

        # super().__init__(input_dim, active_dims, name=name)
        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )
        self.num_features = num_features
        self.num_levels = num_levels
        self.len_examples = self._validate_number_of_features(
            input_dim, num_features
        )
        self.order = (
            num_levels if (order <= 0 or order >= num_levels) else order
        )
        self.jitter = jitter
        self.float_type = float_dtype
        self.int_dtype = int_dtype

        if self.order != 1 and low_rank:
            raise NotImplementedError(
                "Higher-order algorithms not compatible with low-rank mode ("
                "yet)."
            )

        self.normalization = normalization
        self.difference = difference

        self.variances = nn.Parameter(variances * torch.ones(num_levels + 1))
        self.register_constraint("variances", constraints.Positive())

        # self.sigma = Parameter(
        #     1.0, transform=transforms.positive, dtype=settings.float_type
        # )
        self.sigma = nn.Parameter(torch.tensor(1.0, dtype=self.dtype))
        self.register_constraint("sigma", constraints.Positive())

        (
            self.low_rank,
            self.num_components,
            self.rank_bound,
            self.sparsity,
        ) = self._validate_low_rank_params(
            low_rank, num_components, rank_bound, sparsity
        )

        if num_lags is None:
            self.num_lags = 0
        else:
            # check if right value
            if not isinstance(num_lags, int) or num_lags < 0:
                raise ValueError(
                    "The variable num_lags most be a nonnegative integer or "
                    "None."
                )
            else:
                self.num_lags = int(num_lags)
                if num_lags > 0:
                    # self.lags = Parameter(
                    #     0.1 * np.asarray(range(1, num_lags + 1)),
                    #     transform=transforms.Logistic(),
                    #     dtype=settings.float_type,
                    # )
                    # transform is handled by @property(def lags())
                    self._lags = nn.Parameter(
                        0.1
                        * torch.arange(1, num_lags + 1, dtype=self.float_type)
                    )

                    gamma = 1.0 / torch.arange(
                        1, self.num_lags + 2, dtype=self.float_type
                    )
                    gamma /= torch.sum(gamma)
                    self.gamma = nn.Parameter(
                        gamma,
                    )
                    self.register_constraint("gamma", constraints.Positive())

        # ard_num_dims are handled in base class (Kernel)
        # if ard_num_dims is not None:
        #     ard_num_dims = self._validate_signature_param(
        #         "lengthscales", ard_num_dims, self.num_features
        #     )
        #     self.lengthscales = Parameter(
        #         ard_num_dims,
        #         transform=transforms.positive,
        #         dtype=settings.float_type,
        #     )
        # else:
        #     self.lengthscales = None

    ######################
    ## Input validators ##
    ######################

    @property
    def lags(self):
        if self.num_lags > 0:
            return F.sigmoid(self._lags)
        else:
            return None

    @lags.setter
    def lags(self, value):
        if self.num_lags > 0:
            assert isinstance(self._lags, nn.Parameter)
            self._lags.data.copy_(value)
        else:
            raise RuntimeError("Kernel has no lags")

    def _validate_number_of_features(self, input_dim, num_features):
        """
        Validates the format of the input samples.
        """
        if input_dim % num_features == 0:
            len_examples = int(input_dim / num_features)
        else:
            raise ValueError(
                "The arguments num_features and input_dim are not consistent."
            )
        return len_examples

    def _validate_low_rank_params(
        self, low_rank, num_components, rank_bound, sparsity
    ):
        """
        Validates the low-rank options
        """
        if low_rank is not None and low_rank == True:
            if not type(low_rank) == bool:
                raise ValueError(
                    "Unknown low-rank argument: %s. It should be True of False."
                    % low_rank
                )
            if sparsity not in ["log", "sqrt", "lin"]:
                raise ValueError(
                    "Unknown sparsity argument %s. Possible values are 'sqrt', 'log', 'lin'"
                    % sparsity
                )
            if rank_bound is not None and rank_bound <= 0:
                raise ValueError(
                    "The rank-bound in the low-rank algorithm must be either None or a positiv integer."
                )
            if num_components is None or num_components <= 0:
                raise ValueError(
                    "The number of components in the kernel approximation must be a positive integer."
                )
            if rank_bound is None:
                rank_bound = num_components
        else:
            low_rank = False
        return low_rank, num_components, rank_bound, sparsity

    def _validate_signature_param(self, name, value, length):
        """
        Validates signature params
        """
        value = value * np.ones(length, dtype=self.dtype)
        correct_shape = () if length == 1 else (length,)
        if np.asarray(value).squeeze().shape != correct_shape:
            raise ValueError(
                "shape of parameter {} is not what is expected ({})".format(
                    name, length
                )
            )
        return value

    ########################################
    ## Autoflow functions for interfacing ##
    ########################################

    def compute_K(self, X, Y):
        return self.K(X, Y)

    def compute_K_symm(self, X):
        return self.K(X)

    def compute_base_kern_symm(self, X: Tensor):
        # num_examples = tf.shape(X)[0]
        num_examples = X.size(0)
        # X = tf.reshape(X, (num_examples, -1, self.num_features))
        X = X.reshape(num_examples, -1, self.num_features)
        # len_examples = tf.shape(X)[1]
        len_examples = X.size(1)
        # X = tf.reshape(
        #     self._apply_scaling_and_lags_to_sequences(X),
        #     (-1, self.num_features),
        # )
        X = X.reshape(
            self._apply_scaling_and_lags_to_sequences(X), -1, self.num_features
        )
        # M = tf.transpose(
        #     tf.reshape(
        #         self._base_kern(X),
        #         [num_examples, len_examples, num_examples, len_examples],
        #     ),
        #     [0, 2, 1, 3],
        # )
        M = torch.permute(
            self._base_kern(X).reshape(
                num_examples, len_examples, num_examples, len_examples
            ),
            [0, 2, 1, 3],
        )
        return M

    def compute_K_level_diags(self, X):
        return self.Kdiag(X, return_levels=True)

    def compute_K_levels(self, X, X2):
        return self.K(X, X2, return_levels=True)

    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    def compute_K_tens(self, Z):
        return self.K_tens(Z, return_levels=False)

    def compute_K_tens_vs_seq(self, Z, X):
        return self.K_tens_vs_seq(Z, X, return_levels=False)

    def compute_K_incr_tens(self, Z):
        return self.K_tens(Z, increments=True, return_levels=False)

    def compute_K_incr_tens_vs_seq(self, Z, X):
        return self.K_tens_vs_seq(Z, X, increments=True, return_levels=False)

    def _K_seq_diag(self, X):
        """
        # Input
        :X:             (num_examples, len_examples, num_features) tensor of sequences
        # Output
        :K:             (num_levels+1, num_examples) tensor of (unnormalized) diagonals of signature kernel
        """

        len_examples = X.size(-2)

        M = self._base_kern(X)

        if self.order == 1:
            K_lvls_diag = signature_algs.signature_kern_first_order(
                M, self.num_levels, difference=self.difference
            )
        else:
            K_lvls_diag = signature_algs.signature_kern_higher_order(
                M, self.num_levels, order=self.order, difference=self.difference
            )

        return K_lvls_diag

    def _K_seq(self, X, X2=None):
        """
        # Input
        :X:             (num_examples, len_examples, num_features) tensor of  sequences
        :X2:            (num_examples2, len_examples2, num_features) tensor of sequences
        # Output
        :K:             (num_levels+1, num_examples, num_examples2) tensor of (unnormalized) signature kernel matrices
        """

        num_examples, len_examples, num_features = X.shape[:3]
        num_samples = num_examples * len_examples

        if X2 is not None:
            # num_examples2, len_examples2 = tf.shape(X2)[0], tf.shape(X2)[1]
            num_examples2, len_examples2 = X2.shape[:2]
            num_samples2 = num_examples2 * len_examples2

        if X2 is None:
            X = X.reshape(num_samples, num_features)
            # M = tf.reshape(
            #     self._base_kern(X),
            #     [num_examples, len_examples, num_examples, len_examples],
            # )
            M = self._base_kern(X).reshape(
                num_examples, len_examples, num_examples, len_examples
            )
        else:
            X = X.reshape(num_samples, num_features)
            X2 = X2.reshape(num_samples2, num_features)
            M = self._base_kern(X, X2).reshape(
                num_examples, len_examples, num_examples2, len_examples2
            )
            # X = tf.reshape(X, [num_samples, num_features])
            # X2 = tf.reshape(X2, [num_samples2, num_features])
            # M = tf.reshape(
            #     self._base_kern(X, X2),
            #     [num_examples, len_examples, num_examples2, len_examples2],
            # )

        if self.order == 1:
            K_lvls = signature_algs.signature_kern_first_order(
                M, self.num_levels, difference=self.difference
            )
        else:
            K_lvls = signature_algs.signature_kern_higher_order(
                M, self.num_levels, order=self.order, difference=self.difference
            )

        return K_lvls

    def _K_seq_lr_feat(self, X, nys_samples=None, seeds=None):
        """
        # Input
        :X:                 (num_examples, len_examples, num_features) tensor of sequences
        :nys_samples:       (num_samples, num_features) tensor of samples to use in Nystrom approximation
        :seeds:             (num_levels-1, 2) array of ints for seeding randomized projection matrices
        # Output
        :Phi_lvls:          a (num_levels+1,) list of low-rank factors for each signature level
        """

        num_examples, len_examples, num_features = X.shape[-3:]
        num_samples = num_examples * len_examples

        # X = tf.reshape(X, [num_samples, num_features])
        X = X.reshape(num_samples, num_features)
        X_feat = low_rank_calculations.Nystrom_map(
            X, self._base_kern, nys_samples, self.num_components
        )
        # X_feat = tf.reshape(
        #     X_feat, [num_examples, len_examples, self.num_components]
        # )
        X_feat = X_feat.reshape(num_examples, len_examples, self.num_components)

        if self.order == 1:
            Phi_lvls = signature_algs.signature_kern_first_order_lr_feature(
                X_feat,
                self.num_levels,
                self.rank_bound,
                self.sparsity,
                seeds,
                difference=self.difference,
            )
        else:
            raise NotImplementedError(
                "Low-rank mode not implemented for order higher than 1."
            )

        return Phi_lvls

    def _K_tens(self, Z, increments=False):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors, if not increments
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        # Output
        :K_lvls:        (num_levels+1,) list of (num_tensors, num_tensors) kernel matrices (for each T.A. level)
        """

        len_tensors, num_tensors, num_features = (
            Z.size(0),
            Z.size(1),
            Z.size(-1),
        )

        if increments:
            # Z = tf.reshape(Z, [len_tensors, 2 * num_tensors, num_features])
            Z = Z.reshape(len_tensors, 2 * num_tensors, num_features)
            # M = tf.reshape(
            #     self._base_kern(Z),
            #     [len_tensors, num_tensors, 2, num_tensors, 2],
            # )
            M = self._base_kern(Z).reshape(
                len_tensors, num_tensors, 2, num_tensors, 2
            )
            M = (
                M[:, :, 1, :, 1]
                + M[:, :, 0, :, 0]
                - M[:, :, 1, :, 0]
                - M[:, :, 0, :, 1]
            )
        else:
            M = self._base_kern(Z)

        K_lvls = signature_algs.tensor_kern(M, self.num_levels)

        return K_lvls

    def _K_tens_lr_feat(
        self, Z, increments=False, nys_samples=None, seeds=None
    ):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors, if not increments
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        :nys_samples:   (num_samples, num_features) tensor of samples to use in Nystrom approximation
        :seeds:         (num_levels-1, 2) array of ints for seeding randomized projection matrices
        # Output
        :Phi_lvls:      a (num_levels+1,) list of low-rank factors for cov matrices of inducing tensors on each TA level
        """

        if self.order > 1:
            raise NotImplementedError(
                "higher order not implemented yet for low-rank mode"
            )

        len_tensors, num_tensors, num_features = (
            Z.size(0),
            Z.size(1),
            Z.size(-1),
        )

        if increments:
            # Z = tf.reshape(Z, [num_tensors * len_tensors * 2, num_features])
            Z = Z.reshape(num_tensors * len_tensors * 2, num_features)
            Z_feat = low_rank_calculations.Nystrom_map(
                Z, self._base_kern, nys_samples, self.num_components
            )
            # Z_feat = tf.reshape(
            #     Z_feat, [len_tensors, num_tensors, 2, self.num_components]
            # )
            Z_feat = Z_feat.reshape(
                len_tensors, num_tensors, 2, self.num_components
            )
            Z_feat = Z_feat[:, :, 1, :] - Z_feat[:, :, 0, :]
        else:
            # Z = tf.reshape(Z, [num_tensors * len_tensors, num_features])
            Z = Z.reshape(Z, num_tensors * len_tensors, num_features)
            Z_feat = low_rank_calculations.Nystrom_map(
                Z, self._base_kern, nys_samples, self.num_components
            )
            # Z_feat = tf.reshape(
            #     Z_feat, [len_tensors, num_tensors, self.num_components]
            # )
            Z_feat = Z_feat.reshape(
                len_tensors, num_tensors, self.num_components
            )

        Phi_lvls = signature_algs.tensor_kern_lr_feature(
            Z_feat, self.num_levels, self.rank_bound, self.sparsity, seeds
        )
        return Phi_lvls

    def _K_tens_vs_seq(self, Z, X, increments=False):
        """
        # Input
        :Z:             (num_levels*(num_levels+1)/2, num_tensors, num_features) tensor of inducing tensors, if not increments
                        else (num_levels*(num_levels+1)/2, num_tensors, 2, num_features)
        :X:             (num_examples, len_examples, num_features) tensor of sequences
        Output
        :K_lvls:        (num_levels+1,) list of inducing tensors vs input sequences covariance matrices on each T.A. level
        """

        len_tensors, num_tensors, num_features = (
            Z.size(0),
            Z.size(1),
            Z.size(-1),
        )
        # num_examples, len_examples = tf.shape(X)[-3], tf.shape(X)[-2]
        num_examples, len_examples = X.shape[-3:-1]

        # X = tf.reshape(X, [num_examples * len_examples, num_features])
        X = X.reshape(num_examples * len_examples, num_features)
        if increments:
            # Z = tf.reshape(Z, [2 * num_tensors * len_tensors, num_features])
            Z = Z.reshape(2 * num_tensors * len_tensors, num_features)
            # M = tf.reshape(
            #     self._base_kern(Z, X),
            #     (len_tensors, num_tensors, 2, num_examples, len_examples),
            # )
            M = self._base_kern(Z, X).reshape(
                len_tensors, num_tensors, 2, num_examples, len_examples
            )
            M = M[:, :, 1] - M[:, :, 0]
        else:
            # Z = tf.reshape(Z, [num_tensors * len_tensors, num_features])
            Z = Z.reshape(num_tensors * len_tensors, num_features)
            # M = tf.reshape(
            #     self._base_kern(Z, X),
            #     (len_tensors, num_tensors, num_examples, len_examples),
            # )
            M = self._base_kern(Z, X).reshape(
                len_tensors, num_tensors, num_examples, len_examples
            )

        if self.order == 1:
            K_lvls = signature_algs.signature_kern_tens_vs_seq_first_order(
                M, self.num_levels, difference=self.difference
            )
        else:
            K_lvls = signature_algs.signature_kern_tens_vs_seq_higher_order(
                M, self.num_levels, order=self.order, difference=self.difference
            )

        return K_lvls

    # @params_as_tensors
    def _apply_scaling_and_lags_to_sequences(self, X):
        """
        Applies scaling and lags to sequences.
        """

        # num_examples, len_examples, _ = tf.unstack(tf.shape(X))
        num_examples, len_examples = X.shape[:2]

        num_features = self.num_features * (self.num_lags + 1)

        if self.num_lags > 0:
            X = lags.add_lags_to_sequences(X, self.lags)

        # X = tf.reshape(
        #     X,
        #     (num_examples, len_examples, self.num_lags + 1, self.num_features),
        # )
        X = X.reshape(
            num_examples, len_examples, self.num_lags + 1, self.num_features
        )

        # TODO
        if self.lengthscales is not None:
            X /= self.lengthscales[None, None, None, :]

        if self.num_lags > 0:
            X *= self.gamma[None, None, :, None]

        # X = tf.reshape(X, (num_examples, len_examples, num_features))
        X = X.reshape(num_examples, len_examples, num_features)
        return X

    # @params_as_tensors
    def _apply_scaling_to_tensors(self, Z):
        """
        Applies scaling to simple tensors of shape (num_levels*(num_levels+1)/2, num_tensors, num_features*(num_lags+1))
        """

        # len_tensors, num_tensors = tf.shape(Z)[0], tf.shape(Z)[1]
        len_tensors, num_tensors = Z.shape[:2]

        # TODO
        if self.lengthscales is not None:
            Z = Z.reshape(
                len_tensors,
                num_tensors,
                self.num_lags + 1,
                self.num_features,
            )
            Z /= self.lengthscales[None, None, None, :]
            if self.num_lags > 0:
                Z *= self.gamma[None, None, :, None]
            Z = Z.reshape(len_tensors, num_tensors, -1)

        return Z

    # @params_as_tensors
    def _apply_scaling_to_incremental_tensors(self, Z):
        """
        Applies scaling to incremental tensors of shape (num_levels*(num_levels+1)/2, num_tensors, 2, num_features*(num_lags+1))
        """

        len_tensors, num_tensors, num_features = (
            Z.size(0),
            Z.size(1),
            Z.size(-1),
        )

        # TODO
        if self.lengthscales is not None:
            Z = Z.reshape(
                len_tensors,
                num_tensors,
                2,
                self.num_lags + 1,
                self.num_features,
            )
            Z /= self.lengthscales[None, None, None, None, :]
            if self.num_lags > 0:
                Z *= self.gamma[None, None, None, :, None]

        # Z = tf.reshape(Z, (len_tensors, num_tensors, 2, num_features))
        Z = Z.reshape(len_tensors, num_tensors, 2, num_features)
        return Z

    # @params_as_tensors
    def K(
        self,
        X,
        X2=None,
        presliced=False,
        return_levels=False,
        presliced_X=False,
        presliced_X2=False,
    ):
        """
        Computes signature kernel between sequences
        """

        if presliced:
            presliced_X = True
            presliced_X2 = True

        if not presliced_X and not presliced_X2:
            X, X2 = self._slice(X, X2)
        elif not presliced_X:
            X, _ = self._slice(X, None)
        elif not presliced_X2 and X2 is not None:
            X2, _ = self._slice(X2, None)

        # num_examples = tf.shape(X)[0]
        num_examples = X.size(0)
        X = X.reshape(num_examples, -1, self.num_features)
        len_examples = X.size(1)
        # len_examples = tf.shape(X)[1]

        X_scaled = self._apply_scaling_and_lags_to_sequences(X)

        if X2 is None:
            if self.low_rank:
                Phi_lvls = self._K_seq_lr_feat(X)
                # K_lvls = tf.stack(
                #     [tf.matmul(P, P, transpose_b=True) for P in Phi_lvls],
                #     axis=0,
                # )
                K_lvls = torch.stack(
                    [torch.matmul(P, P.T) for P in Phi_lvls],
                    dim=0,
                )
            else:
                K_lvls = self._K_seq(X_scaled)

            if self.normalization:
                # K_lvls += (
                #     settings.jitter
                #     * tf.eye(num_examples, dtype=settings.float_type)[None]
                # )
                K_lvls = (
                    K_lvls
                    + self.jitter
                    * torch.eye(num_examples, dtype=self.dtype)[None]
                )
                # K_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(K_lvls))
                K_lvls_diag_sqrt = torch.sqrt(torch.diag(K_lvls))
                K_lvls = K_lvls / (
                    K_lvls_diag_sqrt[:, :, None] * K_lvls_diag_sqrt[:, None, :]
                )

        else:
            # num_examples2 = tf.shape(X2)[0]
            num_examples2 = X2.size(0)
            # X2 = tf.reshape(X2, [num_examples2, -1, self.num_features])
            X2 = X2.reshape(num_examples2, -1, self.num_features)
            # len_examples2 = tf.shape(X2)[1]
            len_examples2 = X2.size(1)

            X2_scaled = self._apply_scaling_and_lags_to_sequences(X2)

            if self.low_rank:
                # seeds = torch.distributions.Uniform(0, torch.iinfo.max).sample(
                #     (self.num_levels - 1, 2)
                # )
                # TODO: fix this
                INT_MAX = 1 << 31
                seeds = torch.randint(
                    low=0, high=INT_MAX, size=(self.num_levels - 1, 2)
                )

                # seeds = tf.random_uniform(
                #     (self.num_levels - 1, 2),
                #     minval=0,
                #     maxval=torch.iinfo.max,
                #     dtype=self.int32,
                # )
                idx, _ = low_rank_calculations._draw_indices(
                    num_examples * len_examples + num_examples2 * len_examples2,
                    self.num_components,
                )

                # nys_samples = tf.gather(
                #     tf.concat(
                #         (
                #             tf.reshape(X, [num_examples * len_examples, -1]),
                #             tf.reshape(X2, [num_examples2 * len_examples2, -1]),
                #         ),
                #         axis=0,
                #     ),
                #     idx,
                #     axis=0,
                # )
                nys_samples = torch.gather(
                    torch.cat(
                        [
                            X.reshape(num_examples * len_examples, -1),
                            X2.reshape(num_examples2 * len_examples2, -1),
                        ]
                    ),
                    dim=0,
                    index=idx,
                )

                Phi_lvls = self._K_seq_lr_feat(
                    X, nys_samples=nys_samples, seeds=seeds
                )
                Phi2_lvls = self._K_seq_lr_feat(
                    X2, nys_samples=nys_samples, seeds=seeds
                )

                # K_lvls = tf.stack(
                #     [
                #         tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True)
                #         for i in range(self.num_levels + 1)
                #     ],
                #     axis=0,
                # )
                K_lvls = torch.stack(
                    [
                        torch.matmul(Phi_lvls[i], Phi2_lvls[i].T)
                        for i in range(self.num_levels + 1)
                    ],
                    dim=0,
                )
            else:
                K_lvls = self._K_seq(X_scaled, X2_scaled)

            if self.normalization:
                if self.low_rank:
                    K1_lvls_diag = torch.stack(
                        [
                            # tf.reduce_sum(tf.square(P), axis=-1)
                            torch.sum(torch.square(P), dim=-1)
                            for P in Phi_lvls
                        ],
                        dim=0,
                    )
                    K2_lvls_diag = torch.stack(
                        [
                            # tf.reduce_sum(tf.square(P), axis=-1)
                            torch.sum(torch.square(P), dim=-1)
                            for P in Phi2_lvls
                        ],
                        dim=0,
                    )
                else:
                    K1_lvls_diag = self._K_seq_diag(X_scaled)
                    K2_lvls_diag = self._K_seq_diag(X2_scaled)

                # TODO: settings.jitter
                K1_lvls_diag += self.jitter
                K2_lvls_diag += self.jitter

                K1_lvls_diag_sqrt = K1_lvls_diag.sqrt()
                K2_lvls_diag_sqrt = K2_lvls_diag.sqrt()

                K_lvls /= (
                    K1_lvls_diag_sqrt[:, :, None]
                    * K2_lvls_diag_sqrt[:, None, :]
                )

        K_lvls *= self.sigma * self.variances[:, None, None]

        if return_levels:
            return K_lvls
        else:
            # return tf.reduce_sum(K_lvls, axis=0)
            return torch.sum(K_lvls, dim=0)

    # @params_as_tensors
    def Kdiag(self, X, presliced=False, return_levels=False):
        """
        Computes the diagonal of a square signature kernel matrix.
        """

        # num_examples = tf.shape(X)[0]
        num_examples = X.size(0)

        if self.normalization:
            # TODO: add self.variances
            if return_levels:
                # return tf.tile(
                #     self.sigma * self.variances[:, None], [1, num_examples]
                # )
                return torch.tile(
                    self.sigma * self.variances[:, None], dims=(1, num_examples)
                )
                # return torch.tile(self.sigma, [1, num_examples])
            else:
                # return tf.fill(
                #     (num_examples,), self.sigma * tf.reduce_sum(self.variances)
                # )
                return torch.fill(
                    torch.empty((num_examples,)),
                    self.sigma * torch.sum(self.variances),
                )
                # return torch.fill(self.sigma)

        if not presliced:
            X, _ = self._slice(X, None)

        # X = tf.reshape(X, (num_examples, -1, self.num_features))
        X = X.reshape(num_examples, -1, self.num_features)

        X = self._apply_scaling_and_lags_to_sequences(X)

        if self.low_rank:
            Phi_lvls = self._K_seq_lr_feat(X)
            # K_lvls_diag = tf.stack(
            #     [tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_lvls], axis=0
            # )
            K_lvls_diag = torch.stack(
                [torch.sum(torch.square(P), dim=-1) for P in Phi_lvls],
                dim=0,
            )
        else:
            K_lvls_diag = self._K_seq_diag(X)

        K_lvls_diag *= self.sigma * self.variances[:, None]

        if return_levels:
            return K_lvls_diag
        else:
            # return tf.reduce_sum(K_lvls_diag, axis=0)
            # return K_lvls_diag.sum(dim=0)
            return torch.sum(K_lvls_diag, dim=0)

    # @params_as_tensors
    def K_tens(self, Z, return_levels=False, increments=False):
        """
        Computes a square covariance matrix of inducing tensors Z.
        """

        # num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0]
        len_tensors, num_tensors = Z.size()[:2]

        if increments:
            Z = self._apply_scaling_to_incremental_tensors(Z)
        else:
            Z = self._apply_scaling_to_tensors(Z)

        if self.low_rank:
            Phi_lvls = self._K_tens_lr_feat(Z, increments=increments)
            # K_lvls = tf.stack(
            #     [tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0
            # )
            K_lvls = torch.stack(
                [torch.matmul(P, P.T) for P in Phi_lvls], dim=0
            )
        else:
            K_lvls = self._K_tens(Z, increments=increments)

        K_lvls *= self.sigma * self.variances[:, None, None]

        if return_levels:
            return K_lvls
        else:
            # return tf.reduce_sum(K_lvls, axis=0)
            return K_lvls.sum(dim=0)

    # @params_as_tensors
    def K_tens_vs_seq(
        self, Z, X, return_levels=False, increments=False, presliced=False
    ):
        """
        Computes a cross-covariance matrix between inducing tensors and sequences.
        """

        if not presliced:
            X, _ = self._slice(X, None)

        num_examples = X.size(0)
        # X = tf.reshape(X, (num_examples, -1, self.num_features))
        X = X.reshape(num_examples, -1, self.num_features)
        len_examples = X.size(1)

        # num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0]
        len_tensors, num_tensors = Z.size()[:2]

        if increments:
            Z = self._apply_scaling_to_incremental_tensors(Z)
        else:
            Z = self._apply_scaling_to_tensors(Z)

        X = self._apply_scaling_and_lags_to_sequences(X)

        if self.low_rank:
            # seeds = tf.random_uniform(
            #     (self.num_levels - 1, 2),
            #     minval=0,
            #     maxval=np.iinfo(settings.int_type).max,
            #     dtype=settings.int_type,
            # )
            INT_MAX = 1 << 31
            seeds = torch.randint(
                low=0, high=INT_MAX, size=(self.num_levels - 1, 2)
            )
            idx, _ = low_rank_calculations._draw_indices(
                num_tensors * len_tensors * (int(increments) + 1)
                + num_examples * len_examples,
                self.num_components,
            )
            # nys_samples = tf.gather(
            #     tf.concat(
            #         (
            #             tf.reshape(
            #                 Z,
            #                 [
            #                     num_tensors
            #                     * len_tensors
            #                     * (int(increments) + 1),
            #                     -1,
            #                 ],
            #             ),
            #             tf.reshape(X, [num_examples * len_examples, -1]),
            #         ),
            #         axis=0,
            #     ),
            #     idx,
            #     axis=0,
            # )
            nys_samples = torch.gather(
                torch.cat(
                    [
                        Z.reshape(
                            num_tensors * len_tensors * (int(increments) + 1),
                            -1,
                        ),
                        X.reshape(num_examples * len_examples, -1),
                    ],
                    dim=0,
                ),
                index=idx,
                dim=0,
            )

            Phi_Z_lvls = self._K_tens_lr_feat(
                Z, increments=increments, nys_samples=nys_samples, seeds=seeds
            )
            Phi_X_lvls = self._K_seq_lr_feat(
                X, nys_samples=nys_samples, seeds=seeds
            )

            # Kzx_lvls = tf.stack(
            #     [
            #         tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True)
            #         for i in range(self.num_levels + 1)
            #     ],
            #     axis=0,
            # )
            Kzx_lvls = torch.stack(
                [
                    torch.matmul(Phi_Z_lvls[i], Phi_X_lvls[i].T)
                    for i in range(self.num_levels + 1)
                ],
                dim=0,
            )
        else:
            Kzx_lvls = self._K_tens_vs_seq(Z, X, increments=increments)

        if self.normalization:
            if self.low_rank:
                # Kxx_lvls_diag = tf.stack(
                #     [tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls],
                #     axis=0,
                # )
                Kxx_lvls_diag = torch.stack(
                    [torch.sum(torch.square(P), dim=-1) for P in Phi_X_lvls]
                )
            else:
                Kxx_lvls_diag = self._K_seq_diag(X)

            # TODO
            Kxx_lvls_diag += self.jitter

            # Kxx_lvls_diag_sqrt = tf.sqrt(Kxx_lvls_diag)
            Kxx_lvls_diag_sqrt = torch.sqrt(Kxx_lvls_diag)
            Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]

        Kzx_lvls *= self.sigma * self.variances[:, None, None]

        if return_levels:
            return Kzx_lvls
        else:
            # return tf.reduce_sum(Kzx_lvls, axis=0)
            return torch.sum(Kzx_lvls, dim=0)

    # @params_as_tensors
    def K_tens_n_seq_covs(
        self,
        Z,
        X,
        full_X_cov=False,
        return_levels=False,
        increments=False,
        presliced=False,
    ):
        """
        Computes and returns all three relevant matrices between inducing tensors tensors and input sequences, Kzz, Kzx, Kxx. Kxx is only diagonal if not full_X_cov
        """

        if not presliced:
            X, _ = self._slice(X, None)

        # num_examples = tf.shape(X)[0]
        num_examples = X.size(0)
        # X = tf.reshape(X, (num_examples, -1, self.num_features))
        X = X.reshape(num_examples, -1, self.num_features)
        # len_examples = tf.shape(X)[1]
        len_examples = X.size(1)

        # num_tensors, len_tensors = tf.shape(Z)[1], tf.shape(Z)[0]
        len_tensors, num_tensors = Z.size()[:2]

        if increments:
            Z = self._apply_scaling_to_incremental_tensors(Z)
        else:
            Z = self._apply_scaling_to_tensors(Z)

        X = self._apply_scaling_and_lags_to_sequences(X)

        if self.low_rank:
            INT_MAX = 1 << 31
            # seeds = tf.random_uniform(
            #     (self.num_levels - 1, 2),
            #     minval=0,
            #     maxval=np.iinfo(settings.int_type).max,
            #     dtype=settings.int_type,
            # )
            seeds = torch.randint(
                low=0, high=INT_MAX, size=(self.num_levels - 1, 2)
            )
            idx, _ = low_rank_calculations._draw_indices(
                num_tensors * len_tensors * (int(increments) + 1)
                + num_examples * len_examples,
                self.num_components,
            )
            # nys_samples = tf.gather(
            #     tf.concat(
            #         (
            #             tf.reshape(
            #                 Z,
            #                 [
            #                     num_tensors
            #                     * len_tensors
            #                     * (int(increments) + 1),
            #                     -1,
            #                 ],
            #             ),
            #             tf.reshape(X, [num_examples * len_examples, -1]),
            #         ),
            #         axis=0,
            #     ),
            #     idx,
            #     axis=0,
            # )
            nys_samples = torch.gather(
                torch.cat(
                    [
                        Z.reshape(
                            num_tensors * len_tensors * (int(increments) + 1),
                            -1,
                        ),
                        X.reshape(num_examples * len_examples, -1),
                    ],
                    dim=0,
                ),
                index=idx,
                dim=0,
            )

            Phi_Z_lvls = self._K_tens_lr_feat(
                Z, increments=increments, nys_samples=nys_samples, seeds=seeds
            )
            Phi_X_lvls = self._K_seq_lr_feat(
                X, nys_samples=nys_samples, seeds=seeds
            )

            # Kzz_lvls = tf.stack(
            #     [tf.matmul(P, P, transpose_b=True) for P in Phi_Z_lvls], axis=0
            # )
            Kzz_lvls = torch.stack(
                [torch.matmul(P, P.T) for P in Phi_Z_lvls],
                dim=0,
            )
            # Kzx_lvls = tf.stack(
            #     [
            #         tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True)
            #         for i in range(self.num_levels + 1)
            #     ],
            #     axis=0,
            # )
            Kzx_lvls = torch.stack(
                [
                    torch.matmul(Phi_Z_lvls[i], Phi_X_lvls[i].T)
                    for i in range(self.num_levels + 1)
                ],
                dim=0,
            )
        else:
            Kzz_lvls = self._K_tens(Z, increments=increments)
            Kzx_lvls = self._K_tens_vs_seq(Z, X, increments=increments)

        if full_X_cov:
            if self.low_rank:
                # Kxx_lvls = tf.stack(
                #     [tf.matmul(P, P, transpose_b=True) for P in Phi_X_lvls],
                #     axis=0,
                # )
                Kxx_lvls = torch.stack(
                    [torch.matmul(P, P.T) for P in Phi_X_lvls]
                )
            else:
                Kxx_lvls = self._K_seq(X)

            if self.normalization:
                Kxx_lvls += (
                    self.jitter
                    * torch.eye(num_examples, dtype=self.float_type)[None]
                )

                # Kxx_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(Kxx_lvls))
                Kxx_lvls_diag_sqrt = torch.sqrt(torch.diag(Kxx_lvls))

                Kxx_lvls /= (
                    Kxx_lvls_diag_sqrt[:, :, None]
                    * Kxx_lvls_diag_sqrt[:, None, :]
                )
                Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]

            Kxx_lvls *= self.sigma * self.variances[:, None, None]
            Kzz_lvls *= self.sigma * self.variances[:, None, None]
            Kzx_lvls *= self.sigma * self.variances[:, None, None]

            if return_levels:
                return Kzz_lvls, Kzx_lvls, Kxx_lvls
            else:
                return (
                    torch.sum(Kzz_lvls, dim=0),
                    torch.sum(Kzx_lvls, dim=0),
                    torch.sum(Kxx_lvls, dim=0),
                    # tf.reduce_sum(Kzz_lvls, axis=0),
                    # tf.reduce_sum(Kzx_lvls, axis=0),
                    # tf.reduce_sum(Kxx_lvls, axis=0),
                )

        else:
            if self.low_rank:
                Kxx_lvls_diag = torch.stack(
                    [torch.sum(torch.square(P), dim=-1) for P in Phi_X_lvls],
                    dim=0,
                )
                # Kxx_lvls_diag = tf.stack(
                #     [tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls],
                #     axis=0,
                # )

            else:
                Kxx_lvls_diag = self._K_seq_diag(X)

            if self.normalization:
                Kxx_lvls_diag += self.jitter

                Kxx_lvls_diag_sqrt = torch.sqrt(Kxx_lvls_diag)

                Kzx_lvls /= Kxx_lvls_diag_sqrt[:, None, :]
                # TODO
                # Kxx_lvls_diag = tf.tile(
                #     self.sigma * self.variances[:, None], [1, num_examples]
                # )
                Kxx_lvls_diag = torch.tile(
                    self.sigma * self.variances[:, None], (1, num_examples)
                )
            else:
                Kxx_lvls_diag *= self.sigma * self.variances[:, None]

            Kzz_lvls *= self.sigma * self.variances[:, None, None]
            Kzx_lvls *= self.sigma * self.variances[:, None, None]

            if return_levels:
                return Kzz_lvls, Kzx_lvls, Kxx_lvls_diag
            else:
                return (
                    torch.sum(Kzz_lvls, dim=0),
                    torch.sum(Kzx_lvls, dim=0),
                    torch.sum(Kxx_lvls_diag, dim=0),
                    # tf.reduce_sum(Kzz_lvls, axis=0),
                    # tf.reduce_sum(Kzx_lvls, axis=0),
                    # tf.reduce_sum(Kxx_lvls_diag, axis=0),
                )

    # @params_as_tensors
    def K_seq_n_seq_covs(
        self, X, X2, full_X2_cov=False, return_levels=False, presliced=False
    ):
        """
        Computes and returns all three relevant matrices between inducing sequences and input sequences, Kxx, Kxx2, Kx2x2. Kx2x2 is only diagonal if not full_X2_cov
        """

        if not presliced:
            X2, _ = self._slice(X2, None)

        # num_examples = tf.shape(X)[0]
        num_examples = X.size(0)
        # X = tf.reshape(X, (num_examples, -1, self.num_features))
        X = X.reshape(num_examples, -1, self.num_features)
        # len_examples = tf.shape(X)[1]
        len_examples = X.size(1)

        num_examples2 = X2.size(0)
        X2 = X2.reshape(num_examples2, -1, self.num_features)
        len_examples2 = X2.size(1)
        # num_examples2 = tf.shape(X2)[0]
        # X2 = tf.reshape(X2, (num_examples2, -1, self.num_features))
        # len_examples2 = tf.shape(X2)[1]

        X = self._apply_scaling_and_lags_to_sequences(X)
        X2 = self._apply_scaling_and_lags_to_sequences(X2)

        if self.low_rank:
            # seeds = tf.random_uniform(
            #     (self.num_levels - 1, 2),
            #     minval=0,
            #     maxval=np.iinfo(settings.int_type).max,
            #     dtype=settings.int_type,
            # )
            INT_MAX = 1 << 31
            seeds = torch.randint(
                low=0, high=INT_MAX, size=(self.num_levels - 1, 2)
            )
            idx, _ = low_rank_calculations._draw_indices(
                num_examples * len_examples + num_examples2 * len_examples2,
                self.num_components,
            )
            # nys_samples = tf.gather(
            #     tf.concat(
            #         (
            #             tf.reshape(X, [num_examples * len_examples, -1]),
            #             tf.reshape(X2, [num_examples2 * len_examples2, -1]),
            #         ),
            #         axis=0,
            #     ),
            #     idx,
            #     axis=0,
            # )
            nys_samples = torch.gather(
                torch.cat(
                    [
                        X.reshape(num_examples * len_examples, -1),
                        X2.reshape(num_examples2 * len_examples2, -1),
                    ],
                    dim=0,
                ),
                index=idx,
                dim=0,
            )

            Phi_lvls = self._K_seq_lr_feat(
                X, nys_samples=nys_samples, seeds=seeds
            )
            Phi2_lvls = self._K_seq_lr_feat(
                X2, nys_samples=nys_samples, seeds=seeds
            )

            # Kxx_lvls = tf.stack(
            #     [tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0
            # )
            Kxx_lvls = torch.stack(
                [torch.matmul(P, P.T) for P in Phi_lvls], dim=0
            )
            # Kxx2_lvls = tf.stack(
            #     [
            #         tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True)
            #         for i in range(self.num_levels + 1)
            #     ],
            #     axis=0,
            # )
            Kxx2_lvls = torch.stack(
                [
                    torch.matmul(Phi_lvls[i], Phi2_lvls[i].T)
                    for i in range(self.num_levels + 1)
                ],
                dim=0,
            )
        else:
            Kxx_lvls = self._K_seq(X)
            Kxx2_lvls = self._K_seq(X, X2)

        if self.normalization:
            Kxx_lvls += (
                self.jitter
                * torch.eye(num_examples, dtype=self.float_type)[None]
            )

            # Kxx_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(Kxx_lvls))
            Kxx_lvls_diag_sqrt = torch.sqrt(torch.diag(Kxx_lvls))
            Kxx_lvls /= (
                Kxx_lvls_diag_sqrt[:, :, None] * Kxx_lvls_diag_sqrt[:, None, :]
            )
            Kxx2_lvls /= Kxx_lvls_diag_sqrt[:, :, None]

        if full_X2_cov:
            if self.low_rank:
                # Kx2x2_lvls = tf.stack(
                #     [tf.matmul(P, P, transpose_b=True) for P in Phi2_lvls],
                #     axis=0,
                # )
                Kx2x2_lvls = torch.stack(
                    [
                        torch.matmul(
                            P,
                            P.T,
                        )
                        for P in Phi2_lvls
                    ],
                    dim=0,
                )
            else:
                Kx2x2_lvls = self._K_seq(X2)

            if self.normalization:
                # K_x2x2_lvls += (
                #     settings.jitter
                #     * tf.eye(num_examples2, dtype=settings.float_type)[None]
                # )
                Kx2x2_lvls += (
                    self.jitter
                    * torch.eye(num_examples2, dtype=self.float_type)[None]
                )

                # Kx2x2_lvls_diag_sqrt = tf.sqrt(tf.matrix_diag_part(K_x2x2_lvls))
                Kx2x2_lvls_diag_sqrt = torch.sqrt(torch.diag(Kx2x2_lvls))
                # TODO

                # Kxx2_lvls /= Kx2x2_lvls_diags_sqrt[:, None, :]
                # Kx2x2_lvls /= (
                #     Kx2x2_lvls_diags_sqrt[:, :, None]
                #     * Kx2x2_lvls_diags_sqrt[:, None, :]
                # )
                Kxx2_lvls /= Kx2x2_lvls_diag_sqrt[:, None, :]
                Kx2x2_lvls /= (
                    Kx2x2_lvls_diag_sqrt[:, :, None]
                    * Kx2x2_lvls_diag_sqrt[:, None, :]
                )

            Kxx_lvls *= self.sigma * self.variances[:, None, None]
            Kxx2_lvls *= self.sigma * self.variances[:, None, None]
            Kx2x2_lvls *= self.sigma * self.variances[:, None, None]

            if return_levels:
                return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls
            else:
                return (
                    torch.sum(Kxx_lvls, dim=0),
                    torch.sum(Kxx2_lvls, dim=0),
                    torch.sum(Kx2x2_lvls, dim=0),
                    # tf.reduce_sum(Kxx_lvls, axis=0),
                    # tf.reduce_sum(Kxx2_lvls, axis=0),
                    # tf.reduce_sum(Kx2x2_lvls, axis=0),
                )

        else:
            if self.low_rank:
                Kx2x2_lvls_diag = torch.stack(
                    [torch.sum(torch.square(P), dim=-1) for P in Phi2_lvls],
                    dim=0,
                )
                # Kx2x2_lvls_diag = tf.stack(
                #     [tf.reduce_sum(tf.square(P), axis=-1) for P in Phi2_lvls],
                #     axis=0,
                # )
            else:
                Kx2x2_lvls_diag = self._K_seq_diag(X2)

            if self.normalization:
                Kx2x2_lvls_diag += self.jitter

                # Kx2x2_lvls_diag_sqrt = tf.sqrt(Kx2x2_lvls_diag)
                Kx2x2_lvls_diag_sqrt = torch.sqrt(Kx2x2_lvls_diag)

                Kxx2_lvls /= (
                    Kxx_lvls_diag_sqrt[:, :, None]
                    * Kx2x2_lvls_diag_sqrt[:, None, :]
                )
                # TODO
                # Kx2x2_lvls_diag = tf.tile(
                #     self.sigma * self.variances[:, None], [1, num_examples2]
                # )
                Kx2x2_lvls_diag = torch.tile(
                    self.sigma * self.variances[:, None], (1, num_examples2)
                )
            else:
                Kx2x2_lvls_diag *= self.sigma * self.variances[:, None]

            Kxx_lvls *= self.sigma * self.variances[:, None, None]
            Kxx2_lvls *= self.sigma * self.variances[:, None, None]

            if return_levels:
                return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls_diag
            else:
                return (
                    torch.sum(Kxx_lvls, dim=0),
                    torch.sum(Kxx2_lvls, dim=0),
                    torch.sum(Kx2x2_lvls_diag, dim=0),
                    # tf.reduce_sum(Kxx_lvls, axis=0),
                    # tf.reduce_sum(Kxx2_lvls, axis=0),
                    # tf.reduce_sum(Kx2x2_lvls_diag, axis=0),
                )

    ##### Helper functions for base kernels

    def _square_dist(self, X, X2=None):
        # TODO: refactor this
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>

        # batch = tf.shape(X)[:-2]
        batch = X.size()[:-2]
        # Xs = tf.reduce_sum(tf.square(X), axis=-1)
        # Xs = (X * X).sum(dim=-1)
        Xs = torch.sum(torch.square(X), dim=-1)
        if X2 is None:
            dist = -2 * torch.matmul(X, X.T)
            # dist += tf.reshape(
            #     Xs, tf.concat((batch, [-1, 1]), axis=0)
            # ) + tf.reshape(Xs, tf.concat((batch, [1, -1]), axis=0))
            dist = dist + (
                Xs.reshape(*batch, -1, 1) + Xs.reshape(*batch, 1, -1)
            )
            return dist

        # X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        # X2s = (X2 * X2).sum(dim=-1)
        X2s = torch.sum(torch.square(X2), dim=-1)
        dist = -2 * torch.matmul(X, X2.T)
        # dist += tf.reshape(
        #     Xs, tf.concat((batch, [-1, 1]), axis=0)
        # ) + tf.reshape(X2s, tf.concat((batch, [1, -1]), axis=0))
        dist = dist + (Xs.reshape(*batch, -1, 1) + X2s.reshape(*batch, 1, -1))
        return dist

    def _euclid_dist(self, X, X2=None):
        r2 = self._square_dist(X, X2)
        # return tf.sqrt(tf.maximum(r2, 1e-40))
        # return torch.sqrt(torch.maximum(r2, torch.fill(r2, 1e-40)))
        return torch.sqrt(torch.clamp_min(r2, 1e-40))

    ##### Base kernel implementations:
    # To-do: use GPflow kernels for vector valued-data as base kernels
