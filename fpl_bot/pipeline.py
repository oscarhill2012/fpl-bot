from __future__ import annotations

import logging
import warnings

import torch

from .features import Features, ScalingMode, FeatureSpec

logger = logging.getLogger(__name__)


class _LinearScaler:
    """
    Standard linear scaler for a tensor of shape [n_samples, n_features].

    Scales by subtracting the mean and dividing by the standard deviation.
    Provides fit(), transform(), and inverse_transform() operations.
    """
    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        """
        Initialise a linear scaler.

        Args:
            device: Torch device string; all tensors will be moved here.
            eps: Small value clamped to scale to avoid division by zero.
            max_value: Unused; present for interface consistency with _BoundedScaler.
            min_value: Unused; present for interface consistency with _BoundedScaler.
        """
        self.eps = eps              # tolerance to ensure no NAN outputs
        self.device = device        # all tensors must be on device or error
        self._fitted = False        # fitted state to stop transforms before fit
        self.location = None
        self.scale = None

    #================================================
    # Hooks
    # Overridden by subclasses to apply non-linear transforms
    # before and after the linear scaling step.
    #================================================

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Identity hook; subclasses override to apply a pre-transform before scaling."""
        return x

    def _inverse_transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Identity hook; subclasses override to apply a post-transform after inverse scaling."""
        return x

    #================================================
    # Properties
    #================================================

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fitted (location, scale) tensors, shape [1, n_scaled_features]; raises RuntimeError if called before fit()."""
        if not self._fitted:
            raise RuntimeError('Must call fit() before accessing params.')

        return self.location, self.scale
        # WARNING: order of output is convention

    #================================================
    # Public Functions
    #================================================

    def fit(self, x: torch.Tensor) -> '_LinearScaler':
        """
        Fit the scaler to a data tensor.

        Args:
            x: Input tensor of shape [n_samples, n_features].

        Returns:
            Self, to allow method chaining.
        """
        # clone to avoid mutation and send to device
        x = x.clone().to(self.device)
        # send through hook, so for inherited classes can apply transform before linear fit
        x_sel = self._transform_input(x)

        # calculate location and mean, output as [1, n_scaled_features]
        self.location = x_sel.mean(dim=0, keepdim=True)
        self.scale = x_sel.std(dim=0, keepdim=True, correction=0).clamp(min=self.eps)

        self._fitted = True
        return self

    def fit_transform_from_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Transform the input using externally provided parameters.

        Used for test sets to avoid leaking distribution information to the model.

        Args:
            x: Input tensor of shape [n_samples, n_features].
            params: Parameter tensor of shape [2, n_features];
                row 0 is location (or min), row 1 is scale (or max).

        Returns:
            Scaled tensor of shape [n_samples, n_features].
        """
        # validate input
        # params must be shape [1, n_scaled_features]
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        # make user aware they overwriting a scaler
        if self._fitted:
            warnings.warn('You have overwritten scaler params')
        # NAN input will corrupt data
        if torch.isnan(params).any().item():
            raise RuntimeError('Scaling parameters contain NaN')

        self.location = params[0, :].to(self.device)
        self.scale = params[1, :].to(self.device)
        self._fitted = True
        # transform input with given params, output is [n_samples, n_scaled_features]
        return self.transform(x)

    def fit_inverse_from_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform the input using externally provided parameters.

        Args:
            x: Scaled tensor of shape [n_samples, n_features].
            params: Parameter tensor of shape [2, n_features];
                row 0 is location (or min), row 1 is scale (or max).

        Returns:
            Unscaled tensor of shape [n_samples, n_features].
        """
        # validate input
        # params are required to be shape [1, n_features]
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        # make user aware they overwriting a scaler
        if self._fitted:
            warnings.warn('You have overwritten scaler params')
        # NAN input will corrupt data
        if torch.isnan(params).any().item():
            raise RuntimeError('Scaling parameters contain NaN')

        self.location = params[0, :].to(self.device)
        self.scale = params[1, :].to(self.device)

        # set fitted state and transform input with given params, output is [n_samples, n_scaled_features]
        self._fitted = True
        return self.inverse_transform(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the fitted scaling to a tensor.

        Args:
            x: Input tensor of shape [n_samples, n_features].

        Returns:
            Scaled tensor of the same shape.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform().")

        # clone to avoid mutation and set to device
        x = x.clone().to(self.device)
        # send through hook to allow non-linear transforms before linear scale in inherited classes
        x = self._transform_input(x)

        # transform tensor, output is [n_samples, n_scaled_features]
        return (x - self.location) / self.scale

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fit the scaler and transform the input in one step.

        Args:
            x: Input tensor of shape [n_samples, n_features].

        Returns:
            Scaled tensor of the same shape.
        """
        return self.fit(x).transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the scaling transformation.

        Args:
            x: Scaled tensor of shape [n_samples, n_features].

        Returns:
            Unscaled tensor of the same shape.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before inverse_transform().")

        x = x.clone().to(self.device)

        # inverse transform
        x = ((x * self.scale) + self.location)

        # return through hook, note hook second as transform and hook dont commute, output is [n_samples, n_scaled_features]
        return self._inverse_transform_input(x)


class _LogScaler(_LinearScaler):
    """
    Log scaler: applies ln(1+x) before standard linear scaling.

    Suitable for skewed, non-negative features.
    """
    def __init__(self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,  # unused, for interface consistency
        min_value: torch.Tensor | None = None,  # unused, for interface consistency
    ):
        """Initialise log scaler; delegates to _LinearScaler."""
        super().__init__(device=device, eps=eps, max_value=max_value, min_value=min_value)

    #================================================
    # Validation
    #================================================

    def _validate(self, x: torch.Tensor):
        """Ensure no values below -1 to prevent log producing NaN."""
        # ensures no values below -1 so log doesn't produce NaN
        if (x < -1).any().item():
            raise ValueError('Feature to be Log scaled contains value < -1')

    #================================================
    # Hooks
    #================================================

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ln(1+x) transform before linear scaling."""
        # validate
        self._validate(x)
        # ln(1+x)
        return torch.log1p(x)

    def _inverse_transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply e^x - 1 to reverse the log transform."""
        # e^x - 1
        return torch.expm1(x)


class _RobustScaler(_LinearScaler):
    """
    Robust scaler; uses median and IQR instead of mean and standard deviation.

    More resilient to outliers than linear scaling.
    """
    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        """Initialise robust scaler; delegates to _LinearScaler."""
        super().__init__(device=device, eps=eps, max_value=max_value, min_value=min_value)

    #================================================
    # Public Functions
    #================================================

    def fit(self, x: torch.Tensor) -> '_RobustScaler':
        """
        Fit the scaler using median and IQR.

        Args:
            x: Input tensor of shape [n_samples, n_features].

        Returns:
            Self, to allow method chaining.
        """
        # clone to avoid mutation and set to device
        x = x.clone().to(self.device)
        x_sel = self._transform_input(x)

        # find median and iqr, both shape [1, n_scaled_features]
        self.location = x_sel.median(dim=0, keepdim=True).values
        q1 = torch.quantile(x_sel, 0.25, dim=0, keepdim=True)
        q3 = torch.quantile(x_sel, 0.75, dim=0, keepdim=True)
        self.scale = (q3 - q1).clamp(min=self.eps)

        # set fitted state and return
        self._fitted = True
        return self


class _RobustLog(_RobustScaler):
    """
    Log-robust scaler: applies ln(1+x) before robust (median/IQR) scaling.

    Combines outlier resilience with handling of skewed distributions.
    """
    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        """Initialise log-robust scaler; delegates to _RobustScaler."""
        super().__init__(device=device, eps=eps, max_value=max_value, min_value=min_value)

    #================================================
    # Validation
    #================================================

    def _validate(self, x: torch.Tensor):
        """Ensure no values below -1 to prevent log producing NaN."""
        # ensure no values below -1 so log doesn't produce NaN
        if (x < -1).any().item():
            raise ValueError('Feature to be Log-Robust scaled contains value < -1')

    #================================================
    # Hooks
    #================================================

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ln(1+x) transform before robust scaling."""
        # validate
        self._validate(x)
        # log(1+x)
        return torch.log1p(x)

    def _inverse_transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply e^x - 1 to reverse the log transform."""
        # e^x - 1
        return torch.expm1(x)


class _BoundedScaler(_LinearScaler):
    """
    Bounded scaler: maps features linearly to [0, 1] using known min/max values.

    Unlike other scalers, parameters are provided at initialisation rather than
    fitted from data. fit() is a no-op, present for interface consistency.
    """
    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        """
        Initialise a bounded scaler.

        Args:
            device: Torch device string.
            eps: Unused; present for interface consistency.
            max_value: Tensor of max values, shape [1, n_bounded_features].
            min_value: Tensor of min values, shape [1, n_bounded_features].
        """
        # Must NOT call super().__init__ because it sets _fitted=False
        # and we want _fitted=True for bounded scalers
        self.eps = eps
        self.device = device
        self.location = None
        self.scale = None

        # max_value generally not 0, but maybe we'd call for data < 0 and want min value < 0 and max = 0
        self.max_value = (
            max_value.to(self.device)
            if max_value is not None
            else torch.zeros(1, device=self.device)
        )
        # min generally 0 but can be set to non-zero if required
        self.min_value = (
            min_value.to(self.device)
            if min_value is not None
            else torch.zeros(1, device=self.device)
        )
        self._fitted = True     # bounded scaler does not fit to params, min and max inputted,
                                # self._fitted is unused for interface consistency

    #================================================
    # Properties
    #================================================

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Min and max scaling parameters (min_value, max_value); redundant but present for interface consistency."""
        # redundant, as cannot call params without user inputs min and max,
        # but required for interface consistency when storing params for all scalers
        if not self._fitted:
            raise RuntimeError('Must call fit() before accessing params.')

        return self.min_value, self.max_value
        # WARNING: order of output is convention

    #================================================
    # Public Functions
    #================================================

    def fit(self, x: torch.Tensor) -> '_BoundedScaler':
        """
        No-op; bounded scaler uses user-supplied parameters rather than fitting.

        Returns:
            Self, for interface consistency.
        """
        # user inputs scaling params so this function is redundant
        # make explicitly do nothing as inherits _LinearScaler's fit() else
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the input to [0, 1] using the stored min and max values.

        Args:
            x: Input tensor of shape [n_samples, n_features].

        Returns:
            Scaled tensor of the same shape.
        """
        x = x.clone().to(self.device)
        x_sel = self._transform_input(x)

        # return transformed tensor, output is [n_samples, n_scaled_features]
        return (x_sel - self.min_value) / (self.max_value - self.min_value)

    def fit_transform_from_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Transform using externally provided min/max parameters.

        Redundant but present for interface consistency; uses min and max
        rather than location and scale.

        Args:
            x: Input tensor of shape [n_samples, n_features].
            params: Parameter tensor of shape [2, n_features];
                row 0 is min_value, row 1 is max_value.

        Returns:
            Scaled tensor of shape [n_samples, n_features].
        """
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        if torch.isnan(params).any().item():
            raise RuntimeError('Scaling parameters contain NAN')

        self.min_value = params[0, :].to(self.device)
        self.max_value = params[1, :].to(self.device)

        # returns transform tensor, output is [n_samples, n_scaled_features]
        return self.transform(x)

    def fit_inverse_from_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform using externally provided min/max parameters.

        Args:
            x: Scaled tensor of shape [n_samples, n_features].
            params: Parameter tensor of shape [2, n_features];
                row 0 is min_value, row 1 is max_value.

        Returns:
            Unscaled tensor of shape [n_samples, n_features].
        """
        # validate
        # params must be shape [1, n_scaled_features]
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        if torch.isnan(params).any().item():
            raise RuntimeError('Scaling parameters contain NaN')

        self.min_value = params[0, :].to(self.device)
        self.max_value = params[1, :].to(self.device)

        # return inverse tensor, output is [n_samples, n_scaled_features]
        return self.inverse_transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the bounded scaling transformation.

        Args:
            x: Scaled tensor of shape [n_samples, n_features].

        Returns:
            Unscaled tensor of the same shape.
        """
        x = x.clone().to(self.device)
        # inverse scale
        x = (x * (self.max_value - self.min_value)) + self.min_value

        # return through hook, output is [n_samples, n_scaled_features]
        return self._inverse_transform_input(x)


_MODE_TO_SCALER = {
    ScalingMode.LINEAR:     _LinearScaler,
    ScalingMode.LOG:        _LogScaler,
    ScalingMode.ROBUST:     _RobustScaler,
    ScalingMode.LOG_ROBUST: _RobustLog,
    ScalingMode.BOUNDED:    _BoundedScaler,
    # IDENTITY intentionally absent
}


class FeatureScaler:
    """
    Mask-aware scaler for tensors of shape [batch, time, numneric_features].
    
    Handles all scaling for numeric features.

    Orchestrates multiple underlying scalers, one per ScalingMode, applying
    each only to the features it governs. Handles presence masking so that
    absent player rows are excluded from fitting and transformation.

    Note:
        - All mask operations are handled internally.
        - Downstream scalers receive 2D tensors as a result.
    """

    def __init__(
        self,
        features: Features,
        eps: float = 1e-8,
        device: str = 'cpu',
    ):
        """
        Initialise the FeatureScaler.

        Args:
            features_cls: Features instance defining the feature registry.
            scaling_masks: Dict mapping each ScalingMode to a boolean mask tensor.
            eps: Small value for numerical stability in underlying scalers.
            device: Torch device string; all tensors are moved here.
        """
        self.features = features
        self.eps = eps
        self.device = device

        # generates scaling and presence masks 
        self.scaling_masks = features.build_scaling_masks()
        self._build_presence_indices()

        # specs gives dict{ScalingMode: list of feature names for that mode}
        self.specs_by_mode = self.features.specs_by_mode

        # bound tensor is collection of min and max for bound scaling
        self._build_bound_tensor()

        # _fitted now refers to the collection of scalers generated upon init
        self._fitted = False

        # n_scaled is how many features each scaling mode acts on
        self.n_scaled = {}
        # dict of scaler instances
        self._scalers = {}

        for mode, cls in _MODE_TO_SCALER.items():
            self.n_scaled[mode] = self.scaling_masks[mode].sum().item()
            # initialise scaler instances
            if self.scaling_masks[mode].any().item():
                self._scalers[mode] = cls(
                    device=self.device,
                    eps=self.eps,
                    min_value=self.min_vals,
                    max_value=self.max_vals,
                )

    #================================================
    # init Helpers
    #================================================

    def _validate_input(self, x: torch.Tensor):
        """Validate that the input tensor has the correct shape, type, and contains only finite values."""
        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a tensor')
        if not torch.is_floating_point(x):
            raise TypeError(f'Input must be type float, received {x.dtype}')
        if x.dim() != 3:
            raise ValueError(f'Expected shape [Batch, Time, numeric_Features], received {x.shape}')
        if not x.shape[-1] == len(self.features):
            # masks generated internally, so check features give correct shape
            raise RuntimeError('Tensor[Features] dimensions do not match len(features.specs)')
        if not torch.isfinite(x).all().item():
            raise ValueError('Input tensor contains NaN or infinite value(s).')

    def _build_bound_tensor(self):
        """Collect min and max values from bounded specs into tensors for _BoundedScaler."""
        min_vals = []
        max_vals = []

        for feature in self.features.specs:
            if feature.scaling_mode == ScalingMode.BOUNDED:
                max_vals.append(feature.max_value)
                min_vals.append(feature.min_value)

        # None makes bound scaler set to 0s
        if not max_vals:
            self.max_vals = None
            self.min_vals = None
            return self

        # _BoundedScaler requires tensor input [1, n_scaled_features], even if None
        self.max_vals = torch.tensor(max_vals, dtype=torch.float32).unsqueeze(0)
        self.min_vals = torch.tensor(min_vals, dtype=torch.float32).unsqueeze(0)
        return self

    #================================================
    # Public Functions
    #================================================

    def train_scale(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Fit all scalers on the training data and return the scaled tensor.

        Fitting parameters are stored inside each FeatureSpec's scaling_params field.
        Absent rows (determined by presence mask) are excluded from fitting.

        Args:
            x: Input tensor of shape [players, gameweeks, features].

        Returns:
            Tuple of (scaled tensor, features dict with fitted scaling params).
        """
        # validate
        self._validate_input(x)
        # clone to avoid mutation
        x = x.clone()
        # build presence mask now we have data
        presence_mask = self._build_presence_mask(x).to(self.device)

        # scaler gets data after presence and scaling masks applied
        # scaling mask is just which type of scaling is applied to which feature
        x_present = x[presence_mask]
        for mode, scaler in self._scalers.items():
            x_present[:, self.scaling_masks[mode]] = scaler.fit_transform(x_present[:, self.scaling_masks[mode]])

            param1, param2 = scaler.params
            self._append_params(mode, param1, param2)

        # write back scaled values for present rows
        x[presence_mask] = x_present

        # now fitted
        self._fitted = True

        # return scaled tensor shape = [n_samples, n_timesteps, n_features], and return features dict, now with scaling params
        # features dict must be given with x as this is the meta data for tensor
        return x, self.features.to_dict()

    def test_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the input using parameters fitted on training data.

        Must be called after train_scale() to avoid leaking test distribution
        information into the model.

        Args:
            x: Input tensor of shape [players, gameweeks, features].

        Returns:
            Scaled tensor of the same shape.

        Raises:
            RuntimeError: If called before train_scale().
        """
        # test data must be scaled with params from training, or we leak distribution information to model
        if not self._fitted:
            raise RuntimeError('Fit training data before scaling Test')

        self._validate_input(x)
        x = x.clone()
        presence_mask = self._build_presence_mask(x).to(self.device)
        x_present = x[presence_mask]

        for mode, scaler in self._scalers.items():
            x_present[:, self.scaling_masks[mode]] = scaler.transform(x_present[:, self.scaling_masks[mode]])

        x[presence_mask] = x_present
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse the scaling transformation.

        Can be called after fitting, or will reconstruct parameters from the
        scaling_params stored in each FeatureSpec.

        Args:
            x: Scaled tensor of shape [players, gameweeks, features].

        Returns:
            Unscaled tensor of the same shape.
        """
        self._validate_input(x)
        x = x.clone()

        #TODO: check this, if presence feature is scaled, now min_value has changed...
        presence_mask = self._build_presence_mask(x).to(self.device)
        x_inv = x[presence_mask]

        # allow inverse to be called after fitting scalers, or with scaling parameters in feature dict
        for mode, scaler in self._scalers.items():
            if self._fitted:
                x_inv[:, self.scaling_masks[mode]] = scaler.inverse_transform(x_inv[:, self.scaling_masks[mode]])

            else:
                # transpose so shape is [n_params, n_features], for broadcasting convenience
                params = torch.tensor(
                    [s.scaling_params for s in self.specs_by_mode[mode]],
                    dtype = torch.float32,
                ).T
                x_inv[:, self.scaling_masks[mode]] = scaler.fit_inverse_from_params(x_inv[:, self.scaling_masks[mode]], params)

        x[presence_mask] = x_inv
        return x

    #================================================
    # Private Helpers
    #================================================

    def _build_presence_indices(self):
        """Build index tensors for presence-checked features, used in presence masking."""
        # presence indices built from features input upon initialisation
        indices = []
        min_values = []

        for i, spec in enumerate(self.features.specs):
            if spec.presence_check:
                indices.append(i)
                min_values.append(spec.min_value)

        if indices:
            # torch.long is signed integers, both shape [1, n_presence_features]
            self._presence_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
            self._presence_min_values = torch.tensor(min_values, dtype=torch.float32, device=self.device)
        else:
            # None makes presence act as identity
            self._presence_indices = None
            self._presence_min_values = None

    def _build_presence_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Build a [batch, time] boolean mask; True where all presence-checked features exceed their min_value."""
        # presence mask built from data and presence indices,
        # so can only be called upon scale, inverse as requires data tensor x
        if self._presence_indices is None:
            # returns all True, allows all data through, no presence filtering
            # output shape is [n_samples, n_timesteps]
            return torch.ones(x.shape[0], x.shape[1], dtype=torch.bool)

        # if sample at time step has presence feature > threshold (min_val)
        # that sample for that time is given to scaler
        x_check = x[:,:, self._presence_indices].to(self.device)
        mask = (x_check > self._presence_min_values).all(dim=-1)
        return mask

    def _append_params(self, mode: ScalingMode, param1: float, param2: float) -> "FeatureScaler":
        """Adds params to features instance"""
        for spec, p1, p2 in zip(self.specs_by_mode[mode], param1[0], param2[0], strict=True):
            spec.scaling_params = [p1.item(), p2.item()]

        return self
