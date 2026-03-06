from typing import Tuple
import warnings
import torch
import logging
from enum import Enum  
import math
from .features import Features, ScalingMode, FeatureSpec
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _LinearScaler:
    '''
    Recieves a tensor [n_samples, n_features]
    fits(), transforms() and inverse_transorms() 
    standardisation
    '''
    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        self.eps = eps
        self.device = device
        self._fitted = False
        self.location = None
        self.scale = None

    '''
    Hook, for subclass. 
    '''

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _inverse_transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return x 

    '''
    Return scaling params
    '''

    @property
    def get_params(self) -> [torch.Tensor, torch.Tensor]:
        # return scaling parameters to user upon request
        if not self._fitted:
            raise RuntimeError('Must call fit() before get_params()')

        return self.location, self.scale
        # WARNING: order of output is convention

    '''
    Fit, Transform and Inverse, 
    fit_from_params allows scaling sample populations with different parameters
    '''

    def fit(self, x: torch.Tensor) -> '_LinearScaler':
        # clone to avoid mutation and send to device 
        x = x.clone().to(self.device)
        # send through hook, so for inherited classes can apply transform before linear fit
        x_sel = self._transform_input(x)

        # calculate location and mean, ouput as [1, n_scaled_features]
        self.location = x_sel.mean(dim=0, keepdim=True)
        self.scale = x_sel.std(dim=0, keepdim=True, correction=0).clamp(min=self.eps)

        # return state, fitted allows transform, inverse and get params
        self._fitted = True
        return self

    def fit_transform_from_params(self, x: torch.Tensor, params: torch.Tensor):
        # validate input
        # params must be shape [1, n_scaled_features]
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        # make user aware they overwritting a scaler
        if self._fitted:
            warnings.warn('You have overwritten scaler params')
        # NAN input will corrupt data
        if torch.isnan(x).any().item():
            raise RuntimeError('Scaling parameters contains NAN')

        # ensure params are on device
        self.location = params[0, :].to(self.device)
        self.scale = params[1, :].to(self.device)

        # set fitted state and transform input with given params, ouput is [n_samples, n_scaled_features]
        self._fitted = True
        return self.transform(x)

    def fit_inverse_from_params(self, x: torch.Tensor, params: torch.Tensor):
        # validate input
        # params are required to be shape [1, n_features]
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        # make user aware they overwritting a scaler
        if self._fitted:
            warnings.warn('You have overwritten scaler params')
        # NAN input will corrupt data
        if torch.isnan(params).any().item():
            raise RuntimeError('Scaling parameters contains NAN')

        # ensure params are on device        
        self.location = params[0, :].to(self.device)
        self.scale = params[1, :].to(self.device)

        # set fitted state and transform input with given params, ouput is [n_samples, n_scaled_features]
        self._fitted = True
        return self.inverse_transform(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # cannot transform if no parameters
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform().")

        # clone to avoid mutation and set to deivice
        x = x.clone().to(self.device)        
        # send through transform to allow transform before linear scale
        x = self._transform_input(x)

        # transform tensor, ouput is [n_samples, n_scaled_features]
        return (x - self.location) / self.scale

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # cannot inverse with no parameters
        if not self._fitted:
            raise RuntimeError("Must call fit() before inverse_transform().")

        # clone to avoid mutation and set to deivice
        x = x.clone().to(self.device)

        # inverse transform 
        x = ((x * self.scale) + self.location)

        #return through hook, note hook second as transform and hook dont commute, ouput is [n_samples, n_scaled_features]
        return self._inverse_transform_input(x)


class _LogScaler(_LinearScaler):

    def __init__(self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,  # unused, for interface consistency
        min_value: torch.Tensor | None = None,  # unused, for interface consistency
    ):
        # inherit from linear scaler
        super().__init__(device=device, eps=eps, max_value=max_value, min_value=min_value)


    '''
    Validate 
    '''

    def _validate(self, x: torch.Tensor):
        # ensures no values below -1 so log doesnt produce NAN
        if (x < -1).any().item():
            raise ValueError('Feature to be Log scaled contains value < -1')

    '''
    Re-write hooks
    '''

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        # validate 
        self._validate(x)
        # log (1 + x)
        return torch.log1p(x)

    def _inverse_transform_input(self, x: torch.Tensor) -> torch.Tensor:
        # e^(1+x)
        return torch.expm1(x)


class _RobustScaler(_LinearScaler):

    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        # inherit from linear scaler
        super().__init__(device=device, eps=eps, max_value=max_value, min_value=min_value)

    '''
    Re-write fit for IQR and median
    '''

    def fit(self, x: torch.Tensor) -> '_RobustScaler':
        # clone to avoid mutation and set to deivice
        x = x.clone().to(self.device)        
        # send through transform to allow transform before linear scale
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

    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        # inherit from linear scaler
        super().__init__(device=device, eps=eps, max_value=max_value, min_value=min_value)

    '''
    Re-write hooks
    '''

    def _validate(self, x: torch.Tensor):
        # ensure no values below -1 so log doesnt NAN
        if (x < -1).any().item():
            raise ValueError('Feature to be Log-Robust scaled contains value < -1')

    '''
    Re-write hooks
    '''

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        # validate
        self._validate(x)
        # log(1+x)
        return torch.log1p(x)

    def _inverse_transform_input(self, x: torch.Tensor) -> torch.Tensor:
        # e^(1+x)
        return torch.expm1(x)


class _BoundedScaler(_LinearScaler):

    def __init__(
        self,
        device: str,
        eps: float = 1e-8,
        max_value: torch.Tensor | None = None,   # unused, for interface consistency
        min_value: torch.Tensor | None = None,   # unused, for interface consistency
    ):
        # inherit from lienar scaler
        super().__init__(device=device, eps=eps, max_value=max_value, min_value=min_value)
        # since bounded we now need max and min values as scaling params
        self.max_value = max_value.to(self.device)
        # min generally 0 but can be set to non-zero if required
        self.min_value = (
            min_value.to(self.device)
            if min_value is not None 
            else torch.zeros(1, self.max_value.shape[1], device=self.device)
        )
        self._fitted = True     # bounded scaler does not fit to params, min and max inputted, 
                                # self._fitted is unused for inferface consistency
    
    '''
    Return scaling params
    '''

    @property
    def get_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # cannot call params without fitting, user inputs min and max so redundant in that sense
        # but required for interface consistency when storing params for all scalers
        if not self._fitted:
            raise RuntimeError('Must call fit() before get_params()')

        return self.min_value, self.max_value
        # WARNING: order of output is convention

    '''
    Fit, Transform, Inverse
    fit_from_params allows scaling sample populations with different parameters
    '''

    def fit(self, x: torch.Tensor) -> '_BoundedScaler':
        # user inputs scaling params so this function is redundant
        # make explicitly do nothing as inherits linear scaler fit() else
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        # clone to avoid mutation and set to deivice
        x = x.clone().to(self.device)        
        # send through transform to allow transform before linear scale
        x_sel = self._transform_input(x)

        # return transformed tensor, ouput is [n_samples, n_scaled_features]
        return (x_sel - self.min_value) / (self.max_value - self.min_value)

    def fit_transform_from_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # validate
        # params must be shape [1, n_scaled_features]
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        # ensure scaling params are NAN
        if torch.isnan(params).any().item():
            raise RuntimeError('Scaling parameters contain NAN')

        # ensure params are on device
        self.min_value = params[0, :].to(self.device)
        self.max_value = params[1, :].to(self.device)

        # returns transform tensor, ouput is [n_samples, n_scaled_features]
        return self.transform(x)

    def fit_inverse_from_params(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # validate
        # params must be shape [1, n_scaled_features]
        if not x.shape[-1] == params.shape[-1]:
            raise ValueError('Params and Data must both be same n_features')
        if torch.isnan(x).any().item():
            raise RuntimeError('Scaling parameters contains NAN')

        # ensure params are on device
        self.min_value = params[0, :].to(self.device)
        self.max_value = params[1, :].to(self.device)

        # return inverse tensor, ouput is [n_samples, n_scaled_features]
        return self.inverse_transform(x)
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # clone to avoid mutation and set to device
        x = x.clone().to(self.device) 
        # inverse scale
        x = (x * (self.max_value - self.min_value)) + self.min_value

        # return through hook, ouput is [n_samples, n_scaled_features]
        return self._inverse_transform_input(x)


class FeatureScaler:
    '''
    Mask aware scaler for tensors shaped [batch, time, features]
    Note:   his class handles all mask operations, 
            works with masks values = None as long as torch.Tensor of correct shape,
            scalers are given 2D tensor as a result of this.
    '''

    def __init__(
        self,
        features_cls: Features,
        scaling_masks: dict[ScalingMode, torch.Tensor],
        eps: float = 1e-8,
        device: str = 'cpu',
    ):

        self.features = features_cls
        self.eps = eps
        self.device = device
        self.scaling_masks = {mode: mask.to(self.device) for mode, mask in scaling_masks.items()}
        self.specs_by_mode = self.features.specs_by_mode
        self._build_bound_tensor()
        self._build_presence_indices()
        self._fitted = False

        _MODE_TO_SCALER = {
            ScalingMode.LINEAR:     _LinearScaler,
            ScalingMode.LOG:        _LogScaler,
            ScalingMode.ROBUST:     _RobustScaler,
            ScalingMode.LOG_ROBUST: _RobustLog,
            ScalingMode.BOUNDED:    _BoundedScaler,
            # IDENTITY intentionally absent
        }

        self.n_scaled = {}
        self._scalers = {}

        for mode, cls in _MODE_TO_SCALER.items():
            self.n_scaled[mode] = self.scaling_masks[mode].sum().item()

            if self.scaling_masks[mode].any().item():
                self._scalers[mode] = cls(
                    device=self.device,
                    eps=self.eps,
                    min_value=self.min_vals,
                    max_value=self.max_vals,
                )
            
    '''
    validation
    '''

    def _validate_input(self, x: torch.Tensor):

        if not isinstance(x, torch.Tensor):
            raise TypeError('Input must be a tensor')

        if not torch.is_floating_point(x):
            raise TypeError(
                f'Input must be type float, recieved {x.dtype}'
            )
        
        if x.dim() != 3:
            raise ValueError(
                f'Expected shape [Batch, Time, Feature], recieved {x.shape}'
            )

        if not x.shape[-1] == len(self.features):
            raise RuntimeError('Tensor[Features] dimensions do not match Mask Dimensions')

        if not torch.isfinite(x).all().item():
            raise ValueError('Input tensor contains NaN or infinite value(s).')

    '''
    Build any internal features 
    '''

    def _build_bound_tensor(self):
         
        min_vals = []
        max_vals = []

        for feature in self.features.specs:
            if feature.scaling_mode == ScalingMode.BOUNDED:
                max_vals.append(feature.max_value)
                min_vals.append(feature.min_value)
        
        if not max_vals:
            self.max_vals = None
            self.min_vals = None
            return self
            
        self.max_vals = torch.tensor(max_vals, dtype=torch.float32).unsqueeze(0)
        self.min_vals = torch.tensor(min_vals, dtype=torch.float32).unsqueeze(0)
        return self

    def _build_presence_indices(self):
        indices = []
        min_values = []

        for i, spec in enumerate(self.features.specs):
            if spec.presence_check:
                indices.append(i)
                min_values.append(spec.min_value)
        
        if indices: 
            self._presence_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
            self._presence_min_values = torch.tensor(min_values, dtype=torch.float32, device=self.device)
        else:
            self._presence_indices = None
            self._presence_min_values = None
        
    def _build_presence_mask(self, x: torch.Tensor):
        if self._presence_indices is None:    
            return torch.ones(x.shape[0], x.shape[1], dtype=torch.bool)

        x_check = x[:,:, self._presence_indices].to(self.device)
        mask = (x_check != self._presence_min_values).all(dim=-1)
        return mask       

    '''
    Main class functions
    '''
    
    def train_scale(self, x: torch.Tensor) -> [torch.Tensor, dict] :
        self._validate_input(x)
        x = x.clone()
        presence_mask = self._build_presence_mask(x).to(self.device)

        x_scale = x[presence_mask]   
        for mode, scaler in self._scalers.items():
            x_scale[:, self.scaling_masks[mode]] = scaler.fit_transform(x_scale[:, self.scaling_masks[mode]])
            param1, param2 = scaler.get_params
            

            for spec, p1, p2 in zip(self.specs_by_mode[mode], param1[0], param2[0], strict=True):
                spec.scaling_params = [p1.item(), p2.item()]        
                # convention p1 is location or min value, p2 is scale or max_value

        x[presence_mask] = x_scale

        self._fitted = True
        return x, self.features.to_dict()

    
    def test_scale(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError('Fit training data before scaling Test')

        self._validate_input(x)
        x = x.clone()
        presence_mask = self._build_presence_mask(x).to(self.device)

        x_scale = x[presence_mask]
        for mode, scaler in self._scalers.items():
            x_scale[:, self.scaling_masks[mode]] = scaler.transform(x_scale[:, self.scaling_masks[mode]])
        x[presence_mask] = x_scale

        return x
                

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        x = x.clone()
        
        presence_mask = self._build_presence_mask(x).to(self.device)
        x_inv = x[presence_mask]

        for mode, scaler in self._scalers.items():
            if self._fitted:
                x_inv[:, self.scaling_masks[mode]] = scaler.inverse_transform(x_inv[:, self.scaling_masks[mode]])      
            
            else:
                params = torch.tensor(
                    [s.scaling_params for s in self.specs_by_mode[mode]],
                    dtype = torch.float32,
                ).T

                x_inv[:, self.scaling_masks[mode]] = scaler.fit_inverse_from_params(x_inv[:, self.scaling_masks[mode]], params)
                
        x[presence_mask] = x_inv
        return x





