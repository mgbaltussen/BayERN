from typing import Dict, List, Any, Optional, Tuple
import copy
import itertools

import numpy as np
import pandas as pd

import pytensor
import pytensor.tensor as pt
import theano
import theano.tensor as tt

import pymc as pm
from pymc.distributions import generate_samples, draw_values, transforms
from pymc.model import Model
# import pymc3 as pm
# from pymc3.distributions import generate_samples, draw_values, transforms
# from pymc3.model import Model
import amici


def tag_value_or_number(p):
    if isinstance(p, theano.tensor.TensorConstant):
        return p.eval()
    if isinstance(p, (int, float)):
        return p
    elif isinstance(p, (list, tuple)):
        return np.array(p)
    elif isinstance(p, theano.compile.SharedVariable):
        return p.get_value()
    else:
        try:
            return p.tag.test_value
        except AttributeError:
            try:
                return p.mean.eval()  # TODO: Rewrite to pymc get_test_val
            except AttributeError:
                raise TypeError(f"Type {type(p)} of {p} cannot be used as parameter")


class ODESystem(pm.distributions.distribution.Continuous):
    """" PyMC distribution to infer dynamical systems using an Amici model

    Model inputs such as parameters, fixed parameteres, initial values, and timepoints
    should be given at distribution initialization as dictionaries 
    with keys corresponding to identifiers in the Amici model. 

    Examples
    --------
        .. code-block:: python

            model = amici.import_model_module(model_name, model_dir).getModel()
            solver = model.getSolver()m 
            # Set model and solver settings before passing to PyMC

            with pm.Model():
                ODESystem(
                    'model_name',
                    amici_model=model,
                    solver=solver,
                    parameters={
                            'param_1': pm.Normal('param_1', 0, 1),
                            'param_2': pm.Normal('param_2', 0, 1),
                            'sigma': pm.Exponential('sigma', 2.0) # The sigma parameter should always be present for noise estimation
                        },
                    fixed=={
                            'fixed_1': 1.0
                            'fixed_1': 2.0
                        },,
                    initial_states={'x': 0.0, 'y': 0.0},
                    timepoints=np.linspace(0, 10, 100),
                    observed={'x': data.x.values, 'y': data.y.values},
                )
                trace = pm.sample()

        If direct inference of the dynamical system is to expensive, and only predictions are required, 
        we can instantiate the ODESystem distribution after inference exclusively for posterior predictive checks.
        The ODESystem class supports random sampling either from a given trace, or directly from random distributions (sample_prior_predictive).

        .. code-block:: python
            with pm.Model():
                param_1 = pm.Normal('param_1', 0, 1)
                param_2 = pm.Normal('param_2', 0, 1)
                sigma = pm.Exponential('sigma', 2.0)

                ### First perform inference on a (non-dynamical system) model to obtain parameter estimates

                pm.Normal("obs", mu=some_non_ode_model(param_1, param_2), sigma=sigma, obs=data)
                trace = pm.sample()

                ### Then register the ODESystem distribution and perform posterior predictive sampling on it.

                ODESystem(
                    'model_name',
                    amici_model=model,
                    solver=solver,
                    parameters={
                            'param_1': param_1
                            'param_2': param_2
                            'sigma': sigma
                        },
                    fixed=={
                            'fixed_1': 1.0
                            'fixed_1': 2.0
                        },,
                    initial_states={'x': 0.0, 'y': 0.0},
                    timepoints=np.linspace(0, 10, 100),
                    observed={'x': data.x.values, 'y': data.y.values},
                )
                post_pred = pm.sample_posterior_predictive(trace, var_names=['model_name])

    """

    def __init__(
        self,
        amici_model: amici.Model,
        solver: amici.AmiciSolver,
        parameters: Dict[str, Any],
        fixed: Dict[str, Any],
        initial_states: Dict[str, Any],
        timepoints: np.array,
        coordinates: Optional[Dict[str, pd.Index]] = None,
        dimensions: Tuple[str] = ("time", "observable"),
        dtype=None,
        transform=transforms.log,
        *args,
        **kwds,
    ):
        try:
            model: pm.Model = Model.get_context()
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed to "
                "instantiate distributions. Add variable inside "
                "a 'with model:' block, or use the '.dist' syntax "
                "for a standalone distribution."
            )
        # if len(model.RV_dims) != 0:
        #     # TODO: Add handling of variable experiment coordinates
        #     print(f"model.RV_dims: {model.RV_dims}")
        #     print(f"model.observed_RVs: {model.observed_RVs}")
        #     # Get observable dims from model context,
        #     # get corresponding coordinates from model.coords
        #     print(f"model.coords: {model.coords}")
        #     print(f"model.deterministics: {model.deterministics}")

        # print(f"dimensions: {dimensions}")
        # print(f"shape_from_dims: {model.shape_from_dims(dimensions)}")
        shape = model.shape_from_dims(dimensions)
        # print(f"ODESystem shape: {shape}")

        self._model = amici_model
        self._solver = solver

        self._parameter_names = self._model.getParameterNames()
        self._fixed_names = self._model.getFixedParameterNames()
        self._state_names = self._model.getStateNames()
        self._observable_ids = self._model.getObservableIds()

        if not set(parameters.keys()).issubset(set(self._parameter_names)):
            raise KeyError(
                f"Parameter {set(parameters.keys()).difference(set(self._parameter_names))} not in `model`!"
            )
        if not set(fixed.keys()).issubset(set(self._fixed_names)):
            raise KeyError(
                f"Fixed parameter {set(fixed.keys()).difference(set(self._fixed_names))} not in `model`!"
            )
        if not set(initial_states.keys()).issubset(set(self._state_names)):
            raise KeyError(
                f"Initial state {set(initial_states.keys()).difference(set(self._state_names))} not in `model`!"
            )
        if self.data and set(self.data.keys()) != set(self._observable_ids):
            raise KeyError(
                f"Observable {set(self._observable_ids).difference(set(self.data.keys()))} missing from `observed`!"
            )

        if self.data:
            data_shapes = [tag_value_or_number(v).shape for k, v in self.data.items()]
            data_dims = [model.RV_dims[k] for k in self.data.keys()]
            # print(data_shapes)
            # print(data_dims)
            # print([d for tup in zip(data_dims, data_shapes) for t in tup ])
            # data_coords= {d:s for tup in zip(data_dims, data_shapes) for dims, shapes in tup}
            # print(data_coords)
            # print(data_shapes, data_dims)
            for i, data_dim in enumerate(data_dims):
                time_idx = data_dim.index("time")
                if data_shapes[i][time_idx] != len(tag_value_or_number(timepoints)):
                    raise ValueError(
                        f"Observed {list(self.data.keys())[i]} shape does not correspond to timepoints shape: {data_shape[0]} != {len(tag_value_or_number(timepoints))}"
                    )

        ### TODO: Every dimension not 'time' or 'observable' should be iterated over
        dimensions_red = list(dimensions)
        if dimensions_red[0] != "time":
            raise ValueError(
                f"First dimension of ODESystem should correspond to 'time'), not {dimensions_red[0]}"
            )
        if dimensions_red[1] != "observable":
            raise ValueError(
                f"Second dimension of ODESystem should correspond to 'observable'), not {dimensions_red[1]}"
            )
        dimensions_red = dimensions_red[2:]

        dim_coords = [model.coords[dim] for dim in dimensions_red]
        dim_coords_prod = itertools.product(*dim_coords)

        # print(dim_coords)
        # print(list())
        # for i, coord in enumerate(model.coords[dim]):
        #     print(i, dim, coord)

        self._parameters = [
            parameters.get(param_name) for param_name in self._parameter_names
        ]  # TODO: Separate sigma/std argument from other inputs?
        self._sigma_idx = [
            i for i, p in enumerate(self._parameter_names) if ("sigma" in p)
        ]
        # print(self._sigma_idx)
        # print(self._parameter_names)

        self._fixed = [
            fixed.get(fixed_name, self._model.getFixedParameterByName(fixed_name))
            for fixed_name in self._fixed_names
        ]

        # print(np.array([tag_value_or_number(fixed) for fixed in self._fixed]))

        fixed_dims = {
            fixed_name: model.RV_dims.get(fixed_name, ())
            for fixed_name in self._fixed_names
        }

        self._initial_states = [
            initial_states.get(
                state_name,
                self._model.getInitialStates()[self._state_names.index(state_name)],
            )
            for state_name in self._state_names
        ]

        self.timepoints = timepoints
        # self.shape = (
        #     len(tag_value_or_number(self.timepoints)),
        #     len(self._observable_ids),
        # )
        # print(self.shape)
        # print(args)
        # print(kwds)

        # Test simulation to acquire testval
        self._model.setTimepoints(tag_value_or_number(self.timepoints))
        # print(tag_value_or_number(self._fixed[0]))
        # print(np.array([tag_value_or_number(param) for param in self._parameters]))
        self._model.setParameters(
            np.array([tag_value_or_number(param) for param in self._parameters])
        )
        self._model.setFixedParameters(
            np.array([tag_value_or_number(fixed) for fixed in self._fixed])
        )
        self._model.setInitialStates(
            np.array([tag_value_or_number(initial) for initial in self._initial_states])
        )
        self._model.setAllStatesNonNegative()
        rdata = amici.runAmiciSimulation(self._model, self._solver)
        self.testval = rdata["y"] + np.random.normal(
            0,
            [tag_value_or_number(self._parameters[idx]) for idx in self._sigma_idx],
            rdata["y"].shape,
        )  # TODO: Add handling of noise estimate per observation
        # print(self.testval.shape)
        # print(
        #     np.array(
        #         [tag_value_or_number(self._parameters[idx]) for idx in self._sigma_idx]
        #     )
        # )
        if dtype is None:
            dtype = theano.config.floatX
        super().__init__(
            shape=shape,
            dtype=dtype,
            transform=transform,
            testval=self.testval,
            *args,
            **kwds,
        )

    def logp(self, **values):
        if len(values) == 0:
            edata = amici.ExpData(
                len(self._observable_ids),
                len(tag_value_or_number(self.timepoints)),
                0,  # Number of events
                tag_value_or_number(self.timepoints),
            )
        else:
            observed = np.array(
                [
                    tag_value_or_number(values.get(observable_id))
                    for observable_id in self._observable_ids
                ]
            ).T.flatten()  # Cast observed dictionary flattened array

            edata = amici.ExpData(
                len(self._observable_ids),
                len(tag_value_or_number(self.timepoints)),
                0,  # Number of events
                tag_value_or_number(self.timepoints),
            )
            edata.setObservedData(observed)

        # TODO: Create Log Op at initialization, not for every evaluation, separate edata input from model and solver input
        # TODO: add Fixed parameters, initial_states, timepoints, as extra Op input
        return LogLike(model=self._model, solver=self._solver, edata=edata)(
            *self._parameters
        )

    def random(self, point=None, size=None):
        parameter_values = draw_values(self._parameters, point=point, size=size)

        def generator(params, size):
            params_vec = np.array(params).T
            if len(params_vec.shape) == 1:
                self._model.setParameters(params_vec)
                rdata = amici.runAmiciSimulation(self._model, self._solver)
                return rdata["y"] + np.random.normal(
                    0,
                    np.array([params_vec[idx] for idx in self._sigma_idx]),
                    rdata["y"].shape,
                )

            else:
                out = []
                for param in params_vec:
                    self._model.setParameters(param)
                    rdata = amici.runAmiciSimulation(self._model, self._solver)
                    out.append(
                        rdata["y"]
                        + np.random.normal(
                            0,
                            np.array([param[idx] for idx in self._sigma_idx]),
                            rdata["y"].shape,
                        )
                    )
            return np.array(out)
        
        self._model.setAllStatesNonNegative()
        self._model.setTimepoints(tag_value_or_number(self.timepoints))
        self._model.setFixedParameters(
            np.array([tag_value_or_number(fixed) for fixed in self._fixed])
        )
        self._model.setInitialStates(
            np.array([tag_value_or_number(initial) for initial in self._initial_states])
        )
        samples = generate_samples(
            generator,
            params=parameter_values,
            dist_shape=(
                len(tag_value_or_number(self.timepoints)),
                len(self._observable_ids),
            ),
            size=size,
            broadcast_shape=(
                len(tag_value_or_number(self.timepoints)),
                len(self._observable_ids),
            ),
        )
        return samples

    def _distr_parameters_for_repr(self):
        return []


class LogLike(tt.Op):
    def __init__(
        self, model: amici.Model, solver: amici.AmiciSolver, edata: amici.AmiciExpData
    ):

        self._model = model
        self._solver = solver
        self._edata = edata

        self._loglikegrad = LogLikeGrad(
            model=self._model, solver=self._solver, edata=self._edata
        )

    def make_node(self, *inputs):
        """should make sure that the inputs correspond to parameters to be set in amici model, including std_dev or sigma"""
        if len(inputs) != len(self._model.getParameterList()):
            raise TypeError(
                f"Model requires {len(self._model.getParameterList())} input parameters, but {len(inputs)} were given"
            )
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        for i in inputs:
            if i.type != tt.dscalar:
                raise TypeError(f"Input {i} must be a double scalar!")
        return theano.graph.basic.Apply(
            self,
            [*inputs],
            [
                tt.dscalar(),
                # tt.dvector('grad')
            ],
        )

    def perform(self, node, inputs, output_storage):
        """Calculate loglikelihood 'llh' for every observation/set of observations -> sum"""

        # print(inputs)
        self._model.setParameters(np.array(inputs))

        rdata = amici.runAmiciSimulation(self._model, self._solver, self._edata)
        output_storage[0][0] = np.array(rdata["llh"])
        # output_storage[1][0] = rdata['sllh']
        # print(np.array(rdata['sllh']))

    def grad(self, inputs, g):
        # return [g[0] for g_out in self._loglikegrad(*inputs)]
        g_out = self._loglikegrad(*inputs)
        return [g[0] * gll for gll in g_out]
        # g[0]*g_out[0], g[0]*g_out[1], g[0]*g_out[2]]


class LogLikeGrad(tt.Op):
    def __init__(
        self, model: amici.Model, solver: amici.AmiciSolver, edata: amici.AmiciExpData
    ):
        self._model = model
        self._solver = solver
        self._edata = edata

    def make_node(self, *inputs):
        """should make sure that the inputs correspond to parameters to be set in amici model, including std_dev"""
        if len(inputs) != len(self._model.getParameterList()):
            raise TypeError(
                f"Model requires {len(self._model.getParameterList())} input parameters, but {len(inputs)} were given"
            )
        inputs = [tt.as_tensor_variable(i) for i in inputs]
        for i in inputs:
            if i.type != tt.dscalar:
                raise TypeError(f"Input {i} must be a double scalar!")
        return theano.graph.basic.Apply(self, [*inputs], [tt.dscalar() for i in inputs])

    def perform(self, node, inputs, output_storage):
        """Calculate loglikelihood 'llh' for every observation/set of observations -> sum"""
        self._model.setParameters(np.array(inputs))
        rdata = amici.runAmiciSimulation(self._model, self._solver, self._edata)
        for i, j in enumerate(inputs):
            output_storage[i][0] = np.array(rdata["sllh"][i])
