import numpy as np
import pytensor
import pytensor.tensor as tt
from pytensor.graph.op import Op
# import theano.tensor as tt
# import theano


class SteadyStateDatasetOp(Op):
    itypes = [tt.dvector]
    otypes = [tt.dmatrix]

    """ Creates an operator that calculates the expected steady-states for multiple experimental conditions.

    This operator iterates over all included experimental conditions to calculate the expected steady-state concentrations.
    It also calculates the gradients (or rather, assembles the computational graph) of the steady-state concentrations with respect to the kinetic parameters. 
    This is implemented using theano.scan as the iteration algorithm, which is notoriously slow, but the only method currently available.   

    itypes: (K) vector of kinetic parameters (as theano operators)
    otypes: M x N matrix of (experimental condition) x (steady-state concentration)

    Parameters
    ----------
    f: Numeric rate equation function
    j: Numeric jacobian wrt species
    grad_phi: Numeric gradient wrt kinetics
    grad_theta: Numeric gradient wrt controls
    find_root: Numeric root optimizer
    theta_set: Array of control parameters values used during experiments
    """

    def __init__(self, f, j, grad_phi, grad_theta, find_root, theta_set) -> None:
        self.f = f
        self.j = j
        self.grad_theta = grad_theta
        self.grad_phi = grad_phi
        self.theta_set = theta_set

        self.gradx_phi = SteadyStateDatasetGradOp(self.grad_phi, self.theta_set)

        self.find_root = find_root

    def perform(self, node, inputs, output_storage):
        phi, = inputs
        output_storage[0][0] = np.array([self.find_root(self.f, self.j, phi, theta) for theta in self.theta_set])

    def grad(self, inputs, output_grads):
        phi,  = inputs
        x = self(phi)

        output, _ = pytensor.scan(
            lambda a, b: np.dot(a.T, b),
            sequences=[self.gradx_phi(x,phi), output_grads[0]],
        )
        return [
            output.sum(axis=[0,2])
        ]

class SteadyStateDatasetGradOp(Op):
    itypes = [tt.dmatrix, tt.dvector]
    otypes = [tt.dtensor3]

    """
    itypes[0]: (M x N) matrix of (experimental condition) x (steady-state concentration)
    itypes[1]: (K) vector of kinetic parameters (as theano operators)
    otypes: (M x N x K) tensor of (experimental condition) x (steady-state concentration) x (kinetic parameter)

    """
    def __init__(self, gradx, theta_set) -> None:
        self.gradx = gradx
        self.theta_set = theta_set
    def perform(self, node, inputs, output_storage):
        xs, phi = inputs
        output_storage[0][0] = np.array([self.gradx(x, phi, theta) for theta, x in zip(self.theta_set, xs)])

class SteadyStateOp(Op):
    itypes = [tt.dvector, tt.dvector]
    otypes = [tt.dvector]

    """ Creates an operator that calculates the expected steady-state for a single experimental condition.

    Parameters
    ----------
    phi: system parameters
    theta: control parameters
    """

    def __init__(self, f, j, grad_phi, grad_theta, find_root) -> None:
        """
            Args
            ----
            f: Numeric rate equation function
            j: Numeric jacobian wrt species
            grad_phi: Numeric gradient wrt kinetics
            grad_theta: Numeric gradient wrt controls
            find_root: Numeric root optimizer
        """
        self.f = f
        self.j = j
        self.grad_theta = grad_theta
        self.grad_phi = grad_phi

        self.gradx_theta = SteadyStateDiffOp(self.grad_theta)
        self.gradx_phi = SteadyStateDiffOp(self.grad_phi)

        self.find_root = find_root

    def perform(self, node, inputs, output_storage):
        phi, theta = inputs
        output_storage[0][0] = self.find_root(self.f, self.j, phi, theta)

    def grad(self, inputs, output_grads):
        phi, theta = inputs
        x = self(phi, theta)

        return [
            np.dot(self.gradx_phi(x, phi, theta).T, output_grads[0]).sum(axis=1),
            np.dot(self.gradx_theta(x, phi, theta).T, output_grads[0]).sum(axis=1)
        ]

class SteadyStateDiffOp(Op):
    itypes = [tt.dvector, tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, gradx) -> None:
        self.gradx = gradx
    def perform(self, node, inputs, output_storage):
        x, phi, theta = inputs
        outputs[0][0] = self.gradx(x, phi, theta)