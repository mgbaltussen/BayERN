import numpy as np
import scipy.optimize as optimize
import theano.tensor as tt
import theano

# def find_root(fun, jac, *args):
#     return optimize.root(fun=fun, x0=[args[0][1],0.0], jac=jac, args=args).x

class RootFinderDatasetOp(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dmatrix]

    """ Creates an operator that calculates the expected steady-states for multiple conditions

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

        self.gradx_phi = FindRootDatasetDiffOp(self.grad_phi, self.theta_set)

        self.find_root = find_root

    def perform(self, node, inputs, output_storage):
        phi, = inputs
        output_storage[0][0] = np.array([self.find_root(self.f, self.j, phi, theta) for theta in self.theta_set])

    def grad(self, inputs, output_grads):
        phi,  = inputs
        x = self(phi)
        # xs = np.array([self.find_root(self.f, self.j, phi, theta) for theta in self.theta_set])

        output, _ = theano.scan(
            lambda a, b: np.dot(a.T, b),
            sequences=[self.gradx_phi(x,phi), output_grads[0]],
            # non_sequences=[output_grads[0]]
        )
        return [
            output.sum(axis=[0,2])
        ]

class FindRootDatasetDiffOp(tt.Op):
    itypes = [tt.dmatrix, tt.dvector]
    otypes = [tt.dtensor3]

    def __init__(self, gradx, theta_set) -> None:
        self.gradx = gradx
        self.theta_set = theta_set
    def perform(self, node, inputs, output_storage):
        xs, phi = inputs
        output_storage[0][0] = np.array([self.gradx(x, phi, theta) for theta, x in zip(self.theta_set, xs)])

class RootFinderOp(tt.Op):
    itypes = [tt.dvector, tt.dvector]
    otypes = [tt.dvector]

    """ Creates an operator that calculates the expected steady-state
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

        self.gradx_theta = FindRootDiffOp(self.grad_theta)
        self.gradx_phi = FindRootDiffOp(self.grad_phi)

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

class FindRootDiffOp(tt.Op):
    itypes = [tt.dvector, tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    def __init__(self, gradx) -> None:
        self.gradx = gradx
    def perform(self, node, inputs, outputs):
        x, phi, theta = inputs
        outputs[0][0] = self.gradx(x, phi, theta)