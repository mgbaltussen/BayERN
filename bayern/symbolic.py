import sympy as sp


class SymbolicSystem():

    def __init__(self,
        species, kinetic_parameters,
        control_parameters, rate_equations
        ) -> None:
        self.species = species
        self.kinetic_parameters = kinetic_parameters
        self.control_parameters = control_parameters
        self.sym_rate_equations = rate_equations

        sym_f = sp.Matrix(rate_equations)
        self.sym_jac_species = sym_f.jacobian(species)
        self.sym_jac_kinetics = sym_f.jacobian(kinetic_parameters)
        self.sym_jac_controls = sym_f.jacobian(control_parameters)
        inv_sym_jac_species = self.sym_jac_species.inv().T
        self.sym_grad_kinetics = -inv_sym_jac_species * self.sym_jac_kinetics
        self.sym_grad_controls = -inv_sym_jac_species * self.sym_jac_controls

        self.num_rate_equations = sp.lambdify([species, kinetic_parameters, control_parameters], self.sym_rate_equations, "numpy")
        self.num_jac_species = sp.lambdify([species, kinetic_parameters, control_parameters], self.sym_jac_species, "numpy")
        self.num_jac_kinetics = sp.lambdify([species, kinetic_parameters, control_parameters], self.sym_jac_kinetics, "numpy")
        self.num_jac_controls = sp.lambdify([species, kinetic_parameters, control_parameters], self.sym_jac_controls, "numpy")
        self.num_grad_kinetics = sp.lambdify([species, kinetic_parameters, control_parameters], self.sym_grad_kinetics, "numpy")
        self.num_grad_controls = sp.lambdify([species, kinetic_parameters, control_parameters], self.sym_grad_controls, "numpy")