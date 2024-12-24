__version__ = "0.999"
__author__ = "Soroush Farokhnia"

from subprocess import call

from wmipa.integration.command_line_integrator import CommandLineIntegrator
from wmipa.integration.expression import Expression
from wmipa.integration.polytope import Polytope
from wmipa.wmiexception import WMIRuntimeException, WMIIntegrationException
from wmipa.integration.faza import faza
import sympy as sym
import logging

# TODO: Add is_faza_installed
_FAZA_INSTALLED = True


class FazaIntegrator(CommandLineIntegrator):

    # Error threshold
    threshold=0.1
    
    # Handelman degree
    degree=None
    
    # Max workers
    max_workers = 1

    def __init__(self,
                degree=degree,
                threshold=threshold,
                max_workers=max_workers,
                **options
                ):
        """Default constructor.
        """
        CommandLineIntegrator.__init__(self, **options)
        self.threshold = threshold
        self.degree = degree
        self.max_workers=max_workers
    
    def _integrate_problem(self, integrand, polytope):
        """Generates the input files and calls integrator executable
            to calculate the integral. Then, reads back the result and returns it
            as a float.

        Args:
            integrand (Integrand): The integrand of the integration.
            polytope (Polytope): The polytope of the integration.

        Returns:
            real: The integration result.

        """
        
        variables = sorted(integrand.variables.union(polytope.variables))
        
        # Create the string representation of the polytope (LattE format)
        var_bounds = []
        
        for var in variables:
            lower = -float('inf')
            upper = float('inf')
            for _, bound in enumerate(polytope.bounds):
                if var in bound.coefficients:
                    if bound.coefficients[var] < 0:
                        lower = max(bound.constant/bound.coefficients[var], lower)
                    elif bound.coefficients[var] > 0:
                        upper = min(upper, bound.constant/bound.coefficients[var])
                        
            var_bounds.append([lower+0, upper+0])
        
        integrand = sym.parse_expr(str(integrand))
        
        if self.degree is None:
            total_degree = sym.total_degree(integrand)
            logging.info(f"Total degree of {integrand} is {total_degree}!") 
        else:
            total_degree = self.degree
        
        volume = faza.calculate_approximate_wmi(
            degree=total_degree,
            max_workers=self.max_workers,
            integrand=integrand,
            vars=[sym.var(str(var)) for var in variables],
            bounds=var_bounds,
            threshold=self.threshold
        )
        
        print(integrand, "on", polytope, "=", volume)

        return volume
        
    def _write_integrand_file(self, integrand, variables, path):
        """Writes the integrand to the given file.

        Args:
            integrand (Expression): The integrand.
            variables (list): The list of variables.
            path (str): The path where to write the integrand.

        """
        
        
        with open(path, "w") as f:
            f.write("{} \n {} \n".format(" ".join(variables), str(integrand)))

    @classmethod
    def _make_problem(cls, weight, bounds, aliases):
        """Makes the problem to be solved by VolEsti.
        Args:
            weight (FNode): The weight function.
            bounds (list): The polytope.
            aliases (dict): The aliases of the variables.
        Returns:
            integrand (Expression): The integrand.
            polytope (Polytope): The polytope.
        """
        integrand = Expression(weight, aliases)
        polytope = Polytope(bounds, aliases)
        return integrand, polytope

    def _call_integrator(self, polynomial_file, polytope_file, output_file):
        """Calls VolEsti executable to calculate the integrand of the problem
            represented by the given files.

        Args:
            polynomial_file (str): The path where to find the file with the
                representation of the polynomial.
            polytope_file (str): The path where to find the file with representation
                of the polytope.
            output_file (str): The file where to write the result of the computation.

        """
        
        if not _FAZA_INSTALLED:
            raise WMIIntegrationException(WMIIntegrationException.INTEGRATOR_NOT_INSTALLED, "Faza")
        
        cmd = [
            "python3", "/workspaces/wmi-pa-w-volappx/wmipa/integration/faza/wmi.py",
            "--integrand",
            polynomial_file,
            "--bounds",
            polytope_file,
            "--degree",
            str(self.degree),
            "--max-workers",
            str(self.max_workers),
            "--threshold",
            str(self.threshold),
        ]

        with open(output_file, "w") as f:
            return_value = call(cmd, stdout=f)
            if return_value != 0:
                raise WMIIntegrationException(WMIIntegrationException.OTHER_ERROR, "Error while calling VolEsti")

    def to_json(self):
        return {"name": "volesti",
                "algorithm": self.algorithm,
                "error": self.error,
                "walk_type": self.walk_type,
                "walk_length": self.walk_length,
                "seed": self.seed,
                "N": self.N,
                "n_threads": self.n_threads}

    def to_short_str(self):
        return "faza_{}_{}_{}_{}_{}_{}".format(self.algorithm, self.error, self.walk_type, self.walk_length,
                                                  self.seed, self.N)
