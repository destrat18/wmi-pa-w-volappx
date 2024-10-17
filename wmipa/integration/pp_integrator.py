__version__ = "0.999"
__author__ = "Paolo Morettin"

from subprocess import call
from uuid import uuid4
from wmipa.integration import _is_latte_installed
from wmipa.integration.command_line_integrator import CommandLineIntegrator
from wmipa.integration.polytope import Polynomial, Polytope
from wmipa.wmiexception import WMIRuntimeException, WMIIntegrationException

_LATTE_INSTALLED = _is_latte_installed()


class PPIntegrator(CommandLineIntegrator):
    """This class is a wrapper for the PP integrator.

    It handles the integration of polynomial functions over (convex) polytopes, using an approximation algorithms.

    PP Integrale is required.

    Attributes:
        n_threads (int): The number of threads to use.
    """

    ALG_TRIANGULATE = "--triangulate"
    ALG_CONE_DECOMPOSE = "--cone-decompose"

    INPUT_TEMPLATE = """
    (declare-const c_1 Real)
    (declare-const c_2 Real)
    (declare-const c_3 Real)
    (declare-const c_4 Real)

    (assert (forall ({variables}) (=> (
            and {bounds}
        ) 
        (
            and ({newVar}>=0) ({integrand} > {newVar}) ({polytope})  
        )
    )))

    (check-sat)
    (get-model)    
    """
    
    def __init__(self, **options):
        """Default constructor.

        It calls the init method of the parent.

        Args:
            algorithm (str): The algorithm to use when computing the integrals.
            options: @see CommandLineIntegrator.__init__

        """
        super().__init__(**options)
        
        # TODO remove it for multithreading
        self.polytope = None
    
    @classmethod
    def _make_problem(cls, weight, bounds, aliases):
        """Makes the problem to be solved by LattE.
        Args:
            weight (FNode): The weight function.
            bounds (list): The polytope.
            aliases (dict): The aliases of the variables.
        Returns:
            integrand (Polynomial): The integrand.
            polytope (Polytope): The polytope.
        """
        integrand = Polynomial(weight, aliases)
        polytope = Polytope(bounds, aliases)

        return integrand, polytope

    def _write_integrand_file(self, integrand, variables, path):
        """Writes the integrand to the given file.

        Args:
            integrand (Polynomial): The integrand.
            variables (list): The list of variables.
            path (str): The path where to write the file.

        """

        new_variable = f"var{str(uuid4()).split('-')[0]}"

        variables_repr = " ".join([ f"({var} Real)" for var in variables+[new_variable]])

        # Create the string representation of the polytope (LattE format)

        bound_repr_list = []
        for _, bound in enumerate(self.polytope.bounds):
            
            bound_repr = ">= (+ {constant} {coefficients} ) 0".format(
                constant = bound.constant,
                coefficients = " ".join([f"(* {var} (* -1 {bound.coefficients[var]}) )" for var in variables if var in bound.coefficients])
            )
            bound_repr_list.append(bound_repr)

        phi_repr = f"and {' '.join([f'({b})' for b in bound_repr_list])}"
        

        # Create the string representation of the integrand (LattE format)
        monomials_repr_list = []
        
        for monomial in integrand.monomials:
            monomials_repr = "* {constant} {coefficients}".format(
                constant = monomial.coefficient,
                coefficients = " ".join([f"(* 1 {' '.join([var for i in range(int(monomial.exponents[var]))])})" for var in variables if var in monomial.exponents])
            )
            monomials_repr_list.append(monomials_repr)
                
        integrand_repr = "+ " + " ".join([f"({m})" for m in monomials_repr_list])


        pp_str = """
        
        (declare-const c_1 Real)
        (declare-const c_2 Real)
        (declare-const c_3 Real)
        (declare-const c_4 Real)

        (assert (forall ({variables}) (=> (
                and {chi}
            ) 
            (
                and (>= {newVar} 0) (>= {integrand} {newVar}) ({phi})  
            )
        )))

        (check-sat)
        (get-model)    
        """.format(
            variables=variables_repr,
            chi = 0,
            newVar = new_variable,
            integrand=integrand_repr,
            phi = phi_repr
        )
        
        # Write the string on the file
        with open(path, "w") as f:
            f.write(phi_repr)
            
        
    def _write_polytope_file(self, polytope, variables, path):
        """Writes the polytope into a file from where the integrator will read.

        Args:
            polytope (Polytope): The polytope of the integration.
            variables (list): The sorted list of all the variables involved in the
                integration.
            path (str): The path of the file to write.
        """
        self.polytope = polytope
        

    def _call_integrator(self, polynomial_file, polytope_file, output_file):
        """Calls LattE executable to calculate the integrand of the problem
            represented by the given files.

        Args:
            polynomial_file (str): The path where to find the file with the
                representation of the polynomial.
            polytope_file (str): The path where to find the file with representation
                of the polytope.
            output_file (str): The file where to write the result of the computation.

        """
        if not _LATTE_INSTALLED:
            raise WMIIntegrationException(WMIIntegrationException.INTEGRATOR_NOT_INSTALLED, "LattE")

        cmd = [
            "integrate",
            "--valuation=integrate",
            self.algorithm,
            "--monomials=" + polynomial_file,
            polytope_file,
        ]

        with open(output_file, "w") as f:
            return_value = call(cmd, stdout=f, stderr=f)
            if return_value != 0:
                # print(open(output_file).read())
                """
                if return_value != 0:
                    msg = "LattE returned with status {}"
                    # LattE returns an exit status != 0 if the polytope is empty.
                    # In the general case this may happen, raising an exception
                    # is not a good idea.
                """

    def to_json(self):
        return {"name": "latte", "algorithm": self.algorithm, "n_threads": self.n_threads}

    def to_short_str(self):
        return "latte"
