from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times
from pysmt.typing import BOOL, REAL

from wmipa import WMI
from wmipa.integration import LatteIntegrator
# from wmipa.integration import SymbolicIntegrator
from wmipa.integration import VolestiIntegrator
from wmipa.integration import FazaIntegrator


# from wmipa.integration.symbolic_integrator import SymbolicIntegrator

# variables definition
a = Symbol("A", BOOL)
x = Symbol("x", REAL)

# formula definition
# fmt: off
phi = And(Iff(a, GE(x, Real(0))),
          GE(x, Real(-1)),
          LE(x, Real(1)))

# weight function definition
w = Ite(GE(3 * x, Real(0)),
        x,
        Times(Real(-1), x))
# fmt: on

chi = Bool(True)

print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())


import time, argparse
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        prog='Mega Miners',
        description='I am approximating!'
        )
        
    parser.add_argument("--degree", help="Handelman degree", type=int, default=None)
    parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
    parser.add_argument("--threshold", help="Error threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    for integrator in [
            FazaIntegrator(max_workers=args.max_workers, threshold=args.threshold)
        ]:
        for mode in [
                WMI.MODE_SAE4WMI
            ]:
            try:
                start_time = time.time()
                wmi = WMI(chi, w, integrator=integrator)
                result, n_integrations = wmi.computeWMI(phi, mode=mode)
                print(
                    "WMI with mode {}, \t integrator = {}, \t result = {}, \t # integrations = {}, \t time = {:.2f}s({:.2f}h)".format(
                        mode, integrator.__class__.__name__, result, n_integrations, time.time()-start_time,(time.time()-start_time)/3600
                    )
                )
            except Exception as e:
                print(
                    "WMI with mode {}, \t integrator = {}, \t failed = {}".format(
                        mode, integrator.__class__.__name__, str(e)
                    )
                )