from pysmt.shortcuts import LE, And, Bool, Implies, Ite, Not, Plus, Real, Symbol, Times
from pysmt.typing import REAL

from wmipa import WMI

# variables definition
x = Symbol("x", REAL)
y = Symbol("y", REAL)

# formula definition
# fmt: off
phi = And(Implies(LE(y, Real(1)), And(LE(Real(0), x), LE(x, Real(2)))),
          Implies(Not(LE(y, Real(1))), And(LE(Real(1), x), LE(x, Real(3)))),
          LE(Real(0), y), LE(y, Real(2)))

# weight function definition
w = Ite(LE(y, Real(1)),
        Plus(x, y),
        Times(Real(2), y))

chi = Bool(True)
# fmt: on

print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())


from wmipa.integration import FazaIntegrator
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