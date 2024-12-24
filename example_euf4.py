from pysmt.shortcuts import GE, LE, LT, And, Bool, Iff, Ite, Plus, Real, Symbol, Times
from pysmt.typing import BOOL, REAL

from wmipa import WMI

# variables definition
a = Symbol("A", BOOL)
x1 = Symbol("x1", REAL)
x2 = Symbol("x2", REAL)

# formula definition
# fmt: off
phi = Bool(True)

# weight function definition
w = Plus(Ite(GE(x1, Real(0)),
             Ite(GE(x1, Real((1, 2))),
                 Times(x1, Real(3)),
                 Times(Real(-2), x1)),
             Times(Real(-1), x1)
             ),
         Ite(a,
             Times(Real(3), x2),
             Times(Real(-1), Times(x2, Real(5)))))

chi = And(LE(Real(-1), x1), LT(x1, Real(1)),
          LE(Real(-1), x2), LT(x2, Real(1)),
          Iff(a, GE(x2, Real(0))))
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