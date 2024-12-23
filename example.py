from pysmt.shortcuts import Iff, Implies, Ite, Symbol
from pysmt.typing import BOOL, REAL
# from wmipa.integration import VolestiIntegrator, LatteIntegrator
from wmipa.integration import FazaIntegrator
import argparse

from wmipa import WMI
import time

# variables definition
a = Symbol("A", BOOL)
b = Symbol("B", BOOL)
c = Symbol("C", BOOL)
x = Symbol("x", REAL)
y = Symbol("y", REAL)

# formula definition
# fmt: off
phi = Implies(a | b, x >= 1) & Implies(
    a | c, x <= 2) & Ite(b, Iff(a & c, y <= 2), y <= 1)

print("Formula:", phi.serialize())

# weight function definition
w = Ite(b,
        Ite(x >= 0.5,
            x * y,
            Ite((x >= 1),
                x + 2 * y,
                2 * x + y
                )
            ),
        Ite(a | c,
            x * x * y,
            2 * x + y
            )
        )
# fmt: on

chi = (x >= 0) & (x <= 3) & (y >= 0) & (y <= 4)
print("Weight function:", w.serialize())
print("Support:", chi.serialize())


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