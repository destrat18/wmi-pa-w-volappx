
from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div, Pow
from pysmt.typing import BOOL, REAL
import time
from wmipa import WMI
from wmipa.integration import LatteIntegrator
from wmipa.integration import FazaIntegrator
from wmipa.integration import VolestiIntegrator


x = Symbol("x", REAL)

w = x*x

phi = And(
    GE(x, Real(0)),
    LE(x, Real(1)),
)

chi = Bool(True)


print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())

for integrator in [
    LatteIntegrator(),
    VolestiIntegrator(), 
    FazaIntegrator(degree=2, max_workers=8)
    ]:
    for mode in [
        WMI.MODE_PA, 
        ]:
        try:
            start_time = time.time()
            wmi = WMI(chi, w, integrator=integrator)
            result, n_integrations = wmi.computeWMI(phi, mode=mode)
            print(
                "WMI with mode {}, \t integrator = {}, \t result = {}, \t # integrations = {}, \t time: {:.3f}s".format(
                    mode, integrator.__class__.__name__, result, n_integrations, time.time()-start_time
                )
            )
        except Exception as e:
            print(
                "WMI with mode {}, \t integrator = {}, \t failed = {}".format(
                    mode, integrator.__class__.__name__, str(e)
                )
            )