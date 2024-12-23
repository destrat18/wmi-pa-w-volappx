from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div
from pysmt.typing import BOOL, REAL
import time

from wmipa import WMI
from wmipa.integration import LatteIntegrator
from wmipa.integration import SymbolicIntegrator
from wmipa.integration import VolestiIntegrator

x = Symbol("x", REAL)


w = x

# Here we are showing in addition to that they cann't take integral over algeabric set, we also showing that even on polytope they can not do some functions
w = Div(Real(1),x)

phi = Bool(True)

# This breaks
chi = And(
    GE(x, Real(0.01)),
    LE(x, Real(1)),
)

#This doesn't break
# chi = And(
#     GE(x, Real(1)),
#     LE(x, Real(2)),
# )


print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())

for integrator in [VolestiIntegrator(), LatteIntegrator()]:
    for mode in [
            WMI.MODE_ALLSMT, 
            WMI.MODE_PA, 
            WMI.MODE_SA_PA, 
            WMI.MODE_SAE4WMI

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
                "WMI with mode {}, \t integrator = {}, \t failed = {}, \t time = {:.2f}".format(
                    mode, integrator.__class__.__name__, str(e), time.time()-start_time
                )
            )