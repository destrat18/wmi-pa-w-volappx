from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div
from pysmt.typing import BOOL, REAL

from wmipa import WMI
from wmipa.integration import LatteIntegrator
from wmipa.integration import VolestiIntegrator

# variables definition
x = Symbol("x", REAL)

# formula definition
# fmt: off
phi = And(GE(x, Real(1)),
          LE(x, Real(2)))

# weight function definition
w = x
# fmt: on

chi = And(GE(x, Real(1)),
          LE(x, Real(2)))

print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())

print()
for mode in [WMI.MODE_ALLSMT, WMI.MODE_PA, WMI.MODE_SA_PA, WMI.MODE_SAE4WMI]:
    for integrator in (LatteIntegrator(), VolestiIntegrator()):
        try:
            wmi = WMI(chi, w, integrator=integrator)
            result, n_integrations = wmi.computeWMI(phi, mode=mode)
            print(
                "WMI with mode {:10} (integrator: {:20})\t "
                "result = {}, \t # integrations = {}".format(
                    mode, integrator.__class__.__name__, result, n_integrations
                )
            )
        except Exception as e:
            print(
                "WMI with mode {:10} (integrator: {:20})\t "
                "Faild = {}".format(
                    mode, integrator.__class__.__name__, str(e)
                )
            )