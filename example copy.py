
from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div, Pow
from pysmt.typing import BOOL, REAL

from wmipa import WMI
from wmipa.integration import LatteIntegrator
from wmipa.integration import SymbolicIntegrator
from wmipa.integration import VolestiIntegrator


x = Symbol("x", REAL)
y = Symbol("y", REAL)
z = Symbol("z", REAL)


w = Real(1)
phi = Bool(True)


# They can not handle integral over algeabric set
chi = And(
    GE(Real(1) - Pow(x, Real(2)) - Pow(y, Real(2)) - Pow(z, Real(2)) + Times(Real(2), x, y, z), Real(0)),
    GE(x, Real(-1)),
    LE(x, Real(1)),
    GE(y, Real(-1)),
    LE(y, Real(1)),
    GE(z, Real(-1)),
    LE(z, Real(1)),
)


print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())

for integrator in [VolestiIntegrator(), LatteIntegrator()]:
    for mode in [
        WMI.MODE_PA, 
        # WMI.MODE_SA_PA, WMI.MODE_SAE4WMI
        ]:
        try:
            wmi = WMI(chi, w, integrator=integrator)
            result, n_integrations = wmi.computeWMI(phi, mode=mode)
            print(
                "WMI with mode {}, \t integrator = {}, \t result = {}, \t # integrations = {}".format(
                    mode, integrator.__class__.__name__, result, n_integrations
                )
            )
        except Exception as e:
            print(
                "WMI with mode {}, \t integrator = {}, \t failed = {}".format(
                    mode, integrator.__class__.__name__, str(e)
                )
            )