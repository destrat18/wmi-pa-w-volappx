from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times
from pysmt.typing import BOOL, REAL

from wmipa import WMI

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
