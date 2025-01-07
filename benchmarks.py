from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div, Pow
from pysmt.typing import BOOL, REAL

from wmipa import WMI
from wmipa.integration import LatteIntegrator, VolestiIntegrator, FazaIntegrator
import pandas as pd
import time, argparse


# variables definition
x = Symbol("x", REAL)
y = Symbol("y", REAL)
z = Symbol("z", REAL)


a = Real(4)
b = Real(9)
c = Real(16)

benchmark_group1 = [
        # ##################### 1/x ######################
        {
                "phi": And(GE(x, Real(0.01)),LE(x, Real(1))),
                'w': Div(Real(1), x)*Real(100)        
        },
        
        # ###################### 1/(x^2-1) ######################
        {
                "phi": And(GE(x, Real(1.01)),LE(x, Real(2))),
                'w': Div(Real(1), Pow(x, Real(2)) - Real(1))*Real(100)        
        },
        ###################### x/(4+x^2) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(x, a+ Pow(x, Real(2)))*Real(100)        
        },

        ###################### x^3/(4+x^2) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(Pow(x, Real(3)), a + Pow(x, Real(2)))*Real(100)        
        },
        
        ###################### 1/(a*x^2 + b*x + c) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(Real(1), a*Pow(x, Real(2)) + b*x + c)*Real(100)        
        },
        
        ###################### 1/(a*x^2 + b*x + c) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(Real(1), Times(x+a, x+b) )*Real(100)        
        },
        # ###################### 1/(x+y) ######################
        {
                "phi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0.1)),LE(y, Real(1)),),
                'w': Div(Real(1), x+y)*Real(100)        
        },
]

