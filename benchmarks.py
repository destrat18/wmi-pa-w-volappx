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

group1 = [
        #########################################################
        {
                'bounds': [0.01, 1],
                'w': "(1/x)*100",
                "smt_phi": And(GE(x, Real(0.01)),LE(x, Real(1))),
                'smt_w': Div(Real(1), x)*Real(100)        
        },
        
        #########################################################
        {
                'bounds': [1.01, 2],
                'w': "(1/(x**2 -1))*100",
                "smt_phi": And(GE(x, Real(1.01)),LE(x, Real(2))),
                'smt_w': Div(Real(1), Pow(x, Real(2)) - Real(1))*Real(100)        
        },
        
        #########################################################
        {
                'bounds': [0, 1],
                'w': f"(x/( {a} + x**2))*100",
                "smt_phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'smt_w': Div(x, a+ Pow(x, Real(2)))*Real(100)        
        },

        #########################################################
        {
                'bounds': [0, 1],
                'w': f"((x**3)/({a}+x**2))*100",
                "smt_phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'smt_w': Div(Pow(x, Real(3)), a + Pow(x, Real(2)))*Real(100)        
        },
        
        #########################################################
        {
                'bounds': [0, 1],
                'w': f"(1/({a}*x**2 + {b}*x + {c}))*100",
                "smt_phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'smt_w': Div(Real(1), a*Pow(x, Real(2)) + b*x + c)*Real(100)        
        },
        
        #########################################################
        {
                'bounds': [0, 1],
                'w': f"(1/( ({a}+x) * (x+{b}) ))*100",
                "smt_phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'smt_w': Div(Real(1), Times(x+a, x+b) )*Real(100)        
        },
        
        #########################################################
        {
                'bounds': [[0.01, 1], [0.01, 1]],
                'w': f"(1/(x+y))*100",
                "smt_phi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0.01)),LE(y, Real(1)),),
                'smt_w': Div(Real(1), x+y)*Real(100)        
        },
]


######################################################### 
# Radical Functions

group2 = [  
        ######################################################### 
        {
                'bounds': [0.01, 1],
                'w': f"(1/x)**(1/2)",
                "smt_phi": None,
                'smt_w': None       
        },
]