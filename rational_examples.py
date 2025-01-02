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

examples = [
        # ###################### x^2 ######################
        # {
        #         "phi": And(GE(x, Real(0)),LE(x, Real(1))),
        #         'w': Pow(x, Real(2))        
        # },
        
        # ##################### 1/x ######################
        # {
        #         "phi": And(GE(x, Real(0.01)),LE(x, Real(1))),
        #         'w': Div(Real(1), x)        
        # },
        
        ###################### x/(4+x^2) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(x, a+ Pow(x, Real(2)))        
        },

        ###################### x^3/(4+x^2) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(Pow(x, Real(3)), a + Pow(x, Real(2)))        
        },
        
        ###################### 1/(a*x^2 + b*x + c) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(Real(1), a*Pow(x, Real(2)) + b*x + c)        
        },
        
        ###################### 1/(a*x^2 + b*x + c) ######################
        {
                "phi": And(GE(x, Real(0)),LE(x, Real(1))),
                'w': Div(Real(1), Times(x+a, x+b) )        
        },
        ###################### 1/(x+y) ######################
        {
                "phi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0)),LE(y, Real(1)),),
                'w': Div(Real(1), x+y)        
        },
]


# fmt: on
if __name__ == "__main__":

        parser = argparse.ArgumentParser(
                prog='Faza Integrator',
                description='I am approximating!'
                )
                
        parser.add_argument("--degree", help="Handelman degree", type=int, default=None)
        parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
        parser.add_argument("--threshold", help="Error threshold", type=float, default=0.1)
        
        args = parser.parse_args()
        
        chi = Bool(True)

        print()
        mode = WMI.MODE_SAE4WMI

        results = []
        results_path = f"rational_example_results_{int(time.time())}.csv"

        for example in examples:

                w = example['w']
                phi = example['phi']

                for integrator in (
                        LatteIntegrator(), 
                        VolestiIntegrator(), 
                        FazaIntegrator(max_workers=args.max_workers, threshold=args.threshold)
                        ):
                        
                        start_time = time.time()
                        try:
                                wmi = WMI(chi, w, integrator=integrator)
                                volume, n_integrations = wmi.computeWMI(phi, mode=mode)
                                print(
                                        "Formula {}, \t integrator= {}, \t result = {}, \t time = {:.4f}s".format(
                                                w.serialize(), integrator.__class__.__name__, volume, time.time()-start_time
                                        )
                                )
                                results.append(
                                {
                                        'time': time.time()-start_time,
                                        'formula': w.serialize(),
                                        'integrator': integrator.__class__.__name__,
                                        'mode': mode,
                                        'volume': volume,
                                        'n_integrations': n_integrations,
                                        'logs': [],
                                        'error': None
                                }
                                )               
                        except Exception as e:
                                print(
                                        "Formula {}, \t integrator = {}, \t failed = {}".format(
                                        w.serialize(), integrator.__class__.__name__, str(e)
                                        )
                                )
                                results.append(
                                {
                                        'time': time.time()-start_time,
                                        'formula': w.serialize(),
                                        'integrator': integrator.__class__.__name__,
                                        'mode': mode,
                                        'volume': None,
                                        'n_integrations': None,
                                        'logs': [],
                                        'error': str(e)
                                }
                                )
                        
                        pd.DataFrame(results).to_csv(results_path, index=False) 
                