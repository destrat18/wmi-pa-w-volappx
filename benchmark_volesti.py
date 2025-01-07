from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div, Pow
from pysmt.typing import BOOL, REAL

from wmipa import WMI
from wmipa.integration import LatteIntegrator, VolestiIntegrator, FazaIntegrator
import pandas as pd
import time, argparse, os
from benchmarks import benchmark_group1


# fmt: on
if __name__ == "__main__":

        parser = argparse.ArgumentParser(
                prog='Faza Integrator',
                description='I am approximating!'
                )
                
        parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
        parser.add_argument("--N", help="Number of samples", type=int, default=[100, 1000, 1000], nargs="+")
        parser.add_argument("--seed", help="Random seed for (the first instance of) VolEsti integrator", type=int, default=[666, 667, 668, 669, 670], nargs="+")
        args = parser.parse_args()
        parser.add_argument("--repeat", help="Number of trials", type=int, default=10)
        args = parser.parse_args()
        
        
        chi = Bool(True)

        print()
        mode = WMI.MODE_SAE4WMI

        results = []
        
        benchmark_name = f'{benchmark_group1=}'.split('=')[0]
        results_dir = "experimental_results"
        results_path = os.path.join(results_dir, f"benchmark_{benchmark_name}_volesti_{int(time.time())}.csv")

        for bench in benchmark_group1:

                w = bench['w']
                phi = bench['phi']
                    
                
                for N in args.N:
                    for seed in args.seed:
                        for i in range(args.repeat):
                            
                            integrator = VolestiIntegrator(N=N, seed=seed) 
                            start_time = time.time()
                            try:
                                    wmi = WMI(chi, w, integrator=integrator)
                                    volume, n_integrations = wmi.computeWMI(phi, mode=mode)
                                    print(
                                            "Formula {}, \t integrator= {}(N={}), \t result = {}, \t time = {:.4f}s".format(
                                                    w.serialize(), integrator.__class__.__name__, N, volume, time.time()-start_time
                                            )
                                    )
                                    results.append(
                                    {
                                            "benchmark": benchmark_group1,
                                            'time': time.time()-start_time,
                                            'formula': w.serialize(),
                                            'integrator': integrator.__class__.__name__,
                                            'mode': mode,
                                            'volume': volume,
                                            'n_integrations': n_integrations,
                                            'details': {
                                                'repeat': i,
                                                'seed': seed,
                                                'N': N,
                                                },
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
                                            "benchmark": benchmark_group1,
                                            'time': time.time()-start_time,
                                            'formula': w.serialize(),
                                            'integrator': integrator.__class__.__name__,
                                            'mode': mode,
                                            'volume': None,
                                            'n_integrations': None,
                                            'details': {
                                                'repeat': i,
                                                'seed': seed,
                                                'N': N,
                                                },
                                            'error': str(e)
                                    }
                                    )
                            
                            pd.DataFrame(results).to_csv(results_path, index=False) 
                