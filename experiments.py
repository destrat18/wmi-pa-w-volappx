from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div, Pow
from pysmt.typing import BOOL, REAL

from wmipa import WMI
from wmipa.integration import LatteIntegrator, VolestiIntegrator, FazaIntegrator, faza
import pandas as pd
import time, argparse, os, logging
import sympy as sym


import benchmarks as FazaBenchmarks
from subprocess import check_output
import tempfile
import pandas as pd


def evaluate_volesi(
    benchmarks,
    result_dir,
    repeat         
        
):


    mode = WMI.MODE_SAE4WMI
    
    results = []
    benchmark_name = f'{benchmarks=}'.split('=')[0]
    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_volesti_{int(time.time())}.csv")

    for bench_i, bench in enumerate(benchmarks):
        
        start_time = time.time()
        try:
                details = []
                if "wmipa" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmipa']['w'] is None:
                    raise Exception('N\S')
        
                for N in args.N:
                    for i in range(args.repeat):

                        integrator = VolestiIntegrator(N=N) 
                        start_time = time.time()
                        wmi = WMI(bench['wmipa']['chi'], bench['wmipa']['w'], integrator=integrator)
                        volume, n_integrations = wmi.computeWMI(bench['wmipa']['phi'], mode=mode)

                        details.append(
                            {
                                'N': N,
                                'repeat': i,
                                'n_integrations': n_integrations,
                                'mode': mode,
                                'output': volume,
                                'time': time.time()-start_time
                            }
                        )
                
                results.append({
                        "bechmark": benchmark_name,
                        "formula": bench['faza']['w'],
                        "index": bench_i,
                        'output': (min([d['output'] for d in details]), max([d['output'] for d in details])),
                        'error': None,
                        "time": (min([d['time'] for d in details]), max([d['time'] for d in details])),
                        'details': details
                    })
        
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is done: {results[-1]['output']}")
                       

        except Exception as e:
            logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is failed: {e}")
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['faza']['w'],
                "index": bench_i,
                "output": None,
                'error': str(e),
                "time": time.time()-start_time,
                'details': []
            }) 
        
        
        pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)


def evaluate_latte(
    benchmarks,
    result_dir        
):


    mode = WMI.MODE_SAE4WMI
    
    results = []
    benchmark_name = f'{benchmarks=}'.split('=')[0]
    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_latte_{int(time.time())}.csv")

    for bench_i, bench in enumerate(benchmarks):
        
        start_time = time.time()
        try:
                
                if "wmipa" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmipa']['w'] is None:
                    raise Exception('N\S')


                integrator = LatteIntegrator() 
                start_time = time.time()
                wmi = WMI(bench['wmipa']['chi'], bench['wmipa']['w'], integrator=integrator)
                volume, n_integrations = wmi.computeWMI(bench['wmipa']['phi'], mode=mode)
               
                results.append({
                        "bechmark": benchmark_name,
                        "formula": bench['faza']['w'],
                        "index": bench_i,
                        'output': volume,
                        'error': None,
                        "time": time.time()-start_time,
                        'details': {
                                'n_integrations': n_integrations,
                                'mode': mode,
                                'output': volume,
                }
                    })
        
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is done: {results[-1]['output']}")
                       

        except Exception as e:
            logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is failed: {e}")
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['faza']['w'],
                "index": bench_i,
                "output": None,
                'error': str(e),
                "time": time.time()-start_time,
                'details': []
            }) 
        
        
        pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)


def evaluate_faza(
        benchmarks,
        result_dir,
        epsilon,
        max_workers         
        
):


        chi = Bool(True)
        mode = WMI.MODE_SAE4WMI

        results = []
        
        benchmark_name = f'{benchmarks=}'.split('=')[0]
        result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_faza_{int(time.time())}.csv")

        for bench_i, bench in enumerate(benchmarks):


            try:
                    
                if "faza" not in  bench:
                    raise Exception('Missing input formula')
                if bench['faza']['w'] is None:
                    raise Exception('N\S')
                
                start_time = time.time()

                # If the format is not supported by WMI-PA input format
                if bench['wmipa']['w'] is None:
                
                    output = faza.calculate_approximate_wmi(
                            phi=bench['faza']['phi'],
                            chi=bench['faza']['chi'],
                            max_workers=max_workers,
                            threshold=epsilon,
                            w=bench['faza']['w'],
                            variables=[sym.symbols(v) for v in bench['faza']['variables']]
                    )
                    
                    results.append({
                            "bechmark": benchmark_name,
                            "formula": bench['faza']['w'],
                            "index": bench_i,
                            'output': (output[0], output[1]),
                            'error': None,
                            "time": time.time()-start_time,
                            'details': {
                                    'n_integrations': 1,
                                    'mode': mode,
                                    'output': output,
                        }
                    })

                else:
                    
                    integrator = FazaIntegrator(threshold=epsilon, max_workers=max_workers) 
                    wmi = WMI(bench['wmipa']['chi'], bench['wmipa']['w'], integrator=integrator)
                    volume, n_integrations = wmi.computeWMI(bench['wmipa']['phi'], mode=mode)

                    if len(integrator.logs)==1:
                        output = integrator.logs[0]['volume']
                    else:
                        output = (None, volume)
                    
                    results.append({
                            "bechmark": benchmark_name,
                            "formula": bench['faza']['w'],
                            "index": bench_i,
                            'output': output,
                            'error': None,
                            "time": time.time()-start_time,
                            'details': {
                                    'n_integrations': 1,
                                    'mode': mode,
                                    'output': output,
                        }
                    })
                
                
                
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is done: {results[-1]['output']}")
                
            except Exception as e:
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is failed: {e}")
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['faza']['w'],
                    "index": bench_i,
                    "output": None,
                    'error': str(e),
                    "time": time.time()-start_time,
                    'details': []
                })
                
            pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False) 
                        

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser(
                prog='Faza Integrator',
                description='I am experimenting!'
                )
        
        parser.add_argument("--epsilon", help="Number of workers", type=float, default=0.1)        
        parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
        parser.add_argument("--N", help="Number of samples", type=int, default=[10000], nargs="+")
        parser.add_argument("--seed", help="Random seed for (the first instance of) VolEsti integrator", type=int, default=[666, 667, 668, 669, 670], nargs="+")
        parser.add_argument("--repeat", help="Number of trials", type=int, default=10)
        parser.add_argument('--volesti', action='store_true', default=False)
        parser.add_argument('--latte', action='store_true', default=False)
        parser.add_argument('--faza', action='store_true', default=False)

        parser.add_argument('--result-dir', type=str, default="experimental_results")
        
        args = parser.parse_args()
        
        
        os.makedirs(args.result_dir, exist_ok=True)

        if args.volesti:
                evaluate_volesi(
                        FazaBenchmarks.selected_benchmark,
                        args.result_dir,
                        repeat=args.repeat
                )
        if args.latte:
                evaluate_latte(
                        FazaBenchmarks.selected_benchmark,
                        args.result_dir
                )
                
        if args.faza or True:
                evaluate_faza(
                        benchmarks=FazaBenchmarks.selected_benchmark,
                        result_dir=args.result_dir,
                        epsilon=args.epsilon,
                        max_workers=args.max_workers
                )
                