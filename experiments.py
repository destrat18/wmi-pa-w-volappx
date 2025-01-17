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
import signal


def OutOfTimeHandler(signum, frame):
    raise Exception('Timeout')


def evaluate_psi(
    benchmarks,
    result_dir,
    benchmark_name,
    timeout
):
        results = []    
        
        result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_psi_{int(time.time())}.csv")

        for _, bench in enumerate(benchmarks):
            bench_i = bench['index']
            start_time = time.time()
            try:
                
                if "psi" not in  bench:
                    raise Exception('Missing input formula')
                if bench['psi']['formula'] is None:
                    raise Exception('N\S')
                
                program_path = os.path.join(result_dir, f"psi_template_bench_{benchmark_name}_{bench_i}.psi")
                
                with open(program_path, 'w') as f:
                        if len(bench['faza']['chi']) == 1:
                            f.write(
                                FazaBenchmarks.PSI_SOLVER_ONE_VAR_TEMPLATE.format(
                                    x_lower_bound = bench['faza']['chi'][0][0],
                                    x_upper_bound = bench['faza']['chi'][0][1],
                                    formula = bench['psi']['formula']
                                )
                            )
                        elif len(bench['faza']['chi']) == 2:
                            f.write(
                                FazaBenchmarks.PSI_SOLVER_TWO_VAR_TEMPLATE.format(
                                    x_lower_bound = bench['faza']['chi'][0][0],
                                    x_upper_bound = bench['faza']['chi'][0][1],
                                    y_lower_bound = bench['faza']['chi'][1][0],
                                    y_upper_bound = bench['faza']['chi'][1][1],
                                    formula = bench['psi']['formula']
                                )
                            )
                        elif len(bench['faza']['chi']) == 3:
                            f.write(
                                FazaBenchmarks.PSI_SOLVER_THREE_VAR_TEMPLATE.format(
                                    x_lower_bound = bench['faza']['chi'][0][0],
                                    x_upper_bound = bench['faza']['chi'][0][1],
                                    y_lower_bound = bench['faza']['chi'][1][0],
                                    y_upper_bound = bench['faza']['chi'][1][1],
                                    z_lower_bound = bench['faza']['chi'][2][0],
                                    z_upper_bound = bench['faza']['chi'][2][1],
                                    formula = bench['psi']['formula']
                                )
                            )
                output = check_output([
                    "timeout", str(timeout),
                    'psi', program_path, '--expectation', '--mathematica']).decode("utf-8").strip().replace('\n', '\t')
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['faza']['w'],
                    "bounds": bench['faza']['chi'],
                    "index": bench_i,
                    'output': output,
                    'error': None,
                    "time": time.time()-start_time,
                    'details': []
                })
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is done: {output}")
            
            except Exception as e:
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is failed: {e}")
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['faza']['w'],
                    "bounds": bench['faza']['chi'],
                    "index": bench_i,
                    "output": None,
                    'error': str(e),
                    "time": time.time()-start_time,
                    'details': []
                })      
                
            pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)
            
def evaluate_volesi(
    benchmarks,
    benchmark_name,
    result_dir,
    repeat         
        
):


    mode = WMI.MODE_SAE4WMI
    
    results = []    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_volesti_{int(time.time())}.csv")

    for _, bench in enumerate(benchmarks):
        
        bench_i = bench['index']
        start_time = time.time()
        try:
                details = []
                if "wmipa" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmipa']['w'] is None:
                    raise Exception('N\S')
        
                for N in [1000]:
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
                        "bounds": bench['faza']['chi'],
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
    result_dir,
    benchmark_name        
):


    mode = WMI.MODE_SAE4WMI
    
    results = []    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_latte_{int(time.time())}.csv")

    for _, bench in enumerate(benchmarks):
        
        bench_i = bench['index']
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
                        "bounds": bench['faza']['chi'],
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
                "bounds": bench['faza']['chi'],
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
        max_workers,
        benchmark_name,
        timeout    
        
):


        chi = Bool(True)
        mode = WMI.MODE_SAE4WMI

        results = []
        
        result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_faza_{int(time.time())}.csv")

        for _, bench in enumerate(benchmarks):

            bench_i = bench['index']
            try:
                
                if "faza" not in  bench:
                    raise Exception('Missing input formula')
                if bench['faza']['w'] is None:
                    raise Exception('N\S')
                
                start_time = time.time()
                signal.alarm(int(timeout))

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
                            "bounds": bench['faza']['chi'],
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
                            "bounds": bench['faza']['chi'],
                            "index": bench_i,
                            'output': output,
                            'error': None,
                            "time": time.time()-start_time,
                            'details': {
                                    'n_integrations': 1,
                                    'mode': mode,
                                    'output': output,
                                    "logs": integrator.logs
                        }
                    })
                
                
                
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is done: {results[-1]['output']}")
                print(f"Bench {bench_i} ({bench['faza']['w']}) is done: {results[-1]['output']}")
                
            except Exception as e:
                logging.info(f"Bench {bench_i} ({bench['faza']['w']}) is failed: {e}")
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['faza']['w'],
                    "bounds": bench['faza']['chi'],
                    "index": bench_i,
                    "output": None,
                    'error': str(e),
                    "time": time.time()-start_time,
                    'details': []
                })
                logging.exception(e)
                
            pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False) 
                        

if __name__ == "__main__":
        
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGALRM, OutOfTimeHandler)
    
    
    parser = argparse.ArgumentParser(
            prog='Faza Integrator',
            description='I am experimenting!'
            )

    parser.add_argument("--constant", type=float, default=None)            
    parser.add_argument("--timeout", type=float, default=10)        
    parser.add_argument("--epsilon", help="Number of workers", type=float, default=50)        
    parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
    parser.add_argument("--repeat", help="Number of trials", type=int, default=10)
    parser.add_argument('--volesti', action='store_true', default=False)
    parser.add_argument('--latte', action='store_true', default=False)
    parser.add_argument('--faza', action='store_true', default=False)
    parser.add_argument('--psi', action='store_true', default=False)
    parser.add_argument('--benchmark', choices=['manual', 'rational', 'sqrt', "rational_sqrt", "rational_2"], default="manual")
    parser.add_argument('--benchmark-path', type=str, help="Path to the benchmark")
    

    parser.add_argument('--result-dir', type=str, default="experimental_results")
    
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)


    if args.benchmark == 'manual':
        benchmarks = FazaBenchmarks.selected_benchmark
        args.benchmark = "manual"
    elif args.benchmark == "rational":
        benchmarks = FazaBenchmarks.load_rational_benchmarks(
            benchmak_path=args.benchmark_path,
            constant=args.constant
        )
    elif args.benchmark == "sqrt":
        benchmarks = FazaBenchmarks.load_sqrt_benchmarks(
            benchmak_path=args.benchmark_path,
            constant=args.constant
        )
    elif args.benchmark == "rational_sqrt":
        benchmarks = FazaBenchmarks.load_rational_sqrt_benchmarks(
            benchmak_path=args.benchmark_path,
            constant=args.constant
        )
    elif args.benchmark == "rational_2":
        benchmarks = FazaBenchmarks.load_rational_2_benchmarks(
            benchmak_path=args.benchmark_path,
            constant=args.constant
        )
    else:
        raise NotImplementedError()


    if args.volesti:
            evaluate_volesi(
                benchmarks=benchmarks,
                benchmark_name=args.benchmark,
                result_dir=args.result_dir,
                repeat=args.repeat
            )
    if args.latte:
            evaluate_latte(
                benchmarks=benchmarks,
                result_dir=args.result_dir,
                benchmark_name=args.benchmark,

            )
            
    if args.faza:
            evaluate_faza(
                benchmarks=benchmarks,
                result_dir=args.result_dir,
                epsilon=args.epsilon,
                max_workers=args.max_workers,
                benchmark_name=args.benchmark,
                timeout=args.timeout
            )
            
    if args.psi:
        evaluate_psi(
            benchmarks=benchmarks,
            result_dir=args.result_dir,
            benchmark_name=args.benchmark,
            timeout=args.timeout
        )