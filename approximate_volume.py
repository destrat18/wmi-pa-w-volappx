from wmipa.integration import faza
import argparse, os, time, logging
import sympy as sym
import pandas as pd

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        prog='Faza: Volume Approximation',
        description=''
        )
    
    parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
    parser.add_argument("--input", help="Path to input", type=str)
    parser.add_argument("--bounds", help="Path to bounds", type=str)
    parser.add_argument("--output", help="Path to output", type=str, default=None)
    parser.add_argument("--threshold", help="Error threshold", type=float, default=0.1)
    parser.add_argument("--degree", help="Handelman degree", type=float, default=None)
    
    
    args = parser.parse_args()
    
        
    inputs, bounds, variables = faza.read_input(args.input,args.bounds)
    
    volume, stats = faza.calculate_approximate_volume(
        degrees=[sym.total_degree(i) for i in inputs],
        max_workers=args.max_workers,
        inputs=inputs,
        bounds=bounds,
        variables=variables,
        threshold=args.threshold
    )
    
    
    #     total_degree = sym.total_degree(integrand)
        
    #     while True:
        
    #         logging.info(f"Checking {integrand} with degree {total_degree}!") 
    #         start_time = time.time()
            
    #         volume, stats = faza.calculate_approximate_volume(
    #             degree=total_degree,
    #             max_workers=args.max_workers,
    #             integrand=integrand,
    #             bounds=bounds,
    #             vars=vars,
    #             threshold=args.threshold
    #         )
            
    #         results.append(
    #             {
    #                 'volume_id': volume_id,
    #                 "volume": volume,
    #                 "time": time.time()-start_time,
    #                 "hrect_checked_num": stats["hrect_checked_num"],
    #                 "total_solver_time": stats["total_solver_time"],
    #                 "total_subs_time": stats["total_subs_time"],
    #                 "threshold": args.threshold,
    #                 "max_workers": args.max_workers,
    #                 "integrand": integrand,
    #                 "bounds": bounds,
    #                 "degree": total_degree
    #             }
                
    #         )
            
    #         pd.DataFrame(results).to_csv(output_path, index=False)
            
    #         if stats['error'] > 0:
    #             break
                
    #         total_degree += 1
        

    # except Exception as e:
    #     logging.exception(e)


    # integrand, bounds, vars = faza.read_input(args.integrand, args.bounds)

    # 