from wmipa.integration import faza
import argparse, os, time, logging
import sympy as sym
import pandas as pd

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        prog='Faza Volume',
        description=''
        )
    
    parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
    parser.add_argument("--threshold", help="Error threshold", type=float, default=0.001)
    
    args = parser.parse_args()
    
    volume_dir = "data/volumes"
    output_path = os.path.join(f"data/volume_results_{int(time.time())}.csv")
    
    results = []
    for f_name in sorted(os.listdir(volume_dir)):
        try:
        
            volume_id = f_name
            
            integrand, bounds, vars = faza.read_input(
                    os.path.join(volume_dir, volume_id, 'integrand.txt'), 
                    os.path.join(volume_dir, volume_id, 'bounds.txt')
                )
            total_degree = sym.total_degree(integrand)
            
            while True:
            
                logging.info(f"Checking {integrand} with degree {total_degree}!") 
                start_time = time.time()
                
                volume, stats = faza.calculate_approximate_volume(
                    degree=total_degree,
                    max_workers=args.max_workers,
                    integrand=integrand,
                    bounds=bounds,
                    vars=vars,
                    threshold=args.threshold
                )
                
                results.append(
                    {
                        'volume_id': volume_id,
                        "volume": volume,
                        "time": time.time()-start_time,
                        "hrect_checked_num": stats["hrect_checked_num"],
                        "total_solver_time": stats["total_solver_time"],
                        "total_subs_time": stats["total_subs_time"],
                        "threshold": args.threshold,
                        "max_workers": args.max_workers,
                        "integrand": integrand,
                        "bounds": bounds,
                        "degree": total_degree
                    }
                    
                )
                
                pd.DataFrame(results).to_csv(output_path, index=False)
                
                if stats['error'] > 0:
                    break
            
        
        except Exception as e:
            logging.exception(e)


    # integrand, bounds, vars = faza.read_input(args.integrand, args.bounds)

    # 