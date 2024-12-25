from wmipa.integration import faza
import argparse


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(
        prog='Mega Miners',
        description='I am approximating!'
        )
        
    parser.add_argument("--degree", help="Handelman degree", type=int, default=1)
    parser.add_argument("--bounds", help="Integral bounds file", type=str, default=None)
    parser.add_argument("--integrand", help="Path to integrand file", type=str, default=None)
    
    parser.add_argument("--max-workers", help="Number of workers", type=int, default=1)
    parser.add_argument("--threshold", help="Error threshold", type=float, default=0.001)
    
    args = parser.parse_args()
    
    integrand, bounds, vars = faza.read_input(args.integrand, args.bounds)

    faza.calculate_approximate_volume(
        degree=args.degree,
        max_workers=args.max_workers,
        integrand=integrand,
        bounds=bounds,
        vars=vars,
        threshold=args.threshold
    )
    