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
    
    if args.degree == None:
        degree_list = [sym.total_degree(i) for i in inputs]
    else:
         degree_list = [int(args.degree)]*len(inputs)
    
    volume, stats = faza.calculate_approximate_volume(
        degree_list=degree_list,
        max_workers=args.max_workers,
        inputs=inputs,
        bounds=bounds,
        variables=variables,
        threshold=args.threshold
    )
    
    