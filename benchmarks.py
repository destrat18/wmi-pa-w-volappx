from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div, Pow, Plus
from pysmt.typing import BOOL, REAL
from pysmt.parsing import parse, HRParser

import random, json, argparse, os


# variables definition
x = Symbol("x", REAL)
y = Symbol("y", REAL)
z = Symbol("z", REAL)

a = Real(4)
b = Real(9)
c = Real(16)

PSI_SOLVER_ONE_VAR_TEMPLATE = """
def main(){{
    x := uniform({x_lower_bound}, {x_upper_bound});

    p := {formula};
    
    return p;
}}
"""


PSI_SOLVER_TWO_VAR_TEMPLATE = """
def main(){{
    x := uniform({x_lower_bound}, {x_upper_bound});
    y := uniform({y_lower_bound}, {y_upper_bound});
    
    p := {formula};
    
    return p;
}}
"""

PSI_SOLVER_THREE_VAR_TEMPLATE = """
def main(){{
    x := uniform({x_lower_bound}, {x_upper_bound});
    y := uniform({y_lower_bound}, {y_upper_bound});
    z := uniform({z_lower_bound}, {z_upper_bound});

    p := {formula};
    
    return p;
}}
"""


GUBPI_SOLVER_ONE_VAR_TEMPLATE = """
        # depth 200
        # discretization -0.1 0.1 0.2

        # Taken from PSI repository (modified)
        # Location: PSI/test/fun/coins.psi (modified to return only one element of the tuple)

        let x = sample uniform({x_lower_bound}, {x_upper_bound}) in
        let p = ({formula})  in
                score(p)

"""

GUBPI_SOLVER_TWO_VAR_TEMPLATE = """
        # depth 200
        # discretization -0.1 0.1 0.2

        # Taken from PSI repository (modified)
        # Location: PSI/test/fun/coins.psi (modified to return only one element of the tuple)

        let x = sample uniform({x_lower_bound}, {x_upper_bound}) in
        let y = sample uniform({y_lower_bound}, {y_upper_bound}) in
        let p = ({formula})  in
                score(p)


"""
GUBPI_SOLVER_THREE_VAR_TEMPLATE = """
        # depth 200
        # discretization -0.1 0.1 0.2

        # Taken from PSI repository (modified)
        # Location: PSI/test/fun/coins.psi (modified to return only one element of the tuple)

        let x = sample uniform({x_lower_bound}, {x_upper_bound}) in
        let y = sample uniform({y_lower_bound}, {y_upper_bound}) in
        let z = sample uniform({z_lower_bound}, {z_upper_bound}) in
        let p = ({formula})  in
                score(p)

"""

selected_benchmark = [

        # #########################################################
        {
                "index": 0,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,                        
                        "variables": ["x"],
                        "w": "(x)*1"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1))),
                        'w': (x)*Real(1),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x)*1"        
                },
                "gubpi": {
                    "formula": "(x)*1"
                }
        },
    
        # #########################################################
        {
                "index": 1,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**2)*1"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': (Pow(x, Real(2)))*Real(1),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^2)*1"        
                },
                "gubpi": {
                        "formula": "(x*x)*1"        
                }
        },
    
        # #########################################################
        {
                "index": 2,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**3)*1"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': (Pow(x, Real(3)))*Real(1),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^3)*1"        
                },
                "gubpi": {
                        "formula": "(x*x*x)*1"        
                }
        },
        # #########################################################
                {
                "index": 3,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**3 + x**2 + x + 1)*1"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': Plus(Pow(x, Real(3)), Pow(x, Real(2)), x, Real(1))*Real(1),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^3  + x^2 + x + 1)*1"        
                },
                "gubpi": {
                        "formula": "(x*x*x + x*x + x + 1)*1"        
                }
        },
                
        # #########################################################
        {
                "index": 4,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**4 + x**3 + x**2 + x**1 + 1)*1"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': (Plus(Pow(x, Real(4)), Pow(x, Real(3)), Pow(x,Real(2)), Pow(x,Real(1)), Real(1)))*Real(1),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^4 + x^3 + x^2 + x + 1)*1"        
                },
                "gubpi": {
                        "formula": "((x*x*x*x) + (x*x*x) + (x*x) + (x) + 1)*1"        
                }
        },
        
        # #########################################################
        {
                "index": 5,
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(1/x)"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1))),
                        'w': Div(Real(1), x),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(1/x)"        
                },
                "gubpi": {
                        "formula": "(div(1,x))"        
                }
        },
        
        # #########################################################
        {
                "index": 9,
                "faza":{
                        "chi": [[1.01, 2]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(1/(x**2 -1))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(1.01)),LE(x, Real(2))),
                        'w': Div(Real(1), Pow(x, Real(2)) - Real(1)),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(1/(x^2 -1))"        
                },
                "gubpi": {
                        "formula": "(div(1,(x*x - 1)))"        
                }
        },

        # #########################################################
        {
                "index": 10,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "((x**2+x+1)*1)/(x**3+1)"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': Div(Plus(Pow(x, Real(2)), x, Real(1))*Real(1), Pow(x, Real(3)) + Real(1)),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "((x^2+x+1)*1/(x^3+1))"     
                },
                "gubpi": {
                        "formula": "(div((x*x + x + 1)*1,(x*x*x + 1)))"        
                }
        },
    
        # #########################################################
        {
                "index": 11,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(x*1/( {a} + x**2))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': Div(x*Real(1), a+ Pow(x, Real(2)))    ,
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(x*1/( {a} + x^2))"        
                },
                "gubpi": {
                        "formula": f"(div(x*1,({a} + x*x)))"        
                }
        },

        # #########################################################
        {
                "index": 12,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((x**3)*1/({a}+x**2))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w':  Div(Pow(x, Real(3))*Real(1), a + Pow(x, Real(2))),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"((x^3)*1/({a}+x^2))",        
                },
                "gubpi": {
                        "formula": f"(div(x*x*x*1,({a} + x*x)))"        
                }
                
        },
        
        # #########################################################
        {
                "index": 13,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(1/({a}*x**2 + {b}*x + {c}))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w':   Div(Real(1), a*Pow(x, Real(2)) + b*x + c)*Real(1)  ,
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(1/({a}*(x^2) + {b}*x + {c}))",        
                },
                "gubpi": {
                        "formula": f"(div(1,({a}*x*x) + ({b}*x) + {c}))"        
                }
        },
        
        # #########################################################
        {
                "index": 14,
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(1/( ({a}+x) * (x+{b}) ))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w':   Div(Real(1), Times(x+a, x+b) )*Real(1),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(1/( ({a}+x) * (x+{b}) ))",        
                },
                "gubpi": {
                        "formula": f"(div(1,({a} + x) * ({b} + x)))"        
                }
        },
    
        
        ######################################################### 
        {
                "index": 15,
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((1/x)**(1/2))"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"((1/x)^(1/2))",        
                },
                "gubpi": {
                        "formula": "(sqrt(div(1,x)))"        
                }
        },
        {
                "index": 16,
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((1/x)**(1/3))"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"((1/x)^(1/3))",        
                },
                "gubpi": {
                        "formula": None        
                }
        },
        {
                "index": 17,
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((1/(x+1))**(1/2))"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"(1/(x+1))^(1/2)",        
                },
                "gubpi": {
                        "formula": "sqrt(div(1,x+1))"        
                }
        },
        {
                "index": 18,
                "faza":{
                        "chi": [[1.01, 2]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(1/(x**2-1))**(1/2)"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"(1/(x^2-1))^(1/2)",        
                }
                ,
                "gubpi": {
                        "formula": "sqrt(div(1,x*x - 1))"        
                }
        },
        {
                "index": 19,
                "faza":{
                        "chi": [[0.01, 1], [0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(1/(x+y))**(1/2)"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"(1/(x+y))^(1/2)",        
                },
                "gubpi": {
                        "formula": "sqrt(div(1,x+y))"        
                }
        },
        # #########################################################
        {
                "index": 6,
                "faza":{
                        "chi": [[0.01, 1], [0.01, 1]],
                        "phi": True,
                        "variables": ["x", "y"],
                        "w": f"(1/(x+y))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0.01)),LE(y, Real(1))),
                        'w':   Div(Real(1), x+y),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(1/(x+y))",        
                },
                "gubpi": {
                        "formula": "div(1,(x+y))"        
                }
                
        },
        
        # #########################################################
        {
                "index": 7,
                "faza":{
                        "chi": [[0.01, 1], [0.01, 1]],
                        "phi": True,
                        "variables": ["x", "y"],
                        "w": f"(1/(x+y+(x*y)))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0.01)),LE(y, Real(1))),
                        'w':   Div(Real(1), x+y+(x*y)),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(1 / ( x + y + (x*y) ) )",        
                },
                "gubpi": {
                        "formula": "(div(1,(x+y + x*y)))"        
                }
        },
        
        # #########################################################
        {
                "index": 8,
                "faza":{
                        "chi": [[0.01, 1], [0.01, 1], [0.01, 1]],
                        "phi": True,
                        "variables": ["x", "y", "z"],
                        "w": f"(1/(x+y+z))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0.01)),LE(y, Real(1)), GE(z, Real(0.01)),LE(z, Real(1))),
                        'w':   Div(Real(1), x+y+z),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(1/(x+y+z))",        
                },
                "gubpi": {
                        "formula": "(div(1,(x+y+z)))"        
                }
        },
        # #########################################################
        {
                "index": 20,
                "faza":{
                        "chi": [[0, 1], [0, 1]],
                        "phi": True,
                        "variables": ["x", "y"],
                        "w": "((x**2 + 2*y**2 + 3*y*x + x + 1)*1/(2*y**2 + y*x + y + 2))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1)), GE(y, Real(0)),LE(y, Real(1))),
                        'w': Div(
                                Plus(
                                        Pow(x, Real(2)), 
                                        Real(2)*Pow(y, Real(2)),
                                        Real(3)*y*x,
                                        x,
                                        Real(1)
                                )*Real(1),
                                Plus(
                                        Real(2)*Pow(y, Real(2)),
                                        y*x,
                                        y,
                                        Real(2)
                                )
                                ),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "((x^2 + 2*(y^2) + 3*y*x + x + 1)*1/(2*(y^2) + y*x + y + 2))"     
                },
                "gubpi": {
                        "formula": "(div((x*x + 2*(y*y) + 3*y*x + x + 1)*1,(2*(y*y) + y*x + y + 2)))"    
                }
        },
        
]


######################################################### 
# Radical Functions

# Benchmark in form w = a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0 / b_n*x^n + b_(m-1)*x^(m-1) + ... + b_1*x
# n < m
# 0 < x < 1
# 0 < غ < 1

def generate_rational_bechmarks(number_of_benchmarks, max_den_deg, max_nom_deg, output_path):

        benchmarks = []
        
        for i in range(number_of_benchmarks):
        
                # denuminator must have higher degree than numinator
                # assert(max_m>max_n)
                
                # select degree of numinator and denuminator randomly 
                n = random.randint(0, max_nom_deg)
                m = random.randint(0, max_den_deg)
                
                # generate coefficents randomly
                a_coefficients = [round(random.uniform(0, 10),2) for i in range(n+1)]
                b_coefficients = [round(random.uniform(0, 10),2) for i in range(m+1)]
                
                benchmarks.append(
                        {
                               "a_i": a_coefficients,
                               "b_i": b_coefficients 
                        }
                )
        
        # Save it to
        with open(output_path, 'w') as f:
                json.dump(benchmarks, f)


def load_rational_benchmarks(benchmak_path, bounds=[[0.01, 1]]):
        
        benchmaks = []
        
        with open(benchmak_path, 'r') as f:
                benchmak_coefficients = json.load(f)
        
        for bench_i, c in enumerate(benchmak_coefficients):
                a_i = c['a_i']
                b_i = c['b_i']
                
                benchmaks.append(
                        {
                                "index": bench_i,
                                "faza":{
                                        "chi": bounds,
                                        "phi": True,
                                        "variables": ["x"],
                                        "w": "(" + "+".join([ f"{a}*x**{len(a_i)-i-1}" for i, a in enumerate(a_i)]) + ")" + " / " + "(" + " + ".join([ f"{b}*x**{len(b_i)-i-1}" for i, b in enumerate(b_i)]) + ")"          
                                },
                                "wmipa":{
                                        "chi": And(GE(x, Real(bounds[0][0])),LE(x, Real(bounds[0][1]))),        
                                        'w':   Div(Plus([Times(Real(a), Pow(x, Real(len(a_i)-i-1))) for i, a in enumerate(a_i)]), Plus([Times(Real(b), Pow(x, Real(len(b_i)-i-1))) for i, b in enumerate(b_i)])),
                                        "phi": Bool(True),
                                },
                                "psi": {
                                        "formula": "(" + "+".join([ f"{a}*x^{len(a_i)-i-1}" for i, a in enumerate(a_i)]) + ")" + " / " + "(" + " + ".join([ f"{b}*x^{len(b_i)-i-1}" for i, b in enumerate(b_i)]) + ")" ,        
                                },
                                "gubpi": {
                                        "formula": "div((" + "+".join([f"{a}*{'*'.join(['x']*(len(a_i)-i-1)+['1'])}" for i, a in enumerate(a_i)]) + ")" + " , " + "(" + " + ".join([f"{b}*{'*'.join(['x']*(len(b_i)-i-1)+['1'])}" for i, b in enumerate(b_i)]) + "))"        
                                }
                        }
                )
                
        return benchmaks        
        


# Sqrt Functions
# Benchmark in form w = sqrt(a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0)
# 0 < x < 1

def generate_sqrt_bechmarks(number_of_benchmarks, max_deg, output_path):

        benchmarks = []
        
        for i in range(number_of_benchmarks):
        
                # denuminator must have higher degree than numinator
                # assert(max_m>max_n)
                
                # select degree of numinator and denuminator randomly 
                n = random.randint(0, max_deg)
                
                # generate coefficents randomly
                a_i = [round(random.uniform(0, 10),2) for i in range(n+1)]
                
                benchmarks.append(
                        {
                               "a_i": a_i 
                        }
                )
        
        # Save it to
        with open(output_path, 'w') as f:
                json.dump(benchmarks, f)


def load_sqrt_benchmarks(benchmak_path, bounds=[[0, 1]]):
        
        benchmaks = []
        
        with open(benchmak_path, 'r') as f:
                benchmak_coefficients = json.load(f)
        
        for bench_i, c in enumerate(benchmak_coefficients):
                a_i = c['a_i']
                
                benchmaks.append(
                        {
                                "index": bench_i,
                                "faza":{
                                        "chi": bounds,
                                        "phi": True,
                                        "variables": ["x"],
                                        "w": "(" + "+".join([ f"{a}*x**{len(a_i)-i-1}" for i, a in enumerate(a_i)]) + ")" + "**(1/2)"          
                                },
                                "wmipa":{
                                        "chi": None,        
                                        'w':   None,
                                        "phi": None,
                                },
                                "psi": {
                                        "formula": "(" + "+".join([ f"{a}*x^{len(a_i)-i-1}" for i, a in enumerate(a_i)]) + ")" + "^(1/2)",        
                                },
                                "gubpi": {
                                        "formula": "sqrt(" + "+".join([f"{a}*{'*'.join(['x']*(len(a_i)-i-1)+['1'])}" for i, a in enumerate(a_i)]) + ")"
                                }
                        }
                )
                
        return benchmaks        
        



if __name__ == "__main__":
        
        parser = argparse.ArgumentParser(
                prog='Faza Integrator',
                description='I am generating!'
                )
        

        parser.add_argument('--output', type=str, default="experimental_results")
        parser.add_argument('--max-den-deg', type=int)
        parser.add_argument('--max-deg', type=int)
        parser.add_argument('--num-benchmarks', type=int)

        parser.add_argument('--type', choices=['rational', 'sqrt'], default=False)

        
        
        args = parser.parse_args()
        
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        
        if args.type == 'rational':
                generate_rational_bechmarks(
                        number_of_benchmarks=args.num_benchmarks,
                        max_den_deg=args.max_den_deg,
                        max_nom_deg=args.max_deg,
                        output_path=args.output       
                )
        elif args.type == 'sqrt':
                generate_sqrt_bechmarks(
                        number_of_benchmarks=args.num_benchmarks,
                        max_deg=args.max_deg,
                        output_path=args.output      
                )