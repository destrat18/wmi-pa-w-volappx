from pysmt.shortcuts import GE, LE, And, Bool, Iff, Ite, Real, Symbol, Times, Div, Pow, Plus
from pysmt.typing import BOOL, REAL
from pysmt.parsing import parse, HRParser

import random, json


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
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,                        
                        "variables": ["x"],
                        "w": "(x)*100"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1))),
                        'w': (x)*Real(100),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x)*100"        
                },
                "gubpi": {
                    "formula": "(x)*100"
                }
        },
    
        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**2)*100"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': (Pow(x, Real(2)))*Real(100),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^2)*100"        
                },
                "gubpi": {
                        "formula": "(x*x)*100"        
                }
        },
    
        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**3)*100"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': (Pow(x, Real(3)))*Real(100),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^3)*100"        
                },
                "gubpi": {
                        "formula": "(x*x*x)*100"        
                }
        },
        # #########################################################
                {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**3 + x**2 + x + 1)*100"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': Plus(Pow(x, Real(3)), Pow(x, Real(2)), x, Real(1))*Real(100),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^3  + x^2 + x + 1)*100"        
                },
                "gubpi": {
                        "formula": "(x*x*x + x*x + x + 1)*100"        
                }
        },
                
        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(x**4 + x**3 + x**2 + x**1 + 1)*100"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': (Plus(Pow(x, Real(4)), Pow(x, Real(3)), Pow(x,Real(2)), Pow(x,Real(1)), Real(1)))*Real(100),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(x^4 + x^3 + x^2 + x + 1)*100"        
                },
                "gubpi": {
                        "formula": "((x*x*x*x) + (x*x*x) + (x*x) + (x) + 1)*100"        
                }
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(100/x)"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1))),
                        'w': Div(Real(100), x),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(100/x)"        
                },
                "gubpi": {
                        "formula": "(div(100,x))"        
                }
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[0.01, 1], [0, 1]],
                        "phi": True,
                        "variables": ["x", "y"],
                        "w": f"(100/(x+y))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0)),LE(y, Real(1))),
                        'w':   Div(Real(100), x+y),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(100/(x+y))",        
                },
                "gubpi": {
                        "formula": "div(100,(x+y))"        
                }
                
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[0.01, 1], [0, 1]],
                        "phi": True,
                        "variables": ["x", "y"],
                        "w": f"(100/(x+y+(x*y)))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0)),LE(y, Real(1))),
                        'w':   Div(Real(100), x+y+(x*y)),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(100 / ( x + y + (x*y) ) )",        
                },
                "gubpi": {
                        "formula": "(div(100,(x+y + x*y)))"        
                }
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[0.01, 1], [0, 1], [0, 1]],
                        "phi": True,
                        "variables": ["x", "y", "z"],
                        "w": f"(100/(x+y+z))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0.01)),LE(x, Real(1)), GE(y, Real(0)),LE(y, Real(1)), GE(z, Real(0)),LE(z, Real(1))),
                        'w':   Div(Real(100), x+y+z),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(100/(x+y+z))",        
                },
                "gubpi": {
                        "formula": "(div(100,(x+y+z)))"        
                }
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[1.01, 2]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "(100/(x**2 -1))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(1.01)),LE(x, Real(2))),
                        'w': Div(Real(100), Pow(x, Real(2)) - Real(1)),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "(100/(x^2 -1))"        
                },
                "gubpi": {
                        "formula": "(div(100,(x*x - 1)))"        
                }
        },

        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": "((x**2+x+1)*100)/(x**3+1)"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': Div(Plus(Pow(x, Real(2)), x, Real(1))*Real(100), Pow(x, Real(3)) + Real(1)),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": "((x^2+x+1)*100/(x^3+1))"     
                },
                "gubpi": {
                        "formula": "(div((x*x + x + 1)*100,(x*x*x + 1)))"        
                }
        },
    
        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(x*100/( {a} + x**2))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w': Div(x*Real(100), a+ Pow(x, Real(2)))    ,
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(x*100/( {a} + x^2))"        
                },
                "gubpi": {
                        "formula": f"(div(x*100,({a} + x*x)))"        
                }
        },

        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((x**3)*100/({a}+x**2))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w':  Div(Pow(x, Real(3))*Real(100), a + Pow(x, Real(2))),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"((x^3)*100/({a}+x^2))",        
                },
                "gubpi": {
                        "formula": f"(div(x*x*x*100,({a} + x*x)))"        
                }
                
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(100/({a}*x**2 + {b}*x + {c}))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w':   Div(Real(1), a*Pow(x, Real(2)) + b*x + c)*Real(100)  ,
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(100/({a}*(x^2) + {b}*x + {c}))",        
                },
                "gubpi": {
                        "formula": f"(div(100,({a}*x*x) + ({b}*x) + {c}))"        
                }
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(100/( ({a}+x) * (x+{b}) ))"          
                },
                "wmipa":{
                        "chi": And(GE(x, Real(0)),LE(x, Real(1))),
                        'w':   Div(Real(1), Times(x+a, x+b) )*Real(100),
                        "phi": Bool(True)        
                },
                "psi": {
                        "formula": f"(100/( ({a}+x) * (x+{b}) ))",        
                },
                "gubpi": {
                        "formula": f"(div(100,({a} + x) * ({b} + x)))"        
                }
        },
    
        
        ######################################################### 
        {
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((10000/x)**(1/2))"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"((10000/x)^(1/2))",        
                },
                "gubpi": {
                        "formula": "(sqrt(div(10000,x)))"        
                }
        },
        {
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((1000000/x)**(1/3))"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"((1000000/x)^(1/3))",        
                },
                "gubpi": {
                        "formula": None        
                }
        },
        {
                "faza":{
                        "chi": [[0.01, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"((10000/(x+1))**(1/2))"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"(10000/(x+1))^(1/2)",        
                },
                "gubpi": {
                        "formula": "sqrt(div(10000,x+1))"        
                }
        },
        {
                "faza":{
                        "chi": [[1.01, 2]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(10000/(x**2-1))**(1/2)"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"(10000/(x^2-1))^(1/2)",        
                }
                ,
                "gubpi": {
                        "formula": "sqrt(div(10000,x*x - 1))"        
                }
        },
        {
                "faza":{
                        "chi": [[0.01, 1], [0, 1]],
                        "phi": True,
                        "variables": ["x"],
                        "w": f"(10000/(x+y))**(1/2)"          
                },
                "wmipa":{
                        "chi": None,
                        'w':   None,
                        "phi": None        
                },
                "psi": {
                        "formula": f"(10000/(x+y))^(1/2)",        
                },
                "gubpi": {
                        "formula": "sqrt(div(10000,x+y))"        
                }
        },
        
        # #########################################################
        {
                "faza":{
                        "chi": [[0, 1], [0, 1]],
                        "phi": True,
                        "variables": ["x", "y"],
                        "w": "((x**2 + 2*y**2 + 3*y*x + x + 1)*100/(2*y**2 + y*x + y + 2))"          
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
                                )*Real(100),
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
                        "formula": "((x^2 + 2*(y^2) + 3*y*x + x + 1)*100/(2*(y^2) + y*x + y + 2))"     
                },
                "gubpi": {
                        "formula": "(div((x*x + 2*(y*y) + 3*y*x + x + 1)*100,(2*(y*y) + y*x + y + 2)))"    
                }
        },
        
]


######################################################### 
# Radical Functions

# Benchmark in form w = a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0 / b_n*x^n + b_(m-1)*x^(m-1) + ... + b_1*x
# n < m
# 0 < x < 1
# 0 < غ < 1

def generate_rational_bechmarks(number_of_benchmarks, max_m, max_n, output_path):

        benchmarks = []
        
        for i in range(number_of_benchmarks):
        
                # denuminator must have higher degree than numinator
                # assert(max_m>max_n)
                
                # select degree of numinator and denuminator randomly 
                m = random.randint(1, max_n)
                n = random.randint(0, max_n)
                
                # generate coefficents randomly
                a_coefficients = [round(random.uniform(0, 10),2) for i in range(n+1) ]
                b_coefficients = [round(random.uniform(0, 10),2) for i in range(0,m+1)]
                
                benchmarks.append(
                        {
                               "a_coefficients": a_coefficients,
                               "b_coefficients": b_coefficients 
                        }
                )
        
        # Save it to
        with open(output_path, 'w') as f:
                json.dump(benchmarks, f)


def load_rational_benchmarks(benchmak_path):
        
        benchmaks = []
        
        with open(benchmak_path, 'r') as f:
                benchmak_coefficients = json.load(f)
        
        for c in benchmak_coefficients:
                a_coefs = c['a_coefficients']
                b_coefs = c['b_coefficients']
                
                benchmaks.append({
                'bounds': [0.01, 1],
                'w': "(" + "+".join([ f"{a}*x**{i}" for i, a in enumerate(a_coefs)]) + ")" + " / " + "(" + " + ".join([ f"{b}*x**{j}" for j, b in enumerate(b_coefs)]) + ")",
                'smt_w': Div(Plus([Times(Real(a), Pow(x, Real(i))) for i, a in enumerate(a_coefs)]), Plus([Times(Real(b), Pow(x, Real(j))) for j, b in enumerate(b_coefs)])),
                'smt_phi': And(GE(x, Real(0.01)),LE(x, Real(1))),
                })
        
        return benchmaks        
        
