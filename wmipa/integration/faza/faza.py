import  gurobipy as gp 
import numpy as np
from itertools import product
import time
import copy
import logging
import pandas as pd
import uuid
import queue
import threading
import argparse
logging.basicConfig(level=logging.INFO)
    

import sympy as sym
from sympy.parsing.sympy_parser import parse_expr

import  gurobipy as gp 
import numpy as np
from itertools import product
import time
import copy
import logging
import pandas as pd
import uuid
import queue
import threading
import argparse
import multiprocessing as mp


import sympy as sym
from sympy.parsing.sympy_parser import parse_expr

def generate_monoids(vars, d):

    # Generate all combinations of variables of degree up to d
    monoids = set()

    combinations = product(vars, repeat=d)
    for combo in combinations:
        monoids.add(sym.prod(combo))
    return list(monoids)

def generate_monoids_up_to_degree(vars, d):

    # Generate all combinations of variables of degree up to d
    monoids = set()
    for i in range(0, d+1):
        combinations = product(vars, repeat=i)
        for combo in combinations:
            monoids.add(sym.prod(combo))
            
    monoids = list(set(monoids))
    return list(monoids)


def generate_handelman_equations(degree, f_list, g, vars):
    
    # Generate all possible monoids
    monoids = generate_monoids_up_to_degree(f_list, degree)
    
    # create temp variable for each monoind
    temp_vars = []
    for i in range(len(monoids)):
        l = sym.symbols(f"l_{i}")
        monoids[i] = monoids[i]*l
        temp_vars.append(l)

    # TODO do it for negetive    
    a_pos = sym.simplify(sym.expand(sum(monoids)-(g)))

    pos_equations = []
    for d in range(degree, 0, -1):
        for mn in generate_monoids(vars, d):
            new_assert = a_pos.coeff(mn)
            pos_equations.append(new_assert)
            a_pos = sym.simplify(a_pos - (new_assert)*mn)
    pos_equations.append(a_pos)

    a_neg = sym.simplify(sym.expand(sum(monoids)-(-1*g)))

    neq_equations = []
    for d in range(degree, 0, -1):
        for mn in generate_monoids(vars, d):
            new_assert = a_neg.coeff(mn)
            neq_equations.append(new_assert)
            a_neg = sym.simplify(a_neg - (new_assert)*mn)
    neq_equations.append(a_neg)

    
    return pos_equations, neq_equations, temp_vars


def is_feasible(equations, temp_vars):
    
    coeff_matrix = []
    for a in equations:
        coeff = []
        for v in [1]+temp_vars:
            try:
                coeff.append(float(a.coeff_monomial(v)))
            except:
                coeff.append(0)
        coeff_matrix.append(coeff)

    M = np.array(coeff_matrix)
    RHS = (M[:,:1]*-1).flatten()
    M = M[:,1:]
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            m.setObjective(True, gp.GRB.MAXIMIZE)
            
            lp_l = m.addMVar(shape=len(temp_vars), name="l", ub=float('inf'), lb=0)            
            m.addConstr( M @ lp_l == RHS, name="c")

            m.optimize()
    
            return not(m.status==gp.GRB.INFEASIBLE), m.runtime
    

def read_input(integrand_path, polytope_path):
    
    # Read integrand and list of variables
    with open(integrand_path) as f:
        vars = sym.symbols(f.readline().strip().split(" "))
        integrand = sym.parse_expr(f.readline())
    
    # Read bounds
    bounds = []
    with open(polytope_path) as f:

        n_ineq, n_vars = pd.to_numeric(f.readline().split(" "))
        n_vars = n_vars -1

        bounds = []
        for i in range(n_vars):
            bounds.append([None, None])

        for i in range(n_ineq):
            coeffs = [float(c) for c in f.readline().strip().split(" ")]
            # print(coeffs)
            constant = coeffs[0]
            for j, c in enumerate(coeffs[1:]):
                if c < 0:
                    bounds[j][1] = constant/abs(c)
                elif c > 0:
                    bounds[j][0] = -1*constant/abs(c)    
                # print(j, c, bounds)
                
    return integrand, bounds, vars


def generate_f_list(vars):
    
    f_list = []
    bound_vars = []

    for i, var in enumerate(vars):
        l = sym.Symbol(f"l_{str(var)}")
        u = sym.Symbol(f"u_{str(var)}")
        f_list.append(u-var)
        f_list.append(var-l)
        bound_vars.append([l, u])
    return f_list, bound_vars 


class Checker(mp.Process):
    
    
    command = None
    hardhat_instance = None
    
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
    
    def __init__(self, inside_equations, outside_equations, vars, temp_vars, to_check_queue, checked_queue):
        mp.Process.__init__(self)
        self.check_next = True
        self.inside_equations = inside_equations
        self.outside_equations = outside_equations
        self.temp_vars = temp_vars
        self.vars = vars
        self.to_check_queue = to_check_queue
        self.checked_queue = checked_queue
    
    
    def run(self):
        
        # Read the queue until it finishes
        while True:
            try:
                # Get a hrect from the to_check_queue
                hrect = self.to_check_queue.get()
                if hrect == None: # Time to close the process
                    break
                
                cur_depth, cur_bounds, cur_volume, bound_vars = hrect
                
                # timing
                start_time = time.time()

                # Optimization porblem
                subs_dict = {}
                for i in range(len(cur_bounds)):
                    subs_dict[bound_vars[i][0]] = cur_bounds[i][0] 
                    subs_dict[bound_vars[i][1]] = cur_bounds[i][1]

                cur_inside_equations_ = [
                    sym.Poly(a.subs(subs_dict)) for a in self.inside_equations
                    ]
                cur_outside_equations_ = [
                    sym.Poly(a.subs(subs_dict)) for a in self.outside_equations
                    ]

                subs_time = time.time()-start_time

                start_time = time.time()
                is_inside, inside_runtime = is_feasible(cur_inside_equations_, self.temp_vars)
                is_outside, outside_runtime = is_feasible(cur_outside_equations_, self.temp_vars)
                solver_time = time.time()-start_time
                
                # logging.info(f"Bounds: {cur_bounds}, Inside: {is_inside}, Outside: {is_outside}")

                stats = {
                    "solver_time": solver_time,
                    "subs_time": subs_time
                }
                            
                # if it's inside or outside remove it from error
                if is_inside:
                    self.checked_queue.put((
                            0, # Inside
                            cur_volume, # volume of hyper-rect
                            cur_bounds,
                            stats
                        ))
                elif is_outside:
                    self.checked_queue.put(
                        (
                            1, # mean it is outside
                            cur_volume, # volume of hyper-rect
                            cur_bounds,
                            stats
                        )
                    )
                else:
                    self.checked_queue.put(
                        (
                            2, # means it must be splitted
                            cur_volume, # volume of hyper-rect
                            cur_bounds,
                            stats
                        )
                    )
                    
                    # create two smaller hyper-rects
                    i = cur_depth%len(self.vars)
                    s_bounds = cur_bounds[i]
                    s_bound_middle = (s_bounds[0]+s_bounds[1])/2
                    
                    left_bounds = copy.deepcopy(cur_bounds)
                    left_bounds[i]=[s_bounds[0], s_bound_middle]
                    self.to_check_queue.put((cur_depth+1, left_bounds, cur_volume/2, bound_vars))
                    
                    right_bounds = copy.deepcopy(cur_bounds)
                    right_bounds[i]=[s_bound_middle, s_bounds[1]]
                    self.to_check_queue.put((cur_depth+1, right_bounds, cur_volume/2, bound_vars))
            
            except Exception as e:
                logging.exception(e)
        
        logging.debug("I'm done!")

def calculate_approximate_volume(
        degree,
        max_workers,
        integrand, 
        bounds, 
        vars,
        threshold,
    ):

    start_time = time.time()


    # We apply Handelman below
    # The input form of Handelman is f_i>=0 => g >=0

    # RHS
    g = sym.simplify(integrand)
    
    # Generate symbolic f_is with symbolic variable for upper bound and lower bound
    f_list, bound_vars = generate_f_list(vars)

    # we apply handelman here to generate eqations based on l_is.
    # l_0 + l_1(f_1) + l_2(f_2) + ... + l_n(f_n) = g
    inside_equations, outside_equations, temp_vars = generate_handelman_equations(
        degree=degree,
        f_list=f_list,
        g = g,
        vars=vars
    )
    
    logging.info(f"Integral {g} over {bounds}")
    
    # We run n instance of hardhats which listion on a queue and execute blocks
    # Create two queues
    to_check_queue = mp.Queue()
    checked_queue = mp.Queue()

    checker_list = []
    for _ in range(max_workers):
        checker = Checker(
            inside_equations=inside_equations,
            outside_equations=outside_equations,
            temp_vars=temp_vars,
            vars=vars,
            to_check_queue=to_check_queue,
            checked_queue=checked_queue
        )
        checker.daemon=True
        checker.start()
        checker_list.append(checker)
    
    # We put the first hyper-rectangle
    start_volume = sym.prod([abs(b[1]-b[0]) for b in bounds])
     
    to_check_queue.put(
        (
            0, # start depth
            bounds, # starting bounds
            start_volume, #start volume
            bound_vars # bound vars that need to be replaced
        )
    )
    
    error = start_volume # Start volume
    volume = 0
    
    total_hrect_checked = 0
    total_solver_time = 0
    total_subs_time = 0
    
    while error > threshold:
        res = checked_queue.get()
        
        total_hrect_checked += 1
        total_solver_time += res[-1]["solver_time"]
        total_subs_time += res[-1]["subs_time"]
        
        if res[0]==0:      
            error -= res[1]
            volume += res[1]
        elif res[0]==1:
            error -= res[1]
        
        
        if total_hrect_checked%250 == 0:
            logging.info(f"#Checked: {total_hrect_checked}, Error: {error:.6f}, Volume: {volume+error:.6f}, Total time: {(time.time()-start_time)/60:.2f}m({(time.time()-start_time):.2f}s)")
                     
            logging.info(f"Avg subs time: {(total_subs_time)/(total_hrect_checked*2):.6f}s, Avg solver time: {total_solver_time/(total_hrect_checked*2):.6f}s")


    # Must stop the thread
    logging.info("###################### Done ######################")
    logging.info(f"#Checked: {total_hrect_checked}, Error: {error:.6f}, Volume: {volume+error:.6f}, Total time: {(time.time()-start_time):.2f}")
    
    if total_hrect_checked > 0:
        logging.info(f"Avg subs time: {(total_subs_time)/(total_hrect_checked):.6f}s, Avg solver time: {total_solver_time/(total_hrect_checked):.6f}s")
    
    
    # Clear queue
    while to_check_queue.qsize():
        try:
            to_check_queue.get_nowait()
        except:
            break
        
    for checker in checker_list:
        to_check_queue.put(None)
        to_check_queue.put(None)
    
    for checker in checker_list:
        checker.terminate()
        # checker.join()
    
    
    return volume+error, {
        "hrect_checked_num": total_hrect_checked,
        "total_solver_time": total_solver_time,
        "total_subs_time": total_subs_time        
    }

def find_upper_bound(    
                     
    degree, 
    f_list, 
    g, 
    vars,
    bound_vars,
    bounds
    ):
    
    
     # Optimization porblem
    subs_dict = {}
    for i in range(len(bounds)):
        subs_dict[bound_vars[i][0]] = bounds[i][0] 
        subs_dict[bound_vars[i][1]] = bounds[i][1]

    # Subsitute
    # we apply handelman here to generate eqations based on l_is.
    # l_0 + l_1(f_1) + l_2(f_2) + ... + l_n(f_n) = g
    
    # Since we need to find upper and lower bounds, we introduce two new variables
    U = sym.Symbol(f"U_{str(uuid.uuid4()).split('-')[0]}")

    # g <= U => U - g>=0
    # TODO: add proof rules
    if sym.denom(g)!=1:
        n, d = sym.fraction(g)
        gU = U*d - n
    else:
        gU = U - g
    
        
    upper_equations, _, temp_vars = generate_handelman_equations(
        degree=degree,
        f_list=f_list,
        g = gU,
        vars=vars
    )
    equations = [
        sym.Poly(a.subs(subs_dict)) for a in upper_equations
        ]
    
    coeff_matrix = []
    for a in equations:
        coeff = []
        for v in [1]+temp_vars+[U]:
            try:
                coeff.append(float(a.coeff_monomial(v)))
            except:
                coeff.append(0)
        coeff_matrix.append(coeff)

    M = np.array(coeff_matrix)
    RHS = (M[:,:1]*-1).flatten()
    M = M[:,1:]
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            
            lp_l = m.addMVar(shape=len(temp_vars)+1, name="l", ub=[float('inf')]*len(temp_vars)+[float('inf')], lb=[0]*len(temp_vars)+[0])

            m.addConstr( M @ lp_l == RHS, name="c")

            m.setObjective(lp_l[-1], gp.GRB.MINIMIZE)            
            m.optimize()
            
            if m.status == gp.GRB.OPTIMAL:
                return True, float(lp_l[-1].x), m.runtime
            else:
                return False, None, m.runtime


def find_lower_bound(    
                     
    degree, 
    f_list, 
    g, 
    vars,
    bound_vars,
    bounds
    ):
    
    
     # Optimization porblem
    subs_dict = {}
    for i in range(len(bounds)):
        subs_dict[bound_vars[i][0]] = bounds[i][0] 
        subs_dict[bound_vars[i][1]] = bounds[i][1]

    # Subsitute
    # we apply handelman here to generate eqations based on l_is.
    # l_0 + l_1(f_1) + l_2(f_2) + ... + l_n(f_n) = g
    
    # Since we need to find upper and lower bounds, we introduce two new variables
    L = sym.Symbol(f"L_{str(uuid.uuid4()).split('-')[0]}")

    # g <= U => U - g>=0    
    
    # TODO: add proof rules
    if sym.denom(g)!=1:
        n, d = sym.fraction(g)
        gL = n - L*d
    else:
        gL = g - L
    
    lower_equations, _, temp_vars = generate_handelman_equations(
        degree=degree,
        f_list=f_list,
        g = gL,
        vars=vars
    )
    equations = [
        sym.Poly(a.subs(subs_dict)) for a in lower_equations
        ]
    
    coeff_matrix = []
    for a in equations:
        coeff = []
        for v in [1]+temp_vars+[L]:
            try:
                coeff.append(float(a.coeff_monomial(v)))
            except:
                coeff.append(0)
        coeff_matrix.append(coeff)

    M = np.array(coeff_matrix)
    RHS = (M[:,:1]*-1).flatten()
    M = M[:,1:]
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            
            lp_l = m.addMVar(shape=len(temp_vars)+1, name="l", ub=[float('inf')]*len(temp_vars)+[0], lb=[0]*len(temp_vars)+[float('-inf')])
            
            m.addConstr( M @ lp_l == RHS, name="c")
            m.setObjective(lp_l[-1], gp.GRB.MAXIMIZE)
                        
            m.optimize()
            
            if m.status == gp.GRB.OPTIMAL:
                return True, float(lp_l[-1].x), m.runtime
            else:
                return False, None, m.runtime

def calculate_approximate_wmi(
        degree,
        max_workers,
        integrand, 
        bounds, 
        vars,
        threshold,
    ):

    start_time = time.time()

    # We apply Handelman below
    # The input form of Handelman is f_i>=0 => g >=0

    # RHS
    g = sym.simplify(integrand)
    
    # Generate symbolic f_is with symbolic variable for upper bound and lower bound
    f_list, bound_vars = generate_f_list(vars)
    
    ###### for psi+ ######
    has_upper_bound, upper_bound, runtime = find_upper_bound(
        degree=degree,
        f_list=f_list,
        g=g,
        bound_vars=bound_vars,
        bounds=bounds,
        vars=vars
    )
    
    # We introduce a new variable
    y = sym.Symbol(f"y_{str(uuid.uuid4()).split('-')[0]}")


    # TODO: add proof rules
    if sym.denom(g)!=1:
        n, d = sym.fraction(g)
        new_integrand = n - d*y
    else:
        new_integrand = g - y
        
    new_bounds = bounds+[[0, upper_bound]]
    new_vars = vars+[y]
    
    logging.info(f"Psi+ bounds: [{0}, {upper_bound}]")
    
    
    if upper_bound != 0:
        psi_plus, psi_plus_stats = calculate_approximate_volume(
            degree=degree,
            max_workers=max_workers,
            integrand=new_integrand,
            bounds=new_bounds,
            vars=new_vars,
            threshold=threshold
        )
    else:
        psi_plus = 0
        psi_plus_stats = {
        "hrect_checked_num": 0,
        "total_solver_time": 0,
        "total_subs_time": 0        
        }
    
    ###### for psi- ######    
    # We need to calculate this for psi-
    
    has_lower_bound, lower_bound, runtime = find_lower_bound(
        degree=degree,
        f_list=f_list,
        g=g,
        bound_vars=bound_vars,
        bounds=bounds,
        vars=vars
    )
    
    # We introduce a new variable
    y = sym.Symbol(f"y_{str(uuid.uuid4()).split('-')[0]}")


    # TODO: add proof rules
    if sym.denom(g)!=1:
        n, d = sym.fraction(g)
        new_integrand = d*y - n 
    else:
        new_integrand = y - g
        
    new_bounds = bounds+[[lower_bound, 0]]
    new_vars = vars+[y]
    
    logging.info(f"Psi- bounds: [{lower_bound}, {0}]")
    
    if lower_bound != 0:
        psi_minus, psi_minus_stats = calculate_approximate_volume(
            degree=degree,
            max_workers=max_workers,
            integrand=new_integrand,
            bounds=new_bounds,
            vars=new_vars,
            threshold=threshold
        )
    else:
        psi_minus = 0
        psi_minus_stats = {
        "hrect_checked_num": 0,
        "total_solver_time": 0,
        "total_subs_time": 0        
        }
    
    volume = psi_plus-psi_minus
    logging.info(f"Shape: {integrand}, Volume: {psi_plus}(Psi+) - {psi_minus}(Psi-)={volume}")
    
    return volume, {
        "hrect_checked_num": psi_plus_stats["hrect_checked_num"]+psi_minus_stats["hrect_checked_num"],
        "total_solver_time": psi_plus_stats["total_solver_time"]+psi_minus_stats["total_solver_time"],
        "total_subs_time": psi_plus_stats["total_subs_time"]+psi_minus_stats["total_subs_time"]        
    }
    