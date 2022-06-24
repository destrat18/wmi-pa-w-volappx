from abc import ABC, abstractmethod
from itertools import product

from pysmt.fnode import FNode
from pysmt.rewritings import CNFizer, nnf
from pysmt.shortcuts import (
    And,
    Bool,
    Equals,
    FreshSymbol,
    FunctionType,
    Not,
    Or,
    Real,
    Times,
    get_env,
)
from pysmt.simplifier import Simplifier
from pysmt.typing import BOOL, REAL
from pysmt.walkers import IdentityDagWalker

from wmipa.utils import is_pow


class WeightConverter(ABC):
    def __init__(self, variables):
        self.variables = variables

    @abstractmethod
    def convert(self, weight_func, mode):
        """Convert the weight function into a LRA formula

        Args:
            weight_func (FNode): The weight function

        Returns:
            FNode: The formula representing the weight function
        """
        pass


class WeightConverterEUF(WeightConverter):
    def __init__(self, variables):
        super().__init__(variables)
        self.conv_aliases = set()
        self.prod_fn = FreshSymbol(
            typename=FunctionType(REAL, (REAL, REAL)), template="PROD%s"
        )
        self.counter = 1

    def convert(self, weight_func):
        conversion_list = list()
        w, _ = self.convert_rec(weight_func, Bool(False), conversion_list)
        if not w.is_symbol():
            y = self.variables.new_weight_alias(len(self.conv_aliases))
            conversion_list.append(Equals(y, w))
            w = y
        return And(conversion_list)

    def convert_rec(self, formula, branch_condition, conversion_list):
        # print("CONVERT", formula, get_type(formula))
        if formula.is_ite():
            return self._process_ite(formula, branch_condition, conversion_list)
        elif formula.is_times():
            return self._process_times(formula, branch_condition, conversion_list)
        elif is_pow(formula):
            return self._process_pow(formula, branch_condition, conversion_list)
        elif len(formula.args()) == 0:
            return formula, False
        else:
            has_cond = False
            new_children = []
            for arg in formula.args():
                child, cond = self.convert_rec(arg, branch_condition, conversion_list)
                has_cond = has_cond or cond
                new_children.append(child)
            return (
                get_env().formula_manager.create_node(
                    node_type=formula.node_type(), args=tuple(new_children)
                ),
                has_cond,
            )

    def _process_ite(self, formula, branch_condition, conversion_list):
        phi, left, right = formula.args()
        # update branch conditions for children and recursively convert them
        l_cond = Or(branch_condition, Not(phi))
        r_cond = Or(branch_condition, phi)
        left, l_has_cond = self.convert_rec(left, l_cond, conversion_list)
        right, r_has_cond = self.convert_rec(right, r_cond, conversion_list)

        if not l_has_cond:
            left = self.variables.new_weight_alias(len(self.conv_aliases))
            f = self.variables.new_EUF_alias(self.counter)
            e = Equals(left, f)
            self.counter += 1
            conversion_list.append(Or(branch_condition, e))
        if not r_has_cond:
            right = self.variables.new_weight_alias(len(self.conv_aliases))
            f = self.variables.new_EUF_alias(self.counter)
            e = Equals(right, f)
            self.counter += 1
            conversion_list.append(Or(branch_condition, e))

        # add (    branch_conditions) -> y = left
        # and (not branch_conditions) -> y = right
        y = self.variables.new_weight_alias(len(self.conv_aliases))
        self.conv_aliases.add(y)
        el = Equals(y, left)
        er = Equals(y, right)
        conversion_list.append(Or(Or(branch_condition, Not(phi)), el))
        conversion_list.append(Or(Or(branch_condition, phi), er))
        # add also branch_condition -> (not y = left) or (not y = right)
        # to force the enumeration of relevant branch conditions
        conversion_list.append(Or(branch_condition, Not(el), Not(er)))
        return y, True

    def _process_times(self, formula, branch_condition, conversion_list):
        args = (
            self.convert_rec(arg, branch_condition, conversion_list)
            for arg in formula.args()
        )
        const_val = 1
        others = []
        has_cond = False
        for arg, cond in args:
            has_cond = has_cond or cond
            if arg.is_real_constant():
                const_val *= arg.constant_value()
            else:
                others.append(arg)
        const_val = Real(const_val)
        if not others:
            return const_val, has_cond
        else:
            # abstract product between non-constant nodes with EUF
            return Times(const_val, self._prod(others)), has_cond

    def _process_pow(self, formula, branch_condition, conversion_list):
        args = (
            self.convert_rec(arg, branch_condition, conversion_list)
            for arg in formula.args()
        )
        (base, _), (exponent, _) = args
        if exponent.is_zero():
            return Real(1), False
        else:
            # expand power into product and abstract it with EUF
            n, d = (
                exponent.constant_value().numerator,
                exponent.constant_value().denominator,
            )
            return self._prod([base for _ in range(n // d)]), False

    def _prod(self, args):
        """Abstract the product between args with EUF

        Args:
            args (list(FNode)): List of nodes to multiply. The list is emptied

        Returns:
            FNode: node representing abstracted product
        """
        assert isinstance(args, list)
        if len(args) == 1:
            return args.pop()
        curr = args.pop()
        while args:
            curr = self.prod_fn(args.pop(), curr)
        return curr


class WeightConverterSkeleton(WeightConverter):
    def __init__(self, variables):
        super().__init__(variables)
        self.conv_bools = set()
        self.cnfizer = TseitinCNFizer(self.new_label)
        self.cnfizer2 = DeMorganCNFizer()

    def new_label(self):
        w = self.variables.new_weight_bool(len(self.conv_bools))
        self.conv_bools.add(w)
        return w

    def convert(self, weight_func):
        conversion_list = list()
        LTv2 = False
        if LTv2:
            return self.convert_sk(weight_func)
        self._convert_rec(weight_func, Bool(False), conversion_list)
        return And(conversion_list)

    def _convert_rec(self, formula, branch_condition, conversion_list):
        if formula.is_ite():
            return self._process_ite(formula, branch_condition, conversion_list)
        elif formula.is_theory_op():
            for arg in formula.args():
                self._convert_rec(arg, branch_condition, conversion_list)

    def _process_ite(self, formula, branch_condition, conversion_list):
        phi, left, right = formula.args()
        # print("MIPHI", serialize(phi), branch_condition)
        if self.is_atom(phi):
            # print("ISATOM")
            if branch_condition is not None:
                l_cond = Or(branch_condition, Not(phi))
                r_cond = Or(branch_condition, phi)
                # l_cond = Not(phi)
                # r_cond = phi
            else:
                l_cond = Not(phi)
                r_cond = phi
            conversion_list.append(Or(branch_condition, phi, Not(phi)))
            # print("Clause", Or(branch_condition, phi, Not(phi)))
            self._convert_rec(left, l_cond, conversion_list)
            self._convert_rec(right, r_cond, conversion_list)
        else:
            w = self.new_label()
            # print("IS COMPLEX, WE ADD WEIGHT", w)
            if branch_condition is not None:
                l_cond = Or(branch_condition, Not(w))
                r_cond = Or(branch_condition, w)
                # l_cond = Not(w)
                # r_cond = w
            else:
                l_cond = Not(w)
                r_cond = w
            # print("PHI", phi)
            # print("NNF(PHI)", nnf(phi))
            for clause in self.cnfizer.convert(nnf(phi)):
                # conversion_list.append(Or(*clause))
                conversion_list.append(Or(l_cond, *clause))
                # print("Clause", serialize(Or(l_cond, *clause)))
                # print("tYpe", Or(l_cond, *clause).get_type())

            # Probably error here?
            # print("Not(PHI)", Not(phi))
            # print("NNF(Not(PHI))", nnf(Not(phi)))
            for clause in self.cnfizer.convert(nnf(Not(phi))):
                conversion_list.append(Or(r_cond, *clause))
                # conversion_list.append(Or(*clause))
                # print("Clause", serialize(Or(r_cond, *clause)))
                # print("tYpe", Or(r_cond, *clause).get_type())
            conversion_list.append(Or(branch_condition, w, Not(w)))
            # print("Clause final", serialize(Or(branch_condition, w, Not(w))))
            self._convert_rec(left, l_cond, conversion_list)
            self._convert_rec(right, r_cond, conversion_list)

    def is_atom(self, node):
        return node.is_symbol(BOOL) or node.is_theory_relation()

    def convert_sk(self, formula: FNode):
        if formula.is_ite():
            return self._process_ite_sk(formula)
        elif formula.is_theory_op():
            results = (
                s for a in formula.args() if not (s := self.convert_sk(a)).is_true()
            )
            return And(results)
        else:
            return Bool(True)

    def _process_ite_sk(self, formula):
        phi, left, right = formula.args()

        skl = self.convert_sk(left)
        skr = self.convert_sk(right)
        if skl.is_true() and skr.is_true():
            C = self.variables.new_weight_bool(len(self.conv_bools))
            self.conv_bools.add(C)
            skl = C
            skr = Not(C)
        return Or(And(phi, skl), And(Not(phi), skr))


class CNFPreprocessor(IdentityDagWalker):
    def walk_or(self, formula, args, **kwargs):
        children = []
        for arg in args:
            if arg.is_true():
                return Bool(True)
            elif arg.is_false():
                continue
            elif arg.is_or():
                children.extend(arg.args())
            elif arg.is_not() and arg.arg(0).is_and():
                children.extend(map(Not, arg.arg(0).args()))
            else:
                children.append(arg)
        return Or(children)

    def walk_and(self, formula, args, **kwargs):
        children = []
        for arg in args:
            if arg.is_false():
                return Bool(False)
            elif arg.is_true():
                continue
            elif arg.is_and():
                children.extend(arg.args())
            elif arg.is_not() and arg.arg(0).is_or():
                children.extend(map(Not, arg.arg(0).args()))
            else:
                children.append(arg)
        return And(children)

    def walk_implies(self, formula, args, **kwargs):
        left, right = formula.args()
        left_a, right_a = args
        return self.walk_or(Or(Not(left), right), (Not(left_a), right_a), **kwargs)


class TseitinCNFizer(CNFizer):
    def __init__(self, new_label, environment=None):
        super().__init__(environment)
        self.preprocessor = CNFPreprocessor(env=environment)
        self.new_label = new_label

    def _key_var(self, formula):
        if formula in self._introduced_variables:
            res = self._introduced_variables[formula]
        else:
            res = self.new_label()
            self._introduced_variables[formula] = res
        return res

    def convert(self, formula):
        """Convert formula into an Equisatisfiable CNF.

        Returns a set of clauses: a set of set of literals.
        """
        # print("Preprocessing", formula.serialize())
        # formula = SkeletonSimplifier().simplify(formula)
        formula = self.preprocessor.walk(formula)
        # print("Done:", formula.serialize())
        # if is_cnf(formula):
        #    return frozenset([frozenset(x.args()) for x in formula.args()])
        tl, _cnf = self.walk(formula)
        # print("END PREVIOUS CONVERSION")
        if not _cnf:
            return [frozenset([tl])]
        res = []
        for clause in _cnf:
            if len(clause) == 0:
                return CNFizer.FALSE_CNF
            simp = []
            for lit in clause:
                if lit is tl or lit.is_true():
                    # Prune clauses that are trivially TRUE
                    # and clauses containing the top level label
                    simp = None
                    break
                elif not lit.is_false() and lit is not Not(tl):
                    # Prune FALSE literals
                    simp.append(lit)
            if simp:
                res.append(frozenset(simp))
        # print("CNF:", And(map(Or, res)))
        # print("Res", res)
        return frozenset(res)

    def walk_not(self, formula, args, **kwargs):
        # print("WALK NOT", formula, args)
        a, _cnf = args[0]
        if a.is_true():
            return self.mgr.Bool(False), CNFizer.TRUE_CNF
        elif a.is_false():
            return self.mgr.Bool(True), CNFizer.TRUE_CNF
        else:
            return Not(a), _cnf

    def walk_and(self, formula, args, **kwargs):
        # print("WALK AND", formula, args)
        if len(args) == 1:
            return args[0]

        k = self._key_var(formula)
        # print("KKK", k)
        # _cnf = [frozenset([k] + [self.mgr.Not(a).simplify() for a,_ in args])]
        _cnf = []
        for a, c in args:
            _cnf.append(frozenset([a, self.mgr.Not(k)]))
            for clause in c:
                _cnf.append(clause)
        # print("_CNF", _cnf)
        return k, frozenset(_cnf)

    def walk_or(self, formula, args, **kwargs):
        # print("WALK OR", formula, args)
        if len(args) == 1:
            return args[0]
        k = self._key_var(formula)
        # print("KKK", k)
        _cnf = [frozenset([self.mgr.Not(k)] + [a for a, _ in args])]
        for a, c in args:
            #    _cnf.append(frozenset([k, self.mgr.Not(a)]))
            for clause in c:
                _cnf.append(clause)
        # print("_CNF", _cnf)
        return k, frozenset(_cnf)


class DeMorganCNFizer(IdentityDagWalker):
    def convert(self, formula):
        # convert in Negative Normal Form
        # print("FFF", serialize(formula))
        formula = nnf(formula)
        # print("NNF:", formula.serialize())
        formula = self.walk(formula)
        assert self.is_cnf(formula), formula.serialize()
        # print("CONVERTED INTO (and check {})".format(formula.is_or()))
        res = []
        if formula.is_or():
            # print("ATOMS by clause!", frozenset([x for x in formula.args()]))
            return frozenset([frozenset([x for x in formula.args()])])
        for c in formula.args():
            # print("ANALYZING", c, c.get_type(), is_atom(c) or c.is_not() and is_atom(c.arg(0)))
            if is_atom(c) or c.is_not() and is_atom(c.arg(0)):
                res.append(frozenset([c]))
            else:
                res.append(frozenset([x for x in c.args()]))
            # print("TEMP", res)
        # res = [frozenset([x for x in c.args()]) for c in formula.args()]
        # print("ATOMS", res)
        return frozenset(res)

    def walk_and(self, formula, args, **kwargs):
        # print("Walking and: ", formula.serialize(), "args: ", args)

        and_args = set()
        for a in args:
            if a.is_true():
                continue
            elif a.is_false():
                return Bool(False)
            elif self.is_literal(a) or a.is_or():
                and_args.add(a)
            else:
                assert a.is_and(), "{} {}".format(formula.serialize(), a.serialize())
                # flatten AND
                and_args.update(a.args())
        # print("Returning", And(and_args).serialize())
        return And(and_args)

    def walk_or(self, formula, args, **kwargs):
        # print("Walking or: ", formula.serialize(), "args: ", args)
        # flatten or
        or_literals = set()
        list_of_and = list()
        for a in args:
            if a.is_false():
                continue
            elif a.is_true():
                return Bool(True)
            elif self.is_literal(a):
                or_literals.add(a)
            elif a.is_or():
                or_literals.update(a.args())
            else:
                assert a.is_and(), "{} {}".format(formula.serialize(), a.serialize())
                list_of_and.append(a.args())

        list_of_and.extend((x,) for x in or_literals)
        # print("Flatten_args:", list_of_and)
        and_args = set()

        for comb in product(*list_of_and):
            # print("Comb:", comb)
            or_args = set()
            for item in comb:
                if self.is_literal(item):
                    or_args.add(item)
                elif item.is_or():
                    or_args.update(item.args())
                else:
                    assert item.is_and(), "{}".format(item.serialize())
                    or_args.update(item.args())
            and_args.add(Or(or_args))
        # print("Returning", And(and_args).serialize())

        return And(and_args)

    def is_atom(self, node):
        return node.is_symbol(BOOL) or node.is_theory_relation()

    def is_literal(self, node):
        return is_atom(node) or (node.is_not() and is_atom(node.arg(0)))

    def is_clause(self, formula):
        return is_literal(formula) or (
            formula.is_or() and all(is_literal(l) for l in formula.args())
        )

    def is_cnf(self, formula):
        return is_clause(formula) or (
            formula.is_and() and all(is_clause(c) for c in formula.args())
        )


def is_atom(node):
    return node.is_symbol(BOOL) or node.is_theory_relation()


def is_literal(node):
    return is_atom(node) or (node.is_not() and is_atom(node.arg(0)))


def is_clause(formula):
    return is_literal(formula) or (
        formula.is_or() and all(is_literal(l) for l in formula.args())
    )


def is_cnf(formula):
    return is_clause(formula) or (
        formula.is_and() and all(is_clause(c) for c in formula.args())
    )


class SkeletonSimplifier(Simplifier):
    """Simplifier that does not simplify formulas like Or(phi, Not(phi))"""

    def walk_or(self, formula, args, **kwargs):
        if len(args) == 2 and args[0] == args[1]:
            return args[0]

        new_args = set()
        for a in args:
            if a.is_false():
                continue
            if a.is_true():
                return self.manager.TRUE()
            if a.is_or():
                for s in a.args():
                    new_args.add(s)
            else:
                new_args.add(a)

        if len(new_args) == 0:
            return self.manager.FALSE()
        elif len(new_args) == 1:
            return next(iter(new_args))
        else:
            return self.manager.Or(new_args)
