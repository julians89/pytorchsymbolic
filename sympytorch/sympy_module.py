import sympy
import torch
import numpy as np
import functools as ft

def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)
    return fn_

_global_func_lookup = {
    sympy.Mul: _reduce(torch.mul),
    sympy.Add: _reduce(torch.add),
    sympy.div: torch.div,
    sympy.Abs: torch.abs,
    sympy.sign: torch.sign,
    # Note: May raise error for ints.
    sympy.ceiling: torch.ceil,
    sympy.floor: torch.floor,
    sympy.log: torch.log,
    sympy.exp: torch.exp,
    sympy.sqrt: torch.sqrt,
    sympy.cos: torch.cos,
    sympy.acos: torch.acos,
    sympy.sin: torch.sin,
    sympy.asin: torch.asin,
    sympy.tan: torch.tan,
    sympy.atan: torch.atan,
    sympy.atan2: torch.atan2,
    # Note: May give NaN for complex results.
    sympy.cosh: torch.cosh,
    sympy.acosh: torch.acosh,
    sympy.sinh: torch.sinh,
    sympy.asinh: torch.asinh,
    sympy.tanh: torch.tanh,
    sympy.atanh: torch.atanh,
    sympy.Pow: torch.pow,
    sympy.re: torch.real,
    sympy.im: torch.imag,
    sympy.arg: torch.angle,
    # Note: May raise error for ints and complexes
    sympy.erf: torch.erf,
    sympy.loggamma: torch.lgamma,
    sympy.Eq: torch.eq,
    sympy.Ne: torch.ne,
    sympy.StrictGreaterThan: torch.gt,
    sympy.StrictLessThan: torch.lt,
    sympy.LessThan: torch.le,
    sympy.GreaterThan: torch.ge,
    sympy.And: torch.logical_and,
    sympy.Or: torch.logical_or,
    sympy.Not: torch.logical_not,
    sympy.Max: torch.max,
    sympy.Min: torch.min,
    # Matrices
    sympy.MatAdd: torch.add,
    sympy.HadamardProduct: torch.mul,
    sympy.Trace: torch.trace,
    # Note: May raise error for integer matrices.
    sympy.Determinant: torch.det,
    #sympy.core.numbers.Pi: np.pi,
    sympy.functions.elementary.complexes.conjugate: torch.conj,
}

class _Node(torch.nn.Module):
    def __init__(self, expr, _memodict, _init_vals) -> None:
        super().__init__()

        if issubclass(expr.func, sympy.Symbol): #Symbols are converted to params OR inputs (if not given as init value)!
            
            try:
                setattr(self, str(expr), torch.nn.Parameter(torch.tensor(float(_init_vals[str(expr)]))))
                self._torch_func = lambda: getattr(self, str(expr))

                """
                self._value = torch.nn.Parameter(torch.tensor(float(_init_vals[str(expr)])))
                self._torch_func = lambda: self._value
                """
                self._args = ()
            except KeyError:
                self._torch_func = lambda value: value
                self._args = ((lambda memodict: memodict[str(expr)]),)
        elif issubclass(expr.func, sympy.Float):
            self._torch_func = lambda: torch.tensor(float(expr))
            self._args = ()
        elif issubclass(expr.func, sympy.Rational):
            self.register_buffer('_numerator', torch.tensor(expr.p, dtype=torch.get_default_dtype()))
            self.register_buffer('_denominator', torch.tensor(expr.q, dtype=torch.get_default_dtype()))
            self._torch_func = lambda: self._numerator / self._denominator
            self._args = ()
        elif issubclass(expr.func, type(sympy.pi)):
            self._torch_func = lambda: torch.tensor(np.pi)
            self._args = ()
        else:
            self._torch_func = _global_func_lookup[expr.func]
            args = []
            for arg in expr.args:
                try:
                    arg_ = _memodict[arg]
                except KeyError:
                    arg_ = type(self)(expr=arg, _memodict=_memodict, _init_vals=_init_vals)
                    _memodict[arg] = arg_
                args.append(arg_)
            self._args = torch.nn.ModuleList(args)


    def forward(self, memodict):
        args = []
        for arg in self._args:
            try:
                arg_ = memodict[arg]
            except KeyError:
                arg_ = arg(memodict)
                memodict[arg] = arg_
            args.append(arg_)
        
        return self._torch_func(*args)

class SymPyModule(torch.nn.Module):
    def __init__(self, expressions, init_vals) -> None:
        super().__init__()

        self._init_vals = init_vals
        self._memodict = {}

        self._nodes = torch.nn.ModuleList(
            [_Node(expr=expr, _memodict=self._memodict, _init_vals=self._init_vals) for expr in expressions]
        )

    def forward(self, **symbols):
        out = [node(symbols) for node in self._nodes]
        out = torch.broadcast_tensors(*out)
        return torch.stack(out, dim=-1)