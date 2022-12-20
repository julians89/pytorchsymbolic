<h1 align="center">pytorchsymbolic</h1>

Run symbolic optimization with sympy and pytorch!
Code is largely adopted by <a href="https://github.com/patrick-kidger/sympytorch">sympytorch</a>.
Currently the major change is in the data handling princilpes.
While sympytorch wants to optimize given floats, we optimize symbols.

```python
import sympy, torch, sympytorch

x = sympy.Symbol('x')
y = sympy.Symbol('y')
t = sympy.Symbol('t')

z = x * sympy.cos(y * sympy.sin(x * sympy.erf(y))) * sympy.cos(y) * t + sympy.Float(3)

sp = SymPyModule(z, init_vals={'x': 3, 'y': 1})

sp(t=2)
```

will return: tensor(5.7192, grad_fn=<AddBackward0>)

Symbols that are given in 'init_vals' dict are initialized as trainable variables. All not given as initial variables will be expected to be passed at runtime.