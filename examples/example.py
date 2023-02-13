import sympy
import copy

def gaussian(x_name, mu_name, sigma_name):
    x = sympy.Symbol(x_name, real=True)
    mu = sympy.Symbol(mu_name, real=True)
    sigma = sympy.Symbol(sigma_name, real=True)
    two = sympy.Float('2.0')
    half = sympy.Float('0.5')

    return sympy.exp(-half*((x-mu)/sigma)**2) / (sigma * sympy.sqrt(two * sympy.pi))

def f_fa_func():
    return gaussian('x_fa', 'mu_fa', 'sigma_fa')

def f_d_func():
    return gaussian('x_d', 'mu_d', 'sigma_d')

def f_s_func():

    x = sympy.Symbol('x', real=True)
    y = sympy.Symbol('y', real=True)

    r = sympy.sqrt(x ** 2 + y ** 2)
    #phi = sympy.atan(y/x)
    phi = sympy.atan2(y, x)
    polar = sympy.Matrix([r, phi])

    cartesian = sympy.Matrix([x, y])

    f_fa = f_fa_func()
    f_d = f_d_func()

    f_s = f_fa.subs(sympy.Symbol('x_fa', real=True), phi) * f_d.subs(sympy.Symbol('x_d', real=True), r) * sympy.Abs(sympy.det(polar.jacobian(cartesian))) 
    f_s = sympy.simplify(f_s)

    return f_s

def transform_pdf(pdf, transform_by, to_transform, transformed):

    #Init transformation symbols
    yaw = sympy.Symbol('yaw_' + transform_by, real=True)
    xb = sympy.Symbol('x_' + transform_by, real=True)
    yb = sympy.Symbol('y_' + transform_by, real=True)

    A = sympy.Matrix([[sympy.cos(yaw), -sympy.sin(yaw)], 
                      [sympy.sin(yaw), sympy.cos(yaw)]])

    b = sympy.Matrix([[xb], [yb]])

    xt = sympy.Symbol('x_' + transformed, real=True)
    yt = sympy.Symbol('y_' + transformed, real=True)

    Xt = sympy.Matrix([[xt], [yt]])

    #Compute inverse transformation and jacobian
    f_inv = A.inv() * (Xt - b)
    jacobian = f_inv.jacobian(Xt).det()

    #Substitute pdf with inverse transform


    pdf = pdf.subs(sympy.Symbol('x_' + to_transform if not to_transform is None else 'x', real=True), f_inv[0])
    pdf = pdf.subs(sympy.Symbol('y_' + to_transform if not to_transform is None else 'y', real=True), f_inv[1])


    #Compute entire pdf
    pdf = pdf / sympy.Abs(jacobian)

    #Introduce new random variables for yaw, xb and yb!
    #yaw_pdf = univariate_gaussian('yaw_' + transform_by)
    #xb_pdf = univariate_gaussian('x_' + transform_by)
    #yb_pdf = univariate_gaussian('y_' + transform_by)

    #return pdf * yaw_pdf * xb_pdf * yb_pdf

    return pdf

def f_m_func():
    f_s = f_s_func()
    
    f_m = transform_pdf(f_s, transform_by='sp', 
                             to_transform=None, 
                             transformed='lm')

    f_m = sympy.ln(f_m)
    #return sympy.simplify(f_m)

    return sympy.expand_log(sympy.simplify(f_m, inverse=True), force=True)
    #return sympy.simplify(f_m, inverse=True)

def eq_factory(eq, syms, idx):
   """
   given eq, we create count eqs
   """

   out = []
   for i in idx:
        _eq = copy.deepcopy(eq)
        for s in syms:
            _eq = _eq.subs(sympy.Symbol(s, real=True), sympy.Symbol(s + '_'+str(i), real=True))
        out.append(_eq)

   return out

def unify_eqs(eqs, syms):
    _eqs = []
    for i, eq in enumerate(eqs):
        for s in syms:
            eq = eq.subs(sympy.Symbol(s, real=True), sympy.Symbol(s + '_' + str(i), real=True))
    
        _eqs.append(eq)
    return _eqs
