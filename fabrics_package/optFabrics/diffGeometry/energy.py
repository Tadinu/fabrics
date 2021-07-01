import casadi as ca
import numpy as np

from optFabrics.diffGeometry.spec import Spec

class Lagrangian(object):
    """description"""

    def __init__(self, l : ca.SX, x : ca.SX, xdot : ca.SX):
        assert isinstance(l, ca.SX)
        assert isinstance(x, ca.SX)
        assert isinstance(xdot, ca.SX)
        self._l = l
        self._x = x
        self._xdot = xdot
        self.applyEulerLagrange()

    def applyEulerLagrange(self):
        dL_dx = ca.gradient(self._l, self._x)
        dL_dxdot = ca.gradient(self._l, self._xdot)
        d2L_dx2 = ca.jacobian(dL_dx, self._x)
        d2L_dxdxdot = ca.jacobian(dL_dx, self._xdot)
        d2L_dxdot2 = ca.jacobian(dL_dxdot, self._xdot)

        F = d2L_dxdxdot
        f_e = -dL_dx
        M = d2L_dxdot2
        f = ca.mtimes(ca.transpose(F), self._xdot) + f_e
        self._S = Spec(M, f, self._x, self._xdot)

    def concretize(self):
        self._S.concretize()
        self._l_fun = ca.Function("funs", [self._x, self._xdot], [self._l])

    def evaluate(self, x : np.ndarray, xdot : np.ndarray):
        assert isinstance(x, np.ndarray)
        assert isinstance(xdot, np.ndarray)
        l = float(self._l_fun(x, xdot))
        M, f = self._S.evaluate(x, xdot)
        return M, f, l

class FinslerStructure(Lagrangian):
    def __init__(self, lg : ca.SX, x : ca.SX, xdot : ca.SX):
        self._lg = lg
        l = 0.5 * lg**2
        super().__init__(l, x, xdot)

    def concretize(self):
        super().concretize()
        self._lg_fun = ca.Function("fun_lg", [self._x, self._xdot], [self._lg])

    def evaluate(self, x : np.ndarray, xdot : np.ndarray):
        M, f, l = super().evaluate(x, xdot)
        lg = float(self._lg_fun(x, xdot))
        return M, f, l, lg