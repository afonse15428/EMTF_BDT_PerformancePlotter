import math
import numpy
import scipy
from scipy import optimize
from scipy.integrate import quad

def effFunc(x, a, b):
    E = (0.5) * (1 + math.erf((1 - a/x)/(2**0.5 * b)))

    return E

def effFuncVariableRes(x, a, b, c, d):
    E = (0.5) * (1 + math.erf((1 - a/x)/(2**0.5 * (b * x**c + d))))

    return E

def effFuncVariableResMinus90(x, *args):
    a = args[0] 
    b = args[1]
    c = args[2]
    d = args[3]
    return effFuncVariableRes(x, a, b, c, d) - .9
    # when returns 0, x is at 90%

def findPt_90(a, b, c, d):
    sol = optimize.root_scalar(effFuncVariableResMinus90_v, args = (a, b, c, d), method = "brentq", x0 = 22, bracket = [1, 150])

    return sol.root

def scaleFactorFunc(x, sf_a, sf_b):
    A = sf_a/(1-sf_b*x)

    return A

def effZBConv(x, a, b, c, d):
    return 946308 * (x-.76018)**(-3.66)/(2446103) * effFuncVariableRes(x, a, b, c, d)

def rateFunc(pt_cut, a, b, c, d):
    freq_LHC = (2760*11.246)
    N_mu, err = quad(effZBConv, 1, 1000, args=(a, b, c, d))
    return N_mu*freq_LHC

effFunc_v = numpy.vectorize(effFunc)
effFuncVariableRes_v = numpy.vectorize(effFuncVariableRes)
effFuncVariableResMinus90_v = numpy.vectorize(effFuncVariableResMinus90)
scaleFactorFunc_v = numpy.vectorize(scaleFactorFunc)

