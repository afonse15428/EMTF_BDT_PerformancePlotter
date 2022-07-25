import math
import numpy
import scipy
from scipy import optimize

def effFunc(x, a, b):
    E = (0.5) * (1 + math.erf((1 - a/x)/(2**0.5 * b)))

    return E

def effFuncVariableRes(x, a, b, c, d):
    E = (0.5) * (1 + math.erf((1 - a/x)/(2**0.5 * (b * x**c + d))))

    return E

def effFuncVariableResMinus90(x, *args):
    # print(args)
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

effFunc_v = numpy.vectorize(effFunc)
effFuncVariableRes_v = numpy.vectorize(effFuncVariableRes)
effFuncVariableResMinus90_v = numpy.vectorize(effFuncVariableResMinus90)
scaleFactorFunc_v = numpy.vectorize(scaleFactorFunc)

# print(findPt_90(27.6, .013, .604, .174))

# after the fit, call function of  pt at 90%
# pt_90/pt_cut = scalar factor
# plot scalar factor vs pt

# run the code for pt cut from 5 to 50

# get average of c, b, d using numpy.average(...), then plot the averages with the function b*pt^c+d against pt_cuts
