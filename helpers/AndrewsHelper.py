import math
import numpy

def effFunc(x, a, b):
    E = (0.5) * (1 + math.erf((1 - a/x)/(2**0.5 * b)))

    return E

def effFuncVariableRes(x, a, b, c, d):
    E = (0.5) * (1 + math.erf((1 - a/x)/(2**0.5 * (b * x**c + d)))

    return E

effFunc_v = numpy.vectorize(effFunc)
effFuncVariableRes_v = numpy.vectorize(effFuncVariableRes)
