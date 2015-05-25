
import math
from decimal import Decimal

#NeuralNetowrk structure functions
def makeW(i, j, fill=0.0):
    n = []
    for m in range(i):
        n.append([fill]*j)
    return n

def sigmoid(x):
    return math.tanh(x)

def CALL_TRF_FUNC(input, type):
    return sigmoid(input)


def size( list ):
  try:
    col = len(list[0])
    row = len(list)
    return [row,col]
  except:
    row = 1
    col = len(list)
    return [row,col]
    

def get_multiplier(_from, _to, step):
    digits = []
    for number in [_from, _to, step]:
        pre = Decimal(str(number)) % 1
        digit = len(str(pre)) - 2
        digits.append(digit)
    max_digits = max(digits)
    return float(10 ** (max_digits))


def float_range(_from, _to, step, include=False):
    """Generates a range list of floating point values over the Range [start, stop]
       with step size step
       include=True - allows to include right value to if possible
       !! Works fine with floating point representation !!
    """
    mult = get_multiplier(_from, _to, step)
    # print mult
    int_from = int(round(_from * mult))
    int_to = int(round(_to * mult))
    int_step = int(round(step * mult))
    # print int_from,int_to,int_step
    if include:
        result = range(int_from, int_to + int_step, int_step)
        result = [r for r in result if r <= int_to]
    else:
        result = range(int_from, int_to, int_step)
    # print result
    float_result = [r / mult for r in result]
    return float_result


def find_higher_than( vec, value ):
  return [i for i,x in enumerate(vec) if x > value]
#end


def find_less_than( vec, value ):
  return [i for i,x in enumerate(vec) if x < value]
#end


def find_higher_or_equal_than( vec, value ):
  return [i for i,x in enumerate(vec) if x >= value]
#end


def mapMinMax( x, yMin, yMax ):
  y = []
  xMax = max(x)
  xMin = min(x)
  for i in range( len(x) ):
    y.append( ( (yMax-yMin)*(x[i]-xMin) )/(xMax-xMin)  + yMin)
  return y

def geomean(nums):
  return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))

def mean(nums):
  return (sum(nums)/len(nums))


def getEff( outSignal, outNoise, cut ):
  #[detEff, faEff] = getEff(outSignal, outNoise, cut)
  #Returns the detection and false alarm probabilities for a given input
  #vector of detector's perf for signal events(outSignal) and for noise 
  #events (outNoise), using a decision threshold 'cut'. The result in whithin
  #[0,1].
  detEff = len( find_higher_or_equal_than( outSignal, cut) ) / len(outSignal)
  faEff  = len( find_higher_or_equal_than( outNoise, cut)  )/  len(outNoise)
  return [detEff, faEff]

def calcSP( x, y ):
  #ret  = calcSP(x,y) - Calculates the normalized [0,1] SP value.
  #effic is a vector containing the detection efficiency [0,1] of each
  #discriminating pattern. It effic is a Matrix, the SP will be calculated 
  #for each collumn.
  sp = len(x)*[0]
  for i in range(len(sp)):
    sp[i] = math.sqrt(geomean([x[i],y[i]]) * mean([x[i],y[i]]))
  return sp

def genRoc( outSignal, outNoise, numPts = 1000 ):
  #[spVec, cutVec, detVec, faVec] = genROC(out_signal, out_noise, numPts, doNorm)
  #Plots the RoC curve for a given dataset.
  #Input Parameters are:
  #   out_signal     -> The perf generated by your detection system when
  #                     electrons were applied to it.
  #   out_noise      -> The perf generated by your detection system when
  #                     jets were applied to it
  #   numPts         -> (opt) The number of points to generate your ROC.
  #
  #If any perf parameters is specified, then the ROC is plot. Otherwise,
  #the sp values, cut values, the detection efficiency and false alarm rate 
  # are returned (in that order).
  cutVec = float_range(-1,1,2/float(numPts))
  cutVec = cutVec[0:numPts - 1]
  detVec = len(cutVec)*[0]
  faVec  = len(cutVec)*[0]


  for i in range(len(cutVec)):
    [detVec[i], faVec[i]] = getEff( outSignal, outNoise, cutVec[i] )

  faVecMinusOne = len(faVec)*[0]
  for i in range(len(faVec)):
    faVecMinusOne[i] = 1 - faVec[i]

  spVec = calcSP( detVec, faVecMinusOne )
  maxSP = max(spVec)
  cutIdxMax = spVec.index(maxSP)
  return [spVec, cutVec, detVec, faVec, maxSP, cutIdxMax]

  













