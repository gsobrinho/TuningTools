
"""
  Author: Joao Victor da Fonseca Pinto
  Email: jodafons@cern.ch
 
  Description:
       FastNet: This class is used to connect the python interface and
       the c++ fastnet core. The class FastnetPyWrapper have some methods
       thad can be used to set some train param. Please check the list 
       of methods below:
  
  Class methods:
        FastNet( msgLevel = INFO = 2 )
        setData( trnDatam, valData, tstData )
        initialize()
        execute()
        getNeuralObjectsList()
  Enum:
        VERBOSE
        DEBUG
        INFO
        WARNING
        FATAL
  Functions:
        genRoc( outSignal, outNoise, numPoints = 1000 )

        Where outSignal and outNoise are a numpy array

"""
import sys
import os
import numpy as np
from libFastNetTool     import FastnetPyWrapper
from FastNetTool.Neural import *

"""
  Analysis functions that I used to calculate the SP index or
  generate the ROC (Receive Operation Curve)
"""
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
  
  detEff = np.where(outSignal >= cut)[0].shape[0]/ float(outSignal.shape[0]) 
  faEff  = np.where(outNoise >= cut)[0].shape[0]/ float(outNoise.shape[0])
  return [detEff, faEff]

def calcSP( pd, pf ):
  #ret  = calcSP(x,y) - Calculates the normalized [0,1] SP value.
  #effic is a vector containing the detection efficiency [0,1] of each
  #discriminating pattern.  
  return math.sqrt(geomean([pd,pf]) * mean([pd,pf]))

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
  cutVec = np.arange( -1,1,2 /float(numPts))
  cutVec = cutVec[0:numPts - 1]
  detVec = np.array(cutVec.shape[0]*[float(0)])
  faVec  = np.array(cutVec.shape[0]*[float(0)])
  spVec  = np.array(cutVec.shape[0]*[float(0)])
  for i in range(cutVec.shape[0]):
    [detVec[i],faVec[i]] = getEff( np.array(outSignal), np.array(outNoise),  cutVec[i] ) 
    spVec[i] = calcSP(detVec[i],1-faVec[i])
  return [spVec.tolist(), cutVec.tolist(), detVec.tolist(), faVec.tolist()]

"""
  Enumaration
"""
class Msg:
  """
    Output enum for MsgStream configuration into c++ fastnet core
  """
  VERBOSE = 0
  DEBUG   = 1
  INFO    = 2
  WARNING = 3
  FATAL   = 4


"""
  FastNet interface will be comunicate the python and c++ core using the 
  FastnetPyWrapper. 
"""
#Default contructor
class FastNet(FastnetPyWrapper):

  def __init__( self, msglevel = Msg.INFO ):
    FastnetPyWrapper.__init__(self, msglevel)
    
    self.nNodes              = []
    self.top                 = 2
    self.inputNumber         = 0
    self.batchSize           = 100
    self.trainFcn            = 'trainrp'
    self.doPerformance       = True
    self.trnData             = []
    self.valData             = []
    self.tstData             = []
    self.simData             = []
    self.networksList        = []
    self.trainEvolutionData  = []


  #Set all datasets
  def setData(self, trnData, valData, tstData, simData):
    
    #Get the number of inputs
    self.inputNumber = len(trnData[0][0])
    self.trnData = trnData
    self.valData = valData
    self.simData = simData
    self.setTrainData( trnData )
    self.setValData(   valData )
    if len(tstData) > 0:
      self.tstData = tstData
      self.setTestData(  tstData )
    else:
      self.tstData = valData


  def initialize(self):

    self.nNodes = [self.inputNumber, self.top, 1]
    self.newff(self.nNodes, ['tansig','tansig'], self.trainFcn)


  #Run the training
  def execute(self):

    [DiscriminatorPyWrapperList , TrainDataPyWrapperList] = self.train()
    self.trainEvolutionData = TrainDataPyWrapperList

    for netPyWrapper in DiscriminatorPyWrapperList:

      net  = Neural( netPyWrapper, TrainDataPyWrapperList ) 

      if self.doPerformance:
        out_sim  = [self.sim(netPyWrapper, self.simData[0]), self.sim( netPyWrapper, self.simData[1])]
        out_tst  = [self.sim(netPyWrapper, self.tstData[0]), self.sim( netPyWrapper, self.tstData[1])]

        [spVec, cutVec, detVec, faVec] = genRoc( out_sim[0], out_sim[1], 1000 )
        net.setSimPerformance( spVec,detVec,faVec,cutVec )
      
        [spVec, cutVec, detVec, faVec] = genRoc( out_tst[0], out_tst[1], 1000 )
        net.setTstPerformance(spVec,detVec,faVec,cutVec)
      
      self.networksList.append( net )


  def getNeuralObjectsList(self):
    return self.networksList



