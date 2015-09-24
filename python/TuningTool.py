'''
  Author: Joao Victor da Fonseca Pinto
  Email: jodafons@cern.ch 
  Description:
       TuningTool: This class is used to connect the python interface and
       the c++ tuningtool core. The class TuningToolPyWrapper have some methods
       thad can be used to set some train param. Please check the list 
       of methods below:
      
       - neuron: the number of neurons into the hidden layer.
       - batchSise:
       - doPerf: do performance analisys (default = False)
       - trnData: the train data list
       - valData: the validate data list
       - testData: (optional) The test data list
       - doMultiStop: do sp, fa and pd stop criteria (default = False).

'''
import numpy as np
from RingerCore.Logger  import Logger, LoggingLevel
from libTuningTools     import TuningToolPyWrapper
from TuningTools.Neural import Neural

class TuningTool(TuningToolPyWrapper, Logger):
  """
    TuningTool is the higher level representation of the TuningToolPyWrapper class.
  """

  def __init__( self, **kw ):
    Logger.__init__( self, kw )
    TuningToolPyWrapper.__init__(self, LoggingLevel.toC(self.level))
    from RingerCore.util import checkForUnusedVars
    self.seed                = kw.pop('seed',        None    )
    self.batchSize           = kw.pop('batchSize',    100    )
    self.trainFcn            = kw.pop('trainFcn',  'trainrp' )
    self.doPerf              = kw.pop('doPerf',      False   )
    self.showEvo             = kw.pop('showEvo',       5     )
    self.epochs              = kw.pop('epochs',      1000    )
    self.doMultiStop         = kw.pop('doMultiStop', False   ) 
    checkForUnusedVars(kw, self._logger.warning )
    del kw

  def getMultiStop(self):
    return self._doMultiStop

  def setDoMultistop(self, value):
    """
      doMultiStop: Sets TuningToolPyWrapper self.useAll() if set to true,
        otherwise sets to self.useSP()
    """
    if value: 
      self._doMultiStop = True
      self.useAll()
    else: 
      self._doMultiStop = False
      self.useSP()

  doMultiStop = property( getMultiStop, setDoMultistop )

  def getSeed(self):
    return TuningToolPyWrapper.getSeed(self)

  def setSeed(self, value):
    """
      Set seed value
    """
    import ctypes
    if not value is None: 
      return TuningToolPyWrapper.setSeed(self,value)

  seed = property( getSeed, setSeed )

  def setTrainData(self, trnData):
    """
      Overloads TuningToolPyWrapper setTrainData to change numpy array to its
      ctypes representation.
    """
    if trnData:
      self._logger.debug("Setting trainData to new representation.")
    else:
      self._logger.debug("Emptying trainData.")
    TuningToolPyWrapper.setTrainData(self, trnData)

  def setValData(self, valData):
    """
      Overloads TuningToolPyWrapper setValData to change numpy array to its
      ctypes representation.
    """
    if valData:
      self._logger.debug("Setting valData to new representation.")
    else:
      self._logger.debug("Emptying valData.")
    TuningToolPyWrapper.setValData(self, valData)

  def setTestData(self, testData):
    """
      Overloads TuningToolPyWrapper setTstData to change numpy array to its
      ctypes representation.
    """
    if testData:
      self._logger.debug("Setting testData to new representation.")
    else:
      self._logger.debug("Emptying testData.")
    TuningToolPyWrapper.setTestData(self, testData)

  def newff(self, nodes, funcTrans = ['tansig', 'tansig']):
    """
      Creates new feedforward neural network
    """
    self._logger.info('Initalizing newff...')
    TuningToolPyWrapper.newff(self, nodes, funcTrans, self.trainFcn)

  def train_c(self):
    """
      Train feedforward neural network
    """
    from RingerCore.util import Roc
    netList = []
    [DiscriminatorPyWrapperList , TrainDataPyWrapperList] = \
        TuningToolPyWrapper.train_c(self)
    self._logger.debug('Successfully exited C++ training.')
    for netPyWrapper in DiscriminatorPyWrapperList:
      tstPerf = None
      opPerf  = None
      if self.doPerf:
        self._logger.debug('Calling valid_c to retrieve performance.')
        perfList = self.valid_c(netPyWrapper)
        opPerf   = Roc( perfList[1], 'operation' )
        self._logger.info('Operation: sp = %f, det = %f and fa = %f', \
            opPerf.sp, opPerf.det, opPerf.fa)
        tstPerf  = Roc( perfList[0] , 'test')
        self._logger.info('Test: sp = %f, det = %f and fa = %f', \
            tstPerf.sp, tstPerf.det, tstPerf.fa)
        self._logger.debug('Finished valid_c on python side.')
      netList.append( [Neural(netPyWrapper, train=TrainDataPyWrapperList), \
          tstPerf, opPerf] )
    self._logger.debug("Finished train_c on python side.")
    return netList

