#Author:Joao Victor da Fonseca Pinto
#Helper class to hold the information related to
#the configuration loop over monitoring file.

from RingerCore import calcSP, Logger

#Info class to the iterator
class MonitoringIterator(Logger):

  def __init__(self, obj, **kw):
    from pprint import pprint
    Logger.__init__(self, kw)

  # Private Helper function to extract four inits from the dictinary
  def __getInits(self,sDict):
    initBounds = list(set([sDict['infoOpBest']['init'], sDict['infoOpWorst']['init'], \
                     sDict['infoTstBest']['init'], sDict['infoTstWorst']['init']]))
    initBounds.sort()
    return initBounds

  #Retrieve iterator
  def iterator(self):
    #Extract bound index for config and sort names
    for neuron in self.neuronBounds():
      for sort in self.sortBounds(neuron):
        yield neuron, sort, self.initBounds(neuron,sort)
 
  def iterisize(self):
    size=0
    for neuron,sort,init in self.iterator():
      size+=1
    return size

  #Return config names into a list
  def neuronBounds(self):
    neuronBounds = [int(neuron.replace('config_','')) for neuron in self._summary.keys() if 'config_' in neuron]
    neuronBounds.sort()
    return neuronBounds

  #Return sort names into a list
  def sortBounds(self, neuron):
    sortBounds = [int(sort.replace('sort_','')) for sort in self._summary['config_'+str(neuron).zfill(3)].keys() \
                  if 'sort_' in sort]
    sortBounds.sort()
    return sortBounds

  def initBounds(self, neuron, sort):
    return self.__getInits(self._summary['config_'+str(neuron).zfill(3)]['sort_'+str(sort).zfill(3)])


#Monitoring class
class TuningMonitoringInfo(MonitoringIterator):
  def __init__(self, benchmarkName, rawObj, **kw):
    #Iterator object
    MonitoringIterator.__init__(self,rawObj)
    self._summary = rawObj
    self._benchmarkName = benchmarkName
            
  def name(self):
    return self._benchmarkName

  def reference(self):
    return self._summary['rawBenchmark']['reference']

  def rawBenchmark(self):
    return self._summary['rawBenchmark']

  def eps(self):
    try:
      return self._summary['eps']
    except KeyError:
      return self._summary['rawBenchmark']['eps']

  def etbinidx(self):
    try:
      return self.rawBenchmark()['signalEfficiency']['etBin']
    except:
      return self.rawBenchmark()['signal_efficiency']['etBin']

  def etabinidx(self):
    try: 
      return self.rawBenchmark()['signalEfficiency']['etaBin']
    except:
      return self.rawBenchmark()['signal_efficiency']['etaBin']

  def etbin(self):
    try:
      return self._summary['etBin']
    except KeyError:
      return []

  def etabin(self):
    try:
      return self._summary['etaBin']
    except KeyError:
      return []

  def summary(self):
    return self._summary


#Helper class to hold the operation information
class MonitoringOperationInfo( object ):
  def __init__(self, rawObj):
    self._rawOp = rawObj

  def rawOp(self):
    return self._rawOp

  def getDiscr(self):
    return {'threshold': self._rawOp['cut'],
            'nodes'    : self._rawOp['discriminator']['nodes'],
            'bias'     : self._rawOp['discriminator']['bias'],
            'weights'  : self._rawOp['discriminator']['weights']}

  def print_operation(self):
    return ('Operation: (det = %.2f, sp = %.2f, fa = %.2f)') %\
            (self._rawOp['det']*100,self._rawOp['sp']*100,self._rawOp['fa']*100)


#Helper class to hold all performance values
class MonitoringPerfInfo( MonitoringOperationInfo ):
  
  _keys = ['detMean','faMean','spMean',
           'detStd', 'faStd', 'spStd']
  
  #tobj: tuning object
  #bobj: benchmark object
  #infoOp: infoOpBest or infoOpWorst
  def __init__(self, name, ref, tobj, opObj, bobj,sortDets,sortPfs):
    
    MonitoringOperationInfo.__init__(self, opObj)
    #name
    self._sortDets = sortDets
    self._sortPfs = sortPfs
    self._name   = name
    self._perf = dict()
    self._ref = dict()
    #Retrieve information from benchmark
    self._ref['reference'] = bobj['reference']
    try:
      self._ref['det'] = bobj['signalEfficiency']['efficiency']
      self._ref['fa'] = bobj['backgroundEfficiency']['efficiency']
    except:
      self._ref['det'] = bobj['signal_efficiency']['efficiency']
      self._ref['fa'] = bobj['background_efficiency']['efficiency']
    #Calculate SP from reference
    self._ref['sp'] = calcSP(self._ref['det'], 100-self._ref['fa'])
    #Hold values
    for key in self._keys:  self._perf[key] = tobj[key]*100
  def getSortPerfs(self):
    return [self._sortDets,self._sortPfs]
  def name(self):
    return self._name

  def getRef(self):
    return self._ref

  def getPerf(self):
    return self._perf

  #Display information
  def __str__(self):
    #Get operation string information
    opString = self.print_operation()

    refString = ('Reference: %s (det = %.2f, sp = %.2f fa = %.2f') % (self._ref['reference'],\
                  self._ref['det'],self._ref['sp'],self._ref['fa'])

    tuningString = ('Tuning  (det = %.2f+/-%.2f, sp = %.2f+/-%.2f, fa = %.2f+/-%.2f') % \
                   (self._perf['detMean'], self._perf['detStd'], self._perf['spMean'], \
                    self._perf['spStd'], self._perf['faMean'], self._perf['faStd'])

    return opString+'\n'+refString+'\n'+tuningString+'\n'





