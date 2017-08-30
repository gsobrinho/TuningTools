__all__ = ['CrossValidStatAnalysis','GridJobFilter','PerfHolder',
           'fixReferenceBenchmarkCollection']

from RingerCore import ( checkForUnusedVars, calcSP, save, load, Logger
                       , LoggingLevel, expandFolders, traverse
                       , retrieve_kw, NotSet, csvStr2List, select, progressbar, getFilters
                       , apply_sort, LoggerStreamable, appendToFileName, ensureExtension
                       , measureLoopTime, checkExtension )

from TuningTools.TuningJob import ( TunedDiscrArchieve, ReferenceBenchmark, ReferenceBenchmarkCollection
                                  , ChooseOPMethod 
                                  )
from TuningTools import PreProc
from TuningTools.dataframe.EnumCollection import Dataset
from pprint import pprint
from cPickle import UnpicklingError
from time import time
import numpy as np
import os
import sys

def _localRetrieveList( l, idx ):
  if len(l) == 1:
    return l[0]
  else:
    return l[idx]

def percentile( data, score ):
  """
  val = percentile( data, score )
  Retrieve the data percentile at score
  """
  size = len(data)
  if size:
    pos = score*size
    if pos % 10 or pos == size:
      return data[pos]
    else:
      return data[pos] + data[pos+1]
  else: return None

def fixReferenceBenchmarkCollection( refCol, nBins, nTuned, level = None ):
  """
    Make sure that input data is a ReferenceBenchmarkCollection( ReferenceBenchmarkCollection([...]) ) 
    with dimensions [nBins][nTuned] or transform it to that format if it is possible.
  """
  from copy import deepcopy
  tree_types = (ReferenceBenchmarkCollection, list, tuple )
  try: 
    # Retrieve collection maximum depth
    _, _, _, _, depth = traverse(refCol, tree_types = tree_types).next()
  except GeneratorExit:
    depth = 0
  if depth == 0:
    refCol = ReferenceBenchmarkCollection( [deepcopy(refCol) for _ in range(nTuned)] )
    refCol = ReferenceBenchmarkCollection( [deepcopy(refCol) for _ in range(nBins if nBins is not None else 1) ] )
  elif depth == 1:
    lRefCol = len(refCol)
    if lRefCol == 1:
      refCol = ReferenceBenchmarkCollection( [ deepcopy( refCol[0] ) for _ in range(nTuned ) ] )
      refCol = ReferenceBenchmarkCollection( [ deepcopy( refCol    ) for _ in range(nBins if nBins is not None else 1 ) ] )
    elif lRefCol == nTuned:
      refCol = ReferenceBenchmarkCollection( refCol )
      refCol = ReferenceBenchmarkCollection( [ deepcopy( refCol ) for _ in range(nBins if nBins is not None else 1 ) ] )
    elif lRefCol == nBins:
      refColBase = ReferenceBenchmarkCollection( [ deepcopy( ref ) for ref in refCol for _ in range(nTuned) ] )
      refCol = ReferenceBenchmarkCollection([])
      for i in range(nBins): refCol.append( ReferenceBenchmarkCollection( refColBase[i*nTuned:(i+1)*nTuned] ) )
    else:
      self._fatal(("The ReferenceBenchmark collection size does not " \
          "match either the number of tuned operating points or the number of bins."), ValueError)
  elif depth == 2:
    pass
  else:
    self._fatal("Collection dimension is greater than 2!", ValueError)
  from RingerCore import inspect_list_attrs
  refCol = inspect_list_attrs(refCol, 2,                               tree_types = tree_types,                                level = level,    )
  refCol = inspect_list_attrs(refCol, 1, ReferenceBenchmarkCollection, tree_types = tree_types, dim = nTuned, name = "nTuned",                   )
  refCol = inspect_list_attrs(refCol, 0, ReferenceBenchmarkCollection, tree_types = tree_types, dim = nBins,  name = "nBins",  deepcopy = True   )
  return refCol

class JobFilter( object ):
  def __call__(self, paths):
    return []

class GridJobFilter( JobFilter ):
  """
  Filter grid job files returning each unique job id.
  """

  import re
  pat = re.compile(r'.*user.[a-zA-Z0-9]+.(?P<jobID>[0-9]+)\..*$')
  #pat = re.compile(r'user.(?P<user>[A-z0-9]*).(?P<jobID>[0-9]+).*\.tgz')

  def __call__(self, paths):
    """
      Returns the unique jobIDs
    """
    jobIDs = sorted(list(set([self.pat.match(f).group('jobID') for f in paths if self.pat.match(f) is not None])))
    return jobIDs

class CrossValidStatAnalysis( Logger ):

  ignoreKeys = ( 'benchmark', 'tuningBenchmark', 'eps', 'aucEps'
               , 'modelChooseMethod', 'rocPointChooseMethod', 'etBinIdx', 'etaBinIdx'
               , 'etBin', 'etaBin'
               )

  def __init__(self, paths, **kw):
    """
    Usage: 
    # Create object
    cvStatAna = CrossValidStatAnalysis( 
                                        paths 
                                        [,binFilters=None]
                                        [,logger[,level=INFO]]
                                      )
    # Fill the information and save output file with cross-validation summary
    cvStatAna( refBenchMark, **args...)
    # Call other methods if necessary.
    """
    Logger.__init__(self, kw)    
    self._binFilters            = retrieve_kw(kw, 'binFilters',            None         )
    self._binFilterJobIdxs      = retrieve_kw(kw, 'binFilterIdxs',         None         )
    self._useTstEfficiencyAsRef = retrieve_kw(kw, 'useTstEfficiencyAsRef', False        )
    checkForUnusedVars(kw, self._warning)
    # Check if path is a file with the paths
    self._paths = csvStr2List( paths )
    # Recursively expand all folders in the given paths so that we have all
    # files lists:
    self._paths = expandFolders( self._paths )
    self._nBins = 1
    if self._binFilters:
      self._binFilters = getFilters( self._binFilters, self._paths, 
                                     idxs = self._binFilterJobIdxs, 
                                     printf = self._info )
      if self._binFilters:
        self._paths = select( self._paths, self._binFilters ) 
        self._nBins = len(self._binFilters)
    if self._nBins is 1:
      self._paths = [self._paths]
    #if self._level <= LoggingLevel.VERBOSE:
    #  for binFilt in self._binFilters if self._binFilters is not None else [None]:
    #    self._verbose("The stored files are (binFilter=%s):", binFilt)
    #    for path in self._paths:
    #      self._verbose("%s", path)
    self._nFiles = [len(l) for l in self._paths]
    self._info("A total of %r files were found.", self._nFiles )
    #alloc variables to the TFile and bool flag
    self._sg = None
    self._sgdirs=list()

  def __addPerformance( self, tunedDiscrInfo, path, ref, benchmarkRef,
                              neuron, sort, init, 
                              etBinIdx, etaBinIdx, 
                              tunedDiscr, trainEvolution,
                              tarMember, eps, rocPointChooseMethod,
                              modelChooseMethod, aucEps ):
    refName = ref.name
    self._verbose("Adding performance for <ref:%s, config:%r,sort:%s,init:%s>", refName, neuron, sort, init)
    # We need to make sure that the key will be available on the dict if it
    # wasn't yet there
    if not refName in tunedDiscrInfo:
      tunedDiscrInfo[refName] = { 'benchmark':            ref,
                                  'tuningBenchmark':      benchmarkRef,
                                  'eps':                  eps if eps is not NotSet else ReferenceBenchmark._def_eps,
                                  'aucEps':               aucEps if aucEps is not NotSet else ReferenceBenchmark._def_auc_eps,
                                  'modelChooseMethod':    modelChooseMethod if modelChooseMethod is not NotSet else ReferenceBenchmark._def_model_choose_method,
                                  'rocPointChooseMethod': rocPointChooseMethod if rocPointChooseMethod is not NotSet else ReferenceBenchmark._def_model_choose_method}
      #ref.level = self.level
      #tunedDiscr['benchmark'].level = self.level
    if not neuron in tunedDiscrInfo[refName]:
      tunedDiscrInfo[refName][neuron] = dict()
    if not sort in tunedDiscrInfo[refName][neuron]:
      tunedDiscrInfo[refName][neuron][sort] = { 'headerInfo' : [], 
                                                'initPerfTstInfo' : [], 
                                                'initPerfOpInfo' : [] }
    # The performance holder, which also contains the discriminator
    perfHolder = PerfHolder( tunedDiscr, trainEvolution, level = self.level )
    # Retrieve operating points:
    (spTst, detTst, faTst, aucTst, cutTst, idxTst) = perfHolder.getOperatingBenchmarks( ref
                                                                                      , ds                   = Dataset.Test
                                                                                      , eps                  = eps
                                                                                      , modelChooseMethod    = modelChooseMethod
                                                                                      , rocPointChooseMethod = rocPointChooseMethod
                                                                                      , aucEps               = aucEps
                                                                                      )
    (spOp, detOp, faOp, aucOp, cutOp, idxOp)       = perfHolder.getOperatingBenchmarks( ref
                                                                                      , ds                   = Dataset.Operation
                                                                                      , eps                  = eps
                                                                                      , modelChooseMethod    = modelChooseMethod
                                                                                      , rocPointChooseMethod = rocPointChooseMethod
                                                                                      , aucEps               = aucEps
                                                                                      )
    headerInfo = { 
                   'discriminator': tunedDiscr['discriminator'],
                   'neuron':        neuron,
                   'sort':          sort,
                   'init':          init,
                   'path':          path,
                   'tarMember':     tarMember
                 }
    # Create performance holders:
    iInfoTst = { 'sp' : spTst, 'det' : detTst, 'fa' : faTst, 'auc' : aucTst, 'cut' : cutTst, 'idx' : idxTst, }
    iInfoOp  = { 'sp' : spOp,  'det' : detOp,  'fa' : faOp,  'auc' : aucOp,  'cut' : cutOp,  'idx' : idxOp,  }
    #if self._level <= LoggingLevel.VERBOSE:
    #  self._verbose("Retrieved file '%s' configuration for benchmark '%s' as follows:", 
    #                     os.path.basename(path),
    #                     ref )
    #  pprint({'headerInfo' : headerInfo, 'initPerfTstInfo' : iInfoTst, 'initPerfOpInfo' : iInfoOp })
    # Append information to our dictionary:
    tunedDiscrInfo[refName][neuron][sort]['headerInfo'].append( headerInfo )
    tunedDiscrInfo[refName][neuron][sort]['initPerfTstInfo'].append( iInfoTst )
    tunedDiscrInfo[refName][neuron][sort]['initPerfOpInfo'].append( iInfoOp )

  def __addMonPerformance( self, discr, trainEvolution, refname, neuron, sort, init):
    # Create perf holder
    perfHolder = PerfHolder(discr, trainEvolution, level = self.level)
    # Adding graphs into monitoring file
    dirname = ('%s/config_%s/sort_%s/init_%d') % (refname,str(neuron).zfill(3),str(sort).zfill(3),init)
    if not dirname in self._sgdirs:
      self._sg.mkdir(dirname)
      self._sgdirs.append(dirname)
    if not self._sg.cd(dirname):
      self._fatal("Could not cd to dir %s", dirname )

    graphNames = [ 'mse_trn', 'mse_val', 'mse_tst',
                   'bestsp_point_sp_val', 'bestsp_point_det_val', 'bestsp_point_fa_val',
                   'bestsp_point_sp_tst', 'bestsp_point_det_tst', 'bestsp_point_fa_tst',
                   'det_point_sp_val'   , 'det_point_det_val'   , 'det_point_fa_val'   , # det_point_det_val is det_fitted
                   'det_point_sp_tst'   , 'det_point_det_tst'   , 'det_point_fa_tst'   , 
                   'fa_point_sp_val'    , 'fa_point_det_val'    , 'fa_point_fa_val'    , # fa_point_fa_val is fa_fitted
                   'fa_point_sp_tst'    , 'fa_point_det_tst'    , 'fa_point_fa_tst'    ,  
                   'roc_tst'            , 'roc_operation',]

    # Attach graphs
    for gname in graphNames:
      g = perfHolder.getGraph(gname); g.SetName(gname)
      g.Write()
      #self._sg.attach(g, holdObj = False)
      del g
 
    # Attach stops
    from RingerCore.util import createRootParameter
    createRootParameter("double", "mse_stop", perfHolder.epoch_mse_stop ).Write()
    createRootParameter("double", "sp_stop",  perfHolder.epoch_sp_stop  ).Write()
    createRootParameter("double", "det_stop", perfHolder.epoch_det_stop ).Write()
    createRootParameter("double", "fa_stop",  perfHolder.epoch_fa_stop  ).Write()

  def __call__(self, **kw ):
    """
    Hook for loop method.
    """
    self.loop( **kw )

  def loop(self, **kw ):
    """
    Optional args:
      * refBenchmarkCol: a list of reference benchmark objects which will be used
        as the operation points.
      * toMatlab [True]: also create a matlab file from the obtained tuned discriminators
      * outputName ['crossValStat']: the output file name.
      * test [False]: Run only for a small number of files
      * doMonitoring [True]: Whether to do tuning monitoring file or not.
      * doCompress [True]: Whether to compress output files or not.
      * epsCol [NotSet]: epsolon value (in-bound limit) for accepting value within reference (used only for Pd/Pf)
      * aucEpsCol [NotSet]: as above, but used for calculating the ROC when ChooseOPMethod is InBoundAUC
      * rocPointChooseMethod: The method for choosing the operation point in the ROC curve
      * modelChooseMethod: The method for choosing the various models when
      operating at rocPointChooseMethod
    """
    import gc
    refBenchmarkColKW = 'refBenchmarkCol'
    if not 'refBenchmarkCol' in kw and 'refBenchmarkList' in kw:
      refBenchmarkColKW = 'refBenchmarkList'
    refBenchmarkCol         = retrieve_kw( kw, refBenchmarkColKW,    None           )
    toMatlab                = retrieve_kw( kw, 'toMatlab',           True           )
    outputName              = retrieve_kw( kw, 'outputName',         'crossValStat' )
    test                    = retrieve_kw( kw, 'test',               False          )
    doMonitoring            = retrieve_kw( kw, 'doMonitoring',       True           )
    compress                = retrieve_kw( kw, 'doCompress',         True           )
    epsCol                  = retrieve_kw( kw, 'epsCol'                             )
    aucEpsCol               = retrieve_kw( kw, 'aucEpsCol'                          )
    rocPointChooseMethodCol = retrieve_kw( kw, 'rocPointChooseMethodCol'            )
    modelChooseMethodCol    = retrieve_kw( kw, 'modelChooseMethodCol'               )
    checkForUnusedVars( kw,            self._warning )
    tuningBenchmarks = ReferenceBenchmarkCollection([])
    if not isinstance( epsCol, (list, tuple) ):                  epsCol                  = [epsCol]
    if not isinstance( aucEpsCol, (list, tuple) ):               aucEpsCol               = [aucEpsCol]
    if not isinstance( rocPointChooseMethodCol, (list, tuple) ): rocPointChooseMethodCol = [rocPointChooseMethodCol]
    if not isinstance( modelChooseMethodCol, (list, tuple) ):    modelChooseMethodCol    = [modelChooseMethodCol]

    if not self._paths:
      self._warning("Attempted to run without any file!")
      return

    pbinIdxList=[]
    isMergedList=[]
    etBinDict=dict()
    etaBinDict=dict()
    for binIdx, binPath in enumerate(progressbar(self._paths, 
                                                 len(self._paths), 'Retrieving tuned operation points: ', 30, True,
                                                 logger = self._logger)):
      
      
      tdArchieve = TunedDiscrArchieve.load(binPath[0], 
                                           useGenerator = True, 
                                           ignore_zeros = False, 
                                           skipBenchmark = False).next()

      binFilesMergedDict = {}
      isMergedList.append( binFilesMergedDict )
      for path in binPath:
        if checkExtension( path, 'tgz|tar.gz|gz'):
          isMerged = False
          from subprocess import Popen, PIPE
          from RingerCore import is_tool
          tar_cmd = 'gtar' if is_tool('gtar') else 'tar'
          tarlist_ps = Popen((tar_cmd, '-tzif', path,), 
                             stdout = PIPE, bufsize = 1)
          start = time()
          for idx, line in enumerate( iter(tarlist_ps.stdout.readline, b'') ):
            if idx > 0:
              isMerged = True
              tarlist_ps.kill()
          if isMerged:
            self._debug("File %s is a merged tar-file.", path)
          else:
            self._debug("File %s is a non-merged tar-file.", path)
          binFilesMergedDict[path] = isMerged
          # NOTE: put this debug inside the loop because the start is reset for each loop. Check this change.
          self._debug("Detecting merged file took %.2fs", time() - start)
        elif checkExtension( path, 'pic' ):
          isMerged = False 
          self._debug("File %s is a non-merged pic-file.", path)
          binFilesMergedDict[path] = isMerged

      tunedArchieveDict = tdArchieve.getTunedInfo( tdArchieve.neuronBounds[0],
                                                   tdArchieve.sortBounds[0],
                                                   tdArchieve.initBounds[0] )
      tunedDiscrList = tunedArchieveDict['tunedDiscr']
      try:
        nTuned = len(refBenchmarkCol[0])
        if nTuned  - len(tunedDiscrList) and ( nTuned != 1 or len(tunedDiscrList) != 1 ):
          self._fatal("For now, all bins must have the same number of tuned (%d) benchmarks (%d).",\
              len(tunedDiscrList),nTuned)
      except (NameError, AttributeError, TypeError):
        pass
      nTuned            = len(tunedDiscrList)
      binTuningBench    = ReferenceBenchmarkCollection( 
                             [tunedDiscrDict['benchmark'] for tunedDiscrDict in tunedDiscrList]
                          )

      # Change output level from the tuning benchmarks
      for bench in binTuningBench: bench.level = self.level
      tuningBenchmarks.append( binTuningBench )
      etBinIdx          = tdArchieve.etBinIdx
      etaBinIdx         = tdArchieve.etaBinIdx
      etBinDict['%d'%etBinIdx] =  tdArchieve.etBin
      etaBinDict['%d'%etaBinIdx] = tdArchieve.etaBin
      # pegar o etBin / etaBin

      self._debug("Found a total of %d tuned operation points on bin (et:%d,eta:%d). They are: ", 
          nTuned, etBinIdx, etaBinIdx)

      for bench in binTuningBench:
        self._debug("%s", bench)
        # end of (tuning benchmarks retrieval)
    #Make sure everything is ok with the reference benchmark collection (do not check for nBins):
    if refBenchmarkCol is not None:
      refBenchmarkCol = fixReferenceBenchmarkCollection(refBenchmarkCol, nBins = None,
                                                        nTuned = nTuned, level = self.level )

    # FIXME Moved due to crash when loading latter.
    from ROOT import TFile, gROOT, kTRUE
    gROOT.SetBatch(kTRUE)
   
    # Match between benchmarks from pref and files in path
    # FIXME This shouldn't be needed anymore as this is done by code inserted more ahead
    #if len(refBenchmarkCol) != 1 and refBenchmarkCol[0][0] is not None:
    #  tRefBenchmarkList=[]
    #  for etBinIdx, etaBinIdx in pbinIdxList:
    #    for idx, refBenchmark in enumerate(refBenchmarkCol):
    #      if refBenchmark[0].checkEtBinIdx(etBinIdx) and refBenchmark[0].checkEtaBinIdx(etaBinIdx):
    #        self._info('BenchmarkCollection found in perf file with operation on bin (et:%d,eta:%d). They are:', etBinIdx,etaBinIdx)
    #        for cref in refBenchmark:  self._debug('%s',cref)
    #        tRefBenchmarkList.append(refBenchmarkCol.pop(idx))
    #  refBenchmarkCol=tRefBenchmarkList

    self._info("Started analysing cross-validation statistics...")
    self._summaryInfo = [ dict() for i in range(self._nBins) ]
    self._summaryPPInfo = [ dict() for i in range(self._nBins) ]

    # Loop over the files
    from itertools import product
    # FIXME If job fails, it will not erase expanded files at temporary folder
    for binIdx, binPath in enumerate(self._paths):
      if self._binFilters:
        self._info("Running bin filter '%s'...",self._binFilters[binIdx])
      tunedDiscrInfo = dict()
      cSummaryInfo = self._summaryInfo[binIdx]
      cSummaryPPInfo = self._summaryPPInfo[binIdx]
      binFilesMergedDict = isMergedList[binIdx]

      # Retrieve binning information
      # FIXME: We shouldn't need to read file three times for retrieving basic information...
      tdArchieve = TunedDiscrArchieve.load(binPath[0], 
                                           useGenerator = True, 
                                           ignore_zeros = False).next()
      if tdArchieve.etaBinIdx != -1:
        self._info("File eta bin index (%d) limits are: %r", 
                           tdArchieve.etaBinIdx, 
                           tdArchieve.etaBin, )
      if tdArchieve.etBinIdx != -1:
        self._info("File Et bin index (%d) limits are: %r", 
                           tdArchieve.etBinIdx, 
                           tdArchieve.etBin, )

      self._info("Retrieving summary...")
      # Find the tuned benchmark that matches with this reference
      tBenchIdx = binIdx
      if tdArchieve.etaBinIdx != -1 and tdArchieve.etBinIdx != -1:
        for cBenchIdx, tBenchmarkList in enumerate(tuningBenchmarks):
          tBenchmark = tBenchmarkList[0]
          if tBenchmark.checkEtaBinIdx(tdArchieve.etaBinIdx) and \
              tBenchmark.checkEtBinIdx(tdArchieve.etBinIdx) :
            tBenchIdx = cBenchIdx
        # Retrieved tBenchIdx
      # end of if
      # Retrieve the tuning benchmark list referent to this binning
      tBenchmarkList = tuningBenchmarks[tBenchIdx]

      # Search for the reference binning information that is the same from the
      # benchmark
      # FIXME: Can I be sure that this will work if user enter None as benchmark?
      if refBenchmarkCol is not None:
        rBenchIdx = binIdx
        if tdArchieve.etaBinIdx != -1 and tdArchieve.etaBinIdx != -1:
          for cBenchIdx, rBenchmarkList in enumerate(refBenchmarkCol):
            for rBenchmark in rBenchmarkList:
              if rBenchmark is not None: break
            if rBenchmark is None: break
            if rBenchmark.checkEtaBinIdx(tdArchieve.etaBinIdx) and \
               rBenchmark.checkEtBinIdx(tdArchieve.etBinIdx):
              rBenchIdx = cBenchIdx
          # Retrieved rBenchIdx
        # end of if
        # Retrieve the benchmark list referent to this binning
        cRefBenchmarkList = refBenchmarkCol[rBenchIdx]
      else:
        cRefBenchmarkList = [None] * len(tBenchmarkList)

      # Check if user requested for using the tuning benchmark info by setting
      # any reference value to None
      if None in cRefBenchmarkList:
        # Check if we have only one reference and it is set to None. 
        # In case the user tuned for the SP or MSE, than replace the tuning benchmark to be set 
        # to SP, Pd and Pf
        if len(cRefBenchmarkList) == 1 and  len(tBenchmarkList) == 1 and \
            tBenchmarkList[0].reference in (ReferenceBenchmark.SP, ReferenceBenchmark.MSE):
          self._info("Found a unique tuned MSE or SP reference. Expanding it to SP/Pd/Pf operation points.")
          from copy import deepcopy
          copyRefList = ReferenceBenchmarkCollection( [deepcopy(ref) for ref in cRefBenchmarkList] )
          # Work the benchmarks to be a list with multiple references, using the Pd, Pf and the MaxSP:
          if refBenchmark.signalEfficiency is not None:
            opRefs = [ReferenceBenchmark.SP, ReferenceBenchmark.Pd, ReferenceBenchmark.Pf]
            for ref, copyRef in zip(opRefs, copyRefList):
              copyRef.reference = ref
              if ref is ReferenceBenchmark.SP:
                copyRef.name = copyRef.name.replace("Tuning_", "OperationPoint_") \
                                           .replace("_" + ReferenceBenchmark.tostring(cRefBenchmarkList[0].reference),
                                                    "_" + ReferenceBenchmark.tostring(ref))
          else:
            if copyRefList.reference is ReferenceBenchmark.MSE:
              copyRefList[0].name = "OperationPoint_" + copyRefList[0].split("_")[1] + "_SP"
          # Replace the reference benchmark by the copied list we created:
          cRefBenchmarkList = copyRefList
        # Replace the requested references using the tuning benchmarks:
        for idx, refBenchmark in enumerate(cRefBenchmarkList):
          if refBenchmark is None:
            ccRefBenchmarkList = tBenchmarkList[idx]
            cRefBenchmarkList[idx] = ccRefBenchmarkList
            ccRefBenchmarkList.name = ccRefBenchmarkList.name.replace('Tuning_', 'OperationPoint_')
      # finished checking

      self._info('Using references: %r.', [(ReferenceBenchmark.tostring(ref.reference),ref.refVal) for ref in cRefBenchmarkList])

      # What is the output name we should give for the written files?
      if self._binFilters:
        cOutputName = appendToFileName( outputName, self._binFilters[binIdx] )
      else:
        cOutputName = outputName
   
      # Finally, we start reading this bin files:
      nBreaks = 0
      cMember = 0
      for cFile, path in progressbar( enumerate(binPath),
                                      self._nFiles[binIdx], 'Reading files: ', 60, 1, True,
                                      logger = self._logger ):
        flagBreak = False
        start = time()
        self._info("Reading file '%s'", path )
        isMerged = binFilesMergedDict[path]
        # And open them as Tuned Discriminators:
        try:
          # Try to retrieve as a collection:
          for tdArchieve in measureLoopTime( TunedDiscrArchieve.load(path, useGenerator = True, 
                                                                     extractAll = True if isMerged else False, 
                                                                     eraseTmpTarMembers = False if isMerged else True,
                                                                    ), 
                                             prefix_end = "read all file '%s' members." % path,
                                             prefix = "Reading member",
                                             logger = self._logger ):
            cMember += 1
            if flagBreak: break
            self._info("Retrieving information from %s.", str(tdArchieve))

            # Calculate the size of the list
            barsize = len(tdArchieve.neuronBounds.list()) * len(tdArchieve.sortBounds.list()) * \
                      len(tdArchieve.initBounds.list())

            for neuron, sort, init in progressbar( product( tdArchieve.neuronBounds(), 
                                                            tdArchieve.sortBounds(), 
                                                            tdArchieve.initBounds() ),\
                                                            barsize, 'Reading configurations: ', 60, 1, False,
                                                   logger = self._logger):
              tunedDict      = tdArchieve.getTunedInfo( neuron, sort, init )
              tunedDiscr     = tunedDict['tunedDiscr']
              tunedPPChain   = tunedDict['tunedPP']
              trainEvolution = tunedDict['tuningInfo']
              if not len(tunedDiscr) == nTuned:
                self._fatal("File %s contains different number of tunings in the collection.", ValueError)
              # We loop on each reference benchmark we have.
              from itertools import izip, count
              for idx, refBenchmark, tuningRefBenchmark in izip(count(), cRefBenchmarkList, tBenchmarkList):
                if   neuron == tdArchieve.neuronBounds.lowerBound() and \
                     sort == tdArchieve.sortBounds.lowerBound() and \
                     init == tdArchieve.initBounds.lowerBound() and \
                     idx == 0:
                  # Check if everything is ok in the binning:
                  if not refBenchmark.checkEtaBinIdx(tdArchieve.etaBinIdx):
                    if refBenchmark.etaBinIdx is None:
                      self._warning("TunedDiscrArchieve does not contain eta binning information!")
                    else:
                      self._logger.error("File (%d) eta binning information does not match with benchmark (%r)!", 
                          tdArchieve.etaBinIdx,
                          refBenchmark.etaBinIdx)
                  if not refBenchmark.checkEtBinIdx(tdArchieve.etBinIdx):
                    if refBenchmark.etaBinIdx is None:
                      self._warning("TunedDiscrArchieve does not contain Et binning information!")
                    else:
                      self._logger.error("File (%d) Et binning information does not match with benchmark (%r)!", 
                          tdArchieve.etBinIdx,
                          refBenchmark.etBinIdx)
                # Retrieve some configurations:
                eps                  = _localRetrieveList( epsCol                  , idx )
                aucEps               = _localRetrieveList( aucEpsCol               , idx )
                rocPointChooseMethod = _localRetrieveList( rocPointChooseMethodCol , idx )
                modelChooseMethod    = _localRetrieveList( modelChooseMethodCol    , idx )
                # We always use the first tuned discriminator if we have more
                # than one benchmark and only one tuning
                if type(tunedDiscr) in (list, tuple,):
                  # fastnet core version
                  discr = tunedDiscr[refBenchmark.reference]
                else:
                  # exmachina core version
                  discr = tunedDiscr
                # Retrieve the pre-processing information:
                self.__addPPChain( cSummaryPPInfo,
                                   tunedPPChain, 
                                   sort )                    
                # And the tuning information:
                self.__addPerformance( tunedDiscrInfo = tunedDiscrInfo,
                                       path = tdArchieve.filePath, ref = refBenchmark, 
                                       benchmarkRef = tuningRefBenchmark,
                                       neuron = neuron, sort = sort, init = init,
                                       etBinIdx = tdArchieve.etBinIdx, etaBinIdx = tdArchieve.etaBinIdx,
                                       tunedDiscr = discr, trainEvolution = trainEvolution,
                                       tarMember = tdArchieve.tarMember,
                                       eps = eps,
                                       rocPointChooseMethod = rocPointChooseMethod,
                                       modelChooseMethod = modelChooseMethod, 
                                       aucEps = aucEps ) 
                # Add bin information to reference benchmark
              # end of references
            # end of configurations
            if test and (cMember - 1) == 3:
              break
          # end of (tdArchieve collection)
        except (UnpicklingError, ValueError, EOFError), e:
          import traceback
          # Couldn't read it as both a common file or a collection:
          self._warning("Ignoring file '%s'. Reason:\n%s", path, traceback.format_exc())
        # end of (try)
        if test and (cMember - 1) == 3:
          break
        # Go! Garbage
        gc.collect()
        elapsed = (time() - start)
        self._debug('Total time is: %.2fs', elapsed)
      # Finished all files in this bin
   
      # Print information retrieved:
      if self._level <= LoggingLevel.VERBOSE:
        for refBenchmark in cRefBenchmarkList:
          refName = refBenchmark.name
          self._verbose("Retrieved %d discriminator configurations for benchmark '%s':", 
              len(tunedDiscrInfo[refName]) - 1, 
              refBenchmark)
          for nKey, nDict in tunedDiscrInfo[refName].iteritems():
            if nKey in self.ignoreKeys: 
              continue
            self._verbose("Retrieved %d sorts for configuration '%r'", len(nDict), nKey)
            for sKey, sDict in nDict.iteritems():
              self._verbose("Retrieved %d inits for sort '%d'", len(sDict['headerInfo']), sKey)
            # got number of inits
          # got number of sorts
        # got number of configurations
      # finished all references

      self._info("Creating summary...")

      # Create summary info object
      iPathHolder = dict()
      extraInfoHolder = dict()
      for refKey, refValue in tunedDiscrInfo.iteritems(): # Loop over operations
        refBenchmark = refValue['benchmark']
        # Create a new dictionary and append bind it to summary info
        rawBenchmark = refBenchmark.toRawObj()
        refDict = { 'rawBenchmark' : rawBenchmark,
                    'rawTuningBenchmark' : refValue['tuningBenchmark'].toRawObj(),
                    'etBin' : etBinDict['%d'%rawBenchmark['signalEfficiency']['etBin']], 'etaBin' : etaBinDict['%d'%rawBenchmark['signalEfficiency']['etaBin']],
                  }
        headerKeys = refDict.keys()
        eps, modelChooseMethod = refValue['eps'], refValue['modelChooseMethod']
        # Add some extra values in rawBenchmark...
        refDict['rawBenchmark']['eps']=eps
        refDict['rawBenchmark']['modelChooseMethod']=modelChooseMethod
        cSummaryInfo[refKey] = refDict

        for nKey, nValue in refValue.iteritems(): # Loop over neurons
          if nKey in self.ignoreKeys:
            continue
          nDict = dict()
          refDict['config_' + str(nKey).zfill(3)] = nDict

          for sKey, sValue in nValue.iteritems(): # Loop over sorts
            sDict = dict()
            nDict['sort_' + str(sKey).zfill(3)] = sDict
            self._debug("%s: Retrieving test outermost init performance for keys: config_%s, sort_%s",
                refBenchmark, nKey, sKey )
            # Retrieve information from outermost initializations:
            ( sDict['summaryInfoTst'], \
              sDict['infoTstBest'], sDict['infoTstWorst']) = self.__outermostPerf( 
                                                                                   sValue['headerInfo'],
                                                                                   sValue['initPerfTstInfo'], 
                                                                                   refBenchmark,
                                                                                   eps = eps,
                                                                                   method = modelChooseMethod, 
                                                                                 )
            self._debug("%s: Retrieving operation outermost init performance for keys: config_%s, sort_%s",
                refBenchmark,  nKey, sKey )
            ( sDict['summaryInfoOp'], \
              sDict['infoOpBest'], sDict['infoOpWorst'])   = self.__outermostPerf( 
                                                                                   sValue['headerInfo'],
                                                                                   sValue['initPerfOpInfo'], 
                                                                                   refBenchmark, 
                                                                                   eps = eps,
                                                                                   method = modelChooseMethod, 
                                                                                 )
            wantedKeys = ['infoOpBest', 'infoOpWorst', 'infoTstBest', 'infoTstWorst']
            for key in wantedKeys:
              kDict = sDict[key]
              iPathKey = kDict['path']
              value = (kDict['neuron'], kDict['sort'], kDict['init'], refBenchmark.reference, refBenchmark.name,)
              extraValue = kDict['tarMember']
              if iPathKey in iPathHolder:
                if not(value in iPathHolder[iPathKey]):
                  iPathHolder[iPathKey].append( value )
                  extraInfoHolder[iPathKey].append( extraValue )
              else:
                iPathHolder[iPathKey] = [value]
                extraInfoHolder[iPathKey] = [extraValue]
          ## Loop over sorts
          # Retrieve information from outermost sorts:
          keyVec = [ key for key, sDict in nDict.iteritems() ]
          self._verbose("config_%s unsorted order information: %r", nKey, keyVec )
          sortingIdxs = np.argsort( keyVec )
          sortedKeys  = apply_sort( keyVec, sortingIdxs )
          self._debug("config_%s sorted order information: %r", nKey, sortedKeys )
          allBestTstSortInfo   = apply_sort( 
                [ sDict['infoTstBest' ] for key, sDict in nDict.iteritems() ]
              , sortingIdxs )
          allBestOpSortInfo    = apply_sort( 
                [ sDict['infoOpBest'  ] for key, sDict in nDict.iteritems() ]
              , sortingIdxs )
          self._debug("%s: Retrieving test outermost sort performance for keys: config_%s",
              refBenchmark,  nKey )
          ( nDict['summaryInfoTst'], \
            nDict['infoTstBest'], nDict['infoTstWorst']) = self.__outermostPerf( 
                                                                                 allBestTstSortInfo,
                                                                                 allBestTstSortInfo, 
                                                                                 refBenchmark, 
                                                                                 eps = eps,
                                                                                 method = modelChooseMethod, 
                                                                               )
          self._debug("%s: Retrieving operation outermost sort performance for keys: config_%s",
              refBenchmark,  nKey )
          ( nDict['summaryInfoOp'], \
            nDict['infoOpBest'], nDict['infoOpWorst'])   = self.__outermostPerf( 
                                                                                 allBestOpSortInfo,
                                                                                 allBestOpSortInfo, 
                                                                                 refBenchmark, 
                                                                                 eps = eps,
                                                                                 method = modelChooseMethod, 
                                                                               )
        ## Loop over configs
        # Retrieve information from outermost discriminator configurations:
        keyVec = [ key for key, nDict in refDict.iteritems() if key not in headerKeys ]
        self._verbose("Ref %s unsort order information: %r", refKey, keyVec )
        sortingIdxs = np.argsort( keyVec )
        sortedKeys  = apply_sort( keyVec, sortingIdxs )
        self._debug("Ref %s sort order information: %r", refKey, sortedKeys )
        allBestTstConfInfo   = apply_sort(
              [ nDict['infoTstBest' ] for key, nDict in refDict.iteritems() if key not in headerKeys ]
            , sortingIdxs )
        allBestOpConfInfo    = apply_sort( 
              [ nDict['infoOpBest'  ] for key, nDict in refDict.iteritems() if key not in headerKeys ]
            , sortingIdxs )
        self._debug("%s: Retrieving test outermost neuron performance", refBenchmark)
        ( refDict['summaryInfoTst'], \
          refDict['infoTstBest'], refDict['infoTstWorst']) = self.__outermostPerf( 
                                                                                   allBestTstConfInfo,
                                                                                   allBestTstConfInfo, 
                                                                                   refBenchmark, 
                                                                                   eps = eps,
                                                                                   method = modelChooseMethod,
                                                                                 )
        self._debug("%s: Retrieving operation outermost neuron performance", refBenchmark)
        ( refDict['summaryInfoOp'], \
          refDict['infoOpBest'], refDict['infoOpWorst'])   = self.__outermostPerf( 
                                                                                   allBestOpConfInfo,  
                                                                                   allBestOpConfInfo, 
                                                                                   refBenchmark, 
                                                                                   eps = eps,
                                                                                   method = modelChooseMethod,
                                                                                 )
      # Finished summary information
      #if self._level <= LoggingLevel.VERBOSE:
      #  self._verbose("Priting full summary dict:")
      #  pprint(cSummaryInfo)

      # Build monitoring root file
      if doMonitoring:
        self._info("Creating monitoring file...")
        # Fix root file name:
        mFName = appendToFileName( cOutputName, 'monitoring' )
        mFName = ensureExtension( mFName, '.root' )
        self._sg = TFile( mFName ,'recreate')
        self._sgdirs=list()
        # Just to start the loop over neuron and sort
        refPrimaryKey = cSummaryInfo.keys()[0]
        for iPath in progressbar(iPathHolder, len(iPathHolder), 'Reading configs: ', 60, 1, True, logger = self._logger):
          start = time()
          infoList, extraInfoList = iPathHolder[iPath], extraInfoHolder[iPath]
          self._info("Reading file '%s' which has %d configurations.", iPath, len(infoList))
          # FIXME Check if extension is tgz, and if so, merge multiple tarMembers
          tdArchieve = TunedDiscrArchieve.load(iPath)
          for (neuron, sort, init, refEnum, refName,), tarMember in zip(infoList, extraInfoList):
            tunedDict      = tdArchieve.getTunedInfo(neuron,sort,init)
            trainEvolution = tunedDict['tuningInfo']
            tunedDiscr     = tunedDict['tunedDiscr']
            if type(tunedDiscr) in (list, tuple,):
              if len(tunedDiscr) == 1:
                discr = tunedDiscr[0]
              else:
                discr = tunedDiscr[refEnum]
            else:
              # exmachina core version
              discr = tunedDiscr
            self.__addMonPerformance(discr, trainEvolution, refName, neuron, sort, init)
          elapsed = (time() - start)
          self._debug('Total time is: %.2fs', elapsed)
        self._sg.Close()
      # Do monitoring

      for iPath in iPathHolder:
        # Check whether the file is a original file (that is, it is in binFilesMergedList),
        # or if it was signed as a merged file:
        if os.path.exists(iPath) and ( iPath not in binFilesMergedDict or binFilesMergedDict[iPath] ):
          # Now we proceed and remove all temporary files created
          # First, we need to find all unique temporary folders:
          from shutil import rmtree
          tmpFolder = os.path.dirname( iPath )
          import tempfile
          if iPath.startswith( tempfile.tempdir ):
            self._debug("Removing temporary folder: %s", tmpFolder)
            rmtree( tmpFolder )
          else:
            self._warning("Cowardly refusing to delete possible non-temp folder: %s. Remove it if this is not an analysis file.", tmpFolder)
          # for tmpFolder
      # if isMerged

      #    Don't bother with the following code, just something I was working on in case extractAll is an issue
      #    neuronList, sortList, initList = iPathHolder[iPath]
      #    tarMemberList, refBenchmarkIdxList, refBenchmarkNameList = extraInfoHolder[iPath]
      #    uniqueMemberList, inverseIdxList = np.unique(tarMemberList, return_inverse=True)
      #    # What would happen to tarMember if multiple files are added?
      #    for tdArchieve, cIdx in enumerate( TunedDiscrArchieve.load(iPath, tarMemberList = uniqueMemberList ) ):
      #      repeatIdxList = matlab.find( inverseIdxList == inverseIdxList[cIdx] )
      #      for repeatIdx in repeatIdxList:
      #        neuron, sort, init, refIdx, refName = neuronList[i], sortList[i], initList[i], refBenchmarkIdxList[i], refBenchmarkNameList[i]

      # Strip keys from summary info that are only used for monitoring and
      # shouldn't be at the final file.
      for refKey, refValue in cSummaryInfo.iteritems(): # Loop over operations
        for nKey, nValue in refValue.iteritems():
          if 'config_' in nKey:
            for sKey, sValue in nValue.iteritems():
              if 'sort_' in sKey:
                for key in ['infoOpBest','infoOpWorst','infoTstBest','infoTstWorst']:
                  sValue[key].pop('path',None)
                  sValue[key].pop('tarMember',None)
              else:
                sValue.pop('path',None)
                sValue.pop('tarMember',None)
          elif nKey in ['infoOpBest','infoOpWorst','infoTstBest','infoTstWorst']:
            nValue.pop('path',None)
            nValue.pop('tarMember',None)
      # Remove keys only needed for 
      # FIXME There is probably a "smarter" way to do this
      #holdenParents = []
      #for _, key, parent, _, level in traverse(cSummaryInfo, tree_types = (dict,)):
      #  if key in ('path', 'tarMember') and not(parent in holdenParents):
      #    holdenParents.append(parent)


      if self._level <= LoggingLevel.VERBOSE:
        pprint(cSummaryInfo)
      elif self._level <= LoggingLevel.DEBUG:
        for refKey, refValue in cSummaryInfo.iteritems(): # Loop over operations
          self._debug("This is the summary information for benchmark %s", refKey )
          pprint({key : { innerkey : innerval for innerkey, innerval in val.iteritems() if not(innerkey.startswith('sort_'))} 
                                              for key, val in refValue.iteritems() if ( type(key) is str  
                                                                                   and key not in ('etBinIdx', 'etaBinIdx','etBin','etaBin')
                                                                                   )
                 } 
                , depth=3
                )

      # Append pp collections
      cSummaryInfo['infoPPChain'] = cSummaryPPInfo

      outputPath = save( cSummaryInfo, cOutputName, compress=compress )
      self._info("Saved file '%s'",outputPath)
      # Save matlab file
      if toMatlab:
        try:
          import scipy.io
          scipy.io.savemat( ensureExtension( cOutputName, '.mat'), cSummaryInfo)
        except ImportError:
          self._warning(("Cannot save matlab file, it seems that scipy is not "
              "available."))
          with open(ensureExtension( cOutputName, '.mat'), 'w') as dummy_mat:
            dummy_mat.write("## This is just a dummy file. ##")
      # Finished bin
    # finished all files
  # end of loop

  #def __retrieveFileInfo(self, tdArchieve, 
  #                             cRefBenchmarkList,
  #                             tunedDiscrInfo,
  #                             cSummaryPPInfo):
  #  """
  #  Retrieve tdArchieve information
  #  """
  # end of __retrieveFileInfo

  def __addPPChain(self, cSummaryPPInfo, tunedPPChain, sort):
    if not( 'sort_' + str(sort).zfill(3) in cSummaryPPInfo ) and tunedPPChain:
      ppData = tunedPPChain.toRawObj()
      cSummaryPPInfo['sort_' + str( sort ).zfill(3) ] = ppData
  # end of __addPPChain

  def __outermostPerf(self, headerInfoList, perfInfoList, refBenchmark, **kw):

    summaryDict = {'cut': [], 'sp': [], 'det': [], 'fa': [], 'auc' : [], 'idx': []}
    # Fetch all information together in the dictionary:
    for key in summaryDict.keys():
      summaryDict[key] = [ perfInfo[key] for perfInfo in perfInfoList ]
      if not key == 'idx':
        summaryDict[key + 'Mean'] = np.mean(summaryDict[key],axis=0)
        summaryDict[key + 'Std' ] = np.std( summaryDict[key],axis=0)

    # Put information together on data:
    benchmarks = [summaryDict['sp'], summaryDict['det'], summaryDict['fa'], summaryDict['auc']]

    # The outermost performances:
    refBenchmark.level = self.level # FIXME Something ignores previous level
                                    # changes, but couldn't discover what...
    bestIdx  = refBenchmark.getOutermostPerf(benchmarks, **kw )
    worstIdx = refBenchmark.getOutermostPerf(benchmarks, cmpType = -1., **kw )
    if self._level <= LoggingLevel.DEBUG:
      self._debug('Retrieved best index as: %d; values: (SP:%f, Pd:%f, Pf:%f)', bestIdx, 
          benchmarks[0][bestIdx],
          benchmarks[1][bestIdx],
          benchmarks[2][bestIdx])
      self._debug('Retrieved worst index as: %d; values: (SP:%f, Pd:%f, Pf:%f)', worstIdx,
          benchmarks[0][worstIdx],
          benchmarks[1][worstIdx],
          benchmarks[2][worstIdx])

    # Retrieve information from outermost performances:
    def __getInfo( headerInfoList, perfInfoList, idx ):
      info = dict()
      wantedKeys = ['discriminator', 'neuron', 'sort', 'init', 'path', 'tarMember']
      headerInfo = headerInfoList[idx]
      for key in wantedKeys:
        info[key] = headerInfo[key]
      wantedKeys = ['cut','sp', 'det', 'fa', 'auc', 'idx']
      perfInfo = perfInfoList[idx]
      for key in wantedKeys:
        info[key] = perfInfo[key]
      return info

    bestInfoDict  = __getInfo( headerInfoList, perfInfoList, bestIdx )
    worstInfoDict = __getInfo( headerInfoList, perfInfoList, worstIdx )
    if self._level <= LoggingLevel.VERBOSE:
      self._debug("The best configuration retrieved is: <config:%s, sort:%s, init:%s>",
                           bestInfoDict['neuron'], bestInfoDict['sort'], bestInfoDict['init'])
      self._debug("The worst configuration retrieved is: <config:%s, sort:%s, init:%s>",
                           worstInfoDict['neuron'], worstInfoDict['sort'], worstInfoDict['init'])
    return (summaryDict, bestInfoDict, worstInfoDict)
  # end of __outermostPerf

  def exportDiscrFiles(self, ringerOperation, **kw ):
    """
    Export discriminators operating at reference benchmark list to the
    ATLAS environment using this CrossValidStat information.
    """
    if not self._summaryInfo:
      self._fatal("Create the summary information using the loop method.")
    CrossValidStat.exportDiscrFiles( self._summaryInfo, 
                                     ringerOperation, 
                                     level = self._level,
                                     **kw )

  @classmethod
  def exportDiscrFiles(cls, summaryInfoList, ringerOperation, **kw):
    """
    Export discriminators operating at reference benchmark list to the
    ATLAS environment using summaryInfo. 
    
    If benchmark name on the reference list is not available at summaryInfo, an
    KeyError exception will be raised.
    """
    baseName      = kw.pop( 'baseName'      , 'tunedDiscr'      )
    refBenchCol   = kw.pop( 'refBenchCol'   , None              )
    configCol     = kw.pop( 'configCol'     , []                )
    triggerChains = kw.pop( 'triggerChains' , None              )
    etBins        = kw.pop( 'EtBins'        , None              )
    etaBins       = kw.pop( 'EtaBins'       , None              )
    level         = kw.pop( 'level'         , LoggingLevel.INFO )

    # Initialize local logger
    logger      = Logger.getModuleLogger("exportDiscrFiles", logDefaultLevel = level )
    checkForUnusedVars( kw, logger.warning )

    try:
      nEtBins = len(etBins) - 1
    except ValueError:
      nEtBins = 1

    try:
      nEtaBins = len(etaBins) - 1
    except ValueError:
      nEtaBins = 1

    # Treat the summaryInfoList
    if not isinstance( summaryInfoList, (list,tuple)):
      summaryInfoList = [ summaryInfoList ]
    summaryInfoList = list(traverse(summaryInfoList,simple_ret=True))
    nSummaries = len(summaryInfoList)

    if refBenchCol is None:
      refBenchCol = summaryInfoList[0].keys()

    # Treat the reference benchmark list
    if not isinstance( refBenchCol, (list,tuple)):
      refBenchCol = [ refBenchCol ] * nSummaries

    if len(refBenchCol) == 1:
      refBenchCol = refBenchCol * nSummaries

    nRefs = len(list(traverse(refBenchCol,simple_ret=True)))

    # Make sure that the lists are the same size as the reference benchmark:
    nConfigs = len(list(traverse(configCol,simple_ret=True)))
    if nConfigs == 0:
      configCol = [None for i in range(nRefs)]
    elif nConfigs == 1:
      configCol = configCol * nSummaries

    if nConfigs != nRefs:
      logger.fatal("Summary size is not equal to the configuration list.", ValueError)
    
    if nRefs == nConfigs == nSummaries:
      # If user input data without using list on the configuration, put it as a list:
      for o, idx, parent, _, _ in traverse(configCol):
        parent[idx] = [o]
      for o, idx, parent, _, _ in traverse(refBenchCol):
        parent[idx] = [o]

    configCol   = list(traverse(configCol,max_depth_dist=1,simple_ret=True))
    refBenchCol = list(traverse(refBenchCol,max_depth_dist=1,simple_ret=True))
    nConfigs = len(configCol)
    nSummary = len(refBenchCol)

    if nRefs != nConfigs != nSummary:
      logger.fatal("Number of references, configurations and summaries do not match!", ValueError)

    # Retrieve the operation:
    from TuningTools.dataframe import RingerOperation
    ringerOperation = RingerOperation.retrieve(ringerOperation)
    logger.info(('Exporting discrimination info files for the following '
                'operating point (RingerOperation:%s).'), 
                RingerOperation.tostring(ringerOperation))

    if ringerOperation is RingerOperation.L2:
      if triggerChains is None:
        triggerChains = "custom"
      if type(triggerChains) not in (list,tuple):
        triggerChains = [triggerChains]
      nExports = len(refBenchCol[0])
      if len(triggerChains) == 1 and nExports != 1:
        baseChainName = triggerChains[0]
        triggerChains = ["%s_chain%d" % (baseChainName,i) for i in range(nExports)]
      if nExports != len(triggerChains):
        self._fatal("Number of exporting chains does not match with number of given chain names.", ValueError)

      #output = open('TrigL2CaloRingerConstants.py','w')
      #output.write('def SignaturesMap():\n')
      #output.write('  signatures=dict()\n')
      outputDict = dict()
    elif ringerOperation is RingerOperation.Offline:
      # Import athena cpp information
      try:
        import cppyy
      except ImportError:
        import PyCintex as cppyy
      try:
        cppyy.loadDict('RingerSelectorTools_Reflex')
      except RuntimeError:
        self._fatal("Couldn't load RingerSelectorTools_Reflex dictionary.")
      from copy import deepcopy
      from ROOT import TFile
      ## Import reflection information
      from ROOT import std # Import C++ STL
      from ROOT.std import vector # Import C++ STL
      # Import Ringer classes:
      from ROOT import Ringer
      from ROOT import MsgStream
      from ROOT import MSG
      from ROOT.Ringer import IOHelperFcns
      from ROOT.Ringer import PreProcessing
      from ROOT.Ringer.PreProcessing      import Norm
      from ROOT.Ringer.PreProcessing.Norm import Norm1VarDep
      from ROOT.Ringer import IPreProcWrapperCollection
      from ROOT.Ringer import Discrimination
      from ROOT.Ringer import IDiscrWrapper
      #from ROOT.Ringer import IDiscrWrapperCollection
      from ROOT.Ringer.Discrimination import NNFeedForwardVarDep
      from ROOT.Ringer import IThresWrapper
      from ROOT.Ringer.Discrimination import UniqueThresholdVarDep
      # Create the vectors which will hold the procedures
      BaseVec = vector("Ringer::PreProcessing::Norm::Norm1VarDep*")
      #vec = BaseVec( ); vec += [ Norm1VarDep() for _ in range(nEtaBins) ]
      #vecvec = vector( BaseVec )(); vecvec += [deepcopy(vec) for _ in range(nEtBins) ]
      #norm1Vec.push_back(vecvec)
      vec = BaseVec( 1, Norm1VarDep() ); vecvec = vector( BaseVec )( 1, vec )
      norm1Vec = vector( vector( BaseVec ) )() # We are not using longitudinal segmentation
      norm1Vec.push_back(vecvec)
      ## Discriminator matrix to the RingerSelectorTools format:
      BaseVec = vector("Ringer::Discrimination::NNFeedForwardVarDep*")
      vec = BaseVec( ); vec += [ NNFeedForwardVarDep() for _ in range(nEtaBins) ]
      vecvec = vector( BaseVec )(); vecvec += [deepcopy(vec) for _ in range(nEtBins) ]
      ringerNNVec = vector( vector( BaseVec ) )() # We are not using longitudinal segmentation
      ringerNNVec.push_back(vecvec)
      BaseVec = vector("Ringer::Discrimination::UniqueThresholdVarDep*")
      vec = BaseVec(); vec +=  [UniqueThresholdVarDep()  for _ in range(nEtaBins) ]
      thresVec = vector( BaseVec )(); thresVec += [deepcopy(vec) for _ in range(nEtBins) ]
    else:
      logger.fatal( "Chosen operation (%s) is not yet implemented.", RingerOperation.tostring(ringerOperation) )

    import time
    for summaryInfo, refBenchmarkList, configList in \
                        zip(summaryInfoList,
                            refBenchCol,
                            configCol,
                           ):
      if type(summaryInfo) is str:
        logger.info('Loading file "%s"...', summaryInfo)
        summaryInfo = load(summaryInfo)
      elif type(summaryInfo) is dict:
        pass
      else:
        self._fatal("Cross-valid summary info is not string and not a dictionary.", ValueError)
      from itertools import izip, count
      for idx, refBenchmarkName, config in izip(count(), refBenchmarkList, configList):
        refBenchmarkNameToMatch = summaryInfo.keys()
        for ref in  refBenchmarkNameToMatch:
          if refBenchmarkName in ref:
            refBenchmarkName = ref
            break
        
        # Retrieve raw information:
        try:
          etBin  = summaryInfo[refBenchmarkName]['rawTuningBenchmark']['signalEfficiency']['etBin']
          etaBin = summaryInfo[refBenchmarkName]['rawTuningBenchmark']['signalEfficiency']['etaBin']
        except:
          etBin  = summaryInfo[refBenchmarkName]['rawTuningBenchmark']['signal_efficiency']['etBin']
          etaBin = summaryInfo[refBenchmarkName]['rawTuningBenchmark']['signal_efficiency']['etaBin']

        logger.info('Dumping (etbin=%d, etabin=%d)',etBin,etaBin)
        #FIXME: this retrieve the true value inside the grid. We loop in order but
        #we can not garanti the order inside of the files
        config = configCol[etBin*(len(etaBins)-1) + etaBin][0]
        
        info   = summaryInfo[refBenchmarkName]['infoOpBest'] if config is None else \
                 summaryInfo[refBenchmarkName]['config_' + str(config).zfill(3)]['infoOpBest']
            
        # Check if user specified parameters for exporting discriminator
        # operation information:
        sort =  info['sort']
        init =  info['init']

        ## Write the discrimination wrapper
        if ringerOperation in (RingerOperation.L2, RingerOperation.L2Calo):
          ## Discriminator configuration
          discrData={}
          discrData['datecode']  = time.strftime("%Y-%m-%d %H:%M")
          discrData['configuration']={}
          discrData['configuration']['benchmarkName'] = refBenchmarkName
          discrData['configuration']['etBin']     = ( etBins[etBin]  , etBins[etBin+1]   )
          discrData['configuration']['etaBin']    = ( etaBins[etaBin], etaBins[etaBin+1] )
          discrData['discriminator'] = info['discriminator']
          discrData['discriminator']['threshold'] = info['cut']

          triggerChain = triggerChains[idx]
          if not triggerChain in outputDict:
            cDict={}
            outputDict[triggerChain] = cDict
          else:
            cDict = outputDict[triggerChain]

          # to list because the dict stringfication
          def tolist(l):
            if type(l) is list:
              return l
            else:
              return l.tolist()
          discrData['discriminator']['nodes']    = tolist(discrData['discriminator']['nodes'])
          discrData['discriminator']['bias']     = tolist(discrData['discriminator']['bias'])
          discrData['discriminator']['weights']  = tolist(discrData['discriminator']['weights'])
          cDict['et%d_eta%d' % (etBin, etaBin) ] = discrData

        elif ringerOperation is RingerOperation.Offline:
          logger.debug( 'Exporting information for et/eta bin: %d (%f->%f) / %d (%f->%f)', etBin, etBins[etBin], etBins[etBin+1], 
                                                                                           etaBin, etaBins[etaBin], etaBins[etaBin+1] )
          ## Retrieve the pre-processing chain:
          #norm1VarDep = norm1Vec[0][etBin][etaBin]
          #norm1VarDep.setEtDep( etBins[etBin], etBins[etBin+1] )
          #norm1VarDep.setEtaDep( etaBins[etaBin], etaBins[etaBin+1] )
          ## Retrieve the discriminator collection:
          # Retrieve discriminator
          tunedDiscr = info['discriminator']
          # And get their weights
          nodes = std.vector("unsigned int")(); nodes += tunedDiscr['nodes']
          weights = std.vector("float")(); weights += tunedDiscr['weights']
          bias = vector("float")(); bias += tunedDiscr['bias']
          ringerDiscr = ringerNNVec[0][etBin][etaBin]
          ringerDiscr.changeArchiteture(nodes, weights, bias)
          ringerDiscr.setEtDep( etBins[etBin], etBins[etBin+1] )
          ringerDiscr.setEtaDep( etaBins[etaBin], etaBins[etaBin+1] )
          logger.verbose('Discriminator information: %d/%d (%f->%f) (%f->%f)', etBin, etaBin, \
              ringerDiscr.etMin(), ringerDiscr.etMax(), ringerDiscr.etaMin(), ringerDiscr.etaMax())
          # Print information discriminator information:
          msg = MsgStream('ExportedNeuralNetwork')
          msg.setLevel(LoggingLevel.toC(level))
          ringerDiscr.setMsgStream(msg)
          getattr(ringerDiscr,'print')(MSG.DEBUG)
          ## Add it to Discriminator collection
          ## Add current threshold to wrapper:
          thres = thresVec[etBin][etaBin]
          thres.setThreshold( info['cut'] )
          thres.setEtDep( etBins[etBin], etBins[etBin+1] )
          thres.setEtaDep( etaBins[etaBin], etaBins[etaBin+1] )
          if logger.isEnabledFor( LoggingLevel.DEBUG ):
            thresMsg = MsgStream("ExportedThreshold")
            thresMsg.setLevel(LoggingLevel.toC(level))
            thres.setMsgStream(thresMsg)
            getattr(thres,'print')(MSG.DEBUG)

        logger.info('neuron = %d, sort = %d, init = %d, thr = %f',
                    info['neuron'],
                    info['sort'],
                    info['init'],
                    info['cut'])
        
      # for benchmark
    # for summay in list

    if ringerOperation in (RingerOperation.L2Calo, RingerOperation.L2):
      #for key, val in outputDict.iteritems():
      #  output.write('  signatures["%s"]=%s\n' % (key, val))
      #output.write('  return signatures\n')
      return outputDict
    elif ringerOperation is RingerOperation.Offline: 
      from ROOT.Ringer import RingerProcedureWrapper
      ## Instantiate the templates:
      RingerNorm1IndepWrapper = RingerProcedureWrapper("Ringer::PreProcessing::Norm::Norm1VarDep",
                                                       "Ringer::EtaIndependent",
                                                       "Ringer::EtIndependent",
                                                       "Ringer::NoSegmentation")
      RingerNNDepWrapper = RingerProcedureWrapper("Ringer::Discrimination::NNFeedForwardVarDep",
                                                  "Ringer::EtaDependent",
                                                  "Ringer::EtDependent",
                                                  "Ringer::NoSegmentation")
      RingerThresWrapper = RingerProcedureWrapper("Ringer::Discrimination::UniqueThresholdVarDep",
                                                  "Ringer::EtaDependent",
                                                  "Ringer::EtDependent",
                                                  "Ringer::NoSegmentation")
      ## Create pre-processing wrapper:
      logger.debug('Initiazing norm1Wrapper...')
      norm1Wrapper = RingerNorm1IndepWrapper(norm1Vec)
      ## Add it to the pre-processing collection chain
      logger.debug('Creating PP-Chain...')
      ringerPPCollection = IPreProcWrapperCollection()
      ringerPPCollection.push_back(norm1Wrapper)
      ## Create the discrimination wrapper:
      logger.debug('Exporting RingerNNDepWrapper...')
      nnWrapper = RingerNNDepWrapper( ringerPPCollection, ringerNNVec )
      # Export the discrimination wrapper to a TFile and save it:
      logger.debug('Creating vector collection...')
      discrCol = vector('Ringer::IDiscrWrapper*')() 
      logger.debug('Pushing back discriminator wrappers...')
      discrCol.push_back(nnWrapper)
      fDiscrName = baseName + '_Discr_' + refBenchmarkName + ".root"
      IDiscrWrapper.writeCol(discrCol, fDiscrName)
      logger.info("Successfully created file %s.", fDiscrName)
      ## Create threshold wrapper:
      logger.debug('Initiazing Threshold Wrapper:')
      thresWrapper = RingerThresWrapper(thresVec)
      fThresName = baseName + '_Thres_' + refBenchmarkName + ".root"
      IThresWrapper.writeWrapper( thresWrapper, fThresName )
      logger.info("Successfully created file %s.", fThresName)
    # which operation to export
  # exportDiscrFiles 




  @classmethod
  def printTables(cls, confBaseNameList,
                       crossValGrid,
                       configMap):
    "Print tables for the cross-validation data."
    # TODO Improve documentation

    # We first loop over the configuration base names:
    for confIdx, confBaseName in enumerate(confBaseNameList):
      # And then on et/eta bins:
      for crossList in crossValGrid:

        print "{:-^90}".format("  Starting new Et  ")
        for crossFile in crossList:
          # Load file and then search the benchmark references with the configuration name:
          summaryInfo = load(crossFile)
          etIdx = -1
          etaIdx = -1
          for key in summaryInfo.keys():
            try:
              rawBenchmark = summaryInfo[key]['rawBenchmark']
              try:
                etIdx = rawBenchmark['signalEfficiency']['etBin']
                etaIdx = rawBenchmark['signalEfficiency']['etaBin']
              except KeyError:
                etIdx = rawBenchmark['signal_efficiency']['etBin']
                etaIdx = rawBenchmark['signal_efficiency']['etaBin']
              break
            except (KeyError, TypeError) as e:
              pass
          
          print "{:-^90}".format("  Eta (%d) | Et (%d)  " % (etaIdx, etIdx))
          
          confPdKey = confSPKey = confPfKey = None
          
          # Organize the names 
          for key in summaryInfo.keys():
            if key == 'infoPPChain': continue
            rawBenchmark = summaryInfo[key]['rawBenchmark']
            reference = rawBenchmark['reference']
            if confBaseName in key:
              if reference == 'Pd':
                confPdKey = key 
              if reference == 'Pf':
                confPfKey = key 
              if reference == 'SP':
                confSPKey = key 
          
          # Loop over each one of the cases and print ringer performance:
          print '{:^13}   {:^13}   {:^13} |   {:^13}   |  {}  '.format("Pd (%)","SP (%)","Pf (%)","cut","(ReferenceBenchmark)")
          print "{:-^90}".format("  Ringer  ")
          
          for keyIdx, key in enumerate([confPdKey, confSPKey, confPfKey]):

            confList = configMap[confIdx][etIdx][etaIdx]

            if confList[keyIdx] is None:
              config_str = 'config_'+str(summaryInfo[key]['infoOpBest']['neuron']).zfill(3)
            else:
              config_str = 'config_'+str(confList[0]).zfill(3)

            ringerPerf = summaryInfo[key] \
                                    [config_str] \
                                    ['summaryInfoTst']

            print '%6.3f+-%5.3f   %6.3f+-%5.3f   %6.3f+-%5.3f |   % 5.3f+-%5.3f   |  (%s) ' % ( 
                ringerPerf['detMean'] * 100.,   ringerPerf['detStd']  * 100.,
                ringerPerf['spMean']  * 100.,   ringerPerf['spStd']   * 100.,
                ringerPerf['faMean']  * 100.,   ringerPerf['faStd']   * 100.,
                ringerPerf['cutMean']       ,   ringerPerf['cutStd']        ,
                key+config_str.replace(', config_','Neuron: '))
            ringerPerf = summaryInfo[key] \
                                    [config_str] \
                                    ['infoOpBest']
            print '{:^13.3f}   {:^13.3f}   {:^13.3f} |   {:^ 13.3f}   |  ({}) '.format(
                ringerPerf['det'] * 100.,
                ringerPerf['sp']  * 100.,
                ringerPerf['fa']  * 100.,
                ringerPerf['cut'],
                key+config_str.replace('config_',', Neuron: '))

          print "{:-^90}".format("  Baseline  ")

          # Retrieve baseline values
          try:# treat some key changes applied 
            try:# the latest key is refVal
              reference_pd = rawBenchmark['signalEfficiency']['refVal']
            except:# treat the exception using the oldest key 
              reference_pd = rawBenchmark['signalEfficiency']['efficiency']
          except:
            reference_pd = rawBenchmark['signal_efficiency']['efficiency']
          try:
            try:
              reference_fa = rawBenchmark['backgroundEfficiency']['refVal']
            except:
              reference_fa = rawBenchmark['backgroundEfficiency']['efficiency']
          except:
            reference_fa = rawBenchmark['background_efficiency']['efficiency']


          reference_sp = calcSP(
                                reference_pd / 100.,
                                ( 1. - reference_fa / 100. )
                               )
          print '{:^13.3f}   {:^13.3f}   {:^13.3f} |{:@<43}'.format(
                                    reference_pd
                                    ,reference_sp * 100.
                                    ,reference_fa
                                    ,''
                                   )
          print "{:=^90}".format("")


class PerfHolder( LoggerStreamable ):
  """
  Hold the performance values and evolution for a tuned discriminator
  """
  def __init__(self, tunedDiscrData, tunedEvolutionData, **kw ):
    LoggerStreamable.__init__(self, kw )
    self.roc_tst              = tunedDiscrData['summaryInfo']['roc_test']
    self.roc_operation        = tunedDiscrData['summaryInfo']['roc_operation']
    trainEvo                  = tunedEvolutionData
    self.epoch                = np.array( range(len(trainEvo['mse_trn'])),  dtype ='float_')
    self.nEpoch               = len(self.epoch)
    def toNpArray( obj, key, d, dtype, default = []):
      """
      Set self value to a numpy array of the dict value
      """
      if ':' in key:
        key = key.split(':')
        sKey, dKey = key
      else:
        sKey, dKey = key, key
      setattr(obj, sKey, np.array( d.get(dKey, default), dtype = dtype ) )
    # end of toNpArray
    
    try:
      # Current schema from Fastnet core
      keyCollection = ['mse_trn' ,'mse_val' ,'mse_tst'
                      ,'bestsp_point_sp_val' ,'bestsp_point_det_val' ,'bestsp_point_fa_val' ,'bestsp_point_sp_tst' ,'bestsp_point_det_tst' ,'bestsp_point_fa_tst'
                      ,'det_point_sp_val' ,'det_point_det_val' ,'det_point_fa_val' ,'det_point_sp_tst' ,'det_point_det_tst' ,'det_point_fa_tst'
                      ,'fa_point_sp_val' ,'fa_point_det_val' ,'fa_point_fa_val' ,'fa_point_sp_tst' ,'fa_point_det_tst' ,'fa_point_fa_tst'
                      ]
      # Test if file format is the new one:
      if not 'bestsp_point_sp_val' in trainEvo: raise KeyError
      for key in keyCollection:
        toNpArray( self, key, trainEvo, 'float_' )
    except KeyError:
      # Old schemma
      from RingerCore import calcSP
      self.mse_trn                = np.array( trainEvo['mse_trn'],                                     dtype = 'float_' )
      self.mse_val                = np.array( trainEvo['mse_val'],                                     dtype = 'float_' )
      self.mse_tst                = np.array( trainEvo['mse_tst'],                                     dtype = 'float_' )

      self.bestsp_point_sp_val    = np.array( trainEvo['sp_val'],                                      dtype = 'float_' )
      self.bestsp_point_det_val   = np.array( [],                                                      dtype = 'float_' )
      self.bestsp_point_fa_val    = np.array( [],                                                      dtype = 'float_' )
      self.bestsp_point_sp_tst    = np.array( trainEvo['sp_tst'],                                      dtype = 'float_' )
      self.bestsp_point_det_tst   = np.array( trainEvo['det_tst'],                                     dtype = 'float_' )
      self.bestsp_point_fa_tst    = np.array( trainEvo['fa_tst'],                                      dtype = 'float_' )
      self.det_point_det_val      = np.array( trainEvo['det_fitted'],                                  dtype = 'float_' ) \
                                    if 'det_fitted' in trainEvo else np.array([], dtype='float_')
      self.det_point_fa_val       = np.array( trainEvo['fa_val'],                                      dtype = 'float_' )
      self.det_point_sp_val       = np.array( calcSP(self.det_point_det_val, 1-self.det_point_fa_val), dtype = 'float_' ) \
                                    if 'det_fitted' in trainEvo else np.array([], dtype='float_')
      self.det_point_sp_tst       = np.array( [],                                                      dtype = 'float_' )
      self.det_point_det_tst      = np.array( [],                                                      dtype = 'float_' )
      self.det_point_fa_tst       = np.array( [],                                                      dtype = 'float_' )
      self.fa_point_det_val       = np.array( trainEvo['det_val'],                                     dtype = 'float_' )
      self.fa_point_fa_val        = np.array( trainEvo['fa_fitted'],                                   dtype = 'float_' ) \
                                    if 'fa_fitted' in trainEvo else np.array([],  dtype='float_')
      self.fa_point_sp_val        = np.array( calcSP(self.fa_point_det_val, 1.-self.fa_point_fa_val),  dtype = 'float_' ) \
                                    if 'fa_fitted' in trainEvo else np.array([],  dtype='float_')
      self.fa_point_sp_tst        = np.array( [],                                                      dtype = 'float_' )
      self.fa_point_det_tst       = np.array( [],                                                      dtype = 'float_' )
      self.fa_point_fa_tst        = np.array( [],                                                      dtype = 'float_' )

    # Check if the roc is a raw object
    if type(self.roc_tst) is dict:
      self.roc_tst_det = np.array( self.roc_tst['pds'],              dtype = 'float_'     )
      self.roc_tst_fa  = np.array( self.roc_tst['pfs'],              dtype = 'float_'     )
      self.roc_tst_cut = np.array( self.roc_tst['thresholds'],       dtype = 'float_'     )
      self.roc_op_det  = np.array( self.roc_operation['pds'],        dtype = 'float_'     )
      self.roc_op_fa   = np.array( self.roc_operation['pfs'],        dtype = 'float_'     )
      self.roc_op_cut  = np.array( self.roc_operation['thresholds'], dtype = 'float_'     )
    else: # Old roc save strategy 
      self.roc_tst_det = np.array( self.roc_tst.pdVec,       dtype = 'float_'     )
      self.roc_tst_fa  = np.array( self.roc_tst.pfVec,        dtype = 'float_'     )
      self.roc_tst_cut = np.array( self.roc_tst.cutVec,       dtype = 'float_'     )
      self.roc_op_det  = np.array( self.roc_operation.pdVec, dtype = 'float_'     )
      self.roc_op_fa   = np.array( self.roc_operation.pfVec,  dtype = 'float_'     )
      self.roc_op_cut  = np.array( self.roc_operation.cutVec, dtype = 'float_'     )

    toNpArray( self, 'epoch_mse_stop:epoch_best_mse', trainEvo, 'int_', -1 )
    toNpArray( self, 'epoch_sp_stop:epoch_best_sp',   trainEvo, 'int_', -1 )
    toNpArray( self, 'epoch_det_stop:epoch_best_det', trainEvo, 'int_', -1 )
    toNpArray( self, 'epoch_fa_stop:epoch_best_fa',   trainEvo, 'int_', -1 )


  def getOperatingBenchmarks( self, refBenchmark, **kw ):
    """
      Returns the operating benchmark values for this tunned discriminator
    """
    ds = retrieve_kw( kw, 'ds', Dataset.Test )
    modelChooseMethod = retrieve_kw( kw, 'modelChooseMethod' )
    rocPointChooseMethod = retrieve_kw( kw, 'rocPointChooseMethod' )
    kw['method'] = rocPointChooseMethod
    if modelChooseMethod in ( ChooseOPMethod.InBoundAUC,  ChooseOPMethod.AUC ):
      kw['calcAUCMethod'] = modelChooseMethod
    if ds is Dataset.Test:
      pdVec = self.roc_tst_det
      pfVec = self.roc_tst_fa
      cutVec = self.roc_tst_cut
    elif ds is Dataset.Operation:
      pdVec = self.roc_op_det
      pfVec = self.roc_op_fa
      cutVec = self.roc_op_cut
    else:
      self._fatal("Cannot retrieve maximum ROC SP for dataset '%s'", ds, ValueError)
    spVec = calcSP( pdVec, 1 - pfVec )
    benchmarks = [spVec, pdVec, pfVec]
    if modelChooseMethod in ( ChooseOPMethod.InBoundAUC,  ChooseOPMethod.AUC ):
      idx, auc = refBenchmark.getOutermostPerf(benchmarks, **kw )
    else:
      idx, auc = refBenchmark.getOutermostPerf(benchmarks, **kw ), -1.
    sp  = spVec[idx]
    det = pdVec[idx]
    fa  = pfVec[idx]
    cut = cutVec[idx]
    self._verbose('Retrieved following performances: SP:%r| Pd:%r | Pf:%r | AUC:%r | cut: %r | idx:%r'
                 , sp, det, fa, auc, cut, idx )
    return (sp, det, fa, auc, cut, idx)

  def getGraph( self, graphType ):
    """
      Retrieve a TGraph from the discriminator tuning information.
      perfHolder.getGraph( option )
      The possible options are:
        * mse_trn
        * mse_val
        * mse_tst
        * (bestsp,det or fa)_point_sp_val
        * (bestsp,det or fa)_point_sp_tst
        * (bestsp,det or fa)_point_det_val
        * (bestsp,det or fa)_point_det_tst
        * (bestsp,det or fa)_point_fa_val
        * (bestsp,det or fa)_point_fa_tst
        * roc_val
        * roc_op
        * roc_val_cut
        * roc_op_cut
    """
    from ROOT import TGraph, gROOT, kTRUE
    gROOT.SetBatch(kTRUE)
    def epoch_graph( benchmark ):
      """
      Helper function to create graphics containing benchmarks evolution thorugh tuning epochs
      """
      return TGraph(self.nEpoch, self.epoch, benchmark) if len( benchmark ) else TGraph()
    if hasattr(self, graphType):
      if graphType.startswith('roc'):
        if graphType == 'roc_tst'               : return TGraph(len(self.roc_tst_fa), self.roc_tst_fa, self.roc_tst_det )
        elif graphType == 'roc_operation'       : return TGraph(len(self.roc_op_fa),  self.roc_op_fa,  self.roc_op_det  )
        elif graphType == 'roc_tst_cut'         : return TGraph(len(self.roc_tst_cut),
                                                                np.array(range(len(self.roc_tst_cut) ), 'float_'), 
                                                                self.roc_tst_cut )
        elif graphType == 'roc_op_cut'          : return TGraph(len(self.roc_op_cut), 
                                                             np.array(range(len(self.roc_op_cut) ),  'float_'), 
                                                             self.roc_op_cut  )
      else:
        return epoch_graph( getattr(self, graphType) )
    else: 
      self._fatal( "Unknown graphType '%s'" % graphType, ValueError )
