#uthor: Joao Victo da Fonseca Pinto
#Analysis framework

__all__ = ['TuningMonitoringTool']

#Import necessary classes
from TuningMonitoringInfo import TuningMonitoringInfo
from TuningStyle          import SetTuningStyle
from RingerCore           import calcSP, save, load, Logger, mkdir_p, progressbar,keyboard
from pprint               import pprint
import os
#Main class to plot and analyser the crossvalidStat object
#created by CrossValidStat class from tuningTool package
class TuningMonitoringTool( Logger ):
  """
  Main class to plot and analyser the crossvalidStat object
  created by CrossValidStat class from tuningTool package
  """  
  #Init class
  def __init__(self, crossvalFileName, monFileName, **kw):
    
    from ROOT import TFile, gROOT
    gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;");
    #Set all global setting from ROOT style plot!
    SetTuningStyle()
    Logger.__init__(self, kw)
    #Hold all information abount the monitoring root file
    self._infoObjs = list()
    try:#Protection
      self._logger.info('Reading monRootFile (%s)',monFileName)
      self._rootObj = TFile(monFileName, 'read')
    except RuntimeError:
      self._logger.fatal('Could not open root monitoring file.')
    from RingerCore import load
    try:#Protection
      self._logger.info('Reading crossvalFile (%s)',crossvalFileName)
      crossvalObj = load(crossvalFileName)
    except RuntimeError:
      self._logger.fatal('Could not open pickle summary file.')
    #Loop over benchmarks


    for benchmarkName in crossvalObj.keys():
      #Must skip if ppchain collector
      if benchmarkName == 'infoPPChain':  continue
      #Add summary information into MonTuningInfo helper class
      self._infoObjs.append( TuningMonitoringInfo( benchmarkName, crossvalObj[benchmarkName] ) ) 
      self._logger.info('Creating MonTuningInfo for %s and the iterator object [et=%d, eta=%d]',
                         benchmarkName,self._infoObjs[-1].etbinidx(), self._infoObjs[-1].etabinidx())
    #Loop over all benchmarks

    #Always be the same bin for all infoObjs  
    etabinidx = self._infoObjs[0].etabinidx()
    etbinidx = self._infoObjs[0].etbinidx()
    #Reading the data rings from path or object
    dataPath = kw.pop('dataPath', None)
    if dataPath:
      from TuningTools import TuningDataArchieve
      self._logger.info(('Reading data tuning file with name %s')%(dataPath))
      TDArchieve = TuningDataArchieve.load(dataPath, etBinIdx = etbinidx, etaBinIdx = etabinidx, 
                                           loadEfficiencies = False)
      self._data = (TDArchieve.signalPatterns, TDArchieve.backgroundPatterns)
    else:
      self._logger.fatal('You must pass the ref file as parameter. abort!')

  
  def etbinidx(self):
    return self._infoObjs[0].etbinidx()

  def _ChoosePdPf(self,faList,pdList,name):
    if name == "Pd":
      fa = min(faList)
      pd = pdList[faList.index(fa)]

    if name == "Pf":
      pd = max(pdList)
      fa = faList[pdList.index(pd)]
    if name == "SP":
      spList =  [calcSP(pdList[idx]*100, 100-faList[idx]*100) for idx in xrange(len(pdList))]
      sp = max(spList)
      pd = pdList[spList.index(sp)]
      fa = faList[spList.index(sp)]
    return pd,fa

  def etabinidx(self):
    return self._infoObjs[0].etabinidx()


  def summary(self):
    summary=dict()
    for info in self._infoObjs:
      summary[info.name()]=info.summary()
    return summary

  #Main method to execute the monitoring 
  def __call__(self, **kw):
    """
      Call the Monitoring tool analysis, the args can be:
        basePath: holt the location where all plots and files will
                  be saved. (defalt is Mon/)
        doBeamer: Start beamer Latex presentation maker (Default is True)
        shortSliedes: Presentation only with tables (Default is False)
    """
    self.loop(**kw)

  #Loop over 
  def loop(self, **kw):   
    
    from scipy.io import loadmat
    import gc

    output       = kw.pop('output'      , 'Mon'          ) 
    tuningReport = kw.pop('tuningReport', 'tuningReport' ) 
    doBeamer     = kw.pop('doBeamer'    , True           )
    shortSlides  = kw.pop('shortSlides' , False          )
    debug        = kw.pop('debug'       , False          )
    overwrite    = kw.pop('overwrite'   , False          )
    choicesfile  = kw.pop('choicesfile' ,                )
    basepath=output
    basepath+=('_et%d_eta%d')%(self._infoObjs[0].etbinidx(),self._infoObjs[0].etabinidx())
    choices = loadmat(choicesfile)
    if not overwrite and os.path.isdir( basepath ):
      self._logger.warning("Monitoring output path already exists!")
      return 

    if shortSlides:
      self._logger.warning('Short slides enabled! Doing only tables...')

    if debug:
      self._logger.warning('Debug mode activated!')

    wantedPlotNames = {'allBestTstSorts','allBestOpSorts','allWorstTstSorts', 'allWorstOpSorts',\
                       'allBestTstNeurons','allBestOpNeurons', 'allWorstTstNeurons', 'allWorstOpNeurons'} 

    perfBenchmarks = dict()
    pathBenchmarks = dict()

    from PlotHolder import PlotHolder
    from PlotHelper import plot_4c, plot_rocs, plot_nnoutput
    from TuningMonitoringInfo import MonitoringPerfInfo
    
    #Loop over benchmarks
    for infoObj in self._infoObjs:
      #Initialize all plos
      plotObjects = dict()
      perfObjects = dict()
      infoObjects = dict()
      pathObjects = dict()
      signal_paterns = dict()
      background_paterns = dict()
      #Init PlotsHolder 
      for plotname in wantedPlotNames:  
        if 'Sorts' in plotname:
          plotObjects[plotname] = PlotHolder(label = 'Sort')
        else:
          plotObjects[plotname] = PlotHolder(label = 'Neuron')
      #Retrieve benchmark name
      benchmarkName = infoObj.name()
      #Retrieve reference name
      reference = infoObj.reference()
      #summary
      csummary = infoObj.summary()
      #benchmark object
      cbenchmark = infoObj.rawBenchmark()
      #
      etBin = infoObj.etbin()
      # reference value
      refVal = infoObj.rawBenchmark()['refVal']
      #Eta bin
      etabinidx = infoObj.etabinidx()
      #Et bin
      etbinidx = infoObj.etbinidx()
      #Eta bin
      etabin = infoObj.etabin()
      #Et bin
      etbin = infoObj.etbin()


      self._logger.info(('Start loop over the benchmark: %s and etaBin = %d etBin = %d')%(benchmarkName,etabinidx, etbinidx)  )
      import copy
       
      args = dict()
      args['reference'] = reference
      args['refVal']    = refVal
      args['eps']       = cbenchmark['eps']
 
      self._logger.info('Creating plots...')
      # Creating plots
      neuron = choices ['choices'][infoObj.name().split('_')[-1]][0][0][etbinidx][etabinidx]
      
      # Figure path location
      currentPath =  ('%s/figures/%s/%s') % (basepath,benchmarkName,'neuron_'+str(neuron))
      neuronName = 'config_'+str(neuron).zfill(3)
      # Create folder to store all plot objects
      mkdir_p(currentPath)
      #Clear all hold plots stored
      plotObjects['allBestTstSorts'].clear()
      plotObjects['allBestOpSorts'].clear()
      infoObjects['allInfoOpBest_'+neuronName] = list()
      #plotObjects['allWorstTstSorts'].clear()
      #plotObjects['allWorstOpSorts'].clear()

      for sort in infoObj.sortBounds(neuron):

        sortName = 'sort_'+str(sort).zfill(3)
        #Init bounds 
        initBounds = infoObj.initBounds(neuron,sort)
        #Create path list from initBound list          
        initPaths = [('%s/%s/%s/init_%s')%(benchmarkName,neuronName,sortName,init) for init in initBounds]
        self._logger.debug('Creating init plots into the path: %s, (neuron_%s,sort_%s)', \
                            benchmarkName, neuron, sort)
        obj = PlotHolder(label = 'Init')
        try: #Create plots holder class (Helper), store all inits
          obj.retrieve(self._rootObj, initPaths)
        except RuntimeError:
          self._logger.fatal('Can not create plot holder object')
        #Hold all inits from current sort
        obj.set_index_correction(initBounds)
        
        obj.set_best_index(  csummary[neuronName][sortName]['infoTstBest']['init']  )
        obj.set_worst_index( csummary[neuronName][sortName]['infoTstWorst']['init'] )
        plotObjects['allBestTstSorts'].append(  copy.deepcopy(obj.get_best() ) )
        obj.set_best_index(   csummary[neuronName][sortName]['infoOpBest']['init']   )
        obj.set_worst_index(  csummary[neuronName][sortName]['infoOpWorst']['init']  )
        plotObjects['allBestOpSorts'].append(   copy.deepcopy(obj.get_best()  ) )
        #plotObjects['allWorstTstSorts'].append( copy.deepcopy(tstObj.getBest() )
        #plotObjects['allWorstOpSorts'].append(  copy.deepcopy(opObj.getBest()  )
        infoObjects['allInfoOpBest_'+neuronName].append( copy.deepcopy(csummary[neuronName][sortName]['infoOpBest']) )
        #Release memory
        del obj
      #Loop over sorts
      gc.collect()
      
      plotObjects['allBestTstSorts'].set_index_correction(  infoObj.sortBounds(neuron) )
      plotObjects['allBestOpSorts'].set_index_correction(   infoObj.sortBounds(neuron) )
      #plotObjects['allWorstTstSorts'].setIdxCorrection( infoObj.sortBounds(neuron) )
      #plotObjects['allWorstOpSorts'].setIdxCorrection(  infoObj.sortBounds(neuron) )

      # Best and worst sorts for this neuron configuration
      plotObjects['allBestTstSorts'].set_best_index(  csummary[neuronName]['infoTstBest']['sort']  )
      plotObjects['allBestTstSorts'].set_worst_index( csummary[neuronName]['infoTstWorst']['sort'] )
      plotObjects['allBestOpSorts'].set_best_index(   csummary[neuronName]['infoOpBest']['sort']   )
      plotObjects['allBestOpSorts'].set_worst_index(  csummary[neuronName]['infoOpWorst']['sort']  )

      # Hold the information from the best and worst discriminator for this neuron 
      infoObjects['infoOpBest_'+neuronName] = copy.deepcopy(csummary[neuronName]['infoOpBest'])
      infoObjects['infoOpWorst_'+neuronName] = copy.deepcopy(csummary[neuronName]['infoOpWorst'])
 
      # Best and worst neuron sort for this configuration
      plotObjects['allBestTstNeurons'].append( copy.deepcopy(plotObjects['allBestTstSorts'].get_best()  ))
      plotObjects['allBestOpNeurons'].append(  copy.deepcopy(plotObjects['allBestOpSorts'].get_best()   ))
      plotObjects['allWorstTstNeurons'].append(copy.deepcopy(plotObjects['allBestTstSorts'].get_worst() ))
      plotObjects['allWorstOpNeurons'].append( copy.deepcopy(plotObjects['allBestOpSorts'].get_worst()  ))
      
      # Create perf (tables) Objects for test and operation (Table)
      rname = benchmarkName.split("_")[-1]
      sortdets = dict()
      sortfas = dict()
      for k in csummary[neuronName].keys():
        if "sort" in k: 
          faList = csummary[neuronName][k]['summaryInfoTst']['fa'] 
          pdList = csummary[neuronName][k]['summaryInfoTst']['det']
          sortdets[k],sortfas[k] = self._ChoosePdPf(faList,pdList,rname)


        
          


      perfObjects[neuronName] =  MonitoringPerfInfo(benchmarkName, reference, 
                                                               csummary[neuronName]['summaryInfoTst'], 
                                                               csummary[neuronName]['infoOpBest'], 
                                                               cbenchmark,sortdets,sortfas)
      # Debug information
      self._logger.debug(('Crossval indexs: (bestSort = %d, bestInit = %d) (worstSort = %d, bestInit = %d)')%\
            (plotObjects['allBestTstSorts'].best, plotObjects['allBestTstSorts'].get_best()['bestInit'],
             plotObjects['allBestTstSorts'].worst, plotObjects['allBestTstSorts'].get_worst()['bestInit']))
      self._logger.debug(('Operation indexs: (bestSort = %d, bestInit = %d) (worstSort = %d, bestInit = %d)')%\
            (plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].get_best()['bestInit'],
             plotObjects['allBestOpSorts'].worst, plotObjects['allBestOpSorts'].get_worst()['bestInit']))

    
      # Figure 1: Plot all validation/test curves for all crossval sorts tested during
      # the training. The best sort will be painted with black and the worst sort will
      # be on red color. There is a label that will be draw into the figure to show 
      # the current location (neuron, sort, init) of the best and the worst network.
      args['label']     = ('#splitline{#splitline{Total sorts: %d}{etaBin: %d, etBin: %d}}'+\
                           '{#splitline{sBestIdx: %d iBestIdx: %d}{sWorstIdx: %d iBestIdx: %d}}') % \
                          (plotObjects['allBestTstSorts'].size(),etabinidx, etbinidx, plotObjects['allBestTstSorts'].best, \
                           plotObjects['allBestTstSorts'].get_best()['bestInit'], plotObjects['allBestTstSorts'].worst,\
                           plotObjects['allBestTstSorts'].get_worst()['bestInit'])

      args['cname']        = ('%s/plot_%s_neuron_%s_sorts_val')%(currentPath,benchmarkName,neuron)
      args['set']          = 'val'
      args['operation']    = False
      args['paintListIdx'] = [plotObjects['allBestTstSorts'].best, plotObjects['allBestTstSorts'].worst]
      pname1 = plot_4c(plotObjects['allBestTstSorts'], args)

      # Figure 2: Plot all validation/test curves for all crossval sorts tested during
      # the training. The best sort will be painted with black and the worst sort will
      # be on red color. But, here the painted curves represented the best and the worst
      # curve from the operation dataset. In other words, we pass all events into the 
      # network and get the efficiencis than we choose the best operation and the worst 
      # operation network and paint the validation curve who represent these sorts.
      # There is a label that will be draw into the figure to show 
      # the current location (neuron, sort, init) of the best and the worst network.
      args['label']     = ('#splitline{#splitline{Total sorts: %d (operation)}{etaBin: %d, etBin: %d}}'+\
                          '{#splitline{sBestIdx: %d iBestIdx: %d}{sWorstIdx: %d iBestIdx: %d}}') % \
                         (plotObjects['allBestOpSorts'].size(),etabinidx, etbinidx, plotObjects['allBestOpSorts'].best, \
                          plotObjects['allBestOpSorts'].get_best()['bestInit'], plotObjects['allBestOpSorts'].worst,\
                          plotObjects['allBestOpSorts'].get_worst()['bestInit'])
      args['cname']        = ('%s/plot_%s_neuron_%s_sorts_op')%(currentPath,benchmarkName,neuron)
      args['set']          = 'val'
      args['operation']    = True
      args['paintListIdx'] = [plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].worst]
      pname2 = plot_4c(plotObjects['allBestOpSorts'], args)

      # Figure 3: This figure show us in deteails the best operation network for the current hidden
      # layer and benchmark analysis. Depend on the benchmark, we draw lines who represents the 
      # stops for each curve. The current neuron will be the last position of the plotObjects
      splotObject = PlotHolder()
      args['label']     = ('#splitline{#splitline{Best network neuron: %d}{etaBin: %d, etBin: %d}}'+\
                          '{#splitline{sBestIdx: %d iBestIdx: %d}{}}') % \
                         (neuron,etabinidx, etbinidx, plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].get_best()['bestInit'])
      args['cname']     = ('%s/plot_%s_neuron_%s_best_op')%(currentPath,benchmarkName,neuron)
      args['set']       = 'val'
      args['operation'] = True
      splotObject.append( plotObjects['allBestOpNeurons'][-1] )
      pname3 = plot_4c(splotObject, args)
      
      
      # Figure 4: Here, we have a plot of the discriminator output for all dataset. Black histogram
      # represents the signal and the red onces represent the background. TODO: Apply this outputs
      # using the feedfoward manual method to generate the network outputs and create the histograms.
      args['cname']     = ('%s/plot_%s_neuron_%s_best_op_output')%(currentPath,benchmarkName,neuron)
      args['nsignal']   = self._data[0].shape[0]
      args['nbackground'] = self._data[1].shape[0]
      sbest = plotObjects['allBestOpNeurons'][-1]['bestSort']
      args['cut'] = csummary[neuronName]['sort_'+str(sbest).zfill(3)]['infoOpBest']['cut']
      args['rocname'] = 'roc_operation'
      pname4 = plot_nnoutput(splotObject,args)
 
      # Figure 5: The receive operation test curve for all sorts using the test dataset as base.
      # Here, we will draw the current tunnel and ref value used to set the discriminator threshold
      # when the bechmark are Pd or Pf case. When we use the SP case, this tunnel will not be ploted.
      # The black curve represents the best sort and the red onces the worst sort. TODO: Put the SP
      # point for the best and worst when the benchmark case is SP.
      args['cname']        = ('%s/plot_%s_neuron_%s_sorts_roc_tst')%(currentPath,benchmarkName,neuron)
      args['set']          = 'tst'
      args['paintListIdx'] = [plotObjects['allBestTstSorts'].best, plotObjects['allBestTstSorts'].worst]
      pname5 = plot_rocs(plotObjects['allBestTstSorts'], args)

      # Figure 6: The receive operation  curve for all sorts using the operation dataset (train+test) as base.
      # Here, we will draw the current tunnel and ref value used to set the discriminator threshold
      # when the bechmark are Pd or Pf case. When we use the SP case, this tunnel will not be ploted.
      # The black curve represents the best sort and the red onces the worst sort. TODO: Put the SP
      # point for the best and worst when the benchmark case is SP.
      args['cname']        = ('%s/plot_%s_neuron_%s_sorts_roc_op')%(currentPath,benchmarkName,neuron)
      args['set']          = 'operation'
      args['paintListIdx'] = [plotObjects['allBestOpSorts'].best, plotObjects['allBestOpSorts'].worst]
      pname6 = plot_rocs(plotObjects['allBestOpSorts'], args)

      # Map names for beamer, if you add a plot, you must add into
      # the path objects holder
      pathObjects['neuron_'+str(neuron)+'_sorts_val']      = pname1 
      pathObjects['neuron_'+str(neuron)+'_sort_op']        = pname2
      pathObjects['neuron_'+str(neuron)+'_best_op']        = pname3
      pathObjects['neuron_'+str(neuron)+'_best_op_output'] = pname4
      pathObjects['neuron_'+str(neuron)+'_sorts_roc_tst']  = pname5
      pathObjects['neuron_'+str(neuron)+'_sorts_roc_op']   = pname6
 
      #Loop over neurons

      #External 
      pathBenchmarks[benchmarkName]  = pathObjects
      perfBenchmarks[benchmarkName]  = perfObjects
     
      #Release memory
      for xname in plotObjects.keys():
        del plotObjects[xname]

      gc.collect()
      #if debug:  break
    #Loop over benchmark
          
    #Eta bin
  #  etabinidx = self._infoObjs[0].etabinidx()
    #Et bin
    binBounds = dict()
    if len(etbin) > 0 :
      binBounds['etbinstr'] = r'$%d < E_{T} \text{[Gev]}<%d$'%tuple(etbin)
    else:
      binBounds['etbinstr'] = r'\text{etBin[%d]}' % etbinidx

    if len(etabin) > 0 :
      binBounds['etabinstr'] = r'$%.2f<\eta<%.2f$'%tuple(etabin)
    else:
      binBounds['etabinstr'] = r'\text{etaBin[%d]}' % etabinidx
    anex=dict()
    anex[ 'nsignal'] = args['nsignal']
    anex[ 'nbackground'] = args['nbackground']
    anex['bounds'] = binBounds
    anex['perf'] = perfBenchmarks
    fname = basepath +'/'+ 'anex'
    save(anex,fname)
  #End of loop()





