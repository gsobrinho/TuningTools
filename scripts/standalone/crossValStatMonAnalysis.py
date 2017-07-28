#!/usr/bin/env python

from RingerCore.tex.TexAPI import *
from RingerCore.tex.BeamerAPI import *
from RingerCore import load
def SlideMaker(outputs,nbins,choicesfile):
  from scipy.io import loadmat
  f =  loadmat(choicesfile) 
  choices = dict()
  choices['Pd'] = f['choices']['Pd'][0][0]
  choices['Pf'] = f['choices']['Pf'][0][0]
  choices['SP'] = f['choices']['SP'][0][0]

  nbinchoices = 0
  for etchoices in choices['Pd']:
    nbinchoices += len(etchoices)
  if not nbins == nbinchoices:
    raise NameError('Choices Archieve Error')

  net = len(choices['Pd'])
  neta = len (choices['Pd'][0])
  unbinned = False
  if net == 1 and neta == 1:
    unbinned = True
  f = load('{}/perfBounds.pic.gz'.format(outputs+'_et0_eta0'))
  benchmarkNames = f['perf'].keys()
  
  for benchmarkName in benchmarkNames:
    slideAnex = []
    for et in xrange(net):
      etlist = []
      for eta in xrange( neta):
        etaDict = dict()
        neuron = choices[benchmarkName.split('_')[-1]][et][eta]
        basepath=outputs
        basepath+=('_et%d_eta%d')%(et,eta)
        f =  load ('{}/perfBounds.pic.gz'.format(basepath))
        etstr =  f['bounds']['etbinstr']
        etastr =  f['bounds']['etabinstr']
        perfs  = f['perf'][benchmarkName]

        refBench = perfs['config_'+str(neuron).zfill(3)].getRef()
        detR = r'{:.2f}'.format(refBench['det'])
        spR  = r'{:.2f}'.format(refBench['sp'])
        faR  = r'{:.2f}'.format(refBench['fa'])
        refBench = [detR,spR,faR]

        perfBench = perfs['config_'+str(neuron).zfill(3)].getPerf()
        detP = r'{:.2f}$\pm${:.2f}'.format(perfBench['detMean'],perfBench['detStd'])
        spP  = r'{:.2f}$\pm${:.2f}'.format(perfBench['spMean'],perfBench['spStd'])
        faP  = r'{:.2f}$\pm${:.2f}'.format(perfBench['faMean'],perfBench['faStd'])
        perfBench = [detP,spP,faP]

        bestNetBench = perfs['config_'+str(neuron).zfill(3)].rawOp()
        detB = r'{:.2f}'.format(bestNetBench['det']*100)
        spB  = r'{:.2f}'.format(bestNetBench['sp']*100)
        faB  = r'{:.2f}'.format(bestNetBench['fa']*100)
        bestNetBench = [ detB,spB,faB]
        perfs = [refBench,perfBench,bestNetBench]  
      
        graphSections = [
              'All Sorts(Validation)' ,
              'All ROC Sorts(Validation)',
              'All Sorts(Operation)' ,  
              'All ROC Sorts(Operation)',
              'Best Network',  
              'Best Operation Output',  
              ]

        figures = [
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_val.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_roc_tst.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_op.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron)      ,  
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_sorts_roc_op.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_best_op.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron)       ,  
              '{}/figures/{}/neuron_{}/plot_{}_neuron_{}_best_op_output.pdf'.format(basepath,benchmarkName,neuron,benchmarkName,neuron),  
                 ]
        figuresDict= dict(zip(graphSections,figures))
        etaDict['neuron'] = neuron
        etaDict['figures'] = figuresDict
        etaDict['graphSections'] = graphSections
        etaDict['perfs'] = perfs 
        etaDict['etastr'] = etastr 
        etaDict['etstr'] = etstr 

        etlist.append(etaDict)
        # for eta
      slideAnex.append(etlist)
      #for et
    with BeamerTexReportTemplate1( theme = 'Berlin'
                                 , _toPDF = True
                                 , title = benchmarkName
                                 , outputFile = benchmarkName
                                 , font = 'structurebold' ):
      with BeamerSection(name = 'Performance'):
        if not unbinned:
          l1 = ['','']
          l2 = ['','']
          sideline =  '{|c|c|'
          for et in xrange(net):
            sideline += 'ccc|'
            l1.extend(['',slideAnex[et][0]['etstr'],''])
            l2.extend(['Pd','SP','PF'])
          sideline +='}'
          etlines = []
          for eta in xrange(neta):
            la = [r'\hline'+ slideAnex[0][eta]['etastr'],'CrossValidation' ]
            lb = [r'','Reference']
            lc = [r'', 'bestNetBench']
            for et in xrange(net):
              prfs = slideAnex [et][eta]['perfs']
              refBench = prfs[0]
              perfBench = prfs[1]
              bestNetBench = prfs[2]

              la.extend(perfBench)
              lb.extend(refBench)
              lc.extend(bestNetBench)
            etlines.extend([la,lb,lc])
          linhas=[]
          linhas.append(l1)
          linhas.append(l2)
          linhas.extend(etlines)
          with BeamerSubSection(name= 'All Bins Performance'):
            BeamerTableSlide(title =  'All Bins Performance',
                           linhas = linhas,
                           sideline = sideline,
                           caption = 'Efficiences',
                           )
            
          


        for et in xrange(net):
          l1 = ['','']
          l2 = ['','']
          sideline =  '{|c|c|ccc|}'
          l1.extend(['',slideAnex[et][0]['etstr'],''])
          l2.extend(['Pd','SP','PF'])
          etlines = []
          for eta in xrange(neta):
            if unbinned:
              la = [r'\hline','CrossValidation' ]
            else:
              la = [r'\hline'+ slideAnex[0][eta]['etastr'],'CrossValidation' ]
            lb = [r'Neuron','Reference']
            lc = [r'', 'bestNetBench']
            neuron = slideAnex [et][eta]['neuron']
            prfs = slideAnex [et][eta]['perfs']
            lc[0] = '%d'%neuron
            refBench = prfs[0]

            perfBench = prfs[1]

            bestNetBench = prfs[2]

            la.extend(perfBench)
            lb.extend(refBench)
            lc.extend(bestNetBench)
            etlines.extend([la,lb,lc])
          linhas=[]
          if not unbinned:
            linhas.append(l1)
          linhas.append(l2)
          linhas.extend(etlines)
          if unbinned:
            BeamerTableSlide(title =  'Performance',
                           linhas = linhas,
                           sideline = sideline,
                           caption = 'Efficiences',
                           )
          else:
            with BeamerSubSection (name= slideAnex[et][0]['etstr']):
              BeamerTableSlide(title =  'Performance',
                           linhas = linhas,
                           sideline = sideline,
                           caption = 'Efficiences',
                           )

      with BeamerSection( name = 'Figures' ):
        if unbinned:
          neuron = slideAnex [0][0]['neuron']
          graphSections = slideAnex [0][0]['graphSections']
          figures = slideAnex [0][0]['figures']
          for graph in graphSections:
            BeamerFigureSlide(title = graph + ' Neuron: '+ str(neuron) ,
                          path = figures[graph]
                               )
        else:
          for et in xrange(net):
            with BeamerSubSection (name= slideAnex[et][0]['etstr']):
              for eta in xrange(neta):
                neuron = slideAnex [et][eta]['neuron']
                graphSections = slideAnex [et][eta]['graphSections']
                figures = slideAnex [et][eta]['figures']

                with BeamerSubSubSection (name= slideAnex[et][eta]['etastr'] + ', Neuron: {}'.format(neuron)):
                  for graph in graphSections:
                    BeamerFigureSlide(title = graph,
                                       path = figures[graph]
                                      )
  return

def filterPaths(paths, grid=False):
  oDict = dict()
  import re
  from RingerCore import checkExtension
  if grid is True:
    pat = re.compile(r'.*user.[a-zA-Z0-9]+.(?P<jobID>[0-9]+)\..*$')
    jobIDs = sorted(list(set([pat.match(f).group('jobID')  for f in paths if pat.match(f) is not None]))) 
    for jobID in jobIDs:
      oDict[jobID] = dict()
      for xname in paths:
        if jobID in xname and checkExtension( xname, '.root'): oDict[jobID]['root'] = xname
        if jobID in xname and checkExtension( xname, '.pic|.pic.gz'): oDict[jobID]['pic'] = xname
  else:

    pat = re.compile(r'.*crossValStat_(?P<jobID>[0-9]+)(_monitoring)?\..*$')
    jobIDs = sorted(list(set([pat.match(f).group('jobID')  for f in paths if pat.match(f) is not None]))) 
    if not len( jobIDs):
      oDict['unique'] = {'root':'','pic':''}
      for xname in paths:
        if xname.endswith('.root'): oDict['unique']['root'] = xname
        if '.pic' in xname: oDict['unique']['pic'] = xname
    else:
      for jobID in jobIDs:
        print jobID
        oDict[jobID] = dict()
        for xname in paths:
          if jobID in xname and checkExtension( xname, '.root'): oDict[jobID]['root'] = xname
          if jobID in xname and checkExtension( xname, '.pic|.pic.gz'): oDict[jobID]['pic'] = xname
       

  return oDict


from RingerCore import csvStr2List, str_to_class, NotSet, BooleanStr, emptyArgumentsPrintHelp
from TuningTools.parsers import ArgumentParser, loggerParser, crossValStatsMonParser, LoggerNamespace
from TuningTools import GridJobFilter, TuningMonitoringTool

parser = ArgumentParser(description = 'Retrieve performance information from the Cross-Validation method.',
                       parents = [crossValStatsMonParser, loggerParser])
parser.make_adjustments()

emptyArgumentsPrintHelp( parser )

# Retrieve parser args:
args = parser.parse_args(namespace = LoggerNamespace() )

from RingerCore import Logger, LoggingLevel, printArgs
logger = Logger.getModuleLogger( __name__, args.output_level )

printArgs( args, logger.debug )


#Find files
from RingerCore import expandFolders, ensureExtension,keyboard
logger.info('Expand folders and filter')
paths = expandFolders(args.file)
paths = filterPaths(paths, args.grid)


from pprint import pprint
logger.info('Grid mode is: %s',args.grid)
pprint(paths)


#from TuningTools import TuningDataArchieve
#try:
#  logger.info(('Opening reference file with location: %s')%(args.refFile))
#  TDArchieve = TuningDataArchieve.load(args.refFile)
#  with TDArchieve as data:
#    patterns = data
#except:
#  raise RuntimeError("Can not open the refFile!")


#Loop over job grid, basically loop over user...
#keyboard()
for jobID in paths:
  logger.info( ('Start from job tag: %s')%(jobID))
  #If files from grid, we must put the bin tag
  
  output = args.output+'_'+jobID if args.grid else args.output
  #Create the monitoring object
  monitoring = TuningMonitoringTool( paths[jobID]['pic'], 
                                     paths[jobID]['root'], 
                                     dataPath = args.dataPath,
                                     level = args.output_level)
  #Start!
  #if monitoring.etabin() == 0 and monitoring.etbin() == 1:
  monitoring(
              shortSlides  = args.doShortSlides,
              debug        = args.debug,
              choicesfile  = args.choicesfile,
              output       = output)

  #ibin =  ('et%s_eta%s')%(monitoring.etbin(), monitoring.etabin())
  #logger.info(('holding summary with key: ')%(ibin))
  #cSummaryInfo[ibin] = monitoring.summary()
  del monitoring
if args.doBeamer:
  SlideMaker(args.output,len(paths.keys()),args.choicesfile)
#Loop ove

