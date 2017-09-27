#!/usr/bin/env python


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
from TuningTools import GridJobFilter, TuningMonitoringTool,makeSummaryMonSlides

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
  
  output = args.output
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
  makeSummaryMonSlides(args.output,len(paths.keys()),args.choicesfile)
#Loop ove

