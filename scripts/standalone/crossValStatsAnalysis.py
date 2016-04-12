#!/usr/bin/env python

from TuningTools.parsers import argparse, loggerParser, LoggerNamespace, tuningJobParser

parser = argparse.ArgumentParser(add_help = False, 
                                 description = 'Tune discriminators using input data.',
                                 parents = [tuningJobParser, loggerParser])

import sys
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

# Retrieve parser args:
args = parser.parse_args( namespace = LoggerNamespace() )

from TuningTools import CrossValidStatAnalysis, GridJobFilter, TuningDataArchieve, \
                         ReferenceBenchmark

stat = CrossValidStatAnalysis( 
    '/afs/cern.ch/work/w/wsfreund/public/Online/20.7.3.6_fixET/user.nbullacr.nn.mc14_13TeV.147406.129160.sgn.offLH.bkg.truth.trig.ef.e24_lhmedium_nod0_binned_t0001_tunedDiscrXYZ.tgz',
    binFilters = GridJobFilter,
    level = LoggingLevel.DEBUG,
    )

TDArchieve = TuningDataArchieve('/afs/cern.ch/work/j/jodafons/public/Tuning2016/TuningConfig/mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_nod0_l1etcut20_l2etcut19_efetcut24_binned.pic.npz')
#TDArchieve = TuningDataArchieve('/Users/wsfreund/Documents/Doutorado/CERN/Online/data/mc14_13TeV.147406.129160.sgn.offLikelihood.bkg.truth.trig.e24_lhmedium_nod0_l1etcut20_l2etcut19_efetcut24_binned.pic.npz')
nEtBins = TDArchieve.nEtBins()
nEtaBins = TDArchieve.nEtaBins()
refBenchmarkList = []
from itertools import product
with TDArchieve as data:
  for etBin, etaBin in product( range( nEtBins if nEtBins is not None else 1 ),
                                range( nEtaBins if nEtaBins is not None else 1 )):
    benchmarks = (data['signal_efficiencies'], data['background_efficiencies'])
    #cross_benchmarks = (TDArchieve['signal_cross_efficiencies'], TDArchieve['background_cross_efficiencies'])
    sigEff = data['signal_efficiencies']['EFCaloAccept'][etBin][etaBin]
    bkgEff = data['background_efficiencies']['EFCaloAccept'][etBin][etaBin]
    try:
      sigCrossEff = data['signal_cross_efficiencies']['EFCaloAccept'][etBin][etaBin]
      bkgCrossEff = data['background_cross_efficiencies']['EFCaloAccept'][etBin][etaBin]
    except KeyError:
      sigCrossEff = None; bkgCrossEff = None
    args = (sigEff, bkgEff, sigCrossEff, bkgCrossEff)
    Medium_LH_EFCalo_Pd = ReferenceBenchmark( "Medium_LH_EFCalo_Pd", "Pd", *args )
    Medium_MaxSP        = ReferenceBenchmark( "Medium_MaxSP",        "SP", *args )
    Medium_LH_EFCalo_Pf = ReferenceBenchmark( "Medium_LH_EFCalo_Pf", "Pf", *args )
    references =  [ Medium_LH_EFCalo_Pd,
                    Medium_MaxSP,
                    Medium_LH_EFCalo_Pf ] 
    print ('Et:',etBin, 'eta:', etaBin), [ref.refVal for ref in references]
    refBenchmarkList.append( references )
del data

stat( refBenchmarkList , outputName = 'FixET_Norm1_20.7.3.6' )

