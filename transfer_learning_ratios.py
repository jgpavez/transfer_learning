
import ROOT
from sklearn import svm, linear_model
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from xgboost_wrapper import XGBoostClassifier

import matplotlib.pyplot as plt

import sys

from os import listdir
from os.path import isfile, join
import os.path

from mlp import make_predictions, train_mlp
from utils import printFrame,makePlotName,makeSigBkg,saveFig

import numpy as np
import arff
from sklearn import preprocessing
import pdb

mu_g = []
cov_g = []

mu_g.append([5.,5.,4.,3.,5.,5.,4.5,2.5,4.,3.5])
#mu_g.append([7.,8.,7.,6.,7.,8.,6.5,5.5,7.,6.5])
mu_g.append([2.,4.5,0.6,5.,6.,4.5,4.2,0.2,4.1,3.3])
mu_g.append([1.,0.5,0.3,0.5,0.6,0.4,0.1,0.2,0.1,0.3])

cov_g.append([[3.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,2.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,14.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,17.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,5.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,1.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,9.3]])
cov_g.append([[3.5,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,3.5,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,9.5,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,7.2,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,4.5,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,4.5,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,8.2,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,9.5,3.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,3.5,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,4.5]])
cov_g.append([[13.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,12.,0.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,14.,0.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,6.,0.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,10.,0.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,15.,0.,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,6.3,0.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,11.,0.],
         [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.3]])


def makeModelND(vars_g,cov_l=cov_g,mu_l=mu_g,
    workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root', 
    dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp',
    verbose_printing=False):
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')

  print 'Generating initial distributions'
  cov_m = []
  mu_m = []
  mu_str = []
  cov_root = []
  vec = []
  argus = ROOT.RooArgList() 
  #features
  for i,var in enumerate(vars_g):
    w.factory('{0}[{1},{2}]'.format(var,-25,30))
    argus.add(w.var(var))

  for glob in range(2):
    # generate covariance matrix
    cov_m.append(np.matrix(cov_l[glob]))
    cov_root.append(ROOT.TMatrixDSym(len(vars_g)))
    for i,var1 in enumerate(vars_g):
      for j,var2 in enumerate(vars_g):
        cov_root[-1][i][j] = cov_m[-1][i,j]
    getattr(w,'import')(cov_root[-1],'cov{0}'.format(glob))
    # generate mu vector
    mu_m.append(np.array(mu_l[glob]))
    vec.append(ROOT.TVectorD(len(vars_g)))
    for i, mu in enumerate(mu_m[-1]):
      vec[-1][i] = mu
    mu_str.append(','.join([str(mu) for mu in mu_m[-1]]))
    # multivariate gaussian
    gaussian = ROOT.RooMultiVarGaussian('f{0}'.format(glob),
          'f{0}'.format(glob),argus,vec[-1],cov_root[-1])
    getattr(w,'import')(gaussian)
  # Check Model
  w.Print()

  w.writeToFile('{0}/{1}'.format(dir,workspace))
  if verbose_printing == True:
    printFrame(w,['x0','x1','x7','x8'],[w.pdf('f0'),w.pdf('f1')],'distributions',['f0','f1']
    ,dir=dir,model_g=model_g,range=[-15,20],title='Distributions',x_text='x0',y_text='p(x)',print_pdf=True)


  return w




def makeModel(
    workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root', 
    dir='/afs/cern.ch/user/j/jpavezse/systematics',model_g='mlp',
    verbose_printing=False):
  '''
  RooFit statistical model for the data
  
  '''  
  # Statistical model
  w = ROOT.RooWorkspace('w')
  #w.factory("EXPR::f1('cos(x)**2 + .01',x)")
  w.factory("EXPR::f0('exp(-(x-2.5)**2/1.)',x[0,10])")
  w.factory("EXPR::f1('exp(-(x-5.5)**2/5.)',x)")
  #w.factory("SUM::f2(c1[0.5]*f0,c2[0.5]*f1)")
  
  # Check Model
  w.Print()
  w.writeToFile('{0}/{1}'.format(dir,workspace))
  if verbose_printing == True:
    printFrame(w,['x'],[w.pdf('f0'),w.pdf('f1')],'transfered',['gaussian','transfered']
    ,dir=dir,model_g=model_g,range=[-15,20],title='Single distributions',x_text='x0',y_text='p(x)',
    print_pdf=True)


def makeData(vars_g, data_file='data', num_train=500,num_test=100,no_train=False,
  workspace='workspace_DecomposingTestOfMixtureModelsClassifiers.root',
  dir='/afs/cern.ch/user/j/jpavezse/systematics',
  model_g='mlp'):
  # Start generating data
  ''' 
    Each function will be discriminated pair-wise
    so n*n datasets are needed (maybe they can be reused?)
  ''' 

  f = ROOT.TFile('{0}/{1}'.format(dir,workspace))
  w = f.Get('w')
  f.Close()

  print 'Making Data'
  # Start generating data
  ''' 
    Each function will be discriminated pair-wise
    so n*n datasets are needed (maybe they can be reused?)
  ''' 
   
  # make data from root pdf
  def makeDataFi(x, pdf, num):
    traindata = np.zeros((num,len(vars_g))) 
    data = pdf.generate(x,num)
    traindata[:] = [[data.get(i).getRealValue(var) for var in vars_g]
        for i in range(num)]
    return traindata
  
  # features
  vars = ROOT.TList()
  for var in vars_g:
    vars.Add(w.var(var))
  x = ROOT.RooArgSet(vars)

  # make data from pdf and save to .dat in folder 
  # ./data/{model}/{c1}
  print 'Making data' 
  traindata = np.zeros((num_train*2,len(vars_g) + 1))
  testdata = np.zeros((num_test*2,len(vars_g) + 1))

  if not no_train:
    #traindata[:num_train,0] =  makeDataFi(x,w.pdf('f0'), num_train).reshape(num_train)

    #traindata[num_train:,0] = makeDataFi(x,w.pdf('f1'), num_train).reshape(num_train)
    traindata[:num_train,:len(vars_g)] =  makeDataFi(x,w.pdf('f0'), num_train)

    traindata[num_train:,:len(vars_g)] = makeDataFi(x,w.pdf('f1'), num_train)
 
    traindata[:num_train,-1] = np.ones(num_train)
    np.savetxt('{0}/train_{1}.dat'.format(dir,data_file),
                        traindata,fmt='%f')
  #testdata[:num_test,0] = makeDataFi(x, w.pdf('f0'), num_test).reshape(num_test)
  #testdata[num_test:,0] = makeDataFi(x, w.pdf('f1'), num_test).reshape(num_test)
  testdata[:num_test,:len(vars_g)] =  makeDataFi(x,w.pdf('f0'), num_test)

  testdata[num_test:,:len(vars_g)] = makeDataFi(x,w.pdf('f1'), num_test)

  testdata[num_test:,-1] = np.ones(num_test)

  np.savetxt('{0}/test_{1}.dat'.format(dir,data_file),
                      testdata,fmt='%f')

def findOutliers(x):
  q5, q95 = np.percentile(x, [5,95])  
  iqr = 2.*(q95 - q5)
  outliers = (x <= q95 + iqr) & (x >= q5 - iqr)
  return outliers

def singleRatio(f0,f1):
  ratio = f1 / f0
  ratio[np.abs(ratio) == np.inf] = 0 
  ratio[np.isnan(ratio)] = 0
  return ratio

def evalDist(x,f0,val):
  iter = x.createIterator()
  v = iter.Next()
  i = 0
  while v:
    v.setVal(val[i])
    v = iter.Next()
    i = i+1
  return f0.getVal(x)

def computeRatios(workspace,data_file,model_file,dir,model_g,c1_g,true_dist=False,
      vars_g=None):
  '''
    Use the computed score densities to compute 
    the ratio test.
 
  '''

  f = ROOT.TFile('{0}/{1}'.format(dir,workspace))
  w = f.Get('w')
  f.Close()
  

  print 'Calculating ratios'

  npoints = 50

  score = ROOT.RooArgSet(w.var('score'))
  getRatio = singleRatio

  if true_dist == True:
    vars = ROOT.TList()
    for var in vars_g:
      vars.Add(w.var(var))
    x = ROOT.RooArgSet(vars)

  # NN trained on complete model
  F0pdf = w.function('bkghistpdf_F0_F1')
  F1pdf = w.function('sighistpdf_F0_F1')
  data = np.loadtxt('{0}/train_{1}.dat'.format(dir,data_file)) 
  testdata = data[:,:-1]
  testtarget = data[:,-1]

  '''
  # Make ratio considering tumor size unknown
  ts_idx = 2
  target = testdata[0]
  testdata_size = np.array([x for x in testdata if (np.delete(x,ts_idx) == np.delete(target,ts_idx)).all()])
  '''

  if true_dist == True and len(vars_g) == 1:
      xarray = np.linspace(1,10,npoints)
      # TODO: Harcoded dist names
      F1dist = np.array([evalDist(x,w.pdf('f1'),[xs]) for xs in xarray])
      F0dist = np.array([evalDist(x,w.pdf('f0'),[xs]) for xs in xarray])
      trueRatio = getRatio(F1dist, F0dist)

      outputs = predict('{0}/{1}_F0_F1.pkl'.format(dir,model_file),xarray,model_g=model_g)

      F1fulldist = np.array([evalDist(score,F1pdf,[xs]) for xs in outputs])
      F0fulldist = np.array([evalDist(score,F0pdf,[xs]) for xs in outputs])

      completeRatio = getRatio(F0fulldist,F1fulldist)

      saveFig(xarray, [completeRatio, trueRatio], makePlotName('all','train',type='ratio'),title='Density Ratios',labels=['Trained', 'Truth'], print_pdf=True,dir=dir)
  
  outputs = predict('{0}/{1}_F0_F1.pkl'.format(dir,model_file),testdata,model_g=model_g)

  F1fulldist = np.array([evalDist(score,F1pdf,[xs]) for xs in outputs])
  F0fulldist = np.array([evalDist(score,F0pdf,[xs]) for xs in outputs])

  completeRatio = getRatio(F1fulldist,F0fulldist)
  complete_target = testtarget
  #Histogram F0-f0 for composed, full and true

  # Removing outliers
  numtest = completeRatio.shape[0]
  #decomposedRatio[decomposedRatio < 0.] = completeRatio[decomposedRatio < 0.]

  complete_outliers = np.zeros(numtest,dtype=bool)
  complete_outliers = findOutliers(completeRatio)
  complete_target = testtarget[complete_outliers] 
  completeRatio = completeRatio[complete_outliers]

  bins = 70
  low = 0.6
  high = 1.2

  for l,name in enumerate(['sig','bkg']):
    minimum = completeRatio[complete_target == 1-l].min() 
    maximum = completeRatio[complete_target == 1-l].max()

    low = minimum - ((maximum - minimum) / bins)*10
    high = maximum + ((maximum - minimum) / bins)*10
    w.factory('ratio{0}[{1},{2}]'.format(name, low, high))
    ratio_var = w.var('ratio{0}'.format(name))

    numtest = completeRatio.shape[0] 
    hist = ROOT.TH1F('{0}hist_F0_f0'.format(name),'hist',bins,low,high)
    for val in completeRatio[complete_target == 1-l]:
      hist.Fill(val)
    datahist = ROOT.RooDataHist('{0}datahist_F0_f0'.format(name),'hist',
          ROOT.RooArgList(ratio_var),hist)
    ratio_var.setBins(bins)
    histpdf = ROOT.RooHistFunc('{0}histpdf_F0_f0'.format(name),'hist',
          ROOT.RooArgSet(ratio_var), datahist, 0)

    histpdf.specialIntegratorConfig(ROOT.kTRUE).method1D().setLabel('RooBinIntegrator')
    getattr(w,'import')(hist)
    getattr(w,'import')(datahist) # work around for morph = w.import(morph)
    getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
    #print '{0} {1} {2}'.format(curr,name,hist.Integral())

    if name == 'bkg':
      all_ratios_plots = [w.function('sighistpdf_F0_f0'),
            w.function('bkghistpdf_F0_f0')]
      all_names_plots = ['sig','bkg']
    
  printFrame(w,['ratiosig','ratiobkg'],all_ratios_plots, makePlotName('ratio','comparison',type='hist',dir=dir,model_g=model_g,c1_g=c1_g),all_names_plots,dir=dir,model_g=model_g,y_text='Count',title='Histograms for ratios',x_text='ratio value',print_pdf=True)

  #completeRatio = np.log(completeRatio)
  completeRatio = completeRatio + np.abs(completeRatio.min())
  ratios_list = completeRatio / completeRatio.max()
  legends_list = ['composed','full']
  makeSigBkg([ratios_list],[complete_target],makePlotName('comp','all',type='sigbkg',dir=dir,model_g=model_g,c1_g=c1_g),dir=dir,model_g=model_g,print_pdf=True,legends=legends_list,title='Signal-Background rejection curves')

  # Make transfer learning

  data = np.loadtxt('{0}/train_{1}.dat'.format(dir,data_file)) 
  # Transforming f1 into f0
  data_f1 = data[data[:,-1] == 0.]
  data_f0 = data[data[:,-1] == 1.]
  testdata = data_f1[:,:-1]
  testtarget = data_f1[:,-1]

  '''
  # Make ratio considering tumor size unknown
  ts_idx = 2
  target = testdata[0]
  testdata_size = np.array([x for x in testdata if (np.delete(x,ts_idx) == np.delete(target,ts_idx)).all()])
  pdb.set_trace()
  '''

  xarray = testdata

  outputs = predict('{0}/{1}_F0_F1.pkl'.format(dir,model_file),xarray,model_g=model_g)

  F1fulldist = np.array([evalDist(score,F1pdf,[xs]) for xs in outputs])
  F0fulldist = np.array([evalDist(score,F0pdf,[xs]) for xs in outputs])

  completeRatio = getRatio(F0fulldist,F1fulldist)

  if len(vars_g) == 1:
    F1dist = np.array([evalDist(x,w.pdf('f1'),[xs]) for xs in xarray])
    F0dist = np.array([evalDist(x,w.pdf('f0'),[xs]) for xs in xarray])
  else:
    F1dist = np.array([evalDist(x,w.pdf('f1'),xs) for xs in xarray])
    F0dist = np.array([evalDist(x,w.pdf('f0'),xs) for xs in xarray])

  trueRatio = getRatio(F1dist, F0dist)

  trueIndexes = findOutliers(trueRatio)
  completeIndexes = findOutliers(completeRatio)
  #indexes = np.logical_and(trueIndexes,completeIndexes)
  indexes = completeIndexes
  data_f1_red = data_f1
  #trueRatio = trueRatio[indexes]
  #completeRatio = completeRatio[indexes]
  #data_f1_red = data_f1[indexes]


  for f in range(10):
    feature = f
    # Transfering distributions
    # Doing histogram manipulation
    fig,ax = plt.subplots()
    colors = ['b-','r-','k-']
    colors_rgb = ['blue','red','black']
    
    hist,bins = np.histogram(data_f1[:,feature],bins=20, range=(0.,10.),density=True)


    hist_transfered,bins_1 = np.histogram(data_f1_red[:,feature],weights=trueRatio,bins=20, range=(0.,10.),density=True)
    hist_transfered_clf,bins_2 = np.histogram(data_f1_red[:,feature],bins=20,weights=completeRatio, range=(0.,10.),density=True)
    hist0,bins0 = np.histogram(data_f0[:,feature], bins=20, range=(0.,10.),density=True)

    #hist, bins =  ax.hist(data_f0[:,0],color=colors_rgb[0],label='true',bins=50,histtype='stepfilled',normed=1, alpha=0.5,range=[0,100]) 

    widths = np.diff(bins)
    #hist_transfered = hist*trueRatio
    #hist_transfered_clf = hist*completeRatio

    ax.bar(bins[:-1], hist0,widths,label='f0',alpha=0.5,color='red')
    #ax.bar(bins[:-1], hist_transfered,widths,label='f1 transfered (true)',
    #    alpha=0.5,color='blue')
    ax.bar(bins[:-1], hist_transfered_clf,widths,label='f1 transfered (trained)',
        alpha=0.5,color='green')

    ax.legend(frameon=False,fontsize=11)
    ax.set_xlabel('x') 
    ax.set_ylabel('p(x)') 
    if len(vars_g) > 1:
      ax.set_title('Transfered distributions feature {0}'.format(feature))
    else:
      ax.set_title('Transfered distributions')
    file_plot =  makePlotName('all','transf',type='hist_v{0}'.format(feature),model_g=model_g) 
    fig.savefig('{0}/plots/{1}/{2}.png'.format(dir,model_g,file_plot))
    #saveFig(xarray, [true_transfer, data_f0[:,0]], makePlotName('all','transf',type='hist'),title='Transfered distribution',labels=['Transfer True', 'Truth'],hist=True, print_pdf=True,dir=dir)


def predict(filename, traindata,model_g='mlp', sig=1):
  sfilename,k,j = filename.split('/')[-1].split('_')
  sfilename = '/'.join(filename.split('/')[:-1]) + '/' +  sfilename
  j = j.split('.')[0]
  sig = 1
  if k <> 'F0':
    k = int(k)
    j = int(j)
    sig = 1 if k < j else 0
    filename = '{0}_{1}_{2}.pkl'.format(sfilename,min(k,j),max(k,j))
  if model_g == 'mlp':
    return make_predictions(dataset=traindata, model_file=filename)[:,sig]
  else:
    clf = joblib.load(filename)
    if clf.__class__.__name__ == 'NuSVR':
      output = clf.predict(traindata)
      return np.clip(output,0.,1.)
    else:
      return clf.predict_proba(traindata)[:,sig]


def fit(input_workspace,dir,model_g='mlp',c1_g='breast',data_file='data',
      model_file='train',verbose_printing=True):

  bins = 80
  low = 0.
  high = 1.  
  
  if input_workspace <> None:
    f = ROOT.TFile('{0}/{1}'.format(dir,input_workspace))
    w = f.Get('w')
    # TODO test this when workspace is present
    w = ROOT.RooWorkspace('w') if w == None else w
    f.Close()
  else: 
    w = ROOT.RooWorkspace('w')
  w.Print()

  print 'Generating Score Histograms'

  w.factory('score[{0},{1}]'.format(low,high))
  s = w.var('score')
  
  def saveHisto(w,outputs,s,bins,low,high,k='F0',j='F1'):
    
    print 'Estimating {0} {1}'.format(k,j)
    for l,name in enumerate(['sig','bkg']):
      data = ROOT.RooDataSet('{0}data_{1}_{2}'.format(name,k,j),"data",
          ROOT.RooArgSet(s))
      hist = ROOT.TH1F('{0}hist_{1}_{2}'.format(name,k,j),'hist',bins,low,high)
      values = outputs[l]
      #values = values[self.findOutliers(values)]
      for val in values:
        hist.Fill(val)
        s.setVal(val)
        data.add(ROOT.RooArgSet(s))
      norm = 1./hist.Integral()
      hist.Scale(norm) 
        
      s.setBins(bins)
      datahist = ROOT.RooDataHist('{0}datahist_{1}_{2}'.format(name,k,j),'hist',
            ROOT.RooArgList(s),hist)
      histpdf = ROOT.RooHistFunc('{0}histpdf_{1}_{2}'.format(name,k,j),'hist',
            ROOT.RooArgSet(s), datahist, 1)

      getattr(w,'import')(hist)
      getattr(w,'import')(data)
      getattr(w,'import')(datahist) # work around for morph = w.import(morph)
      getattr(w,'import')(histpdf) # work around for morph = w.import(morph)
      score_str = 'score'
      # Calculate the density of the classifier output using kernel density 
      #w.factory('KeysPdf::{0}dist_{1}_{2}({3},{0}data_{1}_{2},RooKeysPdf::NoMirror,2)'.format(name,k,j,score_str))

  # Full model
  data = np.loadtxt('{0}/train_{1}.dat'.format(dir,data_file)) 
  traindata = data[:,:-1]
  targetdata = data[:,-1]

  numtrain = traindata.shape[0]       
  size2 = traindata.shape[1] if len(traindata.shape) > 1 else 1

  outputs = [predict('/afs/cern.ch/work/j/jpavezse/private/transfer_learning/{0}_F0_F1.pkl'.format(model_file),traindata[targetdata==1],model_g=model_g),
            predict('/afs/cern.ch/work/j/jpavezse/private/transfer_learning/{0}_F0_F1.pkl'.format(model_file),traindata[targetdata==0],model_g=model_g)]

  saveHisto(w,outputs,s, bins, low, high)

  if verbose_printing == True:
    printFrame(w,['score'],[w.function('sighistpdf_F0_F1'),w.function('bkghistpdf_F0_F1')], makePlotName('full','all',type='hist',dir=dir,c1_g=c1_g,model_g=model_g),['signal','bkg'],
  dir=dir,model_g=model_g,y_text='score(x)',print_pdf=True,title='Pairwise score distributions')
 
  w.writeToFile('{0}/{1}'.format(dir,input_workspace))
  w.Print()


def trainClassifier(clf,
      dir,model_file='adaptive',
      data_file='train',
      seed=1234,
    ):
  '''
   Train classifier
  '''
  print 'Training classifier'

  data = np.loadtxt('{0}/train_{1}.dat'.format(dir,data_file)) 
  traindata = data[:,:-1]
  targetdata = data[:,-1]
  pdb.set_trace()

  if model_g == 'mlp':
    train_mlp((traindata, targetdata), save_file='{0}/{1}_F0_F1.pkl'.format(dir,model_file))
  else:
    rng = np.random.RandomState(seed)
    indices = rng.permutation(traindata.shape[0])
    traindata = traindata[indices]
    targetdata = targetdata[indices]
    scores = cross_validation.cross_val_score(clf, traindata, targetdata)
    print "Accuracy: {0} (+/- {1})".format(scores.mean(), scores.std() * 2)
    clf.fit(traindata,targetdata)
    #clf.plot_importance_matrix(vars_names)
    joblib.dump(clf, '{0}/{1}_F0_F1.pkl'.format(dir,model_file))


if __name__ == '__main__':
  #Setting classifier to use
  model_g = None
  classifiers = {'svc':svm.NuSVC(probability=True),'svr':svm.NuSVR(),
        'logistic': linear_model.LogisticRegression(), 
        'bdt':GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
        max_depth=4, random_state=0),
        'mlp':'',
        'xgboost': XGBoostClassifier(num_class=2, nthread=4, silent=0,
          num_boost_round=50, eta=0.1, max_depth=3)}
  clf = None
  if (len(sys.argv) > 1):
    model_g = sys.argv[1]
    clf = classifiers.get(sys.argv[1])
  if clf == None:
    model_g = 'logistic'
    clf = classifiers['logistic']    
    print 'Not found classifier, Using logistic instead'
  c1_g = 'breast'

  dir = '/afs/cern.ch/work/j/jpavezse/private/transfer_learning'
  workspace_file = 'workspace_transfer.root'
  verbose_printing = True

  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsRel(1E-15)
  ROOT.RooAbsPdf.defaultIntegratorConfig().setEpsAbs(1E-15)

  random_seed = 1234
  if (len(sys.argv) > 3):
    print 'Setting seed: {0} '.format(sys.argv[2])
    random_seed = int(sys.argv[2])
    ROOT.RooRandom.randomGenerator().SetSeed(random_seed) 

  data_file = 'transfer_data'
  model_file = 'train'

  #vars_g = ['x']
  vars_g = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']

  makeModelND(vars_g=vars_g,workspace=workspace_file,dir=dir,model_g=model_g,verbose_printing=verbose_printing)
  #makeModel(workspace_file,dir=dir,model_g=model_g,verbose_printing=verbose_printing)
  #makeData(vars_g,data_file,workspace=workspace_file,dir=dir,model_g=model_g,num_train=15000,num_test=2500)
    #Loading data
  #trainClassifier(clf,dir,model_file,data_file)
  #fit(workspace_file, dir, model_g, c1_g, data_file = data_file,
  #      model_file=model_file) 

  #computeRatios(workspace_file,data_file=data_file,model_file=model_file,dir=dir,model_g=model_g,c1_g=c1_g,true_dist=True,vars_g=vars_g) 

