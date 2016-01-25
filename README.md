## Approximated density ratios for Transfer Learning

This work will study how approximated likelihood ratios [Approximating Likelihood Ratios with Calibrated Discriminative Classifiers]
(http://arxiv.org/abs/1506.02169) can be used in transfer learning (or domain adaptation) tasks.

Transfer learning is a field which study how we can improve a task (e.g. classification or regression) using information from a different 
domain. If we have a classifier trained in a dataset, we would like to use this calssifier in a different dataset with different but related 
distribution. A complete survey can be found in [A Survey on Transfer Learning] (http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=5288526&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D5288526). 

A way to do this is to reweight the data using *w = g(x) / f(x)* where *f(x)* is the target domain distribution and *g(x)* is the source domain distribution. 
Then we minimize *sum(g(x)/f(x) * f(x) * l(x,y))*. A similar 
task in physics is known as *event reweighting* and is used to account for differences between simulated and real data.

Using approximated likelihood ratios we can replace *g(x) / f(x)* by *f(s|domain)/f(s|target)* where *s* is the score of a classifier trained to classify between source and domain data.

We will start by showing a simple 1-dimentional case. The target (f1(x)) and source distributions (f0(x)) are shown next. Both are gaussian distributions.

![1D_dist](https://github.com/jgpavez/transfer_learning/blob/master/plots/mlp/transfered.png)

We train a *multilayer perceptron* to classify between data generated from the distributions. The score distribution for background and signal are shown.

![1D_score](https://github.com/jgpavez/transfer_learning/blob/master/plots/mlp/full_all_mlp_hist.png)

We can compare the density ratios obtained with the real ratio (in this case known).

![1D_ratios](https://github.com/jgpavez/transfer_learning/blob/master/plots/mlp/all_train_mlp_ratio.png).

Finally, the source distribution and weighted target distribution (with real and approximated ratios) are shown next.


![1D_tranf](https://github.com/jgpavez/transfer_learning/blob/master/plots/mlp/all_transf_mlp_hist.png)

It can be seen that the weighted target distribution using approximated density ratios is pretty similar to the source distribution.

Lets study now how the method works on multivariate data. We define two 10-dim gaussian distributions. In the next image, the distributions of four of the 
ten features are shown.

![10D_dist](https://github.com/jgpavez/transfer_learning/blob/master/plots/xgboost/distributions.png)

We train a *boosted decision tree* to classify between data generated from the distributions. The score distribution for background and signal are shown next.

![10D_score](https://github.com/jgpavez/transfer_learning/blob/master/plots/xgboost/full_all_xgboost_hist.png)

The source distribution and weighted target distribution using approximated ratios, for the previously shown 4 features area computed.


 feature 0                   | feature 1
:-------------------------:|:-------------------------:
<img src="https://github.com/jgpavez/transfer_learning/blob/master/plots/xgboost/all_transf_xgboost_hist_v0.png" width="350">  | <img src="https://github.com/jgpavez/transfer_learning/blob/master/plots/xgboost/all_transf_xgboost_hist_v1.png" width="350" >
 feature 7                   | feature 8
<img src="https://github.com/jgpavez/transfer_learning/blob/master/plots/xgboost/all_transf_xgboost_hist_v7.png" width="350">  | <img src="https://github.com/jgpavez/transfer_learning/blob/master/plots/xgboost/all_transf_xgboost_hist_v8.png" width="350" >

Again, the distributions for each of the features are pretty similar with the weighted distributions.
