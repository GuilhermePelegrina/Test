" Importing packages "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from math import comb
from scipy.special import bernoulli
from itertools import chain, combinations

import numpy.matlib

from numpy import trapz


def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

    
def powerset(iterable,nAttr):
    '''Return the powerset of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(nAttr+1))


def tr_shap2game(nAttr):
    '''Return the transformation matrix from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(nAttr) #Números de Bernoulli
    k_add_numb = nParam_kAdd(nAttr,nAttr)
    
    coalit = np.zeros((k_add_numb,nAttr))
    
    for i,s in enumerate(powerset(range(nAttr),nAttr)):
        s = list(s)
        coalit[i,s] = 1
        
    matrix_shap2game = np.zeros((k_add_numb,k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2,:]))
            aux3 = int(sum(coalit[i,:] * coalit[i2,:]))
            aux4 = 0
            for i3 in range(int(aux3+1)):
                aux4 += comb(aux3, i3) * nBern[aux2-i3]
            matrix_shap2game[i,i2] = aux4
    return matrix_shap2game

def plot_waterfall(nAttr,values,values_std,names,y_label):
    
    " Waterfall plots "
    values_argsort =  np.abs(values[1:]).argsort()[::-1]
    values_sort =  np.hstack([values[0],values[values_argsort+1]])
    increment = np.zeros((nAttr+2,))
    increment[0:nAttr+1] = values_sort
    increment[-1] = sum(increment)
    start_point = np.zeros((len(increment)))
    position = np.zeros((nAttr+2,))
    position[0] = increment[0]
    position[-1] = sum(increment[0:-1])
    for ii in range(len(increment)-2):
        start_point[ii+1] = start_point[ii] + increment[ii]
        position[ii+1] = position[ii] + increment[ii+1]

    increment, start_point, position, values, values_std = 100*increment, 100*start_point, 100*position, 100*values, 100*values_std
    attr_names_all = names[values_argsort+1].insert(0,attr_names[0])
    attr_names_all = attr_names_all.insert(len(attr_names_all),y_label)

    colors_bar = ["black"]
    for ii in increment[1:-1]:
        if ii >= 0:
            colors_bar.append("green")
        else:
            colors_bar.append("red")
    colors_bar.append("blue")

    width = 0.75
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    bar_plot = ax.bar(attr_names_all, increment, width,bottom=start_point, color=colors_bar, yerr=values_std, edgecolor = 'black', capsize = 7)
    plt.xticks(rotation=90, fontsize=13)
    ax.set_ylim([-5, max(110, max(position)+10)])
    ax.set_ylabel('Feature contribution on the {} (%)'.format(y_label),fontsize=13)
    ii = 0
    for rect in bar_plot:
        if ii == 0:
            plt.text(rect.get_x() + rect.get_width()/15., 0.5+position[ii],'%.2f' % increment[ii],ha='center', va='bottom')
        else:
            plt.text(rect.get_x() + rect.get_width()/15., 0.5+max(position[ii],position[ii-1]),'%.2f' % increment[ii],ha='center', va='bottom')
        ii += 1
    plt.show()

" Definig the threshold values, the region (FPR values) under analysis and the selected ones for visualization "
step = 0.01
thresh = np.arange(0,1+step,step)
fpr_values = thresh
fpr_values_selec = (0.2,0.8)
#cut_show = np.max(8,nAttr) # One defines the number of attributes to show in the plot
threshold_display = None # One defines the threshold used to display the feature contribution. It groups when the contribution is lower then the threshold - use 'None' to avoid this
 
" Defining and reading the dataset and scenarios (if it is the case) - Choose one of them"

" Rice (Commeo - 1 and Osmancik - 0) dataset "
#dataset = pd.read_excel('data_rice.xlsx')
#X = dataset.loc[:, dataset.columns!='Class']
#vals = dataset.values
#y = vals[:,-1]

# Scenarios
#X = dataset.loc[1380:, dataset.columns!='Class']
#y = vals[1380:,-1]

" Red wine quality dataset "
#dataset = pd.read_csv('data_wine_quality_red.csv')
#vals = dataset.values
#X = dataset.loc[:, dataset.columns!='quality']
#y = vals[:,-1]
#y = (y>5)*1

# Imbalanced data
#sort_y = y.argsort()
#X = dataset.loc[sort_y[0:830], dataset.columns!='quality']
#y = vals[sort_y[0:830],-1]
#y = (y>5)*1

# Scenarios
#X['dupl. variance'] = X['alcohol']
#X['random'] = np.random.normal(0, 1, X.shape[0])

" Bank notes dataset "
dataset = pd.read_csv('banknotes.csv')
vals = dataset.values
X = dataset.loc[:, dataset.columns!='authentic']
y = vals[:,4]

# Imbalanced data
#X = dataset.loc[0:850, dataset.columns!='authentic']
#y = vals[0:851,4]

# Scenarios
#X['dupl. variance'] = X['variance']
#X['random'] = np.random.normal(0, 1, X.shape[0])

" Defining the number of simulations and the dispersion parameter (if it is the case) "
nSimul = 3
ci_param = 1 # Confident interval parameter x +- ci_param*std

" Defining the machine learning model - Choose of of them "
#model = MLPClassifier()
model = GaussianNB()
#model = xgb.XGBClassifier()

" Scalling the train and test"
scaler = StandardScaler()
Xt = scaler.fit_transform(X)
X = pd.DataFrame(Xt, columns=X.columns)
nAttr = X.shape[1] # Number of samples (train) and attributes
nCoal = 2**nAttr # Number of coalitions
transf_matrix = np.linalg.inv(tr_shap2game(nAttr)) # Transformation matrix from game to Shapley domain

" Matrices of results "
# Shapley values
shapley_tpr_optim = np.zeros((nAttr,len(thresh),nSimul))
shapley_tpr_pess = np.zeros((nAttr,len(thresh),nSimul))
shapley_tpr_med = np.zeros((nAttr,len(thresh),nSimul))

# AUC
AUC_optim = np.zeros((nAttr+1,nSimul))
AUC_pess = np.zeros((nAttr+1,nSimul))
AUC_med = np.zeros((nAttr+1,nSimul))
AUC_optim[0,:], AUC_pess[0,:], AUC_med[0,:] = 0.5, 0.5, 0.5 # Including the area for the random classifier

# ROC explanation for each selected FPRs
ROC_explain_optim = np.zeros((nAttr+1,len(fpr_values_selec),nSimul))
ROC_explain_pess = np.zeros((nAttr+1,len(fpr_values_selec),nSimul))
ROC_explain_med = np.zeros((nAttr+1,len(fpr_values_selec),nSimul))

# FPRs and TPRs
fpr_simul, tpr_simul = np.zeros((2**nAttr,len(thresh),nSimul)), np.zeros((2**nAttr,len(thresh),nSimul))
tpr_payoffs_optim_simul = np.zeros((len(thresh),nSimul))
tpr_payoffs_pess_simul = np.zeros((len(thresh),nSimul))
tpr_payoffs_med_simul = np.zeros((len(thresh),nSimul))

attr_names = X.columns

for kk in range(nSimul):

    " Spliting the train and test"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_test = np.array(X_train), np.array(X_test)
    nSamp_train = X_train.shape[0] # Number of samples (train) and attributes
    nSamp_test = X_test.shape[0] # Number of samples (test)
    
    " ROC curve Shapley-based decomposition "
    fpr, tpr = np.zeros((nCoal,len(thresh))), np.zeros((nCoal,len(thresh))) # TPR and FPR for all coalitions and thresholds
    tpr_payoffs_optim = np.zeros((nCoal,len(fpr_values))) # TPR for the FPR values under analysis, optimist ROC
    tpr_payoffs_pess = np.zeros((nCoal,len(fpr_values))) # TPR for the FPR values under analysis, pessimist ROC
    tpr_payoffs_med = np.zeros((nCoal,len(fpr_values))) # TPR for the FPR values under analysis, medium ROC
    
    tpr[0,:], fpr[0,:] = np.flip(thresh), np.flip(thresh) # When there is no feature, we have the random classifier
    tpr_payoffs_optim[0,:] = fpr_values # Random classifier, optimist ROC
    tpr_payoffs_pess[0,:] = fpr_values # Random classifier, pessimist ROC
    tpr_payoffs_med[0,:] = fpr_values # Random classifier, medium ROC
    for i,s in enumerate(powerset(range(nAttr),nAttr)):
        s = list(s)
        if len(s) >= 1:
            X_train_aux = X_train[:,s]
            X_test_aux = X_test[:,s]
            
            " Fiting the ML model "
            model.fit(X_train_aux, y_train)
            
            " Calculating the true/false positive/negative rates "
            for jj in range(len(thresh)):
                y_pred = (model.predict_proba(X_test_aux)[:, 1] > thresh[jj]).astype('float')
                tn = sum((y_pred == 0) * (y_test == 0))
                tp = sum((y_pred == 1) * (y_test == 1))
                fn = sum((y_pred == 0) * (y_test == 1))
                fp = sum((y_pred == 1) * (y_test == 0))
                tpr[i,jj] = tp/(tp+fn)
                fpr[i,jj] = fp/(fp+tn)
            
            " Ordering the FPR and TPR according to the FPR values (x-axis in the ROC curve) and removing equal values of FPRs (we keep the maximum TPRs)"
            fpr_ordered, fpr_ordered_ind = np.sort(fpr[i,:]), np.argsort(fpr[i,:]) # Ordering the FPRs
            tpr_ordered = tpr[i,fpr_ordered_ind] # Taking the TPRs ordered by the FPRs
            
            # If one considers the maximum TPR when we have equal FPRs, let this while uncommented
            aux_var = 0
            while aux_var < (len(fpr_ordered) - 1):
                if fpr_ordered[aux_var] == fpr_ordered[aux_var+1]:
                    if tpr_ordered[aux_var] >= tpr_ordered[aux_var+1]:
                        fpr_ordered = np.delete(fpr_ordered,aux_var+1)
                        tpr_ordered = np.delete(tpr_ordered,aux_var+1)
                    else:
                        fpr_ordered = np.delete(fpr_ordered,aux_var)
                        tpr_ordered = np.delete(tpr_ordered,aux_var)
                else:
                    aux_var += 1
            
            for jj in range(len(fpr_values)):
                nEqual_fpr = tpr[i,fpr[i,:]==fpr_values[jj]] # We evaluate if we have already a FPR for the considered FPR value under analysis
                
                if len(nEqual_fpr) == 0: # If we do not have a FPR, we evaluate 3 strategies
                    index_aux = 1
                    fpr_close, fpr_close_ind = np.sort((fpr_ordered - fpr_values[jj])**2), np.argsort((fpr_ordered - fpr_values[jj])**2)
                    fpr_close_ind1, fpr_close_ind2 = fpr_close_ind[0], fpr_close_ind[1]
                    
                    if (fpr_ordered[fpr_close_ind1] - fpr_values[jj]) > 0:
                        while (fpr_ordered[fpr_close_ind2] - fpr_values[jj]) > 0:
                            index_aux += 1
                            fpr_close_ind2 = fpr_close_ind[index_aux]
                            
                        tpr_payoffs_optim[i,jj] = tpr_ordered[fpr_close_ind1]
                        tpr_payoffs_pess[i,jj] = tpr_ordered[fpr_close_ind2]
                        tpr_payoffs_med[i,jj] = tpr_ordered[fpr_close_ind2]+(tpr_ordered[fpr_close_ind1]-tpr_ordered[fpr_close_ind2])*(fpr_values[jj]-fpr_ordered[fpr_close_ind2])/(fpr_ordered[fpr_close_ind1]-fpr_ordered[fpr_close_ind2])
                    
                    else:
                        while (fpr_ordered[fpr_close_ind2] - fpr_values[jj]) < 0:
                            index_aux += 1
                            fpr_close_ind2 = fpr_close_ind[index_aux]
                        
                        tpr_payoffs_optim[i,jj] = tpr_ordered[fpr_close_ind2]
                        tpr_payoffs_pess[i,jj] = tpr_ordered[fpr_close_ind1]
                        tpr_payoffs_med[i,jj] = tpr_ordered[fpr_close_ind1]+(tpr_ordered[fpr_close_ind2]-tpr_ordered[fpr_close_ind1])*(fpr_values[jj]-fpr_ordered[fpr_close_ind1])/(fpr_ordered[fpr_close_ind2]-fpr_ordered[fpr_close_ind1])
                
                else: # If we have a FPR, selec the maximum of the TPRs (as considered before for equal FPRs values). Otherwise, we evaluate 3 strategies
                    tpr_payoffs_optim[i,jj] = nEqual_fpr.max()
                    tpr_payoffs_pess[i,jj] = nEqual_fpr.max()
                    tpr_payoffs_med[i,jj] = nEqual_fpr.max()
                    
                    #tpr_payoffs_optim[i,jj] = nEqual_fpr.max()
                    #tpr_payoffs_pess[i,jj] = nEqual_fpr.min()
                    #tpr_payoffs_med[i,jj] = nEqual_fpr.mean()
    
    " Calculating the TPRs to be explained (and removing the random classifier) "            
    tpr_explain_optim = tpr_payoffs_optim - np.matlib.repmat(tpr_payoffs_optim[0,:],2**nAttr,1) # Eliminating the random classifier (expected payoffs when we have no feature)
    tpr_explain_pess = tpr_payoffs_pess - np.matlib.repmat(tpr_payoffs_pess[0,:],2**nAttr,1) # Eliminating the random classifier (expected payoffs when we have no feature)
    tpr_explain_med = tpr_payoffs_med - np.matlib.repmat(tpr_payoffs_med[0,:],2**nAttr,1) # Eliminating the random classifier (expected payoffs when we have no feature)
    
    " Shapley values calculation "        
    shapley_tpr_aux_optim = transf_matrix @ tpr_explain_optim # Calculating all Shapley indices
    shapley_tpr_optim[:,:,kk] = shapley_tpr_aux_optim[1:nAttr+1] # Taking only the Shapley values
    
    shapley_tpr_aux_pess = transf_matrix @ tpr_explain_pess # Calculating all Shapley indices
    shapley_tpr_pess[:,:,kk] = shapley_tpr_aux_pess[1:nAttr+1] # Taking only the Shapley values
    
    shapley_tpr_aux_med = transf_matrix @ tpr_explain_med # Calculating all Shapley indices
    shapley_tpr_med[:,:,kk] = shapley_tpr_aux_med[1:nAttr+1] # Taking only the Shapley values
    
    for ii in range(nAttr):
        AUC_optim[ii+1,kk] = trapz(shapley_tpr_optim[ii,:,kk], dx=step)
        AUC_pess[ii+1,kk] = trapz(shapley_tpr_pess[ii,:,kk], dx=step)
        AUC_med[ii+1,kk] = trapz(shapley_tpr_med[ii,:,kk], dx=step)

    for ii in range(len(fpr_values_selec)):
        index_selec = np.where(np.abs(thresh-fpr_values_selec[ii])<10**(-6))[0]
        ROC_explain_optim[1:,ii,kk] = np.squeeze(shapley_tpr_optim[:,index_selec,kk])
        ROC_explain_pess[1:,ii,kk] = np.squeeze(shapley_tpr_pess[:,index_selec,kk])
        ROC_explain_med[1:,ii,kk] = np.squeeze(shapley_tpr_med[:,index_selec,kk])
    
    fpr_simul[:,:,kk], tpr_simul[:,:,kk] = fpr, tpr
    tpr_payoffs_optim_simul[:,kk] = tpr_payoffs_optim[-1,:]
    tpr_payoffs_pess_simul[:,kk] = tpr_payoffs_pess[-1,:]
    tpr_payoffs_med_simul[:,kk] = tpr_payoffs_med[-1,:]

    print(kk)
    
AUC_optim_mean, AUC_optim_std = np.mean(AUC_optim,axis=1), ci_param*np.std(AUC_optim,axis=1)
AUC_pess_mean, AUC_pess_std = np.mean(AUC_pess,axis=1), ci_param*np.std(AUC_pess,axis=1)
AUC_med_mean, AUC_med_std = np.mean(AUC_med,axis=1), ci_param*np.std(AUC_med,axis=1)

ROC_explain_optim_mean, ROC_explain_optim_std = np.mean(ROC_explain_optim,axis=2), ci_param*np.std(ROC_explain_optim,axis=2)
ROC_explain_pess_mean, ROC_explain_pess_std = np.mean(ROC_explain_pess,axis=2), ci_param*np.std(ROC_explain_pess,axis=2)
ROC_explain_med_mean, ROC_explain_med_std = np.mean(ROC_explain_med,axis=2), ci_param*np.std(ROC_explain_med,axis=2)

shapley_tpr_optim_mean, shapley_tpr_optim_std = np.mean(shapley_tpr_optim,axis=2), ci_param*np.std(shapley_tpr_optim,axis=2)
shapley_tpr_pess_mean, shapley_tpr_pess_std = np.mean(shapley_tpr_pess,axis=2), ci_param*np.std(shapley_tpr_pess,axis=2)
shapley_tpr_med_mean, shapley_tpr_med_std = np.mean(shapley_tpr_med,axis=2), ci_param*np.std(shapley_tpr_med,axis=2)

fpr_mean, fpr_std = np.mean(fpr_simul,axis=2), ci_param*np.std(fpr_simul,axis=2)
tpr_mean, tpr_std = np.mean(tpr_simul,axis=2), ci_param*np.std(tpr_simul,axis=2)

tpr_payoffs_optim_mean, tpr_payoffs_optim_std = np.mean(tpr_payoffs_optim_simul,axis=1), ci_param*np.std(tpr_payoffs_optim_simul,axis=1)
tpr_payoffs_pess_mean, tpr_payoffs_pess_std = np.mean(tpr_payoffs_pess_simul,axis=1), ci_param*np.std(tpr_payoffs_pess_simul,axis=1)
tpr_payoffs_med_mean, tpr_payoffs_med_std = np.mean(tpr_payoffs_med_simul,axis=1), ci_param*np.std(tpr_payoffs_med_simul,axis=1)

" Plots "
attr_names = attr_names.insert(0, 'RC') # RC means Random Classifier

# ROC curve approximation for each strategy
plt.show()
plt.plot(fpr_mean[0,:],tpr_mean[0,:],'k',fpr_mean[-1,:],tpr_mean[-1,:],'b')
plt.fill_between(fpr_mean[-1,:], tpr_mean[-1,:] - tpr_std[-1,:], tpr_mean[-1,:] + tpr_std[-1,:],facecolor='b',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)'],loc='lower right',fontsize=11)
plt.xlabel('FPR',fontsize=12)
plt.ylabel('TPR',fontsize=12)

plt.show()
plt.plot(fpr_mean[0,:],tpr_mean[0,:],'k',fpr_mean[-1,:],tpr_mean[-1,:],'b',np.hstack([0,fpr_values]),np.hstack([0,tpr_payoffs_optim_mean]),'r')
plt.fill_between(fpr_mean[-1,:], tpr_mean[-1,:] - tpr_std[-1,:], tpr_mean[-1,:] + tpr_std[-1,:],facecolor='b',alpha=0.3)
plt.fill_between(fpr_values, tpr_payoffs_optim_mean - tpr_payoffs_optim_std, tpr_payoffs_optim_mean + tpr_payoffs_optim_std,facecolor='r',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)', 'ROC curve (from optimistic strategy'],loc='lower right',fontsize=10.5)
plt.xlabel('FPR',fontsize=12)
plt.ylabel('TPR',fontsize=12)

plt.show()
plt.plot(fpr_mean[0,:],tpr_mean[0,:],'k',fpr_mean[-1,:],tpr_mean[-1,:],'b',np.hstack([0,fpr_values]),np.hstack([0,tpr_payoffs_pess_mean]),'r')
plt.fill_between(fpr_mean[-1,:], tpr_mean[-1,:] - tpr_std[-1,:], tpr_mean[-1,:] + tpr_std[-1,:],facecolor='b',alpha=0.3)
plt.fill_between(fpr_values, tpr_payoffs_pess_mean - tpr_payoffs_pess_std, tpr_payoffs_pess_mean + tpr_payoffs_pess_std,facecolor='r',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)', 'ROC curve (from pessimistic strategy'],loc='lower right',fontsize=10.5)
plt.xlabel('FPR',fontsize=12)
plt.ylabel('TPR',fontsize=12)

plt.show()
plt.plot(fpr_mean[0,:],tpr_mean[0,:],'k',fpr_mean[-1,:],tpr_mean[-1,:],'b',np.hstack([0,fpr_values]),np.hstack([0,tpr_payoffs_med_mean]),'r')
plt.fill_between(fpr_mean[-1,:], tpr_mean[-1,:] - tpr_std[-1,:], tpr_mean[-1,:] + tpr_std[-1,:],facecolor='b',alpha=0.3)
plt.fill_between(fpr_values, tpr_payoffs_med_mean - tpr_payoffs_med_std, tpr_payoffs_med_mean + tpr_payoffs_med_std,facecolor='r',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)', 'ROC curve (from interpolation strategy'],loc='lower right',fontsize=10.5)
plt.xlabel('FPR',fontsize=12)
plt.ylabel('TPR',fontsize=12)

# AUC waterfalls for each strategy
plot_waterfall(nAttr,AUC_optim_mean,np.hstack([AUC_optim_std,ci_param*np.std(sum(AUC_optim,0))]),attr_names,'AUC')
plot_waterfall(nAttr,AUC_pess_mean,np.hstack([AUC_pess_std,ci_param*np.std(sum(AUC_pess,0))]),attr_names,'AUC')
plot_waterfall(nAttr,AUC_med_mean,np.hstack([AUC_med_std,ci_param*np.std(sum(AUC_med,0))]),attr_names,'AUC')

# ROC waterfalls for each selected FPRs/TPRs and strategy
for ii in range(len(fpr_values_selec)):
    ROC_explain_optim_mean[0,ii], ROC_explain_pess_mean[0,ii], ROC_explain_med_mean[0,ii] = fpr_values_selec[ii], fpr_values_selec[ii], fpr_values_selec[ii] # Including the area for the random classifier
    
    plot_waterfall(nAttr,ROC_explain_optim_mean[:,ii],np.hstack([ROC_explain_optim_std[:,ii],ci_param*np.std(sum(ROC_explain_optim[:,ii,:],0))]),attr_names,'TPR')
    plot_waterfall(nAttr,ROC_explain_pess_mean[:,ii],np.hstack([ROC_explain_pess_std[:,ii],ci_param*np.std(sum(ROC_explain_pess[:,ii,:],0))]),attr_names,'TPR')
    plot_waterfall(nAttr,ROC_explain_med_mean[:,ii],np.hstack([ROC_explain_med_std[:,ii],ci_param*np.std(sum(ROC_explain_med[:,ii,:],0))]),attr_names,'TPR')
    
# Contribution of each attribute in the ROC curve
max_ROC_plot_optim, min_ROC_plot_optim = np.max(shapley_tpr_optim_mean + shapley_tpr_optim_std), np.min(shapley_tpr_optim_mean - shapley_tpr_optim_std)
max_ROC_plot_pess, min_ROC_plot_pess = np.max(shapley_tpr_pess_mean + shapley_tpr_pess_std), np.min(shapley_tpr_pess_mean - shapley_tpr_pess_std)
max_ROC_plot_med, min_ROC_plot_med = np.max(shapley_tpr_med_mean + shapley_tpr_med_std), np.min(shapley_tpr_med_mean - shapley_tpr_med_std)

colors = []

plt.show()
for ii in range(nAttr):
    plt.plot(fpr_values,shapley_tpr_optim_mean[ii,:])
plt.legend(attr_names[1:],fontsize=12)
plt.xlabel('FPR',fontsize=11)
plt.ylabel('Feature contribution on the ROC curve',fontsize=11)
plt.ylim([min_ROC_plot_optim, max_ROC_plot_optim])

plt.show()    
for ii in range(nAttr):
    plt.plot(fpr_values,shapley_tpr_pess_mean[ii,:])
plt.legend(attr_names[1:],fontsize=12)
plt.xlabel('FPR',fontsize=11)
plt.ylabel('Feature contribution on the ROC curve',fontsize=11)
plt.ylim([min_ROC_plot_pess, max_ROC_plot_pess])

plt.show()    
for ii in range(nAttr):
    line, = plt.plot(fpr_values,shapley_tpr_med_mean[ii,:])
    colors.append(line.get_color())
plt.legend(attr_names[1:],fontsize=12)
plt.xlabel('FPR',fontsize=11)
plt.ylabel('Feature contribution on the ROC curve',fontsize=11)
plt.ylim([min_ROC_plot_med, max_ROC_plot_med])

# Contribution of each attribute in the ROC curve (with intervals)
plt.show()
fig = plt.figure()
gs = fig.add_gridspec(nAttr, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for ii in range(nAttr):
    axs[ii].plot(fpr_values,shapley_tpr_optim_mean[ii,:], color=colors[ii], label=attr_names[ii+1])
    axs[ii].fill_between(fpr_values, shapley_tpr_optim_mean[ii,:] - shapley_tpr_optim_std[ii,:], shapley_tpr_optim_mean[ii,:] + shapley_tpr_optim_std[ii,:],facecolor=colors[ii],alpha=0.3)
    axs[ii].legend(loc="upper right",fontsize=12)
    axs[ii].set_ylim([min_ROC_plot_optim, max_ROC_plot_optim])
    plt.xlabel('FPR',fontsize=14)
for ax in axs:
    ax.label_outer()
# set labels
plt.setp(axs[-1], xlabel='FPR')
fig.text(0.04, 0.5, 'Feature contribution on the ROC curve',fontsize=11, va='center', rotation='vertical')

plt.show()
fig = plt.figure()
gs = fig.add_gridspec(nAttr, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for ii in range(nAttr):
    axs[ii].plot(fpr_values,shapley_tpr_pess_mean[ii,:], color=colors[ii], label=attr_names[ii+1])
    axs[ii].fill_between(fpr_values, shapley_tpr_pess_mean[ii,:] - shapley_tpr_pess_std[ii,:], shapley_tpr_pess_mean[ii,:] + shapley_tpr_pess_std[ii,:],facecolor=colors[ii],alpha=0.3)
    axs[ii].legend(loc="upper right",fontsize=12)
    axs[ii].set_ylim([min_ROC_plot_pess, max_ROC_plot_pess])
    plt.xlabel('FPR',fontsize=11)
for ax in axs:
    ax.label_outer()
# set labels
plt.setp(axs[-1], xlabel='FPR')
fig.text(0.04, 0.5, 'Feature contribution on the ROC curve',fontsize=11, va='center', rotation='vertical')

plt.show()
fig = plt.figure()
gs = fig.add_gridspec(nAttr, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for ii in range(nAttr):
    axs[ii].plot(fpr_values,shapley_tpr_med_mean[ii,:], color=colors[ii], label=attr_names[ii+1])
    axs[ii].fill_between(fpr_values, shapley_tpr_med_mean[ii,:] - shapley_tpr_med_std[ii,:], shapley_tpr_med_mean[ii,:] + shapley_tpr_med_std[ii,:],facecolor=colors[ii],alpha=0.3)
    axs[ii].legend(loc="upper right",fontsize=12)
    axs[ii].set_ylim([min_ROC_plot_med, max_ROC_plot_med])
    plt.xlabel('FPR',fontsize=11)
for ax in axs:
    ax.label_outer()
# set labels
plt.setp(axs[-1], xlabel='FPR')
fig.text(0.04, 0.5, 'Feature contribution on the ROC curve',fontsize=11, va='center', rotation='vertical')

" Save (if it is the case) "
#data_save = [attr_names,AUC_med,AUC_optim,AUC_pess,ci_param,fpr_simul,fpr_values,fpr_values_selec,nAttr,nSimul,ROC_explain_med,ROC_explain_optim,ROC_explain_pess,shapley_tpr_med,shapley_tpr_optim,shapley_tpr_pess,step,thresh,tpr_payoffs_med_simul,tpr_payoffs_optim_simul,tpr_payoffs_pess_simul,tpr_simul]
#np.save('results_interpretability_ROC_curve_bank_gaussian.npy', data_save, allow_pickle=True)
