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

" Definig the threshold values, the region (Recall values) under analysis and the selected ones for visualization "
step = 0.01
thresh = np.arange(0,1+step,step)
rec_values = thresh
rec_values_selec = (0.2,0.8)
#cut_show = np.max(8,nAttr) # One defines the number of attributes to show in the plot
threshold_display = None # One defines the threshold used to display the feature contribution. It groups when the contribution is lower then the threshold - use 'None' to avoid this
 
" Defining and reading the dataset - Choose one of them"

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
shapley_pre_optim = np.zeros((nAttr,len(thresh),nSimul))
shapley_pre_pess = np.zeros((nAttr,len(thresh),nSimul))
shapley_pre_med = np.zeros((nAttr,len(thresh),nSimul))

# AUC_PR
AUCPR_optim = np.zeros((nAttr+1,nSimul))
AUCPR_pess = np.zeros((nAttr+1,nSimul))
AUCPR_med = np.zeros((nAttr+1,nSimul))
AUCPR_optim[0,:], AUCPR_pess[0,:], AUCPR_med[0,:] = 0.5, 0.5, 0.5 # Including the area for the random classifier

# PRC explanation for each selected Recalls
PRC_explain_optim = np.zeros((nAttr+1,len(rec_values_selec),nSimul))
PRC_explain_pess = np.zeros((nAttr+1,len(rec_values_selec),nSimul))
PRC_explain_med = np.zeros((nAttr+1,len(rec_values_selec),nSimul))

# Precisions and Recalls
rec_simul, pre_simul = np.zeros((2**nAttr,len(thresh),nSimul)), np.zeros((2**nAttr,len(thresh),nSimul))
pre_payoffs_optim_simul = np.zeros((len(thresh),nSimul))
pre_payoffs_pess_simul = np.zeros((len(thresh),nSimul))
pre_payoffs_med_simul = np.zeros((len(thresh),nSimul))

attr_names = X.columns

for kk in range(nSimul):

    " Spliting the train and test"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_test = np.array(X_train), np.array(X_test)
    nSamp_train = X_train.shape[0] # Number of samples (train) and attributes
    nSamp_test = X_test.shape[0] # Number of samples (test)
    
    " PRC Shapley-based decomposition "
    rec, pre = np.zeros((nCoal,len(thresh))), np.zeros((nCoal,len(thresh))) # Precision and Recall for all coalitions and thresholds
    pre_payoffs_optim = np.zeros((nCoal,len(rec_values))) # Precision for the Recall values under analysis, optimist ROC
    pre_payoffs_pess = np.zeros((nCoal,len(rec_values))) # Precision for the Recall values under analysis, pessimist ROC
    pre_payoffs_med = np.zeros((nCoal,len(rec_values))) # Precision for the Recall values under analysis, medium ROC
    
    pre[0,:], rec[0,:] = 0.5, np.flip(thresh) # When there is no feature, we have the random classifier
    pre_payoffs_optim[0,:] = 0.5 # Random classifier, optimist ROC
    pre_payoffs_pess[0,:] = 0.5 # Random classifier, pessimist ROC
    pre_payoffs_med[0,:] = 0.5 # Random classifier, medium ROC
    for i,s in enumerate(powerset(range(nAttr),nAttr)):
        s = list(s)
        if len(s) >= 1:
            X_train_aux = X_train[:,s]
            X_test_aux = X_test[:,s]
            
            " Fitting the model "
            model.fit(X_train_aux, y_train)
            
            " Calculating the precision and recall "
            for jj in range(len(thresh)):
                y_pred = (model.predict_proba(X_test_aux)[:, 1] > thresh[jj]).astype('float')
                tn = sum((y_pred == 0) * (y_test == 0))
                tp = sum((y_pred == 1) * (y_test == 1))
                fn = sum((y_pred == 0) * (y_test == 1))
                fp = sum((y_pred == 1) * (y_test == 0))
                if (tp+fp) > 0:
                    pre[i,jj] = tp/(tp+fp)
                else:
                    pre[i,jj] = 1
                    
                if (tp+fn) > 0:
                    rec[i,jj] = tp/(tp+fn)
                else:
                    pre[i,jj] = 1
            
            " Ordering the Recall and Precision according to the Recall values (x-axis in the ROC curve) and removing equal values of Recalls (we keep the maximum Precisions)"
            rec_ordered, rec_ordered_ind = np.sort(rec[i,:]), np.argsort(rec[i,:]) # Ordering the Recalls
            pre_ordered = pre[i,rec_ordered_ind] # Taking the Precisions ordered by the Recalls
            
            # If one considers the maximum Precision when we have equal Recalls, let this while uncommented
            aux_var = 0
            while aux_var < (len(rec_ordered) - 1):
                if rec_ordered[aux_var] == rec_ordered[aux_var+1]:
                    if pre_ordered[aux_var] >= pre_ordered[aux_var+1]:
                        rec_ordered = np.delete(rec_ordered,aux_var+1)
                        pre_ordered = np.delete(pre_ordered,aux_var+1)
                    else:
                        rec_ordered = np.delete(rec_ordered,aux_var)
                        pre_ordered = np.delete(pre_ordered,aux_var)
                else:
                    aux_var += 1
            
            for jj in range(len(rec_values)):
                nEqual_rec = pre[i,rec[i,:]==rec_values[jj]] # We evaluate if we have already a Recall for the considered Recall value under analysis
                
                if len(nEqual_rec) == 0: # If we do not have a Recall, we evaluate 3 strategies
                    index_aux = 1
                    rec_close, rec_close_ind = np.sort((rec_ordered - rec_values[jj])**2), np.argsort((rec_ordered - rec_values[jj])**2)
                    rec_close_ind1, rec_close_ind2 = rec_close_ind[0], rec_close_ind[1]
                    
                    if (rec_ordered[rec_close_ind1] - rec_values[jj]) > 0:
                        while (rec_ordered[rec_close_ind2] - rec_values[jj]) > 0:
                            index_aux += 1
                            rec_close_ind2 = rec_close_ind[index_aux]
                            
                        pre_payoffs_optim[i,jj] = pre_ordered[rec_close_ind1]
                        pre_payoffs_pess[i,jj] = pre_ordered[rec_close_ind2]
                        pre_payoffs_med[i,jj] = pre_ordered[rec_close_ind2]+(pre_ordered[rec_close_ind1]-pre_ordered[rec_close_ind2])*(rec_values[jj]-rec_ordered[rec_close_ind2])/(rec_ordered[rec_close_ind1]-rec_ordered[rec_close_ind2])
                    
                    else:
                        while (rec_ordered[rec_close_ind2] - rec_values[jj]) < 0:
                            index_aux += 1
                            rec_close_ind2 = rec_close_ind[index_aux]
                        
                        pre_payoffs_optim[i,jj] = pre_ordered[rec_close_ind2]
                        pre_payoffs_pess[i,jj] = pre_ordered[rec_close_ind1]
                        pre_payoffs_med[i,jj] = pre_ordered[rec_close_ind1]+(pre_ordered[rec_close_ind2]-pre_ordered[rec_close_ind1])*(rec_values[jj]-rec_ordered[rec_close_ind1])/(rec_ordered[rec_close_ind2]-rec_ordered[rec_close_ind1])
                
                else: # If we have a Recall, selec the maximum of the Precisions (as considered before for equal Recalls values). Otherwise, we evaluate 3 strategies
                    pre_payoffs_optim[i,jj] = nEqual_rec.max()
                    pre_payoffs_pess[i,jj] = nEqual_rec.max()
                    pre_payoffs_med[i,jj] = nEqual_rec.max()
                    
                    #pre_payoffs_optim[i,jj] = nEqual_rec.max()
                    #pre_payoffs_pess[i,jj] = nEqual_rec.min()
                    #pre_payoffs_med[i,jj] = nEqual_rec.mean()
    
    " Calculating the Precisions to be explained (and removing the random classifier) "            
    pre_explain_optim = pre_payoffs_optim - np.matlib.repmat(pre_payoffs_optim[0,:],2**nAttr,1) # Eliminating the random classifier (expected payoffs when we have no feature)
    pre_explain_pess = pre_payoffs_pess - np.matlib.repmat(pre_payoffs_pess[0,:],2**nAttr,1) # Eliminating the random classifier (expected payoffs when we have no feature)
    pre_explain_med = pre_payoffs_med - np.matlib.repmat(pre_payoffs_med[0,:],2**nAttr,1) # Eliminating the random classifier (expected payoffs when we have no feature)
    
    " Shapley values calculation "        
    shapley_pre_aux_optim = transf_matrix @ pre_explain_optim # Calculating all Shapley indices
    shapley_pre_optim[:,:,kk] = shapley_pre_aux_optim[1:nAttr+1] # Taking only the Shapley values
    
    shapley_pre_aux_pess = transf_matrix @ pre_explain_pess # Calculating all Shapley indices
    shapley_pre_pess[:,:,kk] = shapley_pre_aux_pess[1:nAttr+1] # Taking only the Shapley values
    
    shapley_pre_aux_med = transf_matrix @ pre_explain_med # Calculating all Shapley indices
    shapley_pre_med[:,:,kk] = shapley_pre_aux_med[1:nAttr+1] # Taking only the Shapley values
    
    for ii in range(nAttr):
        AUCPR_optim[ii+1,kk] = trapz(shapley_pre_optim[ii,:,kk], dx=step)
        AUCPR_pess[ii+1,kk] = trapz(shapley_pre_pess[ii,:,kk], dx=step)
        AUCPR_med[ii+1,kk] = trapz(shapley_pre_med[ii,:,kk], dx=step)

    for ii in range(len(rec_values_selec)):
        index_selec = np.where(np.abs(thresh-rec_values_selec[ii])<10**(-6))[0]
        PRC_explain_optim[1:,ii,kk] = np.squeeze(shapley_pre_optim[:,index_selec,kk])
        PRC_explain_pess[1:,ii,kk] = np.squeeze(shapley_pre_pess[:,index_selec,kk])
        PRC_explain_med[1:,ii,kk] = np.squeeze(shapley_pre_med[:,index_selec,kk])
    
    rec_simul[:,:,kk], pre_simul[:,:,kk] = rec, pre
    pre_payoffs_optim_simul[:,kk] = pre_payoffs_optim[-1,:]
    pre_payoffs_pess_simul[:,kk] = pre_payoffs_pess[-1,:]
    pre_payoffs_med_simul[:,kk] = pre_payoffs_med[-1,:]

    print(kk)
    
AUCPR_optim_mean, AUCPR_optim_std = np.mean(AUCPR_optim,axis=1), ci_param*np.std(AUCPR_optim,axis=1)
AUCPR_pess_mean, AUCPR_pess_std = np.mean(AUCPR_pess,axis=1), ci_param*np.std(AUCPR_pess,axis=1)
AUCPR_med_mean, AUCPR_med_std = np.mean(AUCPR_med,axis=1), ci_param*np.std(AUCPR_med,axis=1)

PRC_explain_optim_mean, PRC_explain_optim_std = np.mean(PRC_explain_optim,axis=2), ci_param*np.std(PRC_explain_optim,axis=2)
PRC_explain_pess_mean, PRC_explain_pess_std = np.mean(PRC_explain_pess,axis=2), ci_param*np.std(PRC_explain_pess,axis=2)
PRC_explain_med_mean, PRC_explain_med_std = np.mean(PRC_explain_med,axis=2), ci_param*np.std(PRC_explain_med,axis=2)

shapley_pre_optim_mean, shapley_pre_optim_std = np.mean(shapley_pre_optim,axis=2), ci_param*np.std(shapley_pre_optim,axis=2)
shapley_pre_pess_mean, shapley_pre_pess_std = np.mean(shapley_pre_pess,axis=2), ci_param*np.std(shapley_pre_pess,axis=2)
shapley_pre_med_mean, shapley_pre_med_std = np.mean(shapley_pre_med,axis=2), ci_param*np.std(shapley_pre_med,axis=2)

rec_mean, rec_std = np.mean(rec_simul,axis=2), ci_param*np.std(rec_simul,axis=2)
pre_mean, pre_std = np.mean(pre_simul,axis=2), ci_param*np.std(pre_simul,axis=2)

pre_payoffs_optim_mean, pre_payoffs_optim_std = np.mean(pre_payoffs_optim_simul,axis=1), ci_param*np.std(pre_payoffs_optim_simul,axis=1)
pre_payoffs_pess_mean, pre_payoffs_pess_std = np.mean(pre_payoffs_pess_simul,axis=1), ci_param*np.std(pre_payoffs_pess_simul,axis=1)
pre_payoffs_med_mean, pre_payoffs_med_std = np.mean(pre_payoffs_med_simul,axis=1), ci_param*np.std(pre_payoffs_med_simul,axis=1)

" Plots "
attr_names = attr_names.insert(0, 'RC') # RC means Random Classifier

# ROC curve approximation for each strategy
plt.show()
plt.plot(rec_mean[0,:],pre_mean[0,:],'k',rec_mean[-1,:],pre_mean[-1,:],'b')
plt.fill_between(rec_mean[-1,:], pre_mean[-1,:] - pre_std[-1,:], pre_mean[-1,:] + pre_std[-1,:],facecolor='b',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)'],fontsize=11)
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)

plt.show()
plt.plot(rec_mean[0,:],pre_mean[0,:],'k',rec_mean[-1,:],pre_mean[-1,:],'b',np.hstack([0,rec_values]),np.hstack([1,pre_payoffs_optim_mean]),'r')
plt.fill_between(rec_mean[-1,:], pre_mean[-1,:] - pre_std[-1,:], pre_mean[-1,:] + pre_std[-1,:],facecolor='b',alpha=0.3)
plt.fill_between(rec_values, pre_payoffs_optim_mean - pre_payoffs_optim_std, pre_payoffs_optim_mean + pre_payoffs_optim_std,facecolor='r',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)', 'ROC curve (from optimistic strategy'],fontsize=10.5)
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)

plt.show()
plt.plot(rec_mean[0,:],pre_mean[0,:],'k',rec_mean[-1,:],pre_mean[-1,:],'b',np.hstack([0,rec_values]),np.hstack([1,pre_payoffs_pess_mean]),'r')
plt.fill_between(rec_mean[-1,:], pre_mean[-1,:] - pre_std[-1,:], pre_mean[-1,:] + pre_std[-1,:],facecolor='b',alpha=0.3)
plt.fill_between(rec_values, pre_payoffs_pess_mean - pre_payoffs_pess_std, pre_payoffs_pess_mean + pre_payoffs_pess_std,facecolor='r',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)', 'ROC curve (from pessimistic strategy'],fontsize=10.5)
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)

plt.show()
plt.plot(rec_mean[0,:],pre_mean[0,:],'k',rec_mean[-1,:],pre_mean[-1,:],'b',np.hstack([0,rec_values]),np.hstack([1,pre_payoffs_med_mean]),'r')
plt.fill_between(rec_mean[-1,:], pre_mean[-1,:] - pre_std[-1,:], pre_mean[-1,:] + pre_std[-1,:],facecolor='b',alpha=0.3)
plt.fill_between(rec_values, pre_payoffs_med_mean - pre_payoffs_med_std, pre_payoffs_med_mean + pre_payoffs_med_std,facecolor='r',alpha=0.3)
plt.legend(['Random Classifier', 'ROC curve (from test data)', 'ROC curve (from interpolation strategy'],fontsize=10.5)
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)

# AUC waterfalls for each strategy
plot_waterfall(nAttr,AUCPR_optim_mean,np.hstack([AUCPR_optim_std,ci_param*np.std(sum(AUCPR_optim,0))]),attr_names,'AUC')
plot_waterfall(nAttr,AUCPR_pess_mean,np.hstack([AUCPR_pess_std,ci_param*np.std(sum(AUCPR_pess,0))]),attr_names,'AUC')
plot_waterfall(nAttr,AUCPR_med_mean,np.hstack([AUCPR_med_std,ci_param*np.std(sum(AUCPR_med,0))]),attr_names,'AUC')

# ROC waterfalls for each selected Recalls/Precisions and strategy
for ii in range(len(rec_values_selec)):
    PRC_explain_optim_mean[0,ii], PRC_explain_pess_mean[0,ii], PRC_explain_med_mean[0,ii] = 0.5, 0.5, 0.5 # Including the area for the random classifier
    
    plot_waterfall(nAttr,PRC_explain_optim_mean[:,ii],np.hstack([PRC_explain_optim_std[:,ii],ci_param*np.std(sum(PRC_explain_optim[:,ii,:],0))]),attr_names,'Precision')
    plot_waterfall(nAttr,PRC_explain_pess_mean[:,ii],np.hstack([PRC_explain_pess_std[:,ii],ci_param*np.std(sum(PRC_explain_pess[:,ii,:],0))]),attr_names,'Precision')
    plot_waterfall(nAttr,PRC_explain_med_mean[:,ii],np.hstack([PRC_explain_med_std[:,ii],ci_param*np.std(sum(PRC_explain_med[:,ii,:],0))]),attr_names,'Precision')
    
# Contribution of each attribute in the ROC curve
max_PRC_plot_optim, min_PRC_plot_optim = np.max(shapley_pre_optim_mean + shapley_pre_optim_std), np.min(shapley_pre_optim_mean - shapley_pre_optim_std)
max_PRC_plot_pess, min_PRC_plot_pess = np.max(shapley_pre_pess_mean + shapley_pre_pess_std), np.min(shapley_pre_pess_mean - shapley_pre_pess_std)
max_PRC_plot_med, min_PRC_plot_med = np.max(shapley_pre_med_mean + shapley_pre_med_std), np.min(shapley_pre_med_mean - shapley_pre_med_std)

colors = []

plt.show()
for ii in range(nAttr):
    plt.plot(rec_values,shapley_pre_optim_mean[ii,:])
plt.legend(attr_names[1:],fontsize=12)
plt.xlabel('Recall',fontsize=11)
plt.ylabel('Feature contribution on the ROC curve',fontsize=11)
plt.ylim([min_PRC_plot_optim, max_PRC_plot_optim])

plt.show()    
for ii in range(nAttr):
    plt.plot(rec_values,shapley_pre_pess_mean[ii,:])
plt.legend(attr_names[1:],fontsize=12)
plt.xlabel('Recall',fontsize=11)
plt.ylabel('Feature contribution on the ROC curve',fontsize=11)
plt.ylim([min_PRC_plot_pess, max_PRC_plot_pess])

plt.show()    
for ii in range(nAttr):
    line, = plt.plot(rec_values,shapley_pre_med_mean[ii,:])
    colors.append(line.get_color())
plt.legend(attr_names[1:],fontsize=12)
plt.xlabel('Recall',fontsize=11)
plt.ylabel('Feature contribution on the ROC curve',fontsize=11)
plt.ylim([min_PRC_plot_med, max_PRC_plot_med])

# Contribution of each attribute in the ROC curve (with intervals)
plt.show()
fig = plt.figure()
gs = fig.add_gridspec(nAttr, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for ii in range(nAttr):
    axs[ii].plot(rec_values,shapley_pre_optim_mean[ii,:], color=colors[ii], label=attr_names[ii+1])
    axs[ii].fill_between(rec_values, shapley_pre_optim_mean[ii,:] - shapley_pre_optim_std[ii,:], shapley_pre_optim_mean[ii,:] + shapley_pre_optim_std[ii,:],facecolor=colors[ii],alpha=0.3)
    axs[ii].legend(loc="upper right",fontsize=12)
    axs[ii].set_ylim([min_PRC_plot_optim, max_PRC_plot_optim])
    plt.xlabel('Recall',fontsize=14)
for ax in axs:
    ax.label_outer()
# set labels
plt.setp(axs[-1], xlabel='Recall')
fig.text(0.04, 0.5, 'Feature contribution on the ROC curve',fontsize=11, va='center', rotation='vertical')

plt.show()
fig = plt.figure()
gs = fig.add_gridspec(nAttr, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for ii in range(nAttr):
    axs[ii].plot(rec_values,shapley_pre_pess_mean[ii,:], color=colors[ii], label=attr_names[ii+1])
    axs[ii].fill_between(rec_values, shapley_pre_pess_mean[ii,:] - shapley_pre_pess_std[ii,:], shapley_pre_pess_mean[ii,:] + shapley_pre_pess_std[ii,:],facecolor=colors[ii],alpha=0.3)
    axs[ii].legend(loc="upper right",fontsize=12)
    axs[ii].set_ylim([min_PRC_plot_pess, max_PRC_plot_pess])
    plt.xlabel('Recall',fontsize=11)
for ax in axs:
    ax.label_outer()
# set labels
plt.setp(axs[-1], xlabel='Recall')
fig.text(0.04, 0.5, 'Feature contribution on the ROC curve',fontsize=11, va='center', rotation='vertical')

plt.show()
fig = plt.figure()
gs = fig.add_gridspec(nAttr, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
for ii in range(nAttr):
    axs[ii].plot(rec_values,shapley_pre_med_mean[ii,:], color=colors[ii], label=attr_names[ii+1])
    axs[ii].fill_between(rec_values, shapley_pre_med_mean[ii,:] - shapley_pre_med_std[ii,:], shapley_pre_med_mean[ii,:] + shapley_pre_med_std[ii,:],facecolor=colors[ii],alpha=0.3)
    axs[ii].legend(loc="upper right",fontsize=12)
    axs[ii].set_ylim([min_PRC_plot_med, max_PRC_plot_med])
    plt.xlabel('Recall',fontsize=11)
for ax in axs:
    ax.label_outer()
# set labels
plt.setp(axs[-1], xlabel='Recall')
fig.text(0.04, 0.5, 'Feature contribution on the ROC curve',fontsize=11, va='center', rotation='vertical')

" Save (if it is the case) "
#data_save = [attr_names,AUCPR_med,AUCPR_optim,AUCPR_pess,ci_param,rec_simul,rec_values,rec_values_selec,nAttr,nSimul,PRC_explain_med,PRC_explain_optim,PRC_explain_pess,shapley_pre_med,shapley_pre_optim,shapley_pre_pess,step,thresh,pre_payoffs_med_simul,pre_payoffs_optim_simul,pre_payoffs_pess_simul,pre_simul]
#np.save('results_interpretability_PRC_curve_bank_gaussian.npy', data_save, allow_pickle=True)
