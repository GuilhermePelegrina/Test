" This is the ShapAUPRC script. We here provide an approach to explain the contribution"
" of features towards the area under the Precision-Recall curve "


" Importing packages "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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
    " Waterfall plot "
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
    attr_names_all = names[values_argsort+1].insert(0,names[0])
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

" Definig the threshold values, the region (rec values) under analysis and the selected ones for visualization "
step = 0.01
thresh = np.arange(0,1+step,step)
#cut_show = np.max(8,nAttr) # One may define the number of attributes to show in the plot
threshold_display = None # One may define the threshold used to display the feature contribution. It groups when the contribution is lower then the threshold - use 'None' to avoid this
 
" Defining and reading the dataset and scenarios (if it is the case) - Choose one of them"

" Red wine quality dataset "
#dataset = pd.read_csv('data_wine_quality_red.csv')
#X = dataset.drop('quality', axis=1)
#y = dataset['quality']
#y = (y>5)*1

# Scenarios
#X['dupl. variance'] = X['alcohol'] # Duplicating a feature
#X['random'] = np.random.normal(0, 1, X.shape[0]) # Including a (normally distributed) random feature

" Bank notes dataset "
dataset = pd.read_csv('banknotes.csv')
vals = dataset.values
X = dataset.loc[:, dataset.columns!='authentic']
y = vals[:,4]

# Scenarios
#X['dupl. variance'] = X['variance'] # Duplicating a feature
#X['random'] = np.random.normal(0, 1, X.shape[0]) # Including a (normally distributed) random feature

" Defining the number of simulations and the dispersion parameter"
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

AUPRC_explain = np.zeros((2**nAttr,nSimul)) # Contributions towards the AUPRC

for kk in range(nSimul):

    " Spliting the train and test"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    X_train, X_test = np.array(X_train), np.array(X_test)
    nSamp_train = X_train.shape[0] # Number of samples (train) and attributes
    nSamp_test = X_test.shape[0] # Number of samples (test)
    
    " PR curve Shapley-based decomposition "
    rec, pre = np.zeros((nCoal,len(thresh))), np.zeros((nCoal,len(thresh))) # pre and rec for all coalitions and thresholds
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
                if (tp+fp) > 0:
                    pre[i,jj] = tp/(tp+fp)
                else:
                    pre[i,jj] = 1
                    
                if (tp+fn) > 0:
                    rec[i,jj] = tp/(tp+fn)
                else:
                    pre[i,jj] = 1
            
            " Ordering the Recall and Precison according to the rec values (x-axis in the PR curve) and removing equal values of recs (we keep the maximum pres)"
            rec_ordered, rec_ordered_ind = np.sort(rec[i,:]), np.argsort(rec[i,:]) # Ordering the recs
            pre_ordered = pre[i,rec_ordered_ind] # Taking the pres ordered by the recs
            
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
            
            " Calculating the AUPRCs to be explained (and removing the random classifier) "
            AUPRC_explain_aux = pre_ordered - 0.5
            AUPRC_explain[i,kk] = trapz(AUPRC_explain_aux, rec_ordered)
    
    print(kk)
    
" Shapley values calculation "
shapley_AUPRC_aux = transf_matrix @ AUPRC_explain # Calculating all Shapley indices
shapley_AUPRC = shapley_AUPRC_aux[1:nAttr+1,:] # Taking only the Shapley values
    
" Plots "
# PR curve
plt.show()
plt.plot((0,1),(0.5,0.5),'k',rec_ordered,pre_ordered,'b')
plt.legend(['Random Classifier', 'PR curve'],fontsize=11)
plt.xlabel('Recall',fontsize=12)
plt.ylabel('Precision',fontsize=12)

# Contributions of features
attr_names = X.columns # Features names

attr_names = attr_names.insert(0, 'RC') # RC means Random Classifier
AUPRC_mean, AUPRC_std = 0.5*np.ones((nAttr+1,)), np.zeros((nAttr+1,))
AUPRC_mean[1:], AUPRC_std[1:] = np.mean(shapley_AUPRC,axis=1), ci_param*np.std(shapley_AUPRC,axis=1)

plot_waterfall(nAttr,AUPRC_mean,np.hstack([AUPRC_std,ci_param*np.std(sum(shapley_AUPRC,0))]),attr_names,'AUPRC')

" Save (if it is the case) "
#data_save = [attr_names,shapley_AUPRC,ci_param]
#np.save('results_interpretability_robustness_AUPRC_bank_gaussian_rand.npy', data_save, allow_pickle=True)
