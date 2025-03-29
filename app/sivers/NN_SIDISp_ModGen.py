import os
import tensorflow as tf
import pandas as pd
import numpy as np
import lhapdf
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import csv
import sys
#import keras_tuner as kt

save_path = './sidis_models/replicas'
#save_path_rest = '/project/UVA-Spin/Ishara/Sivers/PseudoDataNNFit_300rep_LR_00002/Models_Out/rep'

Org_Data_path = './data/'
HERMES2009 = 'HERMES_p_2009.csv'
HERMES2020 = 'HERMES_p_2020.csv'
COMPASS2009 = 'COMPASS_d_2009.csv'
COMPASS2015 = 'COMPASS_p_2015.csv'

#SIDIS_Repl_Folder = '/project/UVA-Spin/Ishara/Sivers/PseudoDataNNFit_300rep_LR_0000110/Replica_Data'
Losses_Folder = './SIDIS_losses'

herm09 = pd.read_csv(Org_Data_path + HERMES2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
herm20 = pd.read_csv(Org_Data_path + HERMES2020).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp09 = pd.read_csv(Org_Data_path + COMPASS2009).dropna(axis=0, how='all').dropna(axis=1, how='all')
comp15 = pd.read_csv(Org_Data_path + COMPASS2015).dropna(axis=0, how='all').dropna(axis=1, how='all')

#df = pd.concat([herm09, herm20, comp09, comp15])
#df = pd.concat([herm09, comp09, comp15])
df = pd.concat([herm09, herm20, comp15])

Hidden_Layers=7
Nodes_per_HL=550
# Learning_Rate = 0.00001
Learning_Rate = 0.00005
EPOCHS = 400
# EPOCHS = 500
L1_reg = 10**(-12)
BATCHSIZE = 300
ACTIVATION = 'relu6'
modify_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=200,mode='auto')
EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, restore_best_weights = "False", patience=1000)
uncert_frac = 1

def create_nn_model(name, hidden_layers=Hidden_Layers, width=Nodes_per_HL, activation=ACTIVATION):
    inp = tf.keras.Input(shape=(1))
    initializer = tf.keras.initializers.RandomUniform(minval=-0.1,maxval=0.1,seed=None)
    x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg), activity_regularizer=tf.keras.regularizers.L1(L1_reg))(inp)
    for i in range(hidden_layers-1):
        x = tf.keras.layers.Dense(width, activation=activation, kernel_initializer = initializer, kernel_regularizer=tf.keras.regularizers.L1(L1_reg), activity_regularizer=tf.keras.regularizers.L1(L1_reg))(x)
    nnout = tf.keras.layers.Dense(1, kernel_initializer = initializer)(x)
    mod = tf.keras.Model(inp, nnout, name=name)
    return mod

def chisquare(y, yhat, err):
    return np.sum(((y - yhat)/err)**2)

        
class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(5., name='m1')
        #self.m1 = tf.constant(1.)
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.e = tf.constant(1.)
        
    def get_config(self):
        config = super().get_config().copy()
        
        config.update({
            'kperp2avg': self.kperp2avg,
            'pperp2avg': self.pperp2avg
        })
        
        return config
        
    def call(self, inputs):
        z = inputs[:, 0]
        pht = inputs[:, 1]
        ks2avg = (self.kperp2avg*self.m1**2)/(self.m1**2 + self.kperp2avg)
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * ks2avg**2
        bottomfirst = (z**2 * ks2avg + self.pperp2avg)**2 * self.kperp2avg 
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg)
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg)
        last = tf.sqrt(2*self.e) * z * pht / self.m1     
        return (topfirst/bottomfirst) * tf.exp(-exptop/expbottom) * last
    
    
class Quotient(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Quotient, self).__init__()
    
    def call(self, inputs):
        if len(inputs) != 2 or inputs[0].shape[1] != 1:
            raise Exception('must be two tensors of shape (?, 1)')
        return inputs[0]/inputs[1]

    

def SIDIS_Model():
    x = tf.keras.Input(shape=(1), name='x')
    z = tf.keras.Input(shape=(1), name='z')
    phT = tf.keras.Input(shape=(1), name='phT')
    uexpr = tf.keras.Input(shape=(1), name='uexpr')
    ubarexpr = tf.keras.Input(shape=(1), name='ubarexpr')
    dexpr = tf.keras.Input(shape=(1), name='dexpr')
    dbarexpr = tf.keras.Input(shape=(1), name='dbarexpr')
    sexpr = tf.keras.Input(shape=(1), name='sexpr')
    sbarexpr = tf.keras.Input(shape=(1), name='sbarexpr')

    NNs=[]
    for i in ['nnu','nnubar','nnd','nndbar','nns','nnsbar']:
        NNs.append(create_nn_model(i)(x))


    nncomb = tf.keras.layers.Concatenate()(NNs)
    denominator = tf.keras.layers.Add()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    exprcomb = tf.keras.layers.Concatenate()([uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr])
    numerator = tf.keras.layers.Dot(axes=1)([nncomb, exprcomb])
    quo = Quotient()([numerator, denominator])
    zpht = tf.keras.layers.Concatenate()([z, phT])
    a0 = A0()(zpht)
    siv = tf.keras.layers.Multiply()([a0, quo])
    return tf.keras.Model([x, z, phT, uexpr, ubarexpr, dexpr, dbarexpr, sexpr, sbarexpr],
                         siv)


       
class DataSIDIS(object):
    # def __init__(self, pdfset='NNPDF40_nlo_as_01180',
    #              ff_PIp='MAPFF10NLOPIp', ff_PIm='MAPFF10NLOPIm', ff_PIsum='MAPFF10NLOPIsum',
    #              ff_KAp='MAPFF10NLOKAp', ff_KAm='MAPFF10NLOKAm'):
    def __init__(self, pdfset='NNPDF40_nlo_as_01180',
                 ff_PIp='DSS14_NLO_Pip', ff_PIm='DSS14_NLO_Pim', ff_PIsum='DSS14_NLO_PiSum',
                 ff_KAp='DSS17_NLO_KaonPlus', ff_KAm='DSS17_NLO_KaonMinus'):
    # def __init__(self, pdfset='NNPDF40_nlo_as_01180',
    #              ff_PIp='NNFF10_PIp_nlo', ff_PIm='NNFF10_PIm_nlo', ff_PIsum='NNFF10_PIsum_nlo',
    #              ff_KAp='NNFF10_KAp_nlo', ff_KAm='NNFF10_KAm_nlo'):
    # def __init__(self, pdfset='cteq61',
    #              ff_PIp='DSS14_NLO_Pip', ff_PIm='DSS14_NLO_Pim', ff_PIsum='DSS14_NLO_PiSum',
    #              ff_KAp='DSS17_NLO_KaonPlus', ff_KAm='DSS17_NLO_KaonMinus'):
        '''
        Get data in proper format for neural network
        '''
        self.pdfData = lhapdf.mkPDF(pdfset)
        self.ffDataPIp = lhapdf.mkPDF(ff_PIp, 0)
        self.ffDataPIm = lhapdf.mkPDF(ff_PIm, 0)
        self.ffDataPIsum = lhapdf.mkPDF(ff_PIsum, 0)
        self.ffDataKAp = lhapdf.mkPDF(ff_KAp, 0)
        self.ffDataKAm = lhapdf.mkPDF(ff_KAm, 0)
        # needs to be extended to generalize for kaons
        self.eu = 2/3
        self.eubar = -2/3
        self.ed = -1/3
        self.edbar = 1/3
        self.es = -1/3
        self.esbar = 1/3

        self.ffDict = {0: self.ffDataPIp,
                       1: self.ffDataPIm,
                       2: self.ffDataPIsum,
                       3: self.ffDataKAp,
                       4: self.ffDataKAm}

    def pdf(self, flavor, x, QQ):
        return np.array([self.pdfData.xfxQ2(flavor, ax, qq) for ax, qq in zip(x, QQ)])

    
    def ff(self, func, flavor, z, QQ):
        return np.array([func.xfxQ2(flavor, az, qq) for az, qq in zip(z, QQ)])

    
    def makeData(self, df, hadrons, dependencies):

        data = {'x': [],
             'z': [],
             'phT': [],
             'uexpr': [],
             'ubarexpr': [],
             'dexpr': [],
             'dbarexpr': [],
             'sexpr': [],
             'sbarexpr': []}
        
        kins = {'hadron':[],
               'Q2': [],
               'x': [],
               'y': [],
               'z': [],
               'phT': [],
               '1D_dependence': []}

        y = []
        err = []
        
        hads = []
        QQs =[]
        yy = []
        deps = []

        df = df.loc[df['hadron'].isin(hadrons), :]
        df = df.loc[df['1D_dependence'].isin(dependencies), :]
        #X = np.array(df[['x', 'z', 'phT', 'Q2', 'hadron']])
        for i, had in enumerate(['pi+', 'pi-', 'pi0', 'k+', 'k-']):
            sliced = df.loc[df['hadron'] == had, :]
            y += list(sliced['Siv'])
            err += list(sliced['tot_err'])

            x = sliced['x']
            z = sliced['z']
            QQ = sliced['Q2']
            hads = sliced['hadron']
            QQs = sliced['Q2']
            #yy = sliced['y']
            phTs = sliced['phT']
            deps = sliced['1D_dependence']
            
            data['uexpr'] += list(self.eu**2 * self.pdf(2, x, QQ) * self.ff(self.ffDict[i], 2, z, QQ))
            data['ubarexpr'] += list(self.eubar**2 * self.pdf(-2, x, QQ) * self.ff(self.ffDict[i], -2, z, QQ))
            data['dexpr'] += list(self.ed**2 * self.pdf(1, x, QQ) * self.ff(self.ffDict[i], 1, z, QQ))
            data['dbarexpr'] += list(self.edbar**2 * self.pdf(-1, x, QQ) * self.ff(self.ffDict[i], -1, z, QQ))
            data['sexpr'] += list(self.es**2 * self.pdf(3, x, QQ) * self.ff(self.ffDict[i], 3, z, QQ))
            data['sbarexpr'] += list(self.esbar**2 * self.pdf(-3, x, QQ) * self.ff(self.ffDict[i], -3, z, QQ))
            data['x'] += list(x)
            data['z'] += list(z)
            data['phT'] += list(phTs)
            
            kins['hadron']+= list(hads)
            kins['Q2']+= list(QQs)
            kins['x']+= list(x)
            #kins['y']+= list(yy)
            kins['z']+= list(z)
            kins['phT']+= list(phTs)
            kins['1D_dependence']+= list(deps)


        for key in data.keys():
            data[key] = np.array(data[key])
            
        for key in kins.keys():
            kins[key] = np.array(kins[key])

        return kins, data, data[dependencies[0]], np.array(y), np.array(err)
    

SIDISdataann = DataSIDIS()

def GenSIDISReplicaData(datasetdf):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'z': [],
                      'phT': [],
                      'Siv':[],
                      'tot_err': [],
                      '1D_dependence': []}
    TempYhat = np.array(datasetdf['Siv'])
    Yerr = np.array(datasetdf['tot_err'])
    Yhat = np.random.normal(TempYhat, uncert_frac*Yerr)
    data_dictionary['hadron']= np.array(datasetdf['hadron'])
    data_dictionary['Q2']= np.array(datasetdf['Q2'])
    data_dictionary['x']= np.array(datasetdf['x'])
    #data_dictionary['y']= np.array(datasetdf['y'])
    data_dictionary['z']= np.array(datasetdf['z'])
    data_dictionary['phT']= np.array(datasetdf['phT'])
    data_dictionary['1D_dependence']= np.array(datasetdf['1D_dependence'])
    data_dictionary['Siv']= np.array(Yhat)
    data_dictionary['tot_err']= Yerr
    return pd.DataFrame(data_dictionary)
    

#################################################################################
###################  Viewing the model points ###################################
#################################################################################

def calc_yhat(model, X):
    return model.predict(X)


def SiversAsym(df,model,hadron,dependence):
    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(df, [hadron], [dependence])
    results = list(calc_yhat(model, T_Xplt))
    #print(results)
    return np.array(results)

def GenReplicaResult(datasetdf, model):
    data_dictionary = {'hadron':[],
                      'Q2': [],
                      'x': [],
                      'z': [],
                      'phT': [],
                      'Siv':[],
                      'tot_err': [],
                      '1D_dependence': []}
    temp_hads = pd.unique(datasetdf['hadron'])
    temp_deps = pd.unique(datasetdf['1D_dependence'])
    SiversA = []
    SiversA_err = []
    for i in temp_hads:
        for j in temp_deps:
            T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(datasetdf, [str(i)], [str(j)])
            #T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = datann.makeData(datasetdf, ['pi+'], ['x'])
            results = SiversAsym(datasetdf, model,i, j)
            Yhat = results.flatten()
            data_dictionary['hadron']+= list(T_Kins['hadron'])
            data_dictionary['Q2']+= list(T_Kins['Q2'])
            data_dictionary['x']+= list(T_Kins['x'])
            #data_dictionary['y']+= list(T_Kins['y'])
            data_dictionary['z']+= list(T_Kins['z'])
            data_dictionary['phT']+= list(T_Kins['phT'])
            data_dictionary['1D_dependence']+= list(T_Kins['1D_dependence'])
            #data_dictionary['Siv']+= list(T_yplt)
            data_dictionary['tot_err']+= list(T_errplt)
            data_dictionary['Siv'] += list(Yhat)
            #print(Yhat)
    return pd.DataFrame(data_dictionary)
    
def plotSivAsymmBands(ModelResultdf, Replicadf, datasetdf, hadron, dependence):
    D_Kins, D_Xplt, D_DEP, D_yplt, D_errplt = SIDISdataann.makeData(datasetdf, [hadron], [dependence])
    R_Kins, R_Xplt, R_DEP, R_yplt, R_errplt = SIDISdataann.makeData(Replicadf, [hadron], [dependence])
    M_Kins, M_Xplt, M_DEP, M_yplt, M_errplt = SIDISdataann.makeData(ModelResultdf, [hadron], [dependence])
    plt.errorbar(D_DEP, D_yplt,yerr=D_errplt, fmt='bo',label='Data')
    plt.plot(R_DEP, R_yplt, 'ro',label='Replica')
    plt.plot(M_DEP, M_yplt, 'go',label='NN_model')
    plt.title('SIDIS Sivers '+str(hadron),fontsize=15)
    #plt.ylim([-0.001,0.001])
    plt.xlabel(str(dependence),fontsize=15)
    plt.legend(loc=2,fontsize=15,handlelength=3)

def Generate_Comparison_Plots(model,replica_number, HR09, HR20, CM09, CM15):
    herm09_r = GenReplicaResult(HR09, model)
    herm20_r = GenReplicaResult(HR20, model)
    comp09_r = GenReplicaResult(CM09, model)
    comp15_r = GenReplicaResult(CM15, model)
    #plotSivAsymmBands(herm09_r,herm09_gen,herm09,'pi+','x')
    #plt.savefig('test_hermes2009.pdf', format='pdf', bbox_inches='tight')
    fig1=plt.figure(1,figsize=(15,30))
    plt.suptitle('HERMES 2009')
    plt.subplot(5,3,1)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi+','x')
    plt.subplot(5,3,2)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi+','z')
    plt.subplot(5,3,3)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi+','phT')
    plt.subplot(5,3,4)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi-','x')
    plt.subplot(5,3,5)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi-','z')
    plt.subplot(5,3,6)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi-','phT')
    plt.subplot(5,3,7)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi0','x')
    plt.subplot(5,3,8)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi0','z')
    plt.subplot(5,3,9)
    plotSivAsymmBands(herm09_r,HR09,herm09,'pi0','phT')
    plt.subplot(5,3,10)
    plotSivAsymmBands(herm09_r,HR09,herm09,'k+','x')
    plt.subplot(5,3,11)
    plotSivAsymmBands(herm09_r,HR09,herm09,'k+','z')
    plt.subplot(5,3,12)
    plotSivAsymmBands(herm09_r,HR09,herm09,'k+','phT')
    plt.subplot(5,3,13)
    plotSivAsymmBands(herm09_r,HR09,herm09,'k-','x')
    plt.subplot(5,3,14)
    plotSivAsymmBands(herm09_r,HR09,herm09,'k-','z')
    plt.subplot(5,3,15)
    plotSivAsymmBands(herm09_r,HR09,herm09,'k-','phT')
    #### Here the user needs to define the plot title based on the data set ####
    #plt.savefig('HERMES09_'+str(LOSSFN)+'_'+str(EPOCHS)+'Ep_'+str(HL)+'H_'+str(NODES)+'N_'+str(LR)+'LR.pdf', format='pdf', bbox_inches='tight')
    #plt.savefig(str(Plots_Folder)+'/'+'HERMES09.pdf', format='pdf', bbox_inches='tight')
    
    fig2=plt.figure(2,figsize=(15,30))
    plt.suptitle('HERMES 2020')
    plt.subplot(5,3,1)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi+','x')
    plt.subplot(5,3,2)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi+','z')
    plt.subplot(5,3,3)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi+','phT')
    plt.subplot(5,3,4)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi-','x')
    plt.subplot(5,3,5)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi-','z')
    plt.subplot(5,3,6)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi-','phT')
    plt.subplot(5,3,7)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi0','x')
    plt.subplot(5,3,8)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi0','z')
    plt.subplot(5,3,9)
    plotSivAsymmBands(herm20_r,HR20,herm20,'pi0','phT')
    plt.subplot(5,3,10)
    plotSivAsymmBands(herm20_r,HR20,herm20,'k+','x')
    plt.subplot(5,3,11)
    plotSivAsymmBands(herm20_r,HR20,herm20,'k+','z')
    plt.subplot(5,3,12)
    plotSivAsymmBands(herm20_r,HR20,herm20,'k+','phT')
    plt.subplot(5,3,13)
    plotSivAsymmBands(herm20_r,HR20,herm20,'k-','x')
    plt.subplot(5,3,14)
    plotSivAsymmBands(herm20_r,HR20,herm20,'k-','z')
    plt.subplot(5,3,15)
    plotSivAsymmBands(herm20_r,HR20,herm20,'k-','phT')
    
    fig3=plt.figure(3,figsize=(15,30))
    plt.suptitle('COMPASS 2009')
    plt.subplot(4,3,1)
    plotSivAsymmBands(comp09_r,CM09,comp09,'pi+','x')
    plt.subplot(4,3,2)
    plotSivAsymmBands(comp09_r,CM09,comp09,'pi+','z')
    plt.subplot(4,3,3)
    plotSivAsymmBands(comp09_r,CM09,comp09,'pi+','phT')
    plt.subplot(4,3,4)
    plotSivAsymmBands(comp09_r,CM09,comp09,'pi-','x')
    plt.subplot(4,3,5)
    plotSivAsymmBands(comp09_r,CM09,comp09,'pi-','z')
    plt.subplot(4,3,6)
    plotSivAsymmBands(comp09_r,CM09,comp09,'pi-','phT')
    plt.subplot(4,3,7)
    plotSivAsymmBands(comp09_r,CM09,comp09,'k+','x')
    plt.subplot(4,3,8)
    plotSivAsymmBands(comp09_r,CM09,comp09,'k+','z')
    plt.subplot(4,3,9)
    plotSivAsymmBands(comp09_r,CM09,comp09,'k+','phT')
    plt.subplot(4,3,10)
    plotSivAsymmBands(comp09_r,CM09,comp09,'k-','x')
    plt.subplot(4,3,11)
    plotSivAsymmBands(comp09_r,CM09,comp09,'k-','z')
    plt.subplot(4,3,12)
    plotSivAsymmBands(comp09_r,CM09,comp09,'k-','phT')
    
    fig4=plt.figure(4,figsize=(15,30))
    plt.suptitle('COMPASS 2015')
    plt.subplot(4,3,1)
    plotSivAsymmBands(comp15_r,CM15,comp15,'pi+','x')
    plt.subplot(4,3,2)
    plotSivAsymmBands(comp15_r,CM15,comp15,'pi+','z')
    plt.subplot(4,3,3)
    plotSivAsymmBands(comp15_r,CM15,comp15,'pi+','phT')
    plt.subplot(4,3,4)
    plotSivAsymmBands(comp15_r,CM15,comp15,'pi-','x')
    plt.subplot(4,3,5)
    plotSivAsymmBands(comp15_r,CM15,comp15,'pi-','z')
    plt.subplot(4,3,6)
    plotSivAsymmBands(comp15_r,CM15,comp15,'pi-','phT')
    plt.subplot(4,3,7)
    plotSivAsymmBands(comp15_r,CM15,comp15,'k+','x')
    plt.subplot(4,3,8)
    plotSivAsymmBands(comp15_r,CM15,comp15,'k+','z')
    plt.subplot(4,3,9)
    plotSivAsymmBands(comp15_r,CM15,comp15,'k+','phT')
    plt.subplot(4,3,10)
    plotSivAsymmBands(comp15_r,CM15,comp15,'k-','x')
    plt.subplot(4,3,11)
    plotSivAsymmBands(comp15_r,CM15,comp15,'k-','z')
    plt.subplot(4,3,12)
    plotSivAsymmBands(comp15_r,CM15,comp15,'k-','phT')


def trn_tst(X, y, err, split=0.0):
    tstidxs = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = {k: v[tstidxs] for k, v in X.items()}
    trn_X = {k: np.delete(v, tstidxs) for k, v in X.items()}
    
    tst_y = y[tstidxs]
    trn_y = np.delete(y, tstidxs)
    
    tst_err = err[tstidxs]
    trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y, trn_err, tst_err

def run_replica():
    replica_number = sys.argv[1]
    #replica_number = 9998 #This is for testing purposes
    herm09_gen = GenSIDISReplicaData(herm09)
    herm20_gen = GenSIDISReplicaData(herm20)
    comp09_gen = GenSIDISReplicaData(comp09)
    comp15_gen = GenSIDISReplicaData(comp15)
    #tempdf = pd.concat([herm09_gen, herm20_gen, comp09_gen, comp15_gen])
    tempdf = pd.concat([herm09_gen, herm20_gen, comp15_gen])
    #print("Here printing pseudo data")
    #tempdf.to_csv(str(SIDIS_Repl_Folder)+'/rep_'+str(replica_number)+'.csv')

    T_Kins, T_Xplt, T_DEP, T_yplt, T_errplt = SIDISdataann.makeData(tempdf, ['pi+', 'pi-', 'pi0', 'k+', 'k-'], ['x', 'z', 'phT'])
    
    trn_X, tst_X, trn_y, tst_y, trn_err, tst_err = trn_tst(T_Xplt, T_yplt, T_errplt)

    sivModel = SIDIS_Model()
    sivModel.compile(
        optimizer = tf.keras.optimizers.Adam(Learning_Rate),
        loss = tf.keras.losses.MeanSquaredError()
        )
    
    #, callbacks=[modify_LR,EarlyStop],EarlyStop
    #history = sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=300, verbose=2)
    # history = sivModel.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=BATCHSIZE, verbose=2)
    
    history = sivModel.fit(trn_X, trn_y, validation_data=(tst_X, tst_y), epochs=EPOCHS, callbacks=[modify_LR], batch_size=BATCHSIZE, verbose=2, shuffle=True)
    
    #sivModel.fit(trn_X, trn_y, sample_weight=(1/trn_err**2), validation_data=(tst_X, tst_y), epochs=EPOCHS, verbose=2)
    #sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')
    #Generate_Comparison_Plots(sivModel,replica_number, herm09_gen, herm20_gen, comp09_gen, comp15_gen

    sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')
    ###########################  This section is to test for investigating outliers ###############
    #if ( (history.history['loss'][-1] < 0.0016) and (history.history['val_loss'][-1] < 0.0016) ):
    #    sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')
    # if ( (history.history['loss'][-1] < 0.0015) and (history.history['val_loss'][-1] < 0.00225) ):
    #     sivModel.save(save_path + str(replica_number) + '.h5', save_format='h5')
    # else:
    #     sivModel.save(save_path_rest + str(replica_number) + '.h5', save_format='h5')
    #     Generate_Comparison_Plots(sivModel,replica_number)
    ###############################################################################################

    tempdfF = pd.DataFrame()
    tempdfF["Train_Loss"] = history.history['loss']
    tempdfF.to_csv(str(Losses_Folder)+'/reploss_'+str(replica_number)+'.csv')
    plt.figure(5)
    plt.plot(history.history['loss'],'-', color = 'blue', label = 'Training Loss')
    plt.ylabel('Loss',fontsize=11)
    plt.xlabel('Epoch',fontsize=11)
    plt.legend(loc=1,fontsize=15,handlelength=3)
    plt.savefig(str(Losses_Folder)+'/Plots'+'/losses'+str(replica_number)+'.pdf')

run_replica()