import numpy as np
import tensorflow as tf
import lhapdf
import matplotlib.pyplot as plt

pdfset = 'NNPDF40_nlo_as_01180'

# (): Make a numerical PDF based on the `pdfset`
pdfData = lhapdf.mkPDF(pdfset)

Model_Path = '../models/rep150.h5'

#############################################################################
######################## Defintions of the SIDIS model ######################

class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(.5, name='m1')
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
        ks2avg = (self.kperp2avg * (self.m1**2)) / (self.m1**2 + self.kperp2avg)
        topfirst = (z**2 * self.kperp2avg + self.pperp2avg) * ks2avg**2
        bottomfirst = (z**2 * ks2avg + self.pperp2avg)**2 * self.kperp2avg
        exptop = pht**2 * z**2 * (ks2avg - self.kperp2avg)
        expbottom = (z**2 * ks2avg + self.pperp2avg) * (z**2 * self.kperp2avg + self.pperp2avg)
        last = tf.sqrt(2 * self.e) * z * pht / self.m1
        return (topfirst / bottomfirst) * tf.exp(-exptop / expbottom) * last


class Quotient(tf.keras.layers.Layer):
    def call(self, inputs):
        if len(inputs) != 2 or inputs[0].shape[1] != 1:
            raise ValueError('Inputs must be two tensors of shape (?, 1)')
        return inputs[0] / inputs[1]

def nnq(model, x, hadronstr):
    if not hadronstr in ['nnu', 'nnd', 'nns', 'nnubar', 'nndbar', 'nnsbar']:
        raise Exception('hadronstr must be one of nnu, nnd, nns, nnubar, nndbar, nnsbar')
    
    print("fafa")
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    
    print('asd')
    return mod_out(x)

def h(model, kperp):
    m1 = model.get_layer('a0').m1.numpy()
    e = model.get_layer('a0').e.numpy()
    return tf.sqrt(2*e) * (kperp/m1) * tf.math.exp(-kperp**2/m1**2)

def pdf(flavor, x, QQ):

    # (1): `.xfxQ2(flavor, )` evaluates x*f(x, Q^{2}) for given quark flavor at a particular Q^{2}:
    return np.array([pdfData.xfxQ2(flavor, figure_axes_1, qq) for figure_axes_1, qq in zip(x, QQ)])


def fqp(x, QQ, kperp2avg, kperp, flavor):

    # (): Obtain the "f" function for a given quark ("q") flavor
    fq = pdf(flavor, x, QQ)
    return fq * (1. / (np.pi * kperp2avg)) * np.exp(-kperp**2 / kperp2avg)


def xsivdist(model, x, QQ, kperp2avg, flavor, kperp):
    refDict = {-3: 'nnsbar',
               -2: 'nnubar',
               -1: 'nndbar',
               1: 'nnd',
               2: 'nnu',
               3: 'nns'}
    nnqval = nnq(model, np.array([x]), refDict[flavor])
    #nnqval = nnq(model , np.array([x]), refDict[flavor])[:,0] np.array([x])
    hval = h(model, kperp)
    if(flavor == -3):
        fqpval = fqpsbar
    if(flavor == -2):
        fqpval = fqpubar
    if(flavor == -1):
        fqpval = fqpdbar
    if(flavor == 1):
        fqpval = fqpd
    if(flavor == 2):
        fqpval = fqpu
    if(flavor == 3):
        fqpval = fqps
    #fqpval = fqp([x], [QQ], kperp2avg, kperp, flavor)
    return ((2*nnqval*hval*fqpval)[0, :])
#############################################################################


## Here is where we need to create the kinematics grid ##
kperp_vals=np.array(list(range(150)))/100
kperp_vals=tf.constant(kperp_vals)

fqpu = fqp([0.1], [2.4], 0.57, kperp_vals, 2)

fqpd = fqp([0.1], [2.4], 0.57, kperp_vals, 1)

fqps = fqp([0.1], [2.4], 0.57, kperp_vals, 3)

fqpubar = fqp([0.1], [2.4], 0.57, kperp_vals, -2)

fqpdbar = fqp([0.1], [2.4], 0.57, kperp_vals, -1)

fqpsbar = fqp([0.1], [2.4], 0.57, kperp_vals, -3)


print("meat")

avg_model=tf.keras.models.load_model(str(Model_Path),custom_objects={'A0': A0, 'Quotient': Quotient})

print('cock')

## For Quarks
siv_u=xsivdist(avg_model, 0.1, 2.4, 0.57, 2, kperp_vals)
siv_d=xsivdist(avg_model, 0.1, 2.4, 0.57, 1, kperp_vals)
siv_s=xsivdist(avg_model, 0.1, 2.4, 0.57, 3, kperp_vals)

## For Anti-Quarks
siv_ubar=xsivdist(avg_model, 0.1, 2.4, 0.57, -2, kperp_vals)
siv_dbar=xsivdist(avg_model, 0.1, 2.4, 0.57, -1, kperp_vals)
siv_sbar=xsivdist(avg_model, 0.1, 2.4, 0.57, -3, kperp_vals)

plt.figure(1)
fontsize =10
figure, figure_axes_1 = plt.subplots(2, 3, sharey='row')

figure_axes_1[0,0].plot(kperp_vals, siv_u, 'b', label='$u$')
figure_axes_1[0,1].plot(kperp_vals, siv_d, 'r', label='$d$')
figure_axes_1[0,2].plot(kperp_vals, siv_s, 'g', label='$s$')
figure_axes_1[1,0].plot(kperp_vals, siv_ubar, 'b', label='$u$')
figure_axes_1[1,1].plot(kperp_vals, siv_dbar, 'r', label='$d$')
figure_axes_1[1,2].plot(kperp_vals, siv_sbar, 'g', label='$s$')


figure_axes_1[0,0].text(0.03, -0.11, '$Q^2=2.4$ GeV$^2$', fontsize=fontsize)
figure_axes_1[0,0].text(0.03, -0.135,'$x=0.1$', fontsize=fontsize)
figure_axes_1[0,1].text(0.03, -0.11, '$Q^2=2.4$ GeV$^2$', fontsize=fontsize)
figure_axes_1[0,1].text(0.03, -0.135,'$x=0.1$', fontsize=fontsize)
figure_axes_1[0,2].text(0.03, -0.11, '$Q^2=2.4$ GeV$^2$', fontsize=fontsize)
figure_axes_1[0,2].text(0.03, -0.135,'$x=0.1$', fontsize=fontsize)

figure_axes_1[0,0].set_ylabel(r'$x \Delta f^N (x,k_{\perp})$', fontsize=fontsize)

figure_axes_1[0,0].set_ylim(-0.15,0.1)

figure_axes_1[0,0].set_yticks(np.arange(-0.15,0.11,0.05))

figure_axes_1[0,0].tick_params(axis='y', labelsize=fontsize)


plt.savefig('Cross-check-Plot.pdf')

# Not sure what this does:
#plt.plot(kperp_vals,test_array1)