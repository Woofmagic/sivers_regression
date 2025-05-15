import numpy as np
import tensorflow as tf
import lhapdf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pdfset = 'NNPDF40_nlo_as_01180'

# (): Make a numerical PDF based on the `pdfset`
pdfData = lhapdf.mkPDF(pdfset)

model_path = '../models/rep150.h5'

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
    
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                       model.get_layer(hadronstr).output)
    
    return mod_out(x)

def h(model, kperp):
    m1 = model.get_layer('a0').m1.numpy()
    e = model.get_layer('a0').e.numpy()
    return tf.sqrt(2*e) * (kperp/m1) * tf.math.exp(-kperp**2/m1**2)

def pdf(flavor, x, QQ):

    for x_value, qvalue in zip(x, QQ):
        print(f"x = {x_value}, q^2 = {qvalue}")

    # (1): `.xfxQ2(flavor, )` evaluates x*f(x, Q^{2}) for given quark flavor at a particular Q^{2}:
    return np.array([pdfData.xfxQ2(flavor, figure_axes_1, qq) for figure_axes_1, qq in zip(x, QQ)])

def fqp(x, QQ, kperp2avg, kperp, flavor):

    # (): Obtain the "f" function for a given quark ("q") flavor
    fq = pdf(flavor, x, QQ)
    print(f"x = {x}, {fq}")
    return fq * (1. / (np.pi * kperp2avg)) * np.exp(-kperp**2 / kperp2avg)

def xsivdist(model, x, QQ, kperp2avg, flavor, kperp):
    # x is passed in as a single number
    # kperp is also a single number...
    
    refDict = {-3: 'nnsbar',
               -2: 'nnubar',
               -1: 'nndbar',
               1: 'nnd',
               2: 'nnu',
               3: 'nns'}
    

    # this then returns what..
    nnqval = nnq(model, np.array([x]), refDict[flavor])
    
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

    # print(f"x = {x}, kperp = {kperp}, fqpval = {fqpval}")
    print(f"Shape of nnqval: {np.shape(nnqval)}")
    print(nnqval)
    print(f"Shape of hval: {np.shape(hval)}")
    print(hval)
    print(f"Shape of nnqval*hval: {np.shape(nnqval*hval)}")
    print(nnqval*hval)
    print(f"Shape of fqpval: {np.shape(fqpval)}")
    print(f"Shape of the entire product: {np.shape(2*nnqval*hval*fqpval)}")
    print(f"Shape of the entire product: {np.shape((2*nnqval*hval*fqpval)[0, :])}")

    # print(2*nnqval*hval*fqpval)
    # print(f"x = {x}, kperp = {kperp}, Total = {((2*nnqval*hval*fqpval)[0, :])}")
    return ((2*nnqval*hval*fqpval)[0, :])
#############################################################################

## Here is where we need to create the kinematics grid ##
kperp_vals = np.array(list(range(150)))/100
kperp_vals = tf.constant(kperp_vals)

fqpu = fqp([0.1], [2.4], 0.57, kperp_vals, 2)

fqpd = fqp([0.1], [2.4], 0.57, kperp_vals, 1)

fqps = fqp([0.1], [2.4], 0.57, kperp_vals, 3)

fqpubar = fqp([0.1], [2.4], 0.57, kperp_vals, -2)

fqpdbar = fqp([0.1], [2.4], 0.57, kperp_vals, -1)

fqpsbar = fqp([0.1], [2.4], 0.57, kperp_vals, -3)

avg_model = tf.keras.models.load_model(str(model_path),custom_objects={'A0': A0, 'Quotient': Quotient})

# Define ranges for x and k_perp
x_vals = np.linspace(0.01, 0.3, 30)  # Adjust as needed
kperp_vals = np.linspace(0, 2, 30)  # Adjust range based on your data

# Create a meshgrid
X, K = np.meshgrid(x_vals, kperp_vals)

# Function to apply xsivdist element-wise
def compute_siv(flavor):
    # Initialize an empty array with the correct shape
    siv_values = np.zeros_like(X)
    
    # Compute values over the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            siv_values[i, j] = xsivdist(avg_model, X[i, j], 2.4, 0.57, flavor, K[i, j])[0]
    
    return siv_values

# Compute distributions for quarks
siv_u = compute_siv(2)
print("DONE MOTHAFUCKA")
siv_d = compute_siv(1)
siv_s = compute_siv(3)

# Compute distributions for anti-quarks
siv_ubar = compute_siv(-2)
siv_dbar = compute_siv(-1)
siv_sbar = compute_siv(-3)

fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={'projection': '3d'})

# Define labels and data
flavors = [
    (siv_u, "u", 'b'),
    (siv_d, "d", 'r'),
    (siv_s, "s", 'g'),
    (siv_ubar, r"\bar{u}", 'b'),
    (siv_dbar, r"\bar{d}", 'r'),
    (siv_sbar, r"\bar{s}", 'g')]

# Loop through axes and plot surfaces
for ax, (siv_data, label, color) in zip(axes.flat, flavors):
    ax.plot_surface(X, K, siv_data, color=color, alpha=0.6)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$k_{\perp}$', fontsize=12)
    ax.set_zlabel(r'$x \Delta f^N (x,k_{\perp})$', fontsize=12)
    ax.set_title(f"Quark: ${label}$", fontsize=14)

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig('cross-check_plot_v4.png')

# Not sure what this does:
#plt.plot(kperp_vals,test_array1)