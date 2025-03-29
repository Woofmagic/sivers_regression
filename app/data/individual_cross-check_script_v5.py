import os 
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

avg_model = tf.keras.models.load_model(str(model_path),custom_objects={'A0': A0, 'Quotient': Quotient})

# Define ranges for x and k_perp
x_vals = np.linspace(0.01, 0.3, 30)  # Adjust as needed
kperp_vals = np.linspace(0, 2, 30)  # Adjust range based on your data

# Create a meshgrid
X, K = np.meshgrid(x_vals, kperp_vals)

np.linspace

# Function to apply xsivdist element-wise
def compute_siv(flavor):
    # Initialize an empty array with the correct shape
    siv_values = np.zeros_like(X)
    
    # Compute values over the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            siv_values[i, j] = xsivdist(avg_model, X[i, j], 2.4, 0.57, flavor, K[i, j])[0]
    
    return siv_values

def predict_with_models(models, flavor):
    """Compute the replica average and uncertainty bounds across multiple models."""

    # Initialize storage arrays with the shape of X
    y_mean = np.zeros_like(X)
    y_min = np.full_like(X, np.inf)
    y_max = np.full_like(X, -np.inf)
    all_predictions = []  # List to store predictions for percentile computation

    # Loop over models to accumulate statistics
    for model in models:

        # (): Obtain the y-predictions for this model.
        y_prediction = xsivdist(model, 0.1, 2.4, 0.57, 2, K.flatten())
        
        all_predictions.append(y_prediction)
        y_min = np.minimum(y_min, y_prediction)
        y_max = np.maximum(y_max, y_prediction)
        y_mean += y_prediction / len(models)

    # Convert all predictions into an array for percentile calculations
    all_predictions = np.array(all_predictions)  # Shape: (num_models, X.shape[0], X.shape[1])

    # Compute percentiles across models
    y_percentile_10 = np.percentile(all_predictions, 10, axis=0)
    y_percentile_90 = np.percentile(all_predictions, 90, axis=0)

    return y_mean, y_min, y_max, y_percentile_10, y_percentile_90

model_paths = [os.path.join("../models", file) for file in os.listdir("../models") if file.endswith(".h5")]
models = [tf.keras.models.load_model(path, custom_objects = {'A0': A0, 'Quotient': Quotient}) for path in model_paths]
print(f"> Obtained {len(models)} models!")

# Compute distributions for quarks
sivers_u_mean = 0.
sivers_u_minimum = 0.
sivers_u_mean, sivers_u_minimum, sivers_u_maximum, sivers_u_10, sivers_u_90 = predict_with_models(models, 2)

siv_d = compute_siv(1)
siv_s = compute_siv(3)

# Compute distributions for anti-quarks
siv_ubar = compute_siv(-2)
siv_dbar = compute_siv(-1)
siv_sbar = compute_siv(-3)

# Create figure and subplots
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

fontsize = 10

# Left plot: Quarks (u, d, s)
axes[0].plot(kperp_vals, sivers_u_mean, 'b', label='$u$')
# axes[0].plot(kperp_vals, sivers_u_maximum, 'b', label='$u$', alpha = 0.2)
# axes[0].plot(kperp_vals, sivers_u_minimum, 'b', label='$u$', alpha = 0.2)
# axes[0].plot(kperp_vals, sivers_u_10, 'b', label='$u$', alpha = 0.3)
# axes[0].plot(kperp_vals, sivers_u_90, 'b', label='$u$', alpha = 0.3)


axes[0].plot(kperp_vals, siv_d, 'r', label='$d$')
axes[0].plot(kperp_vals, siv_s, 'g', label='$s$')
axes[0].set_title("Quarks", fontsize=fontsize)
axes[0].set_xlabel(r'$k_{\perp}$', fontsize=fontsize)
axes[0].set_ylabel(r'$x \Delta f^N (x,k_{\perp})$', fontsize=fontsize)
axes[0].legend(fontsize=fontsize)

# Right plot: Anti-Quarks (ubar, dbar, sbar)
axes[1].plot(kperp_vals, siv_ubar, 'b', label=r'$\bar{u}$')
axes[1].plot(kperp_vals, siv_dbar, 'r', label=r'$\bar{d}$')
axes[1].plot(kperp_vals, siv_sbar, 'g', label=r'$\bar{s}$')
axes[1].set_title("Anti-Quarks", fontsize=fontsize)
axes[1].set_xlabel(r'$k_{\perp}$', fontsize=fontsize)
axes[1].legend(fontsize=fontsize)

# Set y-axis limits and ticks
axes[0].set_ylim(-0.15, 0.1)
axes[0].set_yticks(np.arange(-0.15, 0.11, 0.05))
axes[0].tick_params(axis='y', labelsize=fontsize)

# Add text annotations
for ax in axes:
    ax.text(0.03, -0.11, '$Q^2=2.4$ GeV$^2$', fontsize=fontsize, transform=ax.transAxes)
    ax.text(0.03, -0.135, '$x=0.1$', fontsize=fontsize, transform=ax.transAxes)

plt.tight_layout()

plt.savefig('cross-check_plot_v6.png')
plt.savefig('cross-check_plot_v6.pdf')

# Not sure what this does:
#plt.plot(kperp_vals,test_array1)