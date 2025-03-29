import numpy as np
import pandas as pd
import tensorflow as tf

def get_pdf_values(x, q2, flavor):
    """Retrieve the PDF values from the DataFrame based on x, Q2, and flavor."""
    row = pdf_df[(pdf_df['x'] == x) & (pdf_df['Q2'] == q2)]
    flavor_map = {2: 'fu', 1: 'fd', 3: 'fs', -2: 'fubar', -1: 'fdbar', -3: 'fsbar'}
    return row[flavor_map[flavor]].values[0]

def fqp(x_vals, q2_vals, kperp2avg, kperp, flavor):
    """Compute fq using the loaded PDF values."""
    fq_values = np.array([get_pdf_values(x, q2, flavor) for x, q2 in zip(x_vals, q2_vals)])
    fxk = fq_values[:, np.newaxis] * (1. / (np.pi * kperp2avg)) * np.exp(-kperp**2 / kperp2avg)
    return fxk.flatten()

def nnq(model, x, hadronstr):
    mod_out = tf.keras.backend.function(model.get_layer(hadronstr).input,
                                        model.get_layer(hadronstr).output)
    return mod_out(x)

def h(model, kperp):
    m1 = model.get_layer('a0').m1.numpy()
    e = model.get_layer('a0').e.numpy()
    return tf.sqrt(2. * e) * (kperp / m1) * tf.math.exp(-kperp**2 / m1**2)

def xsivdist(model, x_vals, q2_vals, kperp2avg, flavor, kperp_vals):
    x_repeated = np.tile(pdf_df['x'].values, len(kperp_vals))
    q2_repeated = np.tile(pdf_df['Q2'].values, len(kperp_vals))
    kp_repeated = np.repeat(kperp_vals, len(pdf_df))

    refDict = {2: 'nnu', 1: 'nnd', 3: 'nns', -2: 'nnubar', -1: 'nndbar', -3: 'nnsbar'}
    print("here")
    nnqvals = np.array([nnq(model, np.array([x]), refDict[flavor]) for x in x_repeated])

    print("fafsdfs")
    hvals = h(model, kp_repeated)

    print("after gh")
    fqpvals = fqp(x_vals, q2_vals, kperp2avg, kperp_vals, flavor)

    print('asfasdf after fqp')

    nnqvals = np.array(nnqvals[:, np.newaxis])
    nnqvals = nnqvals.flatten()
    return 2. * nnqvals * hvals * fqpvals

# Define custom Keras layers
class A0(tf.keras.layers.Layer):
    def __init__(self, kperp2avg=.57, pperp2avg=.12, **kwargs):
        super(A0, self).__init__(name='a0')
        self.m1 = tf.Variable(.5, name='m1')
        self.kperp2avg = kperp2avg
        self.pperp2avg = pperp2avg
        self.e = tf.constant(1.)

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
        return inputs[0] / inputs[1]

# Load PDF data from CSV
pdf_df = pd.read_csv('pdf_data.csv')

print('sdfsdf')

# (2): Initialize the kinematic grid for the Sivers Function:

# (2.1): Initialize an array of x-Bjorken values by reading the `x` column of the Pandas DF:
x_vals = np.array(pdf_df['x'])

# (2.2): Initialize an array of Q^{2} values by reading the `Q2` column of the Pandas DF:
q2_vals = np.array(pdf_df['Q2'])

# (2.3): Initalize an array of k_{\perp} values by spacing them equidistantly with an interval of 1.5/N(x-Bjorken values):
kperp_vals = np.linspace(0., 1.5, len(x_vals))

# (3): Initialize a string that refers to the path of the desired TensorFlow model:
model_path = '../models/rep150.h5'

print('zsdfsdfz')

# (4): Load the .h5 TF modle with 
avg_model = tf.keras.models.load_model(model_path, custom_objects = {
    'A0': A0,
    'Quotient': Quotient
    })

print('zfafz')

# Compute Sivers distribution for all quark flavors
flavors = {
    2: 'Siv_u',
    1: 'Siv_d',
    3: 'Siv_s',
    -2: 'Siv_ubar',
    -1: 'Siv_dbar',
    -3: 'Siv_sbar'
    }

print('zz')

# Ensure consistent array lengths by repeating x and Q2 values for each kperp value
x_repeated = np.tile(pdf_df['x'].values, len(kperp_vals))
q2_repeated = np.tile(pdf_df['Q2'].values, len(kperp_vals))
kp_repeated = np.repeat(kperp_vals, len(pdf_df))

print('zfsdfsz')

# Store repeated values in results dictionary
results = {
    'x': x_repeated,
    'Q2': q2_repeated,
    'kp': kp_repeated
}

# Compute Sivers function for each flavor and store results
for flavor, label in flavors.items():
    print(f"> Now doing {flavor} and {label}")
    siv_flavor = xsivdist(avg_model, x_vals, q2_vals, 0.57, flavor, kperp_vals)
    results[label] = siv_flavor.numpy().flatten()

# Convert dictionary to DataFrame and save to CSV
df_siv = pd.DataFrame(results)
df_siv.to_csv('siv_results_all_flavors.csv', index=False)
