SETTING_VERBOSE = True
SETTING_DEBUG = False

does_user_have_lhapdf = False

try:
    import lhapdf
    does_user_have_lhapdf = True
    pdfset = 'NNPDF40_nlo_as_01180'

except ImportError:
    print('> LHAPDF not installed... Generating grids without it.')

PDF_CSV_DATA = 'pdf_data.csv'
PDF_CSV_DATA_COLUMN_HEADER_X = 'x'
PDF_CSV_DATA_COLUMN_HEADER_Q_SQUARED = 'QQ'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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

k_perpendicular_squared_average = 0.57

if SETTING_VERBOSE:
    print(f"> Now running TF version: {tf.__version__}")

def find_hadron_string(quark_flavor):
    dictionary_of_hadron_strings = {
        -3: 'nnsbar',
        -2: 'nnubar',
        -1: 'nndbar',
        1: 'nnd',
        2: 'nnu',
        3: 'nns'}
    return dictionary_of_hadron_strings[quark_flavor]

def evaluate_dnn_representation_of_n_q(tf_model, x_values, hadron_string_identifier):

    # (): 
    model_input = tf_model.get_layer(hadron_string_identifier).input

    # ():
    model_output = tf_model.get_layer(hadron_string_identifier).output

    # (): Define a function u
    model_function = tf.keras.backend.function(model_input, model_output)
    
    # (): Evaluate the Keras model by passing in a value for x:
    model_evaluated_at_specific_x_value = model_function(x_values)

    # (): Return the numerical evaluation:
    return model_evaluated_at_specific_x_value

def evaluate_transverse_momentum_distribution(tf_model, k_perpendicular_values):

    m1 = tf_model.get_layer('a0').m1.numpy()

    e = tf_model.get_layer('a0').e.numpy()

    return tf.sqrt(2. * e) * (k_perpendicular_values / m1) * tf.math.exp(-k_perpendicular_values**2 / m1**2)

def evaluate_parton_distribution_function(quark_flavor, x_values, q_squared_values):
    """
    """
    pdfData = lhapdf.mkPDF(pdfset)

    # (1): `.xfxQ2(quark_flavor, )` evaluates x*f(x, Q^{2}) for given quark flavor at a particular Q^{2}:
    # return np.array([pdfData.xfxQ2(quark_flavor, figure_axes_1, q_squared) for figure_axes_1, q_squared in zip(x_values, q_squared_values)])

    print(x_values)
    print(q_squared_values)
    return pdfData.xfxQ2(quark_flavor, x_values, q_squared_values)
def evaluate_f_quark_fraction_of_proton(quark_flavor, x_values, q_squared_values, k_perpendicular_values):
    """
    ## Description:
    We compute f_{q/N}(x, k_{\perp}).

    ## Notes:
    It is guessed that f_{q/N} may be factorized into the collinear PDF
    which carries the x-dependence and a Gaussian distribution of transverse
    momentum.
    """
    quark_pdf = evaluate_parton_distribution_function(quark_flavor, x_values, q_squared_values)
    
    transverse_momentum_distribution = np.exp(-k_perpendicular_values**2 / k_perpendicular_squared_average) / (np.pi * k_perpendicular_squared_average)

    f_sub_q_over_p = quark_pdf * transverse_momentum_distribution

    return f_sub_q_over_p

def evaluate_dnn_parametrized_sivers_function(quark_flavor, dnn_model, x_values, q_squared_values, k_perpendicular_values):
    """
    ## Description:
    We compute Delta^{N} f_{q/N^{up}}(x, k_{perp}), i.e. the Sivers Function.

    ## Notes:

    ### Decomposition of Sivers Function:
    Remember that our factorization procedure is:
    Delta^{N} f_{q/N^{up}}(x, k_{perp}) = 2 N_{q}(x) h(k_{perp}) f_{q/N}(x, k_{perp}).
    So, there are three ingredients:

        1. N_{q}(x)
        2. h(k_{perp})
        3. f_{q/N}(x, k_{perp})

    ### Decomposition of f_{q/N}(x, k_{perp}):
    We decompose according to:
    
    f_{q/N}(x, k_{perp}) = f_{q}(x) exp{-k_{perp}^{2}/<k_{perp}^{2}>} / \pi <k_{perp}^{2}>
    
    The angle brackets indicate an average value.
    
    ### Decomposition of h(k_{perp}):
    We decompose according to:

    h(k_{perp}) = sqrt{2e} k_{perp} e^{-k_{perp}^{2} / m_{1}^{2}} / m_{1}.
    """
    hadron_string_identifier = find_hadron_string(quark_flavor)

    dnn_representation_of_n_q = evaluate_dnn_representation_of_n_q(dnn_model, x_values, hadron_string_identifier)

    k_perpendicular_distribution = evaluate_transverse_momentum_distribution(dnn_model, k_perpendicular_values)

    f_quark_over_proton = evaluate_f_quark_fraction_of_proton(
        quark_flavor,
        x_values,
        q_squared_values,
        k_perpendicular_values)
    
    return 2. * dnn_representation_of_n_q * k_perpendicular_distribution * f_quark_over_proton

quark_flavors_dictionary = {
    "sivers_up_quark_number": 2,
    "sivers_down_quark_number": 1,
    "sivers_strange_quark_number": 3,
    "sivers_antiup_quark_number": -2,
    "sivers_antidown_quark_number": -1,
    "sivers_antistrange_quark_number": -3,
}

if not does_user_have_lhapdf:
    
    # (): Read the CSV file using Pandas:
    pdf_dataframe = pd.read_csv(PDF_CSV_DATA)

    # (): Extract only the x-values from the DF:
    x_values = pdf_dataframe[PDF_CSV_DATA_COLUMN_HEADER_X]

    # (): Extract only the Q^{2} values in the DF:
    q_squared_values = pdf_dataframe[PDF_CSV_DATA_COLUMN_HEADER_Q_SQUARED]

    # (): Generate an NumPy array of k_perp values from 0. to 1.5 according to number of x values:
    k_perpendicular_values = np.linspace(0., 1.5, len(x_values))

x_values = np.linspace(0.01, 0.3, 30) 
k_perpendicular_values = np.linspace(0, 2, 30)
x_grid_values, k_perpendicular_grid_values = np.meshgrid(x_values, k_perpendicular_values)

q_squared_value = np.array([2.4])
model_path = '../app/models/rep150.h5'

for flavor, label in quark_flavors_dictionary.items():
    
    if SETTING_VERBOSE:
        print(f"> Now analyzing distribution for {flavor} with numerical label {label}...")

    average_model = tf.keras.models.load_model(
        model_path,
        custom_objects = {
            'A0': A0, 
            'Quotient': Quotient
            })

    sivers_data = evaluate_dnn_parametrized_sivers_function(
        label,
        average_model,
        x_values,
        q_squared_value,
        k_perpendicular_values)
    
    figure, axes = plt.subplots(1, 1, figsize = (10, 5))
    axes.plot(k_perpendicular_grid_values, sivers_data[0, :], label = flavor)
    axes.set_title("Quarks")
    axes.set_xlabel(r'$k_{\perp}$')
    axes.set_ylabel(r'$x \Delta f^N (x,k_{\perp})$')
    plt.savefig(f"{flavor}_v1.png")
