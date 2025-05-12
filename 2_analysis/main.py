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
from scipy.optimize import curve_fit

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

def find_quark_string(quark_label):

    dictionary_of_hadron_strings = {
        -3: r'$\bar{s}$',
        -2: r'$\bar{u}$',
        -1: r'$\bar{d}$',
        1: r'$d$',
        2: r'$u$',
        3: r'$s$'
    }
    
    try:
        return dictionary_of_hadron_strings[quark_label]
    except:
        raise Exception("Invalid flavor.")
    
def find_hadron_string(quark_flavor):

    dictionary_of_hadron_strings = {
        -3: 'nnsbar',
        -2: 'nnubar',
        -1: 'nndbar',
        1: 'nnd',
        2: 'nnu',
        3: 'nns'}
    
    try:
        return dictionary_of_hadron_strings[quark_flavor]
    except:
        raise Exception("Invalid flavor.")

def evaluate_dnn_representation_of_n_q(tf_model, x_values, hadron_string_identifier):

    # (): 
    model_input = tf_model.get_layer(hadron_string_identifier).input

    # ():
    model_output = tf_model.get_layer(hadron_string_identifier).output

    # (): Define a function u...
    # - output of this is a NumPy array *of* arrays with a single number...
    model_function = tf.keras.backend.function(model_input, model_output)
    
    # (): Evaluate the Keras model by passing in a value for x:
    model_evaluated_at_specific_x_value = model_function(x_values)
    # print(model_evaluated_at_specific_x_value)
    # print(model_evaluated_at_specific_x_value[0])
    # print(model_evaluated_at_specific_x_value.flatten())

    # (): Return the numerical evaluation:
    return model_evaluated_at_specific_x_value.flatten()

def evaluate_transverse_momentum_distribution(tf_model, k_perpendicular_values):

    # (): For a dumb reason, these parameters are *in* the NN:
    m1 = tf_model.get_layer('a0').m1.numpy()

    # (): Also get the "e" parameter:
    e = tf_model.get_layer('a0').e.numpy()

    # (): Compute the k_{\perp} distribution:
    # k_perpendicular_distribution = tf.sqrt(2. * e) * (k_perpendicular_values / m1) * tf.math.exp(-k_perpendicular_values**2 / m1**2)
    k_perpendicular_distribution = np.sqrt(2. * e) * (k_perpendicular_values / m1) * np.exp(-k_perpendicular_values**2 / m1**2)

    # (): Return the distribution:
    return k_perpendicular_distribution

def evaluate_parton_distribution_function(quark_flavor, x_values, q_squared_values):
    """
    """
    pdfData = lhapdf.mkPDF(pdfset)

    # (1): `.xfxQ2(quark_flavor, )` evaluates x*f(x, Q^{2}) for given quark flavor at a particular Q^{2}:
    # return np.array([pdfData.xfxQ2(quark_flavor, figure_axes_1, q_squared) for figure_axes_1, q_squared in zip(x_values, q_squared_values)])
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
    x_values_flattened = x_values.flatten()
    k_perpendicular_values_flattened = k_perpendicular_values.flatten()

    if SETTING_VERBOSE:
        print(f"> Length of vectorized x values grid is: {len(x_values_flattened)}")

    if SETTING_VERBOSE:
        print(f"> Length of vectorized k-perp values grid is: {len(k_perpendicular_values_flattened)}")

    # (): The "hadron string" is determined:
    hadron_string_identifier = find_hadron_string(quark_flavor)

    # (): Evaluate the DNN for N_{q} for the given hadron string:
    dnn_representation_of_n_q = evaluate_dnn_representation_of_n_q(dnn_model, x_values_flattened, hadron_string_identifier)

    if SETTING_VERBOSE:
        print(f"> Length of DNN computation with vectorized x-values is: {len(dnn_representation_of_n_q)}")

    k_perpendicular_distribution = evaluate_transverse_momentum_distribution(dnn_model, k_perpendicular_values_flattened)

    if SETTING_VERBOSE:
        print(f"> Length of computed k-perp distribution is: {len(k_perpendicular_distribution)}")

    f_quark_over_proton = evaluate_f_quark_fraction_of_proton(
        quark_flavor,
        x_values_flattened,
        q_squared_values,
        k_perpendicular_values_flattened)

    dnn_sivers_function_flattened = 2. * dnn_representation_of_n_q * k_perpendicular_distribution * f_quark_over_proton

    dnn_sivers_function_normal_shape = dnn_sivers_function_flattened.reshape(x_grid_values.shape)

    return dnn_sivers_function_normal_shape

def sivers_ansatz(independent_variables, *function_parameters):
    """
    ## Description:

    ## Notes:

    ### Fitting Parameters:
    In this function, which we will feed into a curve-fitting algorithm,
    we have three parameters:

        1. N_{q}(x, Q^{2})
        2. \Lambda(x, Q^{2})
        3. \sigma_{q}(x, Q^{2})
    """
    _DEFAULT_N_Q_VALUE = 0.3
    _DEFAULT_LAMBDA_VALUE = 0.2
    _DEFAULT_SIGMA_VALUE = 0.5

    parameter_n_q, parameter_lambda, parameter_sigma = function_parameters

    k_perpendicular, x_values = independent_variables

    return parameter_n_q * k_perpendicular / (k_perpendicular**2 + parameter_lambda) * np.exp(-k_perpendicular**2 / parameter_sigma)

def evaluate_function_on_meshgrid(function, variable_ranges, **params):
    """
    ## Description:
    Evaluate a multidimensional function on a meshgrid.

    ## Arguments:
        function: The multidimensional function to evaluate.
        variable_ranges: List of (min, max) tuples for each variable.
        resolution: Number of points per axis.
        **params: Additional parameters for the function.
    
    ## Returns:
        Tuple of (coordinate_matrix, evaluated_function), where:
            grids: tuple of meshgrid arrays
            evaluated_function: evaluated function on the meshgrid
    """

    coordinate_matrix = np.meshgrid(*variable_ranges, indexing = 'ij')
    
    evaluated_function = function(*coordinate_matrix, **params)

    return coordinate_matrix, evaluated_function

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


MINIMUM_X_VALUE = 0.01
MAXIMUM_X_VALUE = 0.3

MINIMUM_Y_VALUE = 0.
MAXIMUM_Y_VALUE = 2.

x_values = np.linspace(MINIMUM_X_VALUE, MAXIMUM_X_VALUE, 30) 
k_perpendicular_values = np.linspace(MINIMUM_Y_VALUE, MAXIMUM_Y_VALUE, 30)

x_grid_values, k_perpendicular_grid_values = np.meshgrid(x_values, k_perpendicular_values)

q_squared_value = np.array([2.4])

model_path = '../app/models/rep150.h5'

average_model = tf.keras.models.load_model(
    model_path,
    custom_objects = {
        'A0': A0, 
        'Quotient': Quotient
        })

# (): We now evaluate the DNN-representation of the Sivers Function:
sivers_prediction_up_quark = evaluate_dnn_parametrized_sivers_function(2, average_model, x_grid_values, q_squared_value, k_perpendicular_grid_values)
sivers_prediction_down_quark = evaluate_dnn_parametrized_sivers_function(3, average_model, x_grid_values, q_squared_value, k_perpendicular_grid_values)
sivers_prediction_strange_quark = evaluate_dnn_parametrized_sivers_function(1, average_model, x_grid_values, q_squared_value, k_perpendicular_grid_values)
sivers_prediction_anti_up_quark = evaluate_dnn_parametrized_sivers_function(-2, average_model, x_grid_values, q_squared_value, k_perpendicular_grid_values)
sivers_prediction_anti_down_quark = evaluate_dnn_parametrized_sivers_function(-3, average_model, x_grid_values, q_squared_value, k_perpendicular_grid_values)
sivers_prediction_anti_strange_quark = evaluate_dnn_parametrized_sivers_function(-1, average_model, x_grid_values, q_squared_value, k_perpendicular_grid_values)

optimized_parameters_up_quark, parameter_covariance_up_quark = curve_fit(
        f = sivers_ansatz, 
        xdata = np.vstack((x_grid_values.ravel(), k_perpendicular_grid_values.ravel())), 
        ydata = sivers_prediction_up_quark.ravel(), 
        p0 = [0.1, 0.2, 0.2])
optimized_parameters_down_quark, parameter_covariance_down_quark = curve_fit(
        f = sivers_ansatz, 
        xdata = np.vstack((x_grid_values.ravel(), k_perpendicular_grid_values.ravel())),
        ydata = sivers_prediction_down_quark.ravel(), 
        p0 = [0.1, 0.2, 0.2])
optimized_parameters_strange_quark, parameter_covariance_strange_quark = curve_fit(
        f = sivers_ansatz, 
        xdata = np.vstack((x_grid_values.ravel(), k_perpendicular_grid_values.ravel())),
        ydata = sivers_prediction_strange_quark.ravel(), 
        p0 = [0.1, 0.2, 0.2])
optimized_parameters_anti_up_quark, parameter_covariance_anti_up_quark = curve_fit(
        f = sivers_ansatz, 
        xdata = np.vstack((x_grid_values.ravel(), k_perpendicular_grid_values.ravel())), 
        ydata = sivers_prediction_anti_up_quark.ravel(), 
        p0 = [0.1, 0.2, 0.2])
optimized_parameters_anti_down_quark, parameter_covariance_anti_down_quark = curve_fit(
        f = sivers_ansatz, 
        xdata = np.vstack((x_grid_values.ravel(), k_perpendicular_grid_values.ravel())),
        ydata = sivers_prediction_anti_down_quark.ravel(), 
        p0 = [0.1, 0.2, 0.2])
optimized_parameters_anti_strange_quark, parameter_covariance_anti_strange_quark = curve_fit(
        f = sivers_ansatz, 
        xdata = np.vstack((x_grid_values.ravel(), k_perpendicular_grid_values.ravel())),
        ydata = sivers_prediction_anti_strange_quark.ravel(), 
        p0 = [0.1, 0.2, 0.2])

print(optimized_parameters_up_quark)
print(len(optimized_parameters_up_quark))
print(f"> Optimized parameters for the up quark Sivers are: {optimized_parameters_up_quark}")
print(f"> Optimized parameters for the down quark Sivers are: {optimized_parameters_down_quark}")
print(f"> Optimized parameters for the sivers quark Sivers are: {optimized_parameters_strange_quark}")
print(f"> Optimized parameters for the anti up quark Sivers are: {optimized_parameters_anti_up_quark}")
print(f"> Optimized parameters for the anti down quark Sivers are: {optimized_parameters_anti_down_quark}")
print(f"> Optimized parameters for the anti strange quark Sivers are: {optimized_parameters_anti_strange_quark}")



sivers_functions_array = [
    (sivers_prediction_up_quark, 2, "up", "red", optimized_parameters_up_quark),
    (sivers_prediction_down_quark, 1, "down", "blue", optimized_parameters_down_quark), 
    (sivers_prediction_strange_quark, 3, "strange", "green", optimized_parameters_strange_quark),
    (sivers_prediction_anti_up_quark, -2, "anti_up", "cyan", optimized_parameters_anti_up_quark),
    (sivers_prediction_anti_down_quark, -1, "anti_down", "orange", optimized_parameters_anti_down_quark),
    (sivers_prediction_anti_strange_quark, -3, "anti_strange", "purple", optimized_parameters_anti_strange_quark),
]


# (X, Y), Z = evaluate_function_on_meshgrid(sivers_ansatz, [ x_values, k_perpendicular_values ], n = 0.1, l = 0.2, sigma = 0.2)

for (sivers_function_representation, quark_numerical_label, quark_label, color, parameters) in sivers_functions_array:
    
    if SETTING_VERBOSE:
        print(f"> Now analyzing distribution 1...")
    
    x_target = 0.1
    x_index = np.abs(x_values - x_target).argmin()
    
    figure, axes = plt.subplots(1, 1, figsize = (10, 5))
    axes.plot(k_perpendicular_values, sivers_function_representation[:, x_index], label = find_quark_string(quark_numerical_label))
    axes.set_xlabel(r'$k_{\perp}$')
    axes.set_ylabel(r'$x \Delta f_{q}/N^{\uparrow} (x,k_{\perp})$')
    for index, item in enumerate(parameters):
        axes.text(0.6, 0.95 -  0.1 * index, fr'$\theta_{index + 1}={item}$', fontsize=12, transform=axes.transAxes)
    axes.text(0.03, 0.95, fr'$Q^2={q_squared_value[0]}$ GeV$^2$', fontsize=12, transform=axes.transAxes)
    axes.text(0.03, 0.90, fr'$x={x_target}$', fontsize=12, transform=axes.transAxes)
    plt.savefig(f"{quark_label}_vs_kperp_at_{x_target}_v3_2.png")
    plt.close()

    figure, axes = plt.subplots(subplot_kw = {"projection": "3d"})
    axes.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_function_representation, color = color, alpha = 0.6)
    axes.set_xlabel(r'$x$', fontsize=12)
    axes.set_ylabel(r'$k_{\perp}$', fontsize=12)
    axes.set_zlabel(r'$x \Delta f^N (x,k_{\perp})$', fontsize=12)
    axes.set_title(rf"Quark: {find_quark_string(quark_numerical_label)}", fontsize=14)
    plt.savefig(f"{quark_label}_quark_surface_plot_v3_2.png")
    plt.close()

    figure, axes = plt.subplots(subplot_kw = {"projection": "3d"})
    axes.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_ansatz((x_grid_values, k_perpendicular_grid_values), *parameters), color = color, alpha = 0.6)
    axes.set_xlabel(r'$x$', fontsize=12)
    axes.set_ylabel(r'$k_{\perp}$', fontsize=12)
    axes.set_zlabel(r'$x \Delta f^N (x,k_{\perp})$', fontsize=12)
    axes.set_title(f"Quark: {find_quark_string(quark_numerical_label)}", fontsize=14)
    plt.savefig(f"{quark_label}_fit_surface_plot_v3_2.png")
    plt.close()

    x_cuts = [0.0, 0.25, 0.5, 0.75, 1.0]

    for x_value in x_cuts:
        
        if SETTING_VERBOSE:
            print(f"> Now making slices at x = {x_value}")

        index = np.argmin(np.abs(x_values - x_value))

        true_slice = sivers_function_representation[:, index]

        if SETTING_VERBOSE:
            print(f"> Calculated the shape of the slice of Sivers Function as: {true_slice.shape}")

        # (): Evaluate the Sivers ansatz across k_perp but at a fixed x_value:
        fit_slice = sivers_ansatz((x_grid_values, k_perpendicular_grid_values), *optimized_parameters_up_quark)

        figure, axes = plt.subplots(1, 1, figsize = (10, 5))

        axes.plot(k_perpendicular_grid_values, true_slice, label = f'{find_quark_string(quark_numerical_label)}')
        axes.plot(k_perpendicular_grid_values, fit_slice[:, index], '-', label='Fit')
        axes.set_title(rf'Projection at $x = {x_value:.2f}$')
        axes.set_xlabel(r'$k_{\perp}$')
        axes.set_ylabel(r'$x \Delta f^N (x, k_{\perp})$')
        axes.text(0.03, 0.95, fr'$Q^2={q_squared_value[0]}$ GeV$^2$', fontsize=12, transform=axes.transAxes)
        axes.text(0.03, 0.90, fr'$x={x_value}$', fontsize=12, transform=axes.transAxes)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"sivers_{quark_label}_{x_value}_cut_v3_2.png")
        plt.close()

flavors = [
    (2, "u", 'b'),
    (1, "d", 'r'),
    (3, "s", 'g'),
    (-2, r"\bar{u}", 'b'),
    (-1, r"\bar{d}", 'r'),
    (-3, r"\bar{s}", 'g')
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={'projection': '3d'})
for ax, (sivers_function_representation, quark_numerical_label, quark_label) in zip(axes.flat, sivers_functions_array):
    ax.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_function_representation, alpha=0.6)
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$k_{\perp}$', fontsize=12)
    ax.set_zlabel(r'$x \Delta f^N (x,k_{\perp})$', fontsize=12)
    ax.set_title(f"Quark: ${quark_label}$", fontsize=14)

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig('combined_surface_plots_v3_2')

# Create figure and 3D axes
fig = plt.figure(figsize=(14, 6))

# Left subplot: Quarks
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_prediction_up_quark, color='b', alpha=0.34, label='$u$')
ax1.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_prediction_down_quark, color='r', alpha=0.34, label='$d$')
ax1.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_prediction_strange_quark, color='g', alpha=0.34, label='$s$')
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$k_{\perp}$')
ax1.set_zlabel(r'$x \Delta f^N (x,k_{\perp})$')
ax1.set_title("Quarks")
ax1.legend()

# Right subplot: Anti-Quarks
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_prediction_anti_up_quark, color='b', alpha=0.34, label=r'$\bar{u}$')
ax2.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_prediction_anti_down_quark, color='r', alpha=0.34, label=r'$\bar{d}$')
ax2.plot_surface(x_grid_values, k_perpendicular_grid_values, sivers_prediction_anti_strange_quark, color='g', alpha=0.34, label=r'$\bar{s}$')
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$k_{\perp}$')
ax2.set_zlabel(r'$x \Delta f^N (x,k_{\perp})$')
ax2.set_title("Anti-Quarks")
ax2.legend()

# Adjust layout and show
plt.tight_layout()

plt.savefig('surface_plots_v3_2.png') 

plt.close()