SETTING_VERBOSE = True
SETTING_DEBUG = False

does_user_have_lhapdf = False

try:
    import lhapdf
    does_user_have_lhapdf = True

except ImportError:
    print('> LHAPDF not installed... Generating grids without it.')

PDF_CSV_DATA = 'pdf_data.csv'
PDF_CSV_DATA_COLUMN_HEADER_X = 'x'
PDF_CSV_DATA_COLUMN_HEADER_Q_SQUARED = 'QQ'

import numpy as np
import pandas as pd
import tensorflow as tf

if SETTING_VERBOSE:
    print(f"> Now running TF version: {tf.__version__}")

def get_values_for_nnq(tf_model, value_of_x, hadron_string_identifier):

    # (): 
    model_input = tf_model.get_layer(hadron_string_identifier).input

    # ():
    model_output = tf_model.get_layer(hadron_string_identifier).output

    # (): Define a function u
    model_function = tf.keras.backend.function(model_input, model_output)
    
    # (): Evaluate the Keras model by passing in a value for x:
    model_evaluated_at_specific_x_value = model_function(value_of_x)

    # (): Return the numerical evaluation:
    return model_evaluated_at_specific_x_value

def evaluate_dnn_parametrized_sivers_function():
    """
    ## Description:
    We compute Delta^{N} f_{q/N^{up}}(x, k_{perp}), i.e. the Sivers Function.

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
    pass

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

for flavor, label in quark_flavors_dictionary.items():
    
    if SETTING_VERBOSE:
        print(f"> Now analyzing distribution for {flavor} with numerical label {label}...")


