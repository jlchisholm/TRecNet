##########################################################################
#                                                                        #
#  convert_model.py                                                      #
#  Author: Jenna Chisholm and Tommy Lubomirski                           #
#  Updated: Feb.25/26                                                    #
#                                                                        #
#  Converts Tensorflow Keras models to ONNX, such that they can be used  #
#  in TopCPToolKit. Additionally converts the relevant maxmean files to  #
#  root files to be read by TopCPToolkit.                                #
#                                                                        #
#  Thoughts for improvements:                                            #
#                                                                        #
##########################################################################

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training
import numpy as np
import uproot
from argparse import ArgumentParser
import tensorflow as tf
import keras
import tf2onnx
import onnx
import onnxruntime as ort

from source.ml.Models.blocks.set_encoder import JetSetEncoder
from source.ml.Models.blocks.transformer_blocks import ObjFFNBottom, SelfAttentionBlock
from source.ml.Models.blocks.objwise import ObjWise
from source.ml.Models.blocks.pooling import AttentionPooling

def patch_gelu(layer):
    # keras 3 exact GELU uses Erfc internally which has no ONNX equivalent
    # swap to tanh approximation which maps to native ONNX ops
    n = 0
    if hasattr(layer, 'activation'):
        name = getattr(layer.activation, '__name__', '') or getattr(layer.activation, 'name', '') or ''
        if 'gelu' in name.lower():
            layer.activation = lambda x: keras.activations.gelu(x, approximate=True)
            layer.activation.__name__ = 'gelu_approx'
            n += 1
    for sub in list(getattr(layer, 'layers', [])) + list(getattr(layer, '_layers', [])) + list(getattr(layer, 'blocks', [])):
        n += patch_gelu(sub)
    return n



if __name__== "__main__":

    ### --- Get all necessary arguments --- ###

    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="Path to the trained model directory (which contains directories 'history', 'info', 'model', etc.).", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Obtain model ID, path to model files, and names of maxmmean files
    in_dir = args.input
    model_id = in_dir.split('/')[-1]
    model_dir = in_dir+'/model/'
    scaling_dir = in_dir+'/scaling/'
    for file in os.listdir(scaling_dir):
        if file.startswith('X_maxmean_'):
            xmm_file = scaling_dir+file
        if file.startswith('Y_maxmean_'):
            ymm_file = scaling_dir+file


    ### --- Convert and save model --- #
    
    print("Converting model...")
    
    keras_model = keras.models.load_model(model_dir+model_id+'.keras', compile=False)

    # Patch GELU activations before conversion (see patch_gelu docstring)
    n_patched = patch_gelu(keras_model)
    print("replaced "+str(n_patched)+" GELU activations")

    input_signature = [tf.TensorSpec(i.shape, name=i.name) for i in keras_model.inputs]
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)
    onnx.save(onnx_model, model_dir+model_id+".onnx")
    
    print(model_id+" converted and saved to "+model_dir)


    ### --- Check conversion worked --- ###

    print("Checking model conversion worked...")

    # Create some input for the model (batch size, shape)
    jet_input = np.zeros((100, 10, 6), np.float32)
    other_input = np.zeros((100, 7), np.float32)

    # Get results for both models
    sess = ort.InferenceSession(model_dir+model_id+".onnx", providers=["CUDAExecutionProvider"])
    results_ort = sess.run(None, {"jet_input": jet_input, "other_input": other_input})
    results_keras = keras_model.predict([jet_input, other_input])

    # Check that results are practically identical
    for ort_res, keras_res in zip(results_ort[0], results_keras):
        np.testing.assert_allclose(ort_res, keras_res, rtol=1e-3, atol=1e-3)

    print("Results match")


    ### --- Convert maxmean files --- ###

    print("Converting maxmean dictionaries...")

    # Load dictionaries
    X_maxmean_dic = np.load(xmm_file,allow_pickle=True).item()
    Y_maxmean_dic = np.load(ymm_file,allow_pickle=True).item()

    # Split into max and mean dictionaries for x and y variables
    x_maxs = {key: [float(vals[0])] for key, vals in X_maxmean_dic.items()}
    x_means = {key: [float(vals[1])] for key, vals in X_maxmean_dic.items()}
    y_maxs = {key: [float(vals[0])] for key, vals in Y_maxmean_dic.items()}
    y_means = {key: [float(vals[1])] for key, vals in Y_maxmean_dic.items()}

    # Create root file
    file = uproot.recreate(scaling_dir+model_id+'_maxmeans.root')
    file.mktree('x_maxs', x_maxs)
    file.mktree('x_means', x_means)
    file.mktree('y_maxs', y_maxs)
    file.mktree('y_means', y_means)

    print("Maxmean root file saved to "+scaling_dir)




    print('done! :)')