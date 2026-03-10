##########################################################################
#                                                                        #
#  convert_model.py                                                      #
#  Author: Jenna Chisholm                                                #
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
import tf2onnx
import onnx
import onnxruntime as ort



if __name__== "__main__":
    
    ### --- Get all necessary arguments --- ###
    
    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help="Path and file name of the model to be converted.", type=str, required=True)
    parser.add_argument('-o','--output', help="Directory (including path) in which to save the results.", type=str, required=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Obtain model ID, path to model files, and names of maxmmean files
    model_id = args.input.split('/')[-1].split('.keras')[0]
    in_dir = args.input.split(model_id+'.keras')[0]
    for file in os.listdir(in_dir):
        if file.startswith('X_maxmean_'):
            xmm_file = in_dir+file
        if file.startswith('Y_maxmean_'):
            ymm_file = in_dir+file
            
    
    ### --- Convert and save model --- #
    
    print("Converting model...")
    
    keras_model = tf.keras.models.load_model(args.input)
    input_signature = [tf.TensorSpec(i.shape, name=i.name) for i in keras_model.inputs]
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)
    onnx.save(onnx_model, args.output+model_id+".onnx")
    
    print(model_id+" converted and saved to "+args.output)
    
    
    ### --- Check conversion worked --- ###
    
    print("Checking model conversion worked...")
    
    # Create some input for the model (batch size, shape)
    jet_input = np.zeros((100, 10, 6), np.float32)
    other_input = np.zeros((100, 7), np.float32)
    
    # Get results for both models
    sess = ort.InferenceSession(args.output+model_id+".onnx", providers=["CUDAExecutionProvider"])
    results_ort = sess.run(None, {"jet_input": jet_input, "other_input": other_input})
    results_keras = keras_model.predict([jet_input, other_input])

    # Check that results are practically identical
    for ort_res, keras_res in zip(results_ort[0], results_keras):
        np.testing.assert_allclose(ort_res, keras_res, rtol=1e-5, atol=1e-5)
        
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
    file = uproot.recreate(args.output+model_id+'_maxmeans.root')
    file.mktree('x_maxs', x_maxs)
    file.mktree('x_means', x_means)
    file.mktree('y_maxs', y_maxs)
    file.mktree('y_means', y_means)
    
    print("Maxmean root file saved to "+args.output)
    
    


    print('done! :)')