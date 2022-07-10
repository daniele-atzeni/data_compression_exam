import tensorflow as tf
from tensorflow import keras
from transformers import BertConfig, TFBertForMaskedLM

import numpy as np

import os
import gc

#-------- FUNCTIONS TO GET MODELS --------#

def get_BERT_model() -> keras.layers.Layer:
    config = BertConfig()
    model = TFBertForMaskedLM(config).from_pretrained('bert-base-uncased')
    return model

def get_RNN_model() -> keras.layers.Layer:
    pass

def get_model(model_type:str) -> keras.layers.Layer:
    if model_type == 'BERT':
        return get_BERT_model()
    if model_type == 'RNN':
        return get_RNN_model()
    else:
        raise ValueError('Unknown model type: {}'.format(model_type))


#-------- FUNCTIONS TO CONVERT MODELS --------#

def float32_quant_conversion(converter:tf.lite.TFLiteConverter) -> tf.lite.TFLiteConverter:
    return converter

def float16_quant_conversion(converter:tf.lite.TFLiteConverter) -> tf.lite.TFLiteConverter:
    # https://www.tensorflow.org/lite/performance/post_training_float16_quant
    converter.optimizations = [tf.lite.Optimize.DEFAULT]    # enable optimizations flag to quantize all fixed parameters
    converter.target_spec.supported_types = [tf.float16]
    return converter

def int_quant_conversion(converter:tf.lite.TFLiteConverter) -> tf.lite.TFLiteConverter:
    # https://www.tensorflow.org/model_optimization/guide/quantization/post_training
    converter.optimizations = [tf.lite.Optimize.DEFAULT]    # enable optimizations flag to quantize all fixed parameters
    return converter

def float_fallback_quant_conversion(converter:tf.lite.TFLiteConverter, representative_data_gen) -> tf.lite.TFLiteConverter:
    # full integer quantization but float operators when no integer implementation
    # https://www.tensorflow.org/lite/performance/post_training_integer_quant
    converter.optimizations = [tf.lite.Optimize.DEFAULT]    # enable optimizations flag to quantize all fixed parameters
    converter.representative_dataset = representative_data_gen
    return converter

def integer_only_quant_conversion(converter:tf.lite.TFLiteConverter, representative_data_gen) -> tf.lite.TFLiteConverter:
    # https://www.tensorflow.org/lite/performance/post_training_integer_quant
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter

def config_converter(converter:tf.lite.TFLiteConverter, method:str, representative_dataset=None) -> tf.lite.TFLiteConverter:
    if method == 'float32':
        return float32_quant_conversion(converter)
    if method == 'float16':
        return float16_quant_conversion(converter)
    if method == 'int':
        return int_quant_conversion(converter)
    if method == 'float_fallback':
        if representative_dataset is None:
            raise ValueError('representative_dataset is required for float_fallback')
        return float_fallback_quant_conversion(converter, representative_dataset)
    if method == 'integer_only':
        if representative_dataset is None:
            raise ValueError('representative_dataset is required for integer_only')
        return integer_only_quant_conversion(converter, representative_dataset)
    
    else:
        raise ValueError('Unknown conversion method: {}'.format(method))


# --------------------------------------------#

def main()-> None:

    #------------------ ONLY CHANGE THESE VARIABLES ------------------#
    MODEL_TYPE = 'RNN'  # 'BERT', 'RNN', TODO 'RNN_CLUST' or 'BERT_CLUST'
    CONVERSION_METHOD = 'float16'
    ''' CONVERTION OPTIONS:
    float32
    float16
    int
    float_fallback
    integer_only
    '''
    CSV = True
    #-----------------------------------------------------------------#
    # Managing folders
    model_type_path = os.path.join('models', MODEL_TYPE)
    base_model_path = os.path.join(model_type_path, 'base_model')
    if not os.path.exists(base_model_path):
        os.makedirs(base_model_path)

    converted_model_path = os.path.join(model_type_path, CONVERSION_METHOD)
    if not os.path.exists(converted_model_path):
        os.makedirs(converted_model_path)
    tflite_model_filename = os.path.join(converted_model_path, 'model.tflite')
    parameters_path = os.path.join(converted_model_path, 'parameters')
    if not os.path.exists(parameters_path):
        os.makedirs(parameters_path)

    # Save base model
    if not os.path.exists(base_model_path):
        print('Getting model...')
        model = get_model(MODEL_TYPE)
        # save the model as recommended by tflite (https://www.tensorflow.org/lite/models/convert)
        model.save(base_model_path)

        del model
        gc.collect()
    else:
        print('Model already saved!')
    
    print('Converter initialization and setup...')
    converter = tf.lite.TFLiteConverter.from_saved_model(base_model_path)
    converter = config_converter(converter, CONVERSION_METHOD)
    print(f'Converting model with {CONVERSION_METHOD}...')
    lite_model = converter.convert()

    # save the converted model
    print('Saving lite model...')
    with open(tflite_model_filename, 'wb') as f:
        f.write(lite_model)
    
    del converter
    del lite_model
    gc.collect()

    print('Interpreter initialization...')
    interpreter = tf.lite.Interpreter(model_path=tflite_model_filename)
    interpreter.allocate_tensors()
    
    print('Invoking the model...')
    # invoke the model (to be sure of the model's correctness)
    input_details = interpreter.get_input_details()
    for detail in input_details:
        interpreter.set_tensor(detail['index'], np.ones(detail['shape'], dtype=detail['dtype']))
    interpreter.invoke()
    print('Invoking OK!')

    tensor_details = interpreter.get_tensor_details()
    for j, i in enumerate([el['index'] for el in tensor_details]):
        if j >= len(interpreter.get_input_details()):   # don't save inputs
            if not 'flatten' in tensor_details[j]['name'] and not 'output' in tensor_details[j]['name']:
                tensor = interpreter.tensor(i)().squeeze()
                if len(tensor.shape) == 2:   # save just the matrices
                    print(f'Saving weight with index {i}, {j}-th of {len(tensor_details)-1}...')
                    if CSV:
                        np.savetxt(os.path.join(parameters_path, f'params_{i}.csv'), tensor, delimiter=',')
                    else:
                        np.save(os.path.join(parameters_path, f'params_{i}.npy'), tensor)

if __name__ == '__main__':
    main()