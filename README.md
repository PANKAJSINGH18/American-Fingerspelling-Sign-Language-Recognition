# American Fingerspelling Sign Language Recognition 

## Steps to Run the Project

1. set up a Python virtual Environment (Python 3.10)
2. Install the pip dependencies through requirements.txt
3. run the finger.py file in order to run the Project


## Model Reference

https://www.kaggle.com/code/pankajkumar2002/361-380-tr-epochs-git-fingerspelling

## About Model Building

The model is being trained over the kaggle. It took 8 days of training with GPUP100 using the Pytorch checkpoints. The Model Architecture is based upon Transformer for which the encoder part is taken as a squeezeormer and for the decoder standard decoder with 2 layers is used.

## Note

There is a high probability of getting a Runtime error with tflite-runtime. This error would be the reason for Registering tensorflow_runtime twice by tflite and medipape. 
In order to prevent the Runtime Error Make the following changes.

Go to '$HOME/pythonvenv/python310/lib/python3.10/site-packages/tensorflow/lite/python/interpreter.py'

$HOME is your user directory

Comment out 'from tensorflow.lite.python.interpreter_wrapper import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper'

It will prevent twice Registration of Interpreter wrapper.

## Note

Making this change will lead to a Runtime Error on the standalone use of TfliteRuntime, So you need to Uncomment the above-commented line.
