# American Fingerspelling Sign Language Recognition 

## Steps to Run the Project

1. set up a Python virtual Environment (Python 3.10)
2. Install the pip dependencies through requirements.txt
3. run the finger.py file in order to run the Project


## Model is being trained over the kaggle and below is the refence of the Notebook

https://www.kaggle.com/code/pankajkumar2002/361-380-tr-epochs-git-fingerspelling

## Note

There is high probability of getting Runtime error with tflite-runtime. This error would be reason on Registerting tensorflow_runtime twice by tflite and medipape. 
Inorder to prevent the Runtime Error Make the following changes.

Go to '$HOME/pythonvenv/python310/lib/python3.10/site-packages/tensorflow/lite/python/interpreter.py'

$HOME is your user directory

Comment out 'from tensorflow.lite.python.interpreter_wrapper import _pywrap_tensorflow_interpreter_wrapper as _interpreter_wrapper'

It will prevent twice Registration of Interpreter wrapper.

## Note

Making this change will lead to Runtime Error on the standalone use of TfliteRuntime, So you need to Uncomment the above commented line.
