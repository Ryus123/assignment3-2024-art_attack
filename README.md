This is a basic code repository for Assignment 3.

The repository contains a basic model and a basic training and testing
procedure. It will work on the testing-platform. The goal of the project is
to train a new model that is as robust as possible.

# Basic usage

Install python dependencies with pip: 

    $ pip install -r requirements.txt

Test the basic model:

    $ ./model.py
    Testing with model from 'models/default_model.pth'. 
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
    100.0%
    Extracting ./data/cifar-10-python.tar.gz to ./data/
    Model natural accuracy (test): 53.07

(Re)train the basic model:

    $ ./model.py --force-train
    Training model
    models/default_model.pth
    Files already downloaded and verified
    Starting training
    [1,   500] loss: 0.576
    [1,  1000] loss: 0.575
    ...

Train/test the basic model and store the weights to a different file:

    $ ./model.py --model-file models/mymodel.pth
    ...

Load the module project and test it as close as it will be tested on the testing plateform:

    $ ./test_project.py

Even safer: do it from a different directory:

    $ mkdir tmp
    $ cd /tmp
    $ ../test_project.py ../