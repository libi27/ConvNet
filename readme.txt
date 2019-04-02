Lirane Bitton
lirane
200024677

EX 4

Submitted files:

visualizeNetwork.m - was given to us. No changes made.
template_script.m - was given to us. No changes made.
lossClass.m - was given to us. No changes made.
demoMNIST.m - was given to us. No changes made.
dataClass.m - was given to us. No changes made.
data_disp_script.m - was given to us. No changes made.

batch_size_script.m - (based on template_script.m), running with batch 
    sizes of 1, 2, 4, 8 and 20, in each case setting the number of 
    iterations to cover exactly 4 epochs (e.g. for batch size 20 we will 
    have 500 iterations). For each batch size independently,
    learning rates of 0.5, 0.1 and 0.05, are checked in a loop. Produces a 
    plot of the train errors as a function of the batch size.

momentum_script.m - in which values of 0.9,
    0.95 and 0.99 will be tried for the momentum variable. For each case 
    independently in a loop and also a loop for learning rates
    {5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4}.

lambda_script.m - the L2 regularization parameter, takes on the values 
    0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2 and 5e-2, in a loop.
    Plots the resulting train and test errors as a function of lambda in 
    a logarithmic distribution of lambda where 0 is pointed to 1e-5.

Part4.m - same as template_script.m but replaces ‘Xavier’ initialization of 
    affine weights with ‘Zeros’.

Part5.m - Modification of the learning rate in template_script.m to 
    decrease by a factor of 10 after 3 epochs (7.5K iterations).

best_small_script.m - is based on template_script.m, where the training 
    hyperparameters below are configured to give optimal test accuracy:
     SGD batch size
     Number of iterations T
     Momentum variable 
     Learning rate
     L2 regularization parameter

small_large_script.m - Part 7 tuning in the exercise with own tuning for best results

best_script.m - configuration for best results

dropout_script.m - configuration for best results using dropout layer

ConvNet.m - was given to us. Dropout case added with relevent functions

readme.txt - this file