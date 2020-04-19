# Tensorflow MPC

## Introduction
This directory contains scripts that test out using tensorflow to compute jacobians that are passed to an optimizer for solving MPC problems. If you define a dynamics model, as well as cost functions and constraints using differentiable tensorflow operations, tf.GradientTape() can automatically compute jacobians which are used in many optimizers. The optimizer used in these experiments is SLSQP from the scipy.optimize() function.

Using tensorflow to compute these gradients can be useful by itself, but using the same structure, an MPC problem with a neural network in the loop (as the dynamics model or part of the dynamics model) can be formed. This is useful for testing model learning tasks with neural networks.

`opt_solver.py` implements the core functionality. It defines a `DiscreteSystemModel` class that implemented dynamics models should inherit (whether it be a neural net or otherwise). `ScipyMPCOptProblem` defines a base class that MPC problems should inherit.
A simple 2d quadrotor example is given in `quadrotor_opt.py`.


![Quadrotor](images/quadrotor_optimization.gif?raw=true "Quadrotor")

## Basic Example Instructions
To run the optimizer once and plot the results, run the following in the directory this README.md is located in.

`python3 quadrotor_opt.py -x 4 -y 10`

where the '-x' flag will set the starting x location, and the '-y' flag will start the starting y location. 

To run a full MPC loop, run 

`python3 quadrotor_opt.py -x 4 -y 10 mpc --niter 7`

This will run an MPC loop, resolving the problem 7 times. This is the command that was used to generate the gif.


## Dependencies
Code has been tested with the following packages:

* Python3.6
* tensorflow 2.1
* numpy 1.18.2
* scipy 1.4.1
* matplotlib 3.2.1
