# Genetic Algorithm for Image Apprxoimation

## Introduction
The main idea is taken from https://rogerjohansson.blog/2008/12/07/genetic-programming-evolution-of-mona-lisa/, where the goal is to approximate an image using colored semi-transparent polygons. In the original code, the approximation is obtained through a genetic algorithm where a batch of polygons is generated. Then the batch of polygons is mutated and compared with the original batch. If it is a better approximation, the mutation replaces the original.

This code is different in a couple ways. First, it looks at a population of batchs of polygons, and retains some number of them to repopulate the next generation. It also uses only triangles. 

One of the most important aspects is how the mutation is implemented. A good mutation method will require less samples to get good results, and a bad mutation method can take forever. The mutation used here is with some probability, an additional triangle (up to a maximum amount) will be added. If a triangle isn't added, then that triangle is either moved or has its color changed.

![SaturnV](images/saturnv_animated.gif?raw=true "Saturn V")

## Usage
To generate run the algorithm to generate the images used in the animation above, run

`python3 main.py  --image images/saturnv.jpg  --nepochs 30000 --max_triangles 200 --pop 120 --use_hist --savedir images/out`

To make the animation from the generated images, run

`python3 make_animation.py --image_dir images/out --target_image images/saturnv.jpg`

## Dependencies
The code has been tested with the following packages:

* Python3.6
  * numpy 1.18.2
  * scipy 1.4.1
  * matplotlib 3.2.1
  * opencv-python 4.0.0.21  
  * imageio 2.5.0
* imagemagick (option, used for saving gif)

