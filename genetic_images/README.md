# Genetic Algorithm for Image Approximation

## Introduction
The main idea is taken from https://rogerjohansson.blog/2008/12/07/genetic-programming-evolution-of-mona-lisa/, where the goal is to approximate an rgb-image using colored semi-transparent polygons. In the original code, the approximation is obtained through a genetic algorithm where a batch of polygons is generated. Then the batch of polygons is mutated and compared with the original batch. If it is a better approximation, the mutation replaces the original.

This code is different in a couple ways. First, it looks at a population of batchs of triangles, and retains some number of them to repopulate the next generation. In addition, there is an option, `--use_hist`, which will compute a histogram of colors for the original image. Then, when generating new triangles, colors that are missing have a higher probability of being generated.

One of the most important aspects is how the mutation is implemented. A good mutation method will require less samples to get good results, but a bad mutation method can take forever. The mutation used here is 1) with some probability, an additional triangle (up to a maximum amount) will be added. 2) If a triangle isn't added, then a random triangle is either moved or has its color changed.

![Pic](images/picture_animated.gif?raw=true "Pic")

## Usage
To generate run the algorithm to generate the images used in the animation above, run

`python3 main.py  --image images/picture.jpg  --nepochs 100000 --max_triangles 250 --pop 120 --use_hist --savedir images/out`

To make the animation from the generated images, run

`python3 make_animation.py --image_dir images/out --target_image images/picture.jpg --save_file images/picture_animated/gif`

## Dependencies
The code has been tested with the following packages:

* Python3.6
  * numpy 1.18.2
  * scipy 1.4.1
  * matplotlib 3.2.1
  * opencv-python 4.0.0.21  
  * imageio 2.5.0
  * tqdm 4.31.1
* imagemagick (optional, used for saving gif)


