import copy
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import imageio
import multiprocessing
from tqdm import tqdm
import pickle


class TriangleImage(object):
    """
    This class represents a set of triangles
    to be rendered onto an image.
    Each triangle is represented by its 3 vertices and an rgba color.
    """
    def __init__(self, im_height, im_width, max_triangles):
        self.im_height = im_height
        self.im_width = im_width
        self.max_triangles = max_triangles
        self.num_triangles = 0

        self.image = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        
        # (x1, y1, x2, y2, x3, y3)
        self.vertices = np.zeros((self.max_triangles, 3, 2), dtype=np.int32)
        
        # rgba. all numbers in range [0, 255]
        # this uses int16 so that arithmetic won't over/underflow
        self.colors = np.zeros((self.max_triangles, 4), dtype=np.int16)

        self.mutate_new_prob = 0.03
        self.mutate_pos_prob = 0.5

    def init(self):
        self._generate_random_triangle(0)
        self.num_triangles = 1

    def get_num_triangles(self):
        return self.num_triangles

    def render(self):
        """
        Renders the triangles onto a numpy array
        
        Returns:
            np.array -- rendered image as a numpy array.
        """
        self.image.fill(0)
        for i in range(self.num_triangles):
            shape_im = np.copy(self.image)
            cv2.fillPoly(
                img=shape_im, 
                pts=[self.vertices[i, :, :]], 
                color=self.colors[i, 0:3].tolist())
            shape_alpha = float(self.colors[i, 3]) / 255
            self.image = cv2.addWeighted(self.image, 1 - shape_alpha, shape_im, shape_alpha, gamma=0)
        return self.image

    def mutate(self, color_data=None):
        """
        Mutates the triangle image.
        The mutated triangles will be randomly chosen.
        The mutated triangles will either have a single vertex changed,
        its color changed.  
        """

        if (self.num_triangles != self.max_triangles) and \
           (np.random.random() < self.mutate_new_prob):
            self._generate_random_triangle(self.num_triangles, color_data=color_data)
            self.num_triangles += 1
            return

        # index of random triangle to mutate
        rand_idx = np.random.randint(0, self.num_triangles)

        # Perturbations are generated from a gaussian distribution
        # with the following standard deviations.
        rand_x_sigma = self.im_width * 0.05
        rand_y_sigma = self.im_height * 0.05
        rand_color_sigma = 20
        rand_alpha_sigma = 10
        if np.random.random() < self.mutate_pos_prob:
            # randomly perturb 1, 2, or 3 vertices
            num_vertices = np.random.randint(1, 4)
            rand_vertex = np.random.randint(0, 3, (num_vertices,))

            new_x = self.vertices[rand_idx, rand_vertex, 0] + np.array(np.rint(np.random.randn(num_vertices) * rand_x_sigma), dtype=np.int32)
            new_x = np.maximum(new_x, 0)
            new_x = np.minimum(new_x, self.im_width)
            self.vertices[rand_idx, rand_vertex, 0] = new_x

            new_y = self.vertices[rand_idx, rand_vertex, 0] + np.array(np.rint(np.random.randn(num_vertices) * rand_y_sigma), dtype=np.int32)
            new_y = np.maximum(new_y, 0)
            new_y = np.minimum(new_y, self.im_width)
            self.vertices[rand_idx, rand_vertex, 0] = new_y
        else:
            # randomly change color
            self.colors[rand_idx, 0:3] += \
                np.array(np.rint(np.random.randn() * rand_color_sigma), dtype=np.int16)
            self.colors[rand_idx, 3] += \
                np.array(np.rint(np.random.randn() * rand_alpha_sigma), dtype=np.int16)
            self.colors[rand_idx, :] = np.minimum(self.colors[rand_idx, :], 255)
            self.colors[rand_idx, :] = np.maximum(self.colors[rand_idx, :], 0)

    def _generate_random_triangle(self, idx, color_data=None):
        self.vertices[idx, :, 0] = np.random.randint(0, self.im_width, (3,))
        self.vertices[idx, :, 1] = np.random.randint(0, self.im_height, (3,))

        if color_data is None:
            self.colors[idx, :] = np.random.randint(0, 256, (4,), dtype=np.int16)
            return

        hist = cv2.calcHist(
            [self.render()], 
            channels=[0, 1, 2], 
            mask=None, 
            histSize=color_data["num_bins"], 
            ranges=color_data["ranges"])

        inv_temp = 1
        normalized_hist_diffs = (color_data["hist"] - hist) / (self.im_height * self.im_width)
        tmp = np.exp(normalized_hist_diffs * inv_temp) 
        prob_hist = tmp / np.sum(tmp)
        prob_hist = prob_hist.flatten()

        rand_color_idx = np.random.choice(len(prob_hist), p=prob_hist)
        unraveled_idx = np.unravel_index(rand_color_idx, color_data["num_bins"])

        color = [color_data["bin_centers"][i][unraveled_idx[i]] for i in range(3)]
        color_delta = np.array(np.rint(np.random.random() * color_data["bin_sizes"] * 2 - color_data["bin_sizes"]), dtype=np.int16)
        color = color + color_delta
        color = np.minimum(color, 255)
        color = np.maximum(color, 0)
        self.colors[idx, :3] = color 

        self.colors[idx, 3] = np.random.randint(0, 256, dtype=np.int16)

    def num_inactive_triangles(self):
        return len(self._get_inactive_triangle_idx())

    def remove_inactive(self):
        inactive_indices, active_indices = self._get_inactive_triangle_idx(return_active=True)
        if (len(inactive_indices) == 0):
            return

        active_vertices = np.copy(self.vertices[active_indices, :, :])
        self.vertices[:len(active_indices), :, :] = active_vertices

        active_colors = np.copy(self.colors[active_indices, :])
        self.colors[:len(active_indices), :] = active_colors

        self.num_triangles -= len(inactive_indices)

    def _get_inactive_triangle_idx(self, return_active=False):
        inactive_indices = []
        active_indices = []
        for i in range(self.num_triangles):
            if self.colors[i, 3] == 0:
                inactive_indices.append(i)
                continue

            pt1 = self.vertices[i, 0, :]
            pt2 = self.vertices[i, 1, :]
            pt3 = self.vertices[i, 2, :]
            area = 0.5 * ((pt1[0] * (pt2[1] - pt3[1])) + \
                          (pt2[0] * (pt3[1] - pt1[1])) + \
                          (pt3[0] * (pt1[1] - pt2[1])))
            area = np.abs(area)
            if area < 0.1:
                inactive_indices.append(i)
                continue
            active_indices.append(i)
        
        if return_active:
            return inactive_indices, active_indices
        else:
            return inactive_indices


def fitness_func(y, y_hat):
    cost = np.mean(np.sum(np.abs(y.astype(np.int16) - y_hat.astype(np.int16)), axis=2))
    return cost

# Costly function that will take a triangle_image, render it,
# and then compute the fitness function.
# This is separated out to be used with multiprocessing submodule
# to vastly speed up computation of large populations.
def _evaluate_item(triangle_image, target_image):
    rendered_image = triangle_image.render()
    cost = fitness_func(target_image, rendered_image)
    return cost

def _mutate(triangle_image, color_data):
    triangle_image.mutate(color_data)
    return triangle_image

def run_genetic_optimization(
    target_image, 
    pop_size, 
    max_triangles, 
    num_epochs=20, 
    save_dir=None,
    color_data=None):

    assert(len(target_image.shape) == 3)

    # Initialize a population of triangles images
    print("Generating Initial Random Population ...")
    im_height = target_image.shape[0]
    im_width = target_image.shape[1]
    population = [[0, TriangleImage(im_height, im_width, max_triangles)] for _ in range(pop_size)]
    for _, item in population:
        item.init()

    # Number of parents to keep from each generation
    num_parents = int(pop_size/10) + 1

    print("Running Optimization ...")
    pool = multiprocessing.Pool(12)
    for i in tqdm(range(num_epochs)):
        # Compute costs for each member of the population
        costs = pool.starmap(
            _evaluate_item,
            [(item[1], target_image) for item in population])
        for idx in range(len(population)):
            population[idx][0] = costs[idx]

        for j in range(num_parents):
            population[j][1].remove_inactive()

        # Sort population by fitness cost.
        # The best members of the population will be 
        # copied and mutated.
        population.sort(key=lambda x: x[0])
        for j in range(num_parents, pop_size):
            population[j] = [0, copy.deepcopy(population[j % num_parents][1])]

        # mutate_list = pool.starmap(
        #     _mutate,
        #     [(population[j][1], color_data) for j in range(num_parents, pop_size)])
        # for j in range(num_parents, pop_size):
        #     population[j][1] = mutate_list[j-num_parents]
        for j in range(num_parents, pop_size):
            population[j][1].mutate(color_data)

        if i % 10 == 0:
            print("Epoch {}: num_triangles={}, num_inactive={}, lowest costs={}".format(
                i, 
                population[0][1].get_num_triangles(),
                population[0][1].num_inactive_triangles(),
                [np.round(x[0], 3) for x in population[:np.minimum(pop_size, 3)]]))

        # Save an intermediate image if needed.
        if save_dir != None:
            save = False
            if i < 300 and i % 20 == 0:
                save = True
            elif i < 1000 and i % 100 == 0:
                save = True
            elif i < 3000 and i % 200 == 0:
                save = True
            elif i < 15000 and i % 500 == 0:
                save = True
            elif i % 1000 == 0:
                save = True
            if save: 
                image_path = os.path.join(save_dir, "image_{}.png".format(i))
                imageio.imwrite(image_path, population[0][1].render())

        if i % 100 == 0 and save_dir != None:
            path = os.path.join(save_dir, "data.pickle")
            pickle.dump(TriangleImage, open(path, "wb"))

    return population[0]

if __name__ == "__main__":
    import argparse    
    parser = argparse.ArgumentParser(description="Approximate images with triangles.")
    parser.add_argument(
        "--image",
        action="store",
        type=str,
        required=True,
        help="Path to image.")
    parser.add_argument(
        "--nepochs",
        action="store",
        type=int,
        default=2000,
        help="Number of epochs for optimization.")
    parser.add_argument(
        "--pop",
        action="store",
        type=int,
        default=50,
        help="Population size.")
    parser.add_argument(
        "--max_triangles",
        action="store",
        type=int,
        default=100,
        help="Number of triangles used to approximate image.")
    parser.add_argument(
        "--savedir",
        action="store",
        type=str,
        default="images/out",
        help="Directory to save final output and intermediate steps.")
    parser.add_argument(
        "--use_hist",
        action="store_true",
        help="Set this flag to randomly generate triangle colors \
              based on target image color histogram.")
    args = parser.parse_args()

    target_image = imageio.imread(args.image)

    # if image is greyscale, copy to all color coordinates
    if len(target_image.shape) == 2:
        target_image = np.stack([target_image for _ in range(3)])

    # if image has an alpha channel, get rid of it
    if target_image.shape[2] == 4:
        target_image = target_image[:, :, :3]

    if args.use_hist:
        color_data = dict()
        color_data["num_bins"] = [10 for _ in range(3)]
        color_data["ranges"] = [0, 255, 0, 255, 0, 255]
        color_data["hist"] = cv2.calcHist(
            [target_image], 
            channels=[0, 1, 2], 
            mask=None, 
            histSize=color_data["num_bins"], 
            ranges=color_data["ranges"])
        color_data["bin_centers"] = []
        for i in range(3):
            edges = np.linspace(0, 255, color_data["num_bins"][i] + 1)
            color_data["bin_centers"].append((edges[:-1] + edges[1:]) / 2)
        color_data["bin_sizes"] = np.array([255./x for x in color_data["num_bins"]])
    else:
        color_data = None

    sol = run_genetic_optimization(
        target_image, 
        pop_size=args.pop, 
        max_triangles=args.max_triangles,
        num_epochs=args.nepochs,
        save_dir=args.savedir,
        color_data=color_data)
    sol_image = sol[1].render()

    image_path = os.path.join(args.savedir, "image_{}.png".format(args.nepochs))
    imageio.imwrite(image_path, sol_image)

    plt.imshow(sol_image)
    plt.show()



