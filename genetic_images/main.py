import copy
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import imageio
import multiprocessing
from tqdm import tqdm


class TriangleImage(object):
    """
    This class represents a set of triangles
    to be rendered onto an image.
    Each triangle is represented by its 3 vertices and an rgba color.
    """
    def __init__(self, im_height, im_width, num_triangles):
        self.im_height = im_height
        self.im_width = im_width
        self.num_triangles = num_triangles

        self.image = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        
        # (x1, y1, x2, y2, x3, y3)
        self.vertices = np.zeros((self.num_triangles, 3, 2), dtype=np.int32)
        
        # rgba. all numbers in range [0, 255]
        # this uses int16 so that arithmetic won't over/underflow
        self.colors = np.zeros((self.num_triangles, 4), dtype=np.int16)

        self.mutate_frac = 0.05

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

    def mutate(self, num_mutate, regen_inactive=False):
        """
        Mutates "num_mutate" triangles.
        The mutated triangles will be randomly chosen.
        The mutated triangles will either have a single vertex changed,
        its color changed.  
        
        Arguments:
            num_mutate {int} -- Number of triangles to mutate.
        """

        num_inactive = 0
        if regen_inactive and np.random.random() < 0.5:
            # with 0.5 probability, try to regenerate as many inactive triangles as possible
            inactive_idx = self._get_inactive_triangle_idx()
            num_inactive = np.minimum(len(inactive_idx), num_mutate)
            rand_inactive_idx = np.random.choice(inactive_idx, (num_inactive,))

            self.vertices[rand_inactive_idx, :, 0] = np.random.randint(0, self.im_width, (num_inactive, 3))
            self.vertices[rand_inactive_idx, :, 1] = np.random.randint(0, self.im_height, (num_inactive, 3))
            self.colors[rand_inactive_idx, :] = np.random.randint(0, 256, (num_inactive, 4), dtype=np.int16)

            if np.min(inactive_idx) < self.num_triangles - len(inactive_idx) - (self.num_triangles / 10):
                self._reorder_inactive()

        # choose "num_mutate" random triangles
        num_mutate = num_mutate - num_inactive
        if num_mutate == 0:
            return 
        rand_idx = np.random.randint(0, self.num_triangles, (num_mutate))
        

        # Mutation parameters:
        # Each vertex location or color
        # will be moved by drawing a difference
        # from a gaussian distribution.
        rand_x_sigma = self.im_width * 0.05
        rand_y_sigma = self.im_height * 0.05
        rand_color_sigma = 20
        rand_alpha_sigma = 10
        if np.random.random() < 0.5:
            rand_vertex = np.random.randint(0, 3, (num_mutate,))
            x_rand = np.zeros((num_mutate, 3), dtype=np.int32)
            x_rand[np.arange(rand_vertex.size), rand_vertex] = \
                np.array(np.rint(np.random.randn(num_mutate,) * rand_x_sigma), dtype=np.int32)
            self.vertices[rand_idx, :, 0] += x_rand
            self.vertices[rand_idx, :, 0] = np.maximum(self.vertices[rand_idx, :, 0], 0)
            self.vertices[rand_idx, :, 0] = np.minimum(self.vertices[rand_idx, :, 0], self.im_width)

            y_rand = np.zeros((num_mutate, 3), dtype=np.int32)
            y_rand[np.arange(rand_vertex.size), rand_vertex] = \
                np.array(np.rint(np.random.randn(num_mutate,) * rand_y_sigma), dtype=np.int32)
            self.vertices[rand_idx, :, 1] += y_rand
            self.vertices[rand_idx, :, 1] = np.maximum(self.vertices[rand_idx, :, 1], 0)
            self.vertices[rand_idx, :, 1] = np.minimum(self.vertices[rand_idx, :, 1], self.im_height)
        else:
            self.colors[rand_idx, 0:3] += \
                np.array(np.rint(np.random.randn(num_mutate, 3) * rand_color_sigma), dtype=np.int16)
            self.colors[rand_idx, 3] += \
                np.array(np.rint(np.random.randn(num_mutate,) * rand_alpha_sigma), dtype=np.int16)
            self.colors = np.minimum(self.colors, 255)
            self.colors = np.maximum(self.colors, 0)

    def get_num_inactive_triangles(self):
        return len(self._get_inactive_triangle_idx())

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
            if area < 0.1:
                inactive_indices.append(i)
                continue
            active_indices.append(i)
        
        if return_active:
            return inactive_indices, active_indices
        else:
            return inactive_indices

    def _reorder_inactive(self):
        inactive_indices, active_indices = self._get_inactive_triangle_idx(return_active=True)

        active_vertices = np.copy(self.vertices[active_indices, :, :])
        inactive_vertices = np.copy(self.vertices[inactive_indices, :, :])
        self.vertices[:len(active_indices), :, :] = active_vertices
        self.vertices[len(active_indices):, :, :] = inactive_vertices

        
        active_colors = np.copy(self.colors[active_indices, :])
        inactive_colors = np.copy(self.colors[inactive_indices, :])
        self.colors[:len(active_indices), :] = active_colors
        self.colors[len(active_indices):, :] = inactive_colors

    def randomize(self):
        self.vertices[:, :, 0] = np.random.randint(0, self.im_width, (self.num_triangles, 3))
        self.vertices[:, :, 1] = np.random.randint(0, self.im_height, (self.num_triangles, 3))
        self.colors = np.random.randint(0, 256, self.colors.shape, dtype=np.int16)


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

def _mutate(triangle_image, num_mutate, regen_inactive):
    triangle_image.mutate(num_mutate, regen_inactive=regen_inactive)

def run_genetic_optimization(target_image, pop_size, num_triangles, num_epochs=20, save_dir=None):
    assert(len(target_image.shape) == 3)

    # Initialize a population of triangles images
    print("Generating Initial Random Population ...")
    im_height = target_image.shape[0]
    im_width = target_image.shape[1]
    population = [[0, TriangleImage(im_height, im_width, num_triangles)] for _ in range(pop_size)]
    for _, item in population:
        item.randomize()

    # Number of parents to keep from each generation
    num_parents = int(pop_size/10) + 1

    # The number of mutations each image will undergo.
    # This number will be annealed down to 1. 
    # It acts somewhat like a learning rate.
    # We start off high, and go low.
    num_mutate = 5

    # We determine when to reduce the number of mutations
    # by looking at how often the best cost has changed.
    # If the best cost has stayed the same for a while,
    # we will reduce the number of mutations.
    prev_best_cost = None
    changed_cost_list = []
    changed_cost_list_max_len = 20
    regen_inactive = False

    print("Running Optimization ...")

    pool = multiprocessing.Pool(12)
    for i in tqdm(range(num_epochs)):
        # Compute costs for each member of the population
        costs = pool.starmap(
            _evaluate_item,
            [(item[1], target_image) for item in population])
        for idx in range(len(population)):
            population[idx][0] = costs[idx]

        # Sort population by fitness cost.
        # The best members of the population will be 
        # copied and mutated.
        population.sort(key=lambda x: x[0])
        for j in range(num_parents, pop_size):
            population[j] = [0, copy.deepcopy(population[j % num_parents][1])]

        mutate_list = pool.starmap(
            _mutate,
            [(population[j][1], num_mutate, regen_inactive) for j in range(num_parents, pop_size)])
    
        for j in range(num_parents, pop_size):
            population[j][1].mutate(5, regen_inactive=regen_inactive)
        
        if i % 10 == 0:
            print("Epoch {}: num_inactive={}, lowest costs={}".format(
                i, 
                population[0][1].get_num_inactive_triangles(),
                [np.round(x[0], 3) for x in population[:np.minimum(pop_size, 3)]]))

        # Compute cost metrics to determine
        # if the number of mutations should change.
        best_cost = population[0][0]
        changed_cost_list.append(prev_best_cost != best_cost)
        if len(changed_cost_list) > changed_cost_list_max_len:
            changed_cost_list.pop(0)
        prev_best_cost = population[0][0]
        
        # When the cost metrics trigger a re-adjustment
        # in how mutations are done, we will 
        # 1) lower the number of mutations until it hits 1
        # 2) lower the number of parents is reduced by half
        # 3) turn on regen_inactive
        frac_changed = float(np.sum(changed_cost_list)) / len(changed_cost_list)
        if frac_changed < 0.7 and \
           len(changed_cost_list) == changed_cost_list_max_len:
            if num_mutate != 1:
                num_mutate = num_mutate - 1
                changed_cost_list = []
                print("Reducing Mutations to {}".format(num_mutate))
            elif num_parents > ((pop_size/20) + 1):
                num_parents = num_parents - 1
                changed_cost_list = []
                print("Reducing number of parents to {}".format(num_parents))
            elif (not regen_inactive):
                regen_inactive = True
                print("Turning on regen_inactive")

        # Save an intermediate image if needed.
        if save_dir != None and i % 100 == 0:
            image_path = os.path.join(save_dir, "image_{}.png".format(i))
            imageio.imwrite(image_path, population[0][1].render())

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
        "--num_triangles",
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
    args = parser.parse_args()



    target_image = imageio.imread(args.image)
    if len(target_image.shape) == 2:
        target_image = np.stack([target_image for _ in range(3)])
    if target_image.shape[2] == 4:
        target_image = target_image[:, :, :3]

    sol = run_genetic_optimization(
        target_image, 
        pop_size=args.pop, 
        num_triangles=args.num_triangles,
        num_epochs=args.nepochs,
        save_dir=args.savedir)
    sol_image = sol[1].render()

    image_path = os.path.join(args.savedir, "image.png")
    imageio.imwrite(image_path, sol_image)

    plt.imshow(sol_image)
    plt.show()


