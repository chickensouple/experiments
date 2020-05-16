import numpy as np
import tensorflow as tf
import time

import opt_solver
from make_animation import make_animation

def quat_mult(q1, q2):
    """
    Quaternion multiplication.
    Computes q1 * q2

    where the quaternions are represented by 
    a 4d array (w, x, y, z)

    Arguments:
        q1 {tf.Tensor}
        q2 {tf.Tensor}
    """
    qw = (q1[0]*q2[0]) - (q1[1]*q2[1]) - (q1[2]*q2[2]) - (q1[3]*q2[3])
    qx = (q1[0]*q2[1]) + (q1[1]*q2[0]) + (q1[2]*q2[3]) - (q1[3]*q2[2])
    qy = (q1[0]*q2[2]) - (q1[1]*q2[3]) + (q1[2]*q2[0]) + (q1[3]*q2[1])
    qz = (q1[0]*q2[3]) + (q1[1]*q2[2]) - (q1[2]*q2[1]) + (q1[3]*q2[0])
    quat = tf.stack([qw, qx, qy, qz])
    return quat

def quat_inv(q):
    quat = tf.stack([q[0], -q[1], -q[2], -q[3]])
    return quat

def compute_ground_effect(x, u):
    com = x[0:3]
    quat = x[6:10]
    motor_vecs = []
    motor_vecs.append(quat_mult(quat_mult(quat, np.array([0, 1, 0, 0.], np.float32)), quat_inv(quat))[1:])
    motor_vecs.append(quat_mult(quat_mult(quat, np.array([0, 0, -1, 0.], np.float32)), quat_inv(quat))[1:])
    motor_vecs.append(quat_mult(quat_mult(quat, np.array([0, -1, 0, 0.], np.float32)), quat_inv(quat))[1:])
    motor_vecs.append(quat_mult(quat_mult(quat, np.array([0, 0, 1, 0.], np.float32)), quat_inv(quat))[1:])

    rotated_vec = quat_mult(quat_mult(quat, np.array([0, 0, 0, -1.], np.float32)), quat_inv(quat))[1:]
    down_vec = np.array([0, 0, -1], dtype=quat.dtype)
    angle = tf.math.acos(tf.tensordot(down_vec, rotated_vec, axes=1))

    alpha = 0.5
    dmax = 2.0
    u_ge = []
    for i in range(4):
        distance = tf.nn.relu(motor_vecs[i][2] + com[2])
        dist_frac = tf.nn.relu(1.0 - (distance / dmax))
        angle = tf.math.minimum(angle, np.pi*0.5)
        angle_frac = tf.math.square(angle - 0.5*np.pi) * 4 / (np.pi**2)

        u_ge.append(u[i] * alpha * angle_frac * dist_frac)
    
    u_ge = tf.stack(u_ge)
    print("u_ge: {}".format(u_ge))
    return u_ge


class Quadrotor3D(opt_solver.DiscreteSystemModel):
    """
    Simple 3D Quadrotor model where the motors are (u0, u1, u2, u2)
    and the body x axis is aligned with the u0 motor.

      x ^ 
        |   
    y <--   O u0
            |
     u3 O---c---O u1
            |
            O u2

    Each of the "arms" of the quadrotor have unit length.
    The state space of the quadrotor is
    [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    where (x, y, z) is the position of the center of mass (COM) in inertial frame
          (vx, vy, vz) is the velocity of the COM in inertial frame
          (qw, qx, qy, qz) is a quarternion representing orientation of the quadrotor
          (wx, wy, wz) is the angular velocity (in body frame)

    the orientation q, transforms vectors from body frame to inertial frame      
    """
    def __init__(self, use_ground_effect=False):
        super().__init__(state_dim=13, control_dim=4)
        self.dt = 0.05
        self.g = 9.81

        self.inertia_mat = np.eye(3, dtype=np.float32)
        self.inertia_mat_inv = np.linalg.inv(self.inertia_mat)
        self.use_ground_effect = use_ground_effect

    def step(self, x, u):
        self._check_input(x, u)
        deriv_list = []

        T = len(x)
        for t in range(T):
            curr_x = x[t, :]
            curr_u = u[t, :]

            if self.use_ground_effect:
                curr_u += compute_ground_effect(curr_x, curr_u)


            # compute forces and torques in body frame
            F = tf.reduce_sum(curr_u)
            Tx = curr_u[3] - curr_u[1]
            Ty = curr_u[2] - curr_u[0]
            Tz = (curr_u[0] + curr_u[2]) - (curr_u[1] + curr_u[3])
            T = tf.stack([Tx, Ty, Tz])

            # computing pos and velocity derivatives
            F_body = tf.stack([0, 0, 0, F])
            curr_quat = curr_x[6:10]
            F_inertial = quat_mult(quat_mult(curr_quat, F_body), quat_inv(curr_quat))

            pos_deriv = curr_x[3:6]
            vel_xy_deriv = F_inertial[1:3]
            vel_z_deriv = F_inertial[3:] - self.g

            # computing quaternion and angular velocity derivatives
            w = tf.stack([curr_x[10], curr_x[11], curr_x[12]])
            tf_zero = tf.zeros((1,), dtype=curr_x.dtype)

            w_q =  tf.concat([tf_zero, curr_x[10:]], axis=0)
            quat_deriv = 0.5 * quat_mult(w_q, curr_quat)
            angular_vel_deriv = T - tf.linalg.cross(w, tf.linalg.matvec(self.inertia_mat, w))
            angular_vel_deriv = tf.linalg.matvec(self.inertia_mat_inv, angular_vel_deriv)
            deriv = tf.concat([pos_deriv, vel_xy_deriv, vel_z_deriv, quat_deriv, angular_vel_deriv], axis=0)

            deriv_list.append(deriv)
        all_deriv = tf.stack(deriv_list)
        x_next = x + all_deriv * self.dt
        
        x_next_0 = x_next[:, 0:6]
        x_next_quat = x_next[:, 6:10]
        x_next_1 = x_next[:, 10:]

        quat_norms = tf.linalg.norm(x_next_quat, axis=1)
        x_next_quat = x_next_quat / tf.reshape(quat_norms, (-1, 1))

        x_next = tf.concat([x_next_0, x_next_quat, x_next_1], axis=1)
        return x_next


    def draw_quad(self, ax, state, color1="blue", color2="red"):
        pos = state[0:3]
        quat = state[6:10]
        try:
            u0_motor = quat_mult(quat_mult(quat, np.array([0, 1, 0, 0.], dtype=np.float32)), quat_inv(quat))
            u1_motor = quat_mult(quat_mult(quat, np.array([0, 0, -1, 0.], dtype=np.float32)), quat_inv(quat))
            u2_motor = quat_mult(quat_mult(quat, np.array([0, -1, 0, 0.], dtype=np.float32)), quat_inv(quat))
            u3_motor = quat_mult(quat_mult(quat, np.array([0, 0, 1, 0.], dtype=np.float32)), quat_inv(quat))
        except:
            u0_motor = quat_mult(quat_mult(quat, np.array([0, 1, 0, 0.], dtype=np.float64)), quat_inv(quat))
            u1_motor = quat_mult(quat_mult(quat, np.array([0, 0, -1, 0.], dtype=np.float64)), quat_inv(quat))
            u2_motor = quat_mult(quat_mult(quat, np.array([0, -1, 0, 0.], dtype=np.float64)), quat_inv(quat))
            u3_motor = quat_mult(quat_mult(quat, np.array([0, 0, 1, 0.], dtype=np.float64)), quat_inv(quat))

        ax.plot3D([pos[0], pos[0]+u0_motor[1]],
                  [pos[1], pos[1]+u0_motor[2]],
                  [pos[2], pos[2]+u0_motor[3]], c=color2)
        ax.plot3D([pos[0], pos[0]+u1_motor[1]],
                  [pos[1], pos[1]+u1_motor[2]],
                  [pos[2], pos[2]+u1_motor[3]], c=color1)
        ax.plot3D([pos[0], pos[0]+u2_motor[1]],
                  [pos[1], pos[1]+u2_motor[2]],
                  [pos[2], pos[2]+u2_motor[3]], c=color1)
        ax.plot3D([pos[0], pos[0]+u3_motor[1]],
                  [pos[1], pos[1]+u3_motor[2]],
                  [pos[2], pos[2]+u3_motor[3]], c=color1)
 
        # F_body = tf.convert_to_tensor(np.array([0., 0., 0., 1.]))
        # F_inertial = quat_mult(quat_mult(quat, F_body), quat_inv(quat))
        # ax.plot3D([pos[0], pos[0]+F_inertial[1]],
        #           [pos[1], pos[1]+F_inertial[2]],
        #           [pos[2], pos[2]+F_inertial[3]], c='green')
        

class Quadrotor3DGoToPointProblem(opt_solver.ScipyMPCOptProblem):
    """
    An MPC problem that tries to get the quadrotor
    to move to the origin and stay there.
    You can use set_initial_state() to give the problem
    different initial states to start the quadrotor at.
    """
    def __init__(self, T=50):
        super().__init__(
            T=T,
            sys_model=Quadrotor3D())

        self.problem_kwargs = {
            "control_max": tf.convert_to_tensor(10.),
            "control_min": tf.convert_to_tensor(0.),
            "initial_loc": 
                tf.convert_to_tensor(np.array([1., 0., 10., 
                                               0, 0, 0,
                                               1., 0., 0., 0.,
                                               0., 0., 0.], dtype=np.float32)),
            "target_loc": tf.convert_to_tensor(np.array([0, 0., 1.], dtype=np.float32))
        }

    def initial_guess(self):
        state = np.zeros((self.T, self.state_dim))
        control = np.zeros((self.T - 1, self.control_dim)) 
        
        state[:, 0] = np.linspace(
            self.problem_kwargs["initial_loc"][0], 
            self.problem_kwargs["target_loc"][0], 
            self.T)
        state[:, 1] = np.linspace(
            self.problem_kwargs["initial_loc"][1], 
            self.problem_kwargs["target_loc"][1], 
            self.T)
        state[:, 2] = np.linspace(
            self.problem_kwargs["initial_loc"][2], 
            self.problem_kwargs["target_loc"][2], 
            self.T)
        state[:, 6] = 1

        state = tf.convert_to_tensor(state)
        control = tf.convert_to_tensor(control)
        return (state, control)

    def eval_obj(self, state, control, **kwargs):
        diff = state[:, 0:3] - kwargs["target_loc"]
        # weighted_diffs = tf.reduce_sum(tf.square(diff), axis=1) * tf.linspace(1., 1000., self.T)
        weighted_diffs = tf.reduce_sum(tf.square(diff), axis=1) * np.logspace(0., 2., self.T).astype(np.float32)

        obj = tf.reduce_mean(weighted_diffs)
        return obj

    def eval_ineq_constraints(self, state, control, **kwargs):
        # control constraints
        u_min = kwargs['control_min'] - control
        u_max = control - kwargs['control_max']
        u_min = tf.reshape(u_min, (-1, 1))
        u_max = tf.reshape(u_max, (-1, 1))

        # make sure com is above ground
        state_constraints = tf.reshape(0.2 - state[:, 2], (-1, 1))

        # orientation constraint on last state
        eps = 0.05
        q_min = 1 - eps - state[-1:, 6]
        q_max = state[-1:, 6] - 1 - eps
        q_min = tf.reshape(q_min, (-1, 1))
        q_max = tf.reshape(q_max, (-1, 1))
        constraints = tf.concat([u_min, u_max, q_min, q_max, state_constraints], axis=0)

        return constraints

    def eval_eq_constraints(self, state, control, **kwargs):
        start_state_constraints = state[0, :] - kwargs["initial_loc"]
        start_state_constraints = tf.reshape(start_state_constraints, (-1, 1))

        return start_state_constraints

    def get_initial_state(self):
        return self.problem_kwargs["initial_loc"]

    def set_initial_state(self, initial_loc):
        initial_loc = np.array(initial_loc, dtype=np.float32)
        initial_loc = np.reshape(initial_loc, (13,))
        self.problem_kwargs["initial_loc"] = tf.convert_to_tensor(initial_loc)

    def set_target_loc(self, target_loc):
        target_loc = np.array(target_loc, dtype=np.float32)
        target_loc = np.reshape(target_loc, (3,))
        self.problem_kwargs["target_loc"] = tf.convert_to_tensor(target_loc)


if __name__ == "__main__":
    # quad = Quadrotor3D(use_ground_effect=True)
    # state = np.array([[0., 0., 0.1, 
    #                           0., 0., 0.,
    #                           0.9659258, 0., 0.258819, 0.,
    #                           0., 0., 0]], dtype=np.float32)
    # control = np.array([[1., 2., 2., 3.]], dtype=np.float32)
    # quad.step(state, control)
    # exit()


    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    def set_axes_equal(ax):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    import argparse
    parser = argparse.ArgumentParser(description="Run Quadrotor optimization")
    parser.add_argument(
        "-T",
        action="store",
        type=int,
        default=50,
        help="Time horizon for problem.")
    parser.add_argument(
        "-x",
        action="store",
        type=float,
        default=4.0, 
        help="Starting x location.")
    parser.add_argument(
        "-y",
        action="store",
        type=float,
        default=10.0,
        help="Starting y location.")
    parser.add_argument(
        "--theta",
        action="store",
        type=float,
        default=-np.pi/2,
        help="Starting orientation (in radians).")
    subparser = parser.add_subparsers(
        help="Run the optimizer once and plot results, or run mpc.",
        dest="cmd")
    once_subparser = subparser.add_parser("once")
    mpc_subparser = subparser.add_parser("mpc")
    mpc_subparser.add_argument(
        "--niter",
        action="store",
        type=int,
        default=20,
        help="Number of times to rerun optimizer for MPC.")
    args = parser.parse_args()


    problem = Quadrotor3DGoToPointProblem(T=args.T)

    initial_state = np.array([5., 5., 1.5, 
                              0., 0., 0.,
                              1., 0., 0., 0.,
                              0., 0., 0])
    # initial_state = np.array([5., 2., 10., 
    #                           0., 0., 0.,
    #                           0.9659258, 0., 0.258819, 0.,
    #                           0., 0., 0])
    problem.set_initial_state(initial_state)

    if args.cmd == "mpc":
        num_steps = 10
        curr_state = problem.get_initial_state()
        # initial solve to do all the tensorflow tracing
        # so that the timing won't take tracing time into consideration
        problem.solve(maxiter=200, ftol=1e-3)
        x_list = []
        u_list = []
        start_time = time.clock()
        for t in range(args.niter):
            problem.set_initial_state(curr_state)
            sol, x, u = problem.solve()
            for i in range(num_steps):
                x_list.append(x[i, :])
                u_list.append(u[i, :])
            curr_state = x_list[-1]
        x_list.append(x[-1, :])
        end_time = time.clock()
        x_list = np.array(x_list)
        u_list = np.array(u_list)
        print("Average time / solve: {} seconds".format((end_time - start_time) / args.niter))
    else:
        start_time = time.clock()
        sol, x_list, u_list = problem.solve(maxiter=500, ftol=1e-3)
        end_time = time.clock()
        print(sol)
        print("Solve time: {} seconds".format(end_time - start_time))

        quad_ge = Quadrotor3D(use_ground_effect=True)
        curr_x = x_list[0]
        ge_x_list = [curr_x]
        for t in range(args.T-1):
            input_x = np.reshape(curr_x, (1, -1)).astype(np.float32)
            input_u = np.array(u_list[t:t+1, :], dtype=np.float32)
            curr_x = quad_ge.step(input_x, input_u)
            ge_x_list.append(curr_x.numpy().squeeze())
        ge_x_list = np.array(ge_x_list)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        quad = Quadrotor3D()
        for t in range(args.T):
            state = x_list[t, :]
            quad.draw_quad(ax, state)

            quad.draw_quad(ax, ge_x_list[t], 
                color1=(1., 0., 1., 0.5), 
                color2=(0., 1., 0., 0.5))
        set_axes_equal(ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

