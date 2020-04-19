import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
import matplotlib.patches as mpatches


def make_animation(x_list, u_list):

    def update_animation(num, fig, x, u, quad_line, state_t_line, state_pts, control_t_line, control_pts):
        ax_list = fig.axes

        # update quadrotor line
        curr_x = x[num, :]
        curr_u = u[num, :]
        theta = curr_x[4]

        x1 = curr_x[0] - np.cos(theta) * 0.5
        y1 = curr_x[1] - np.sin(theta) * 0.5
        x2 = curr_x[0] + np.cos(theta) * 0.5
        y2 = curr_x[1] + np.sin(theta) * 0.5
        quad_line[0].set_data([x1, x2], [y1, y2])


        # update u1 and u2 force arrows
        u1_dx = -np.sin(theta) * curr_u[0] / 10 
        u1_dy = np.cos(theta) * curr_u[0] / 10
        u2_dx = -np.sin(theta) * curr_u[1] / 10 
        u2_dy = np.cos(theta) * curr_u[1] / 10
        if update_animation.u1_patch is not None:
            ax_list[0].patches.remove(update_animation.u1_patch)
        if update_animation.u2_patch is not None:
            ax_list[0].patches.remove(update_animation.u2_patch)
        
        update_animation.u1_patch = FancyArrow(x1, y1, u1_dx, u1_dy, color="xkcd:pinkish red", width=0.03)
        ax_list[0].add_patch(update_animation.u1_patch)

        update_animation.u2_patch = FancyArrow(x2, y2, u2_dx, u2_dy, color="xkcd:pinkish red", width=0.03)
        ax_list[0].add_patch(update_animation.u2_patch)

        # update state graph
        curr_t = num * 0.1
        state_t_line[0].set_data([curr_t, curr_t], [-100, 100])
        pt_arr = np.zeros((6, 2))
        pt_arr[:, 1] = x[num, :]
        pt_arr[:, 0] = num * 0.1
        state_pts.set_offsets(pt_arr)

        # update control graph
        control_t_line[0].set_data([curr_t, curr_t], [-100, 100])
        pt_arr = np.zeros((2, 2))
        pt_arr[:, 1] = u[num, :]
        pt_arr[:, 0] = num * 0.1
        control_pts.set_offsets(pt_arr)

        return (quad_line, 
            update_animation.u1_patch, 
            update_animation.u2_patch, 
            state_t_line, 
            state_pts,
            control_t_line,
            control_pts)

    update_animation.u1_patch = None
    update_animation.u2_patch = None
        
    # quadrotor animation
    fig = plt.figure(figsize=[6.4, 7.2], dpi=100)
    ax1 = plt.subplot(3, 1, 1)

    # scatter all the points to get the xlim and ylim
    # then clear the axis for actual drawing
    plt.scatter(x_list[:, 0], x_list[:, 1])
    curr_xlim = plt.xlim()
    curr_ylim = plt.ylim()
    plt.cla()
    plt.xlim(curr_xlim)
    plt.ylim(curr_ylim)

    plt.xlim(-9, 9)
    plt.ylim(-5, 13)

    plt.grid()
    quad_line = plt.plot(
        [x_list[0, 0], x_list[1, 0]], 
        [x_list[0, 1], x_list[1, 1]],
        c="xkcd:steel blue",
        linewidth=3,
        label="Quadrotor")
    
    quadrotor_patch = mpatches.Patch(color="xkcd:steel blue", label="Quadrotor")
    control_patch = mpatches.Patch(color="xkcd:pinkish red", label="Force Applied")
    plt.legend(handles=[quadrotor_patch, control_patch])

    # State plot
    ax2 = plt.subplot(3, 1, 2)
    # TODO: replace 0.1 with dt
    t = [0.1 * i for i in range(len(u_list))]
    plt.plot(t, x_list[:-1, 0], 
        label="x pos (m)", 
        c="xkcd:ocean blue",
        linestyle="--",
        alpha=0.5)
    plt.plot(t, x_list[:-1, 1], 
        label="y pos (m)", 
        c="xkcd:pinkish red",
        linestyle="--",
        alpha=0.5)
    plt.plot(t, x_list[:-1, 2], 
        label="x vel (m/s)", 
        c="xkcd:moss green",
        linestyle="--",
        alpha=0.5)
    plt.plot(t, x_list[:-1, 3], 
        label="y vel (m/s)", 
        c="xkcd:tangerine",
        linestyle="--",
        alpha=0.5)
    plt.plot(t, x_list[:-1, 4], 
        label="theta (rad)", 
        c="xkcd:medium purple",
        linestyle="--",
        alpha=0.5)
    plt.plot(t, x_list[:-1, 5], 
        label="theta dot (rad/s)", 
        c="xkcd:dark gold",
        linestyle="--",
        alpha=0.5)
    curr_xlim = plt.xlim()
    curr_ylim = plt.ylim()
    state_t_line = plt.plot([0, 0], [-100, 100], c="black")
    state_pts = plt.scatter(
        [0 for _ in range(6)], 
        x_list[0, :], 
        c=["xkcd:ocean blue", 
           "xkcd:pinkish red", 
           "xkcd:moss green", 
           "xkcd:tangerine", 
           "xkcd:medium purple", 
           "xkcd:dark gold"])
    plt.xlim(curr_xlim)
    plt.ylim(curr_ylim)
    plt.legend(loc=1)
    plt.grid()

    # Control plot
    ax3 = plt.subplot(3, 1, 3)
    plt.plot(t, u_list[:, 0], 
        label="u1 (m/s/s)",
        c="xkcd:ocean blue",
        linestyle="--",
        alpha=0.5)
    plt.plot(t, u_list[:, 1], 
        label="u2 (m/s/s)",
        c="xkcd:tangerine",
        linestyle="--",
        alpha=0.5)
    curr_xlim = plt.xlim()
    curr_ylim = plt.ylim()
    control_t_line = plt.plot([0, 0], [-100, 100], c="black")
    control_pts = plt.scatter(
        [0 for _ in range(2)], 
        u_list[0, :], 
        c=["xkcd:ocean blue", 
           "xkcd:tangerine"])
    plt.xlim(curr_xlim)
    plt.ylim(curr_ylim)
    plt.xlabel("time (s)")
    plt.legend(loc=1)
    plt.grid()

    anim = animation.FuncAnimation(
        fig, 
        update_animation, 
        len(u_list), 
        fargs=(fig, x_list, u_list, quad_line, state_t_line, state_pts, control_t_line, control_pts),
        interval=50, 
        blit=False)

    # # Uncomment anim.save() to save the animation.
    # # Requires imagemagick to be installed
    # anim.save('images/quadrotor_optimization.gif', writer='imagemagick', fps=30)

    plt.show()



