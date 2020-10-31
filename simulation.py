"""
Simulation 3d rotation of a T-handle.

Simulation visuals inspired by https://www.youtube.com/watch?v=68aFSgn3LAY.
"""

import numpy as np
import vpython as vp
from scipy.integrate import ode


class RigidBody:
    """ Parent class for all rigid bodies which holds general info like mass and state. """

    def __init__(self, mass=1, x=np.zeros(3), R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), omega=np.zeros(3),
                 part_of_compound=False):
        """
        Note
        ----
        The position vector x will remain constant since linear velocity is ignored.

        Parameters
        ----------
        mass : int or float, optional
        x : ndarray, optional
            Position vector in world-space.
        R : ndarray, optional
            Rotation matrix for principal axes.
        omega : ndarray, optional
            Angular velocity vector in body-space.
        part_of_compound : bool, optional
            If part of a rigid body compound/composite.

        Attributes
        ----------
        self.I_body : ndarray
            Inertia tensor for body space.

        """
        # Constant quantities
        self.mass = mass
        self.I_body = np.zeros((3, 3))

        # State variables
        self.x = x
        self.R = R
        self.omega = omega

        self.part_of_compound = part_of_compound


class Block(RigidBody):
    """ Cuboid rigid body. """

    def __init__(self, dimensions=np.array([1, 1, 1]), **kwargs):
        """
        Parameters
        ----------
        dimensions : ndarray, optional
            Dimension of block (length, width, height)
        **kwargs
            Keyword arguments.

        """
        super().__init__(**kwargs)
        # Constant quantities
        self.dimensions = dimensions
        self.length, self.width, self.height = self.dimensions

        # Calculate and setup initial attributes of the Block
        self.calculate_I_body()
        self.center_of_mass = self.x

    def calculate_I_body(self):
        """ Calculate and set the correct inertia matrix (in body space) for the block. """
        # Define variables for calculation
        M = self.mass
        x_0, y_0, z_0 = self.dimensions
        # Create inertia matrix and calculate correct values
        I_body = np.array([
            [M / 12 * (y_0 ** 2 + z_0 ** 2), 0, 0],
            [0, M / 12 * (x_0 ** 2 + z_0 ** 2), 0],
            [0, 0, M / 12 * (x_0 ** 2 + y_0 ** 2)]
        ], dtype='float64')
        self.I_body = I_body


class TRod(RigidBody):
    """ 3 rod-like Block objects connected in a T-shape. """
    def __init__(self, rod_masses=np.array([1, 1, 1]), rod_dimensions=np.array([[8, 1, 1], [8, 1, 1], [5, 1, 1]]),
                 **kwargs):
        """
        TRod structure of the rods A, B, and C:
        A
        |- C
        B

        Parameters
        ----------
        rod_masses : ndarray, optional
            Contains mass of rod A, B, C.
        rod_dimensions : ndarray, optional
            3x3 array containing dimensions (length, width, height) for each rod.
        **kwargs
            Keyword arguments.

        """
        super().__init__(mass=sum(rod_masses), **kwargs)

        # Construct rods of TRod
        self.rod_masses = rod_masses  # Array containing mass of rod A, B, and C.
        self.rod_dimensions = rod_dimensions
        self.rods = []  # List containing rod object A, B, and C.
        self.construct_rods()  # Create rods and add them to the list 'rods'.

        # Find Center of Mass and it to x
        self.x = self.get_center_of_mass()

        # Calculate inertia in body-space.
        self.calculate_I_body()

        # Set up vpython objects.
        self.vp_object_axes = []  # vpython objects to display axes arrows of the TRod.

        # Angular momentum vpython arrow object with its 3 arrow components.
        self.L_arrow = None  # Angular momentum around principal axes.
        self.L_components = []

        # Create all vpython objects and set vp_object to the TRod vp object.
        self.vp_object = None



    def construct_rods(self):
        """ Creates the each rod (Block) and puts them into self.rods.
        TRod structure of the rods A, B, and C:
        A
        |- C
        B
        """

        a_pos = self.x.copy()
        a_pos[1] = self.rod_dimensions[0, 0]/2
        rod_a = Block(x=a_pos, dimensions=self.rod_dimensions[0],
                      R=np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                      mass=self.rod_masses[0], part_of_compound=True)

        b_pos = self.x.copy()
        b_pos[1] = -self.rod_dimensions[1, 0]/2
        rod_b = Block(x=np.array([0, -4, 0]), dimensions=self.rod_dimensions[1],
                      R=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                      mass=self.rod_masses[1], part_of_compound=True)

        c_pos = self.x.copy()
        c_pos[0] = self.rod_dimensions[2, 0]/2 + self.rod_dimensions[0, 1]/2  # c_length/2 + a_width/2
        rod_c = Block(x=c_pos, dimensions=self.rod_dimensions[2],
                      R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                      mass=self.rod_masses[2], part_of_compound=True)

        self.rods.extend([rod_a, rod_b, rod_c])

    def get_center_of_mass(self):
        """ Calculate center of mass in body space.
        Returns
        -------
        ndarray
            Center of mass position vector in body-space.
        """
        ximi = []
        yimi = []
        zimi = []

        for rod in self.rods:
            # x is the position vector equal to the center of mass of the RigidObject
            ximi.append(rod.x[0] * rod.mass)
            yimi.append(rod.x[1] * rod.mass)
            zimi.append(rod.x[2] * rod.mass)

        xcom = sum(ximi) / self.mass
        ycom = sum(yimi) / self.mass
        zcom = sum(zimi) / self.mass

        return np.array([xcom, ycom, zcom])

    def calculate_I_body(self):
        """ Calculate and set moment of inertia tensor.

        Use tensor generalization of the Parallel Axis Theorem to calculate inertia tensor for the compound.

        Credit to user "melax" from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=246 for algorithm.
        """
        I_body = np.zeros((3, 3))
        identity = np.identity(3)
        for rod in self.rods:
            r = rod.x - self.x
            I_body += rod.I_body + (np.dot(r, r) * identity - np.outer(r, r)) * rod.mass
        self.I_body = I_body

    def create_vpython_object(self, display_axes=True, display_angmomentum=True, axis_length_factor=6/5):
        """ Create vpython compound object from TRod rod components and set it to self.vp_object.
        TODO: Make sure display_axes can be false without error

        Parameters
        ----------
        display_axes : bool, optional
            Create and display vpython arrow objects to display principal axes direction.
        display_angmomentum : bool, optional
            Create and display vpython arrow objects to display body-space angular momentum vector.
        axis_length_factor : float, optional
            Determines length of vpython axes arrows. Ex: value=6/5 --> arrow_length=6/5*(length of rod A)

        """
        rod_a, rod_b, rod_c = self.rods[0], self.rods[1], self.rods[2]

        # Axis and up are set to the rods own x-axis and y-axis respectively
        vp_rod_a = vp.box(pos=vp.vector(*rod_a.x), axis=vp.vector(*rod_a.R[:, 0]), up=vp.vector(*rod_a.R[:, 1]),
                          length=rod_a.length, width=rod_a.width, height=rod_a.height)

        vp_rod_b = vp.box(pos=vp.vector(*rod_b.x), axis=vp.vector(*rod_b.R[:, 0]), up=vp.vector(*rod_b.R[:, 1]),
                          length=rod_b.length, width=rod_b.width, height=rod_b.height)

        vp_rod_c = vp.box(pos=vp.vector(*rod_c.x), axis=vp.vector(*rod_c.R[:, 0]), up=vp.vector(*rod_c.R[:, 1]),
                          length=rod_c.length, width=rod_c.width, height=rod_c.height)

        if display_axes:
            # x_axis is displayed as a red arrow
            x_axis = vp.arrow(pos=vp.vector(*self.x), axis=vp.vector(*self.R[:, 0]),
                              length=(vp_rod_a.length * axis_length_factor), color=vp.vector(1, 0, 0),
                              shaftwidth=0.5)

            # y_axis is displayed as a green arrow
            y_axis = vp.arrow(pos=vp.vector(*self.x), axis=vp.vector(*self.R[:, 1]),
                              length=(vp_rod_a.length * axis_length_factor), color=vp.vector(0, 1, 0),
                              shaftwidth=0.5)

            # z_axis is displayed as a blue arrow
            z_axis = vp.arrow(pos=vp.vector(*self.x), axis=vp.vector(*self.R[:, 2]),
                              length=(vp_rod_a.length * axis_length_factor), color=vp.vector(0, 0, 1),
                              shaftwidth=0.5)

            self.vp_object_axes.extend((x_axis, y_axis, z_axis))

        if display_angmomentum:
            # L = angular momentum
            L_position = self.x + np.array([10, 10, 0])  # Align top right corner.
            L_position = vp.vector(*L_position)
            L_arrow = vp.arrow(pos=L_position, length=1, shaftwidth=0.2, color=vp.color.yellow)  # Initialize without value.
            L_arrow_x = vp.arrow(pos=L_position, length=1, shaftwidth=0.2, color=vp.color.red)
            L_arrow_y = vp.arrow(pos=L_position, length=1, shaftwidth=0.2, color=vp.color.green)
            L_arrow_z = vp.arrow(pos=L_position, length=1, shaftwidth=0.2, color=vp.color.blue)
            self.L_arrow = L_arrow
            self.L_components.extend((L_arrow_x, L_arrow_y, L_arrow_z))

        vp_TRod = vp.compound([vp_rod_a, vp_rod_b, vp_rod_c], origin=vp.vector(*self.get_center_of_mass()))
        self.vp_object = vp_TRod


def star(a):
    """ Returns skew-symmetric matrix of 3-vector.

    TODO: finish docstring

    Parameters
    ----------
    a : ndarray
        3-vector
    Returns
    -------
    ndarray
        Skew-symmetric matrix of 3-vector.

    """
    starred_matrix = np.array([
        [0, a[2], -a[1]],
        [-a[2], 0, a[0]],
        [a[1], -a[0], 0]
    ])
    return starred_matrix


def f(t, y, rb):
    """ Function "f(t, y) = y'(t)" for scipy.integrate.ode.

    Parameters
    ----------
    t : int or float
        Time.
    y : ndarray
        State vector.
    rb : RigidBody

    Returns
    -------
    ndarray
        dy/dt for state vector y.

    """
    # Moments of inertia arround principal axes.
    I1 = rb.I_body[0, 0]
    I2 = rb.I_body[1, 1]
    I3 = rb.I_body[2, 2]

    # Set variables to simplify dydt.
    omega1, omega2, omega3 = y[0:3]
    omega = np.array([omega1, omega2, omega3])
    R = y[3:12].reshape((3, 3))
    Rdot = star(omega) @ R

    dydt = [(I2 - I3) * omega2 * omega3 / I1, (I3 - I1) * omega3 * omega1 / I2, (I1 - I2) * omega1 * omega2 / I3,
            *Rdot.flatten()]
    return dydt


#### USE optimized_animate_TRod() for faster results. ####
# def animate_TRod(y, dt, rb):
#     rb.create_vpython_object
#     animation_graph = vp.graph(scroll=True, fast=True, xmin=-5, xmax=5)
#     t = 0
#     omega1_plot = vp.gcurve(color=vp.color.red, radius=1)
#     omega2_plot = vp.gdots(color=vp.color.green, radius=1)
#     omega2_plot = vp.gdots(color=vp.color.blue, radius=1)
#     tot_axes = len(rb.R)
#     for state in y:
#         vp.rate(30)  # set framerate
#         axes = state[3:12].reshape((3, 3))  # get body-space axes
#
#         # # Plot curves.
#
#         omega1_plot.plot(t, state[0])
#         omega2_plot.plot(t, state[1])
#         omega2_plot.plot(t, state[2])
#         t += dt
#
#         # Loop over axes and rotate object and arrows.
#         for i in range(tot_axes):
#             axis = vp.vector(*axes[i])
#             # Rotate rigid body
#             rb.vp_object.rotate(angle=state[i] * dt, axis=axis)
#             # Rotate axes arrows
#             old_length = rb.vp_object_axes[i].length
#             rb.vp_object_axes[i].axis = axis
#             rb.vp_object_axes[i].length = old_length


def optimized_animate_TRod(y, dt, rb):
    """ Animates RigidBody's vpython object (optimized version).

    TODO: To finally optimize, make sure vp.rate is consistent with real time and make sure to only display
    results needed. Exclude non-displayed results that are left out because of frame-rate."""
    frame_rate = dt**-1

    rb.create_vpython_object()
    animation_graph = vp.graph(scroll=True, fast=True, xmin=-5, xmax=5, ymin=-2.5, ymax=2.5)
    omega1_plot = vp.gcurve(color=vp.color.red)
    omega2_plot = vp.gcurve(color=vp.color.green)
    omega3_plot = vp.gcurve(color=vp.color.blue)
    plot_interval = 1
    t = 0
    tot_axes = len(rb.R)
    omegas1 = y[:, 0].copy()
    omegas2 = y[:, 1].copy()
    omegas3 = y[:, 2].copy()
    np_axes = y[:, 3:12].copy()
    np_axes = np_axes.reshape((len(np_axes), 3, 3))
    vp_axes = []

    for R in np_axes:
        vp_R = []
        for axis in R:
            vp_R.append(vp.vector(*axis))
        vp_axes.append(vp_R)

    for k in range(len(y)):
        vp.rate(frame_rate)  # set framerate

        omega1 = omegas1[k]
        omega2 = omegas2[k]
        omega3 = omegas3[k]
        R = vp_axes[k]

        # Plot curves.
        if not k % plot_interval:  # Ex: If plot_interval = 3, only plot every 3rd point.

            omega1_plot.plot(t, omega1)
            omega2_plot.plot(t, omega2)
            omega3_plot.plot(t, omega3)

        t += dt

        # Loop over axes and rotate object and arrows.
        # Rotate x_arrow
        old_length = rb.vp_object_axes[0].length
        rb.vp_object_axes[0].axis = R[0]
        rb.vp_object_axes[0].length = old_length

        # Rotate y_arrow
        old_length = rb.vp_object_axes[1].length
        rb.vp_object_axes[1].axis = R[1]
        rb.vp_object_axes[1].length = old_length

        # Rotate z_arrow
        old_length = rb.vp_object_axes[2].length
        rb.vp_object_axes[2].axis = R[2]
        rb.vp_object_axes[2].length = old_length

        # # Rotation version (don't use both axis-setting and rotation)
        # rb.vp_object.rotate(angle=omega1*dt, axis=R[0])
        # rb.vp_object.rotate(angle=omega2*dt, axis=R[1])
        # rb.vp_object.rotate(angle=omega3*dt, axis=R[2])

        # Axis-setting version (don't use both axis-setting and rotation)
        rb.vp_object.axis = R[0]
        rb.vp_object.up = R[1]

        # Update angular momentum arrows.
        L = rb.I_body @ np.array((omega1, omega2, omega3))  # Do this calc. outside loop for better optimization.
        L /= np.linalg.norm(L) * 1/5
        L_x = np.zeros(3)
        L_x[0] = L[0]
        L_y = np.zeros(3)
        L_y[1] = L[1]
        L_z = np.zeros(3)
        L_z[2] = L[2]
        rb.L_arrow.axis = vp.vector(*L)
        rb.L_components[0].axis = vp.vector(*L_x)
        rb.L_components[1].axis = vp.vector(*L_y)
        rb.L_components[1].pos = rb.L_components[0].pos + rb.L_components[0].axis
        rb.L_components[2].axis = vp.vector(*L_z)
        rb.L_components[2].pos = rb.L_components[1].pos + rb.L_components[1].axis


def numerical_solver(rb, dt, t0=0, t1=100):
    y0 = [rb.omega[0], rb.omega[1], rb.omega[2],
          *rb.R.flatten()]

    r = ode(f).set_integrator('vode')
    r.set_f_params(rb)
    r.set_initial_value(y0, t0)
    y = np.zeros((int(t1 / dt), 12))
    i = 0
    while r.successful() and not np.isclose(r.t, t1):
        y[i, :] = r.integrate(r.t + dt)
        i += 1
    return y


# Initial simulation values.
omega1 = 2
omega2 = 0.01
omega3 = 0
dt = 1/30


# Create scene and TRod
scene = vp.canvas(title="Intermediate Axis Theorem", autoscale=False)
scene.camera.pos = vp.vector(0, 0, 10)
new_TRod = TRod(omega=np.array([omega1, omega2, omega3]))

# Solve ODEs' and put results in y.
y = numerical_solver(new_TRod, dt)

# Draw and animate the TRod with y.
optimized_animate_TRod(y, dt, new_TRod)

# Save y to file
np.savetxt('y_results.csv', y, delimiter=',')

