from dolfin import*
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge,Arc

def plot_figure(function, name, save=False, map='inferno'):
    """
    Plot a given function with an optional color map and save the plot if specified.

    :param function: The function to plot.
    :type function: :class:`dolfin.function.Function`
    :param name: The title of the plot.
    :type name: str
    :param save: If True, save the plot with the title as the filename.
    :type save: bool, optional
    :param colormap: The name of the colormap to use for the plot.
    :type colormap: str, optional
    
    :returns: None

    :Example:

    >>> plot_figure(u, "Potential Field", save=True, colormap='viridis')
    """
    plt.figure()
    p=plot(function, title=name)
    p.set_cmap(map)
    #p.set_clim(0.0, 4.0)
    plt.colorbar(p)
    if save==True: plt.savefig(name)
    return 



def getBoundaryVertex(mesh,u):
    """
    Get the values of a function `u` at the boundary vertices of a given `mesh`.

    :param mesh: The input mesh.
    :type mesh: Mesh
    :param u: The function to evaluate at the boundary vertices.
    :type u: Function

    :return: A list of function values at the boundary vertices.
    :rtype: list
    """
    u_bvertex=[]
    u_bvertex_ind=[]
    bmesh=BoundaryMesh(mesh, 'exterior', order=True)
    indice_borda=bmesh.entity_map(0).array()
    
    for ind in indice_borda:
        u_bvertex.append(u.compute_vertex_values()[ind]) #pra pegar os valores, apenas na borda, como array
    return u_bvertex


def plot_boundary(mesh, data, name='boundary', line=2, data2=1, save=False, plot=True):
    """
    Plot the boundary of a mesh along with `data`.

    :param mesh: The mesh to plot.
    :type mesh: Mesh
    :param data: The data to plot on the boundary.
    :type data: numpy.ndarray
    :param name: The name of the plot (default 'boundary').
    :type name: str
    :param line: The line width for plotting (default 2).
    :type line: int
    :param data2: Additional data to plot (default 1).
    :type data2: int or numpy.ndarray
    :param save: Whether to save the plot (default False).
    :type save: bool
    :param plot: Whether to display the plot (default True).
    :type plot: bool

    :return: The boundary plot as a numpy array.
    :rtype: numpy.ndarray
    """
    tol=0.0
    bmesh=BoundaryMesh(mesh, 'exterior', order=True)
    indice_borda=bmesh.entity_map(0).array()
    boundary_plot1, boundary_plot2=[], []
    for ind in range(len(indice_borda)):
        vertex = Vertex(mesh, indice_borda[ind])
        if vertex.x(1)>=tol:
            theta=np.arccos(vertex.x(0))
            boundary_plot1.append([theta, data[ind]])
    boundary_plot1=np.array(boundary_plot1)
    max_b=np.max(boundary_plot1[:,0])        

    
    for ind in range(len(indice_borda)):
        vertex = Vertex(mesh, indice_borda[ind])    
        if vertex.x(1)<0:
            theta=np.arccos(-vertex.x(0))+max_b
            boundary_plot2.append([theta, data[ind]])
    
    boundary_plot=np.concatenate((boundary_plot1, boundary_plot2), axis=0)
    
    
    boundary_plot=boundary_plot[boundary_plot[:,0].argsort()]
    #boundary_plot=np.sort(boundary_plot.view('i8'), order=['f1'], axis=0)

    if plot==True:
        plt.figure()
        plt.title(name)
        if type(data2)!=type(1):
            plt.plot(data2[:,0], data2[:,1], marker='.', markersize=2, linewidth=line, label="Dados")    
        plt.plot(boundary_plot[:,0], boundary_plot[:,1], marker='.', markersize=2, linewidth=line, label="Resultado")
        #plt.legend()
        if save==True: plt.savefig(name)
    
    return boundary_plot

def plot_electrodes(mesh, linewidth_mesh = 1, linewidth_elec=5, figsize=(5,5), fontsize=20, elec_num=True, axis=False):
    """
    Plot the electrodes on a mesh.

    :param mesh: A mesh object.
    :type mesh: Mesh
    :param linewidth_mesh: The width of the mesh lines (default 1).
    :type linewidth_mesh: int
    :param linewidth_elec: The width of the electrode lines (default 5).
    :type linewidth_elec: int
    :param figsize: The size of the figure (default (5,5)).
    :type figsize: tuple
    :param fontsize: The font size of the electrode numbers (default 20).
    :type fontsize: int
    :param elec_num: Whether to show electrode numbers (default True).
    :type elec_num: bool
    :param axis: Whether to show the axis (default False).
    :type axis: bool

    :return: The figure object.
    :rtype: matplotlib.figure.Figure
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    mesh_radius = mesh.radius
    theta_vec = np.degrees(np.array(mesh.electrodes.position))

    for index, theta in enumerate(theta_vec):
        theta_start, theta_end = theta[0], theta[1]
        centertheta = (abs(theta_start-theta_end)/2+theta_start)/360*(2*np.pi)
        #Plotting arc
        arc = Arc((0, 0), 2 * mesh_radius * 1.01, 2 * mesh_radius * 1.01, angle=0,
                theta1=theta_start, theta2=theta_end, linewidth=linewidth_elec, color='black')
        #Plotting Electrode number
        if elec_num:
            x, y = mesh_radius*np.cos(centertheta)*1.1, mesh_radius*np.sin(centertheta)*1.1
            ax.annotate(index+1, (x, y), color='black', weight='bold', fontsize=fontsize, ha='center', va='center')
        ax.add_artist(arc)

    ax.set_aspect(1)
    plot(mesh, linewidth=linewidth_mesh)
    
    if not axis:
        plt.xticks([])
        plt.yticks([])
        plt.box(False)
    return 


def EstimateDelta(list_U_noised: np.ndarray, I: np.ndarray) -> float:
    """
    
    Estimate the noise level in potential measurements obtained from a grounded electrode system. The method was based on 
    
    Robert Winkler work: A model-aware inexact Newton scheme for electrical impedance tomography, 2016.

    :param list_U_noised: Noisy potential measurements with shape (l, L).
    :type list_U_noised: numpy.ndarray
    :param I: Current pattern matrix with shape (l, L).
    :type I: numpy.ndarray

    :return: A scalar value representing the estimated noise level in the potential measurements.
    :rtype: float

    :notes:
        The data must be measured through a grounded electrode, i.e., the sum of potential of each of the electrodes must be zero.

    This function estimates the noise level in potential measurements obtained from a grounded electrode system. It takes two
    arguments as inputs: `list_U_noised` and `I`.
    """
    l, L = np.shape(I)
    list_U_noised = list_U_noised.reshape((l, L))
    Iplus = np.linalg.pinv(I)

    ev = (Iplus @ list_U_noised) - (Iplus @ list_U_noised).T
    norm_ev = np.linalg.norm(ev, ord="fro") ** 2
    norm_Iplus = np.linalg.norm(Iplus, ord="fro")
    vCEM = norm_ev / (2 * (L - 1)) * norm_Iplus ** (-2)
    delta_calc = np.sqrt(l * L * vCEM)

    return delta_calc

def ConvertingData(U,method):
    """
    Convert data from different measurement patterns to the ground pattern.

    :param U: The data to be converted.
    :type U: numpy.ndarray
    :param method: The measurement pattern to be converted to. Currently only "KIT4" is supported.
    :type method: str

    :return: The converted data.
    :rtype: numpy.ndarray
    """
    if method=="KIT4": 
        #See: https://arxiv.org/pdf/1704.01178.pdf
        L=len(U)
        U_til = np.zeros(L)
        for i in range(1,L): U_til[i]=np.sum(U[:i])
        c = np.sum(U_til)
        return c/L-U_til
    return U

def EstimateCond(list_U0, I, mesh, z, method="CONT"):
    """
    Estimate the conductivity of the background based on noisy voltage measurements.

    :param list_U0: A list of noisy voltage measurements.
    :type list_U0: list
    :param I: A current pattern matrix.
    :type I: numpy.ndarray
    :param mesh: A mesh object.
    :type mesh: Mesh
    :param z: The background impedance (default 1E-5).
    :type z: numpy.ndarray
    :param method: The method used for estimating conductivity. Options: "CONT", "SHUNT", "CEM1", "CEM2".
    :type method: str

    :return: A tuple containing the estimated conductivity and minimum potential.
    :rtype: tuple(float, float)
    """
    l, L = np.shape(I)
    gamma = Function(FunctionSpace(mesh, "DG", 0))
    gamma.vector()[:]=np.ones(mesh.num_cells())

    #Solver
    VD=FiniteElement('CG',mesh.ufl_cell(),1) #Solution Space Continous Galerkin
    ForwardObject=ForwardProblem(mesh,  z)
    _, list_U = ForwardObject.solve_forward(VD, I, gamma)

    list_U0 = np.array(list_U0).reshape(l,L)
    if method=="CONT":
        list_U=np.array(list_U).flatten()
        list_U0=np.array(list_U0).flatten()
        cond = np.linalg.norm(list_U)**2/np.dot(list_U, list_U0)
        return cond
    if method=="SHUNT":
        U_1=[np.dot(list_U[i],I[i]) for i in np.arange(l)]
        U_2=[np.dot(list_U0[i],I[i]) for i in np.arange(l)]
        cond=np.linalg.norm(U_1)**2/np.dot(U_1,U_2)
        return cond
    if method == "CEM1":
        z0=np.max(z)
        elec_lenght = mesh.radius*(mesh.electrodes.calc_position()[0][1]-mesh.electrodes.calc_position()[0][0])
        U_1=[np.dot(list_U[i]-I[i]*z0/elec_lenght,I[i]) for i in np.arange(l)]
        U_2=[np.dot(list_U0[i]-I[i]*z0/elec_lenght,I[i]) for i in np.arange(l)]
        cond=np.dot(U_1,U_2)/np.linalg.norm(U_2)**2
        return cond
    if method == "CEM2":
        z0=np.max(z)
        elec_lenght = mesh.radius*(mesh.electrodes.calc_position()[0][1]-mesh.electrodes.calc_position()[0][0])

        a_vec=[np.dot(list_U[i]-I[i]*z0/elec_lenght, I[i]) for i in np.arange(l)]
        b_vec=[np.dot(I[i],I[i]/elec_lenght) for i in np.arange(l)]
        c_vec = [np.dot(list_U0[i],I[i])for i in np.arange(l)]

        A = np.array([a_vec, b_vec]).T
        x0 = scipy.optimize.nnls(A, c_vec)
        rho, z = x0[0]
        cond = 1/rho
        return cond, z
    
def EstimateCondIterative(list_U0, I, mesh, z, zmin=1E-5):
    """
    Estimate the conductivity of the background based on noisy voltage measurements using an iterative approach.

    :param list_U0: A list of noisy voltage measurements.
    :type list_U0: list
    :param I: A current pattern matrix.
    :type I: numpy.ndarray
    :param mesh: A mesh object.
    :type mesh: Mesh
    :param z: The background impedance (default 1E-5).
    :type z: numpy.ndarray
    :param zmin: The minimum background impedance value (default 1E-5).
    :type zmin: float

    :return: A tuple containing the estimated conductivity and impedance.
    :rtype: tuple(float, float)
    """
    def fun(zi, list_U0, I, mesh, z):
        l, L = np.shape(I)
        cond, z = EstimateCond(list_U0, I, mesh, zi, method="CEM2")
        z=np.max([zmin, z])

        Q_DG=FunctionSpace(mesh, "DG", 0)
        gamma = Function(Q_DG)
        gamma.vector()[:]=np.ones(mesh.num_cells())*cond

        ForwardObject=ForwardProblem(mesh,  z*np.ones(L))
        VD=FiniteElement('CG',mesh.ufl_cell(),1) #Solution Space Continous Galerkin
        _, list_U = ForwardObject.solve_forward(VD, I, gamma)
        list_U = np.array(list_U).flatten()

        res = np.linalg.norm(list_U - list_U0)/np.linalg.norm(list_U0)*100
        #print(res)
        return  res
    
    data = scipy.optimize.minimize(fun, 1E-5, args=(list_U0, I, mesh, z))
    z_optimal=data['x']
    cond_optimal, z_optimal = EstimateCond(list_U0, I, mesh, z_optimal, method="CEM2")
    return cond_optimal, z_optimal
