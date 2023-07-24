from module1_mesh import*
import matplotlib.pyplot as plt

def plot_boundary(mesh, data, name='boundary', line=2, data2=1, save=False, plot=True):
    """Plot boundary of the function."""
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
    max_b=np.max(boundary_plot1)        
    
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
            plt.plot(data2[:,0], data2[:,1], marker='.', markersize=2, linewidth=line, label="Data")    
        plt.plot(boundary_plot[:,0], boundary_plot[:,1], marker='.', markersize=2, linewidth=line, label="Result")
        plt.legend()
        if save==True: plt.savefig(name)
    
    return boundary_plot

def plot_figure(mesh, V, function, name, save=False, map='inferno'):
    """Plot figure function."""
    Q=FunctionSpace(mesh,V)
    function_Q=interpolate(function, Q)
    
    plt.figure()
    p=plot(function_Q, title=name)
    p.set_cmap(map)
    #p.set_clim(0.0, 4.0)
    plt.colorbar(p)
    if save==True: plt.savefig(name)
    return function_Q


def Verifyg(list_gs, mesh):
    """Verify if current has integral zero and if they are linear independent"""
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    
    for i in range(0,len(list_gs)):
        A=assemble(list_gs[i]*ds)
        print("Integral boundary:", A, i)

    for i in range(len(list_gs)):
        for j in range(i, len(list_gs)):
            if i!=j:
                A=assemble(list_gs[i]*list_gs[j]*ds)
                print("Integral boundary g("+str(i)+")*g("+str(j)+"):", A)
    return
