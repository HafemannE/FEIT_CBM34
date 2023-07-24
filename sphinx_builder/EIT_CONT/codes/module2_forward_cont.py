from module1_mesh import*
from module4_auxiliar import*

def current_method(n_g, value=1, method=1):
    """This function an expression that represent the current in the vertex.
        
    :param n_g: Measurements number.
    :type n_g: int
    :param value: value in the vertex.
    :type value: float
    :param method: Current pattern.
    :type method: int
    :returns:  Expression -- Return list of expressions.
    
    Method Values:           
        1. 1 and -1 in opposite direction, where 50% of the boundary is always 0.
        
    :Example:

        .. code-block:: python

           "Current"
            n_g=2
            list_gs=current_method(n_g, value=1, method=1)

            for i in range(n_g):
                mesh=mesh_direct
                VD=FiniteElement('CG',mesh.ufl_cell(),1) 
                g_u=interpolate(list_gs[i], FunctionSpace(mesh,VD))
                g_u=getBoundaryVertex(mesh, g_u)
                bond=plot_boundary(mesh, data=g_u, name='boundary g'+str(i))

        .. image:: codes/current1.png
           :scale: 75 %       
           """
    #We need implement more methods here, please see the tutorial, there are several examples.
    h=pi/n_g/2 #Angular lenght of each "electrode"
    if method==1:
        #&& - and operator
        list_gs=[Expression(f" x[0]<cos(0+{h*i}) && x[0]>cos({h}+{h*i}) && x[1]>0 ? {value} : "+
                            f"(x[0]>cos(pi+{h*i}) && x[0]<cos(pi+{h}+{h*i}) && x[1]<0 ? -{value} : 0 )"
                            ,degree=1) for i in range(0,n_g*2,2)]
    return list_gs

def fn_addnoise(data, level, noise_type='uniform', seed=42):
    """Function receives a vector which represents the data in the electrodes and returns
    a noised vector with the chosen noise level and the type of noise.
    We use it in :func:`ForwardProblem.add_noise`.
    
    :param data: Vector with potencial in electrodes or any other vector.
    :type data: array
    :param level: Noise level (%), expect values between 0 and 1.
    :type level: float
    :param noise_type: Noise type, uniform or cauchy.
    :type method: str.
    :param seed: Seed for random function.
    :type seed: int.
    
    
    :returns:  Array -- Return noised vector.
    
    :Example:

    >>> print(np.ones(8))
    >>> print(fn_addnoise(data=np.ones(8), level=0.01, noise_type='cauchy', seed=32))
        [1. 1. 1. 1. 1. 1. 1. 1.]
        array([0.99905327, 1.02206251, 1.00356633, 1.00236212, 1.00101231, 0.99904405, 1.0105611 , 0.98656216])
    
    """
    i = len(data)
    # create 1D numpy data:
    npdata = np.asarray(data).reshape((i))
    delta=level*np.linalg.norm(npdata) #delta = noise_level * ||data||_2
    
    #add f normal noise:
    if noise_type=='uniform':
        np.random.seed(seed)                             #Set seed to generate random noise
        noise_f=np.random.randn(npdata.size)             #Generate random noise with vector size of data.
        noise_f=noise_f/np.linalg.norm(noise_f)*delta    #Normalize random noise and multiply by delta.
        noise = npdata + noise_f                         #Add noise to the vector.
    # add cauchy noise:
    elif noise_type=='cauchy':   
    #12 31 59
        np.random.seed(seed)                                     #Set seed to generate random noise
        noise_p = np.random.standard_cauchy(size=npdata.shape)   #Generate random noise with vector size of data.
        noise_p=noise_p/np.linalg.norm(noise_p)*delta            #Normalize random noise and multiply by delta
        noise = npdata + noise_p                                 #Add noise to the vector.
        
    return noise



def GammaCircle(mesh, in_v, out_v, radius,centerx, centery):
    """Function to create a circle in the mesh with some proprieties
    
        :param mesh: Mesh.
        :type mesh: :class:`dolfin.cpp.mesh.Mesh`
        :param in_v: Value inside circle
        :type in_v: float
        :param out_v: Value outside circle
        :type out_v: float
        :param radius: Circle radius
        :type radius: float
        :param centerx: Circle center position x
        :type centerx: float
        :param centery: Circle center position y
        :type centery: float
       
        :returns:  Array -- Return a vector where each position correspond de value of the function in that element.
        
        :Example:

        >>> ValuesCells0=GammaCircle(mesh=mesh_direct, in_v=3.0, out_v=1.0, radius=0.50, centerx=0.25, centery=0.25)
        >>> print(ValuesCells0)
        [1. 1. 1. ... 1. 1. 1.]
        
        >>> "Plot"
        >>> gamma0=CellFunction(mesh_direct, values=ValuesCells0);   
        >>> V_DG=FiniteElement('DG',mesh_direct.ufl_cell(),0)
        >>> plot_figure(mesh_direct, V_DG, gamma0, name="Resposta gamma");
        
        .. image:: codes/gamma.png
           :scale: 75 %

       """
    
    ValuesGamma=np.zeros(mesh.num_cells()) #Null vector
    
    for i in range(0, mesh.num_cells()):
        cell = Cell(mesh, i) #Select cell with index i in the mesh.
        
        vertices=np.array(cell.get_vertex_coordinates()) #Vertex cordinate in the cell.
        x=(vertices[0]+vertices[2]+vertices[4])/3           
        y=(vertices[1]+vertices[3]+vertices[5])/3
        
        #If the baricenter is outside the circle...
        if ((x-centerx)**2+(y-centery)**2>=radius**2):
            ValuesGamma[i]=out_v
        else:
            ValuesGamma[i]=in_v
    
    return ValuesGamma


class CellFunction(UserExpression):
    """Auxiliar function to transform an array to a Function
        We use it with :func:`GammaCircle()`
    
        :param mesh: Mesh.
        :type mesh: :class:`dolfin.cpp.mesh.Mesh`
        :param values: Array with values of the function in the cell.
        :type values: array
        
        :Example:

        >>> ValuesCells0=np.zeros(mesh_inverse.num_cells())        #Define a vector of zeros
        >>> ValuesCells0[5]=1                                      #Cell 5 has value 1
        >>> gamma0=CellFunction(mesh_inverse, values=ValuesCells0);#Get vector and transform in a function cell

        If you want plot the function::
        
        >>> V_DG=FiniteElement('DG',mesh_inverse.ufl_cell(),0)     #Space of Finite Elemente descontinuous garlekin degree 0
        >>> Q=FunctionSpace(mesh_inverse,V_DG)                     #Functionspace to interpolate gamma
        >>> gamma0=interpolate(gamma0, Q)                          #Interpolation gamma to generate a function
        >>> p=plot(gamma0)                                         #plot gamma0
        >>> plot(mesh_inverse)                                     #plot mesh
        >>> plt.colorbar(p)                                        #set colorbar.

        .. image:: codes/cell_function_test.png
          :scale: 75 %

         
        """
    def __init__(self, mesh, values, **kwargs):
        self.mesh = mesh
        self.values=values #Values in a array           
        super().__init__(**kwargs)
    
    #Function that returns value in cell.
    def eval_cell(self, value, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        value[0]=self.values[cell.index()] #Set value of the cell using the array values. (Map between cell.index and values.)
        
    #"Just to erase a warning"            
    def value_shape(self):
        return ()

    
class ForwardProblem(object):
    """ Object Forward Problem EIT 2D Continous Model.
    
        :param mesh: Mesh. We recommend from :func:`MyMesh()`
        :type mesh: :class:`dolfin.cpp.mesh.Mesh`
        
        :Example:
        
        .. code-block:: python
        
            "Basic Definitions"
            VD=FiniteElement('CG',mesh_direct.ufl_cell(),1) 
            F_Problem=ForwardProblem(mesh_direct)

            "Solver"
            list_u0=F_Problem.solve_forward(VD, gamma0, I_all)
            u0_boundary=F_Problem.boundary_array(mesh_inverse)
        
        If you need it, see :func:`GammaCircle` and :func:`CellFunction`.
        
    """
    
    def __init__(self, mesh):
        self.mesh=mesh #Mesh
        
    def solve_forward(self, V, gamma, I_all): 
        """ Solver Forward Problem EIT 2D
            
    :param V: FiniteElement Fenics object
    :param gamma: Finite Element Function
    :param I_all: Current density in each electrode for each measurement
    :type V: FiniteElement
    :type gamma: :func:`CellFunction()`
    :type I_all: :func:`current_method()` or list of arrays
            
    :returns:  (Array) -- Return function that is solution from variational problem.
        
     :Example:
     >>> F_Problem=ForwardProblem(mesh_direct)
     >>> list_u0=F_Problem.solve_forward(VD, gamma0, list_gs)
            
        """
        mesh=self.mesh  #Saving mesh
        n_g=len(I_all)  #Get number of experiments
        
        R=FiniteElement('R',mesh.ufl_cell(),0) #Constant for Lang. Mult

        W=V*R #Mixing Elements HxR
        W=FunctionSpace(mesh,W) #Defining function space.

        (u,c)=TrialFunctions(W) #Functions that will be reconstructed.
        (v,d)=TestFunction(W)   #Test functions

        lagrMult=(v*c+u*d)*ds #If that we have the ground potential. Integral_(dOmega) u ds =0

        a=inner(gamma*grad(u),grad(v))*dx+lagrMult # Integral( gamma*<grad_u,grad_v> ) dOmega + lagrMult
        A=assemble(a) #Make my matriz to solve Ax=b.
        #We only do it only one time, if have mult. measurements, we reuse it.
        
        #Construction my b vector. (Here b=L).
        sol_u=[] #save solution u
        for j in range(n_g): #For each current in I_all
            L=I_all[j]*v*ds #Intregral g_i*v*ds
            b = assemble(L) #Make my b vector
            
            w = Function(W) #Define a zero function based in W.
            U = w.vector()  #Return a vector. (x=U)
            solve(A, U, b) #Solve system AU=b. where A matrix and b vector.
            
            #Split w function in 2 parts, firt the function in H, and the constant lagr
            (u,c)=w.split(deepcopy=True)
            sol_u.append(u)
        
        self.sol_u=sol_u #Save in memory.
        
        return sol_u
    
    def boundary_array(self, mesh_inverse=None, concatenate=True):
        """ Get's the boundary values of function solution and returns array.
        If you set a coarse mesh you will get the values in the vertices that are commum.
        If you set conccatenate=False, will receive a array with separeted results, is usefull if you used more than one current.
        
        :param mesh: Corse Mesh. We recommend from :func:`MyMesh()`
        :type mesh: :class:`dolfin.cpp.mesh.Mesh`
        :param concatenate: Default True
        :type concatenate: bool
        
        :returns:  (Array) -- Vertex values of the function.
        
        :Example:
        
        >>> u0_boundary=F_Problem.boundary_array(mesh_inverse)
        
        """
        sol_u=self.sol_u
        data=[]
    
        for i in range(len(sol_u)):
            if mesh_inverse==None:
                u_dados = getBoundaryVertex(self.mesh,self.sol_u[i])
            else:
                u_dados, u_dados, vertexnumber = getBoundaryVertexTwoMesh(mesh_inverse, self.mesh, sol_u[i], sol_u[i])
            
            if concatenate==False: data.append(u_dados)
            else : data=np.concatenate((data, u_dados))
                
        return data
    
        
    
    def plot_boundary(self, mesh_inverse=None, index=0 ):
        """ Get's the boundary values of function solution and returns a graph.
        If you set a coarse mesh you will get the values in the vertices that are commum and plot it.
        
        :param mesh: Corse Mesh. We recommend from :func:`MyMesh()`
        :type mesh: :class:`dolfin.cpp.mesh.Mesh`
        :param index: Index of solution, if you need it.
        :type index: int
        
        :Example:
        
        >>> data_u0=F_Problem.plot_boundary(mesh_inverse, index=1)
        
        .. image:: codes/boundary_u.png
          :scale: 75 %
        
        """
        
        if mesh_inverse==None: mesh=self.mesh
        else: mesh=mesh_inverse
            
        u_data=self.boundary_array(mesh, concatenate=False)
        plot_boundary(mesh, data=u_data[index], name=f'boundary u_{index}', line=0)
        return
        
    
    def add_noise(self, noise_level=0, noise_type='uniform', seed=42, mesh=None):
        """ Function that add noise in the potential values.
        
            :param data: Vector with potencial in electrodes or any other vector.
            :type data: array
            :param level: Noise level (%), expect values between 0 and 1.
            :type level: float
            :param noise_type: Noise type, uniform or cauchy.
            :type noise_type: str.
            :param mesh: Corse Mesh. We recommend from :func:`MyMesh()`
            :type mesh: :class:`dolfin.cpp.mesh.Mesh`
    
            :returns:  Array -- Return a vector with potentials values concatenated.
            
            :Example:
            
            .. code-block:: python              
            
                "Noise Parameters"
                noise_level=0.01
                noise_type='uniform'
                seed=1
                u0_boundary=F_Problem.add_noise(noise_level noise_type, seed, mesh_inverse)

        """
        if mesh==None: mesh=self.mesh
        u_data=self.boundary_array(mesh, concatenate=False)
        
        #Add same noise in each experiment.
        vec_U=[]
        for i in range(len(u_data)): vec_U=np.concatenate((
            vec_U, fn_addnoise(u_data[i], noise_level, noise_type=noise_type, seed=seed)), axis=0)
        return vec_U
    