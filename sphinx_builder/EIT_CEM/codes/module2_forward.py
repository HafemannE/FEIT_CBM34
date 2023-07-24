from module1_mesh import *
from module4_auxiliar import*
import scipy

def current_method(L,l, method=1, value=1):
    """
    Create a numpy array (or a list of arrays) that represents the current pattern in the electrodes.

    :param L: Number of electrodes.
    :type L: int
    :param l: Number of measurements.
    :type l: int
    :param method: Current pattern. Possible values are 1, 2, 3, or 4 (default=1).
    :type method: int
    :param value: Current density value (default=1).
    :type value: int or float

    :returns: list of arrays or numpy array -- Return list with current density in each electrode for each measurement.

    :Method Values:
        1. 1 and -1 in opposite electrodes.
        2. 1 and -1 in adjacent electrodes.
        3. 1 in one electrode and -1/(L-1) for the rest.
        4. For measurement k, we have: (sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16)).

    :Example:

    Create current pattern 1 with 4 measurements and 4 electrodes:

    >>> I_all = current_method(L=4, l=4, method=1)
    >>> print(I_all)
        [array([ 1.,  0., -1.,  0.]),
        array([ 0.,  1.,  0., -1.]),
        array([-1.,  0.,  1.,  0.]),
        array([ 0., -1.,  0.,  1.])]

    Create current pattern 2 with 4 measurements and 4 electrodes:

    >>> I_all = current_method(L=4, l=4, method=2)
    >>> print(I_all)
        [array([ 1., -1.,  0.,  0.]),
        array([ 0.,  1., -1.,  0.]),
        array([0.,  0.,  1., -1.]),
        array([ 1.,  0.,  0., -1.])]

    """
    I_all=[]
    #Type "(1,0,0,0,-1,0,0,0)"
    if method==1:
        if L%2!=0: raise Exception("L must be odd.")
                                   
        for i in range(l):
            if i<=L/2-1:
                I=np.zeros(L)
                I[i], I[i+int(L/2)]=value, -value
                I_all.append(I)
            elif i==L/2:
                print("This method only accept until L/2 currents, returning L/2 currents.")
    #Type "(1,-1,0,0...)"
    if method==2:
        for i in range(l):
            if i!=L-1:
                I=np.zeros(L)
                I[i], I[i+1]=value, -value
                I_all.append(I)
            else: 
                I=np.zeros(L)
                I[0], I[i]=-value, value
                I_all.append(I)
    #Type "(1,-1/15, -1/15, ....)"
    if method==3:
        for i in range(l):
            I=np.ones(L)*-value/(L-1)
            I[i]=value
            I_all.append(I)
    #Type "(sin(k*2*pi/16) sin(2*k*2*pi/16) ... sin(16*k*2*pi/16))"
    if method==4:
        for i in range(l):
            I=np.ones(L)
            for k in range(L): I[k]=I[k]*sin((i+1)*(k+1)*2*pi/L) 
            I_all.append(I)

            
    if l==1: I_all=I_all[0]
    return np.array(I_all)

def fn_addnoise(data, level, noise_type='uniform', seed=42):
    """
    Add noise to a vector representing data in the electrodes.

    :param data: Vector with potential in electrodes or any other vector.
    :type data: array_like
    :param level: Noise level (%), expects values between 0 and 1.
    :type level: float
    :param noise_type: Noise type, 'uniform' or 'cauchy' (default='uniform').
    :type noise_type: str
    :param seed: Seed for the random number generator (default=42).
    :type seed: int

    :returns: Array -- Return the noised vector.

    :Example:

    >>> data = np.ones(8)
    >>> print(data)
        [1. 1. 1. 1. 1. 1. 1. 1.]
    >>> noised_data = fn_addnoise(data=data, level=0.01, noise_type='cauchy', seed=32)
    >>> print(noised_data)
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

class electrode_domain(SubDomain):
    """
    Auxiliary function for the ForwardProblem to define an electrode domain.
    It is expected that the domain is a circle.
    This routine determines the vertices where the electrodes are defined and marks the mesh.
    We use it in :func:`ForwardProblem.electrodes`.

    :param mesh_vertex: Vertex coordinates of the electrode.
    :type mesh_vertex: array-like
    :param radius: Circle radius.
    :type radius: float
    :param L: Number of electrodes.
    :type L: int
    """
    def __init__(self, mesh_vertex, radius, L):   #Observe that mesh_vertex is from electrode i.
        super(electrode_domain, self).__init__()  #init subclass
        self.mesh_vertex=np.array(mesh_vertex).T  #Getting vertex electrodes from mesh.
        self.L=L                                  #Setting electrodes number.
        self.X=np.max(self.mesh_vertex[0])        #Max value axis x
        self.X1=np.min(self.mesh_vertex[0])       #Min value axis x 
        self.Y=np.max(self.mesh_vertex[1])        #Max value axis y
        self.Y1=np.min(self.mesh_vertex[1])       #Max value axis y
        
    def inside(self, x, on_boundary):  #Fenics functions that evals where is the Subdomain setting True os false on the vertex.
        #Here we implemented a strategy to verify if the vertex is part of electrode or nop.
        #Fenics get only vertex on boundary for us. After that we verify if the vertex is inside a "box" at (X1,X) x (Y1,Y).
        if on_boundary:  #If elemente is on boundary.
            #If vertex coordinate x are is bewteen...
            if between(x[0],((self.X),(self.X1))) or  between(x[0],((self.X1,(self.X)))):
                #If vertex coordinate y are is bewteen...
                if between(x[1],((self.Y),(self.Y1))) or  between(x[1],((self.Y1,(self.Y)))):
                    return True  #Ok, this vertex is part of my electrode.
                else:
                    return False #Nope, this isn't into electrode.


def GammaCircle(mesh, in_v, out_v, radius,centerx, centery):
    """
    Function to create a circle in the mesh with specified properties.

    :param mesh: Mesh.
    :type mesh: :class:`dolfin.cpp.mesh.Mesh`
    :param in_v: Value inside the circle.
    :type in_v: float
    :param out_v: Value outside the circle.
    :type out_v: float
    :param radius: Circle radius.
    :type radius: float
    :param centerx: Circle center position x.
    :type centerx: float
    :param centery: Circle center position y.
    :type centery: float
    
    :returns:  numpy.array -- Return a vector where each position corresponds to the value of the function in that element.
    
    :Example:

    >>> ValuesCells0 = GammaCircle(mesh=mesh_refined, in_v=3.0, out_v=1.0, radius=0.50, centerx=0.25, centery=0.25)
    >>> print(ValuesCells0)
        array([1., 1., 1., ..., 1., 1., 1.])
    >>> Q = FunctionSpace(mesh, "DG", 0) #Define Function space with basis Descontinuous Galerkin
    >>> gamma = Function(Q)
    >>> gamma.vector()[:]=ValuesCells0
    >>> plot_figure(gamma, name="", map="jet");
    
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
 

class ForwardProblem(object):
    """
    Object representing the Forward Problem in 2D EIT.

    :param mesh: Mesh.
    :type mesh: :func:`MyMesh()`
    :param z: Vector of impedances in electrodes.
    :type z: array-like

    :Example:

    >>> # Basic Definitions
    >>> L = 16
    >>> l = int(L)  # Measurements number.
    >>> z = np.ones(L) * 0.025  # Impedance
    >>> I_all = current_method(L, l, method=1)  # Current pattern

    >>> # Solver
    >>> VD = FiniteElement('CG', mesh_refined.ufl_cell(), 1)  # Space Solution
    >>> DirectProblem = ForwardProblem(mesh_refined, z)
    >>> list_u0, list_U0 = DirectProblem.solve_forward(VD, gamma0, I_all)
    >>> list_U0 = DirectProblem.sol_asarray()
    >>> print(list_U0[0:L])
        [1.0842557  0.32826713 0.19591977 0.13158264 0.06214628 -0.03412964
        -0.17331413 -0.40308837 -1.18449889 -0.42369776 -0.21120216 -0.08218106
        0.01735219 0.10789938 0.20976791 0.37492101]
    """
    
    def __init__(self, mesh, z):
        self.mesh=mesh
        self.z=z
        self.radius=mesh.radius
        self.ele_pos=mesh.electrodes.position     #electrodes position
        self.L=len(self.ele_pos)          #L electrodes.
        self.electrodes()                 #See function below, but it use class electrode_domain to set electrodes region.
        self.assembled=False
            
        
    def electrodes(self):
        """Auxiliar function, define subdomains with electrodes and calculates the size."""
        sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1) #MeshFunction 
        sub_domains.set_all(0) #Marking all vertex/edges with false.
 
        #Pass electrode position to mesh        
        list_e = [electrode_domain(self.mesh.vertex_elec[i], self.radius, self.L) for i in range(self.L)]
        #Here we have a array with objects that give us the information where is the vertex of each electrode in the mesh.
        
        #Mark electrodes in subdomain
        for index, elec in enumerate(list_e,1): elec.mark(sub_domains, index);
        #Noe we pass the information to the sub_domains with .mark(), where index is the electrode_index. index>=1
        
        #Generate 
        self.de=Measure('ds', domain=self.mesh, subdomain_data=sub_domains) #Defining integration Domain on electrodes.
        self.ele_size=np.array([assemble(Constant(1)*self.de(i+1)) for i in range(self.L)]) #Calc elec_size.
        self.list_e=list_e
        return
    
        
    def solve_forward(self, V, I_all, gamma): 
        """
        Solver for the Forward Problem of 2D Electrical Impedance Tomography (EIT).

        :param V: FiniteElement FEniCS object.
        :type V: :class:`dolfin.cpp.fem.FiniteElement`
        :param gamma: Finite Element Function representing the electrical conductivity distribution.
        :type gamma: :class:`dolfin.function.Function`
        :param I_all: Current density in each electrode for each measurement.
        :type I_all: :func:`current_method()` or list of arrays
    
        :returns: tuple -- A tuple containing two FEniCS objects, representing the potential distribution in the domain and the potentials at the electrodes, respectively.

        :Example:
        >>> DirectProblem = ForwardProblem(mesh_refined, z)
        >>> list_u0, list_U0 = DirectProblem.solve_forward(VD, I_all, gamma0)
        """
        de=self.de                        #Getting integral domain from memory.
        Intde=self.ele_size               #Size of electrodes.
        mesh=self.mesh
        
        #Verify if is a matrix or a simply vector
        I_all=np.array(I_all)
        self.I_forward=I_all
        if I_all.ndim==2:
            l=len(I_all)
        else: l=1
        
        #Finite element definition
        Rn=VectorElement("R", mesh.ufl_cell(), 0, dim=int(self.L)) #Vector in R_L for electrodes.
        R=FiniteElement("R", mesh.ufl_cell(), 0)                   #Constant for Lang. Mult    
        W=FunctionSpace(mesh, MixedElement([V, Rn, R]))       #Defining Mixed space V x R_l x R
        V_FuncSpace=FunctionSpace(mesh, V)       #Defining Mixed space V x R_l x R
        save_u=[]
        save_U=[]

        u0=TrialFunction(W) #Functions that will be reconstructed.
        v0=TestFunction(W)  #Test functions

        u, un, ul = split(u0)
        v, vn, vl = split(v0)

        # Integral( gamma*<grad_u,grad_v> ) dOmega + lagrMult
        A_inner=assemble(gamma* inner(grad(u),grad(v))*dx)

        if not self.assembled:
            lagrMult=0 #If that we have the ground potential. sum(electrode_i) = 0.
            for i in range(0,self.L): lagrMult+=(vn[i]*ul+un[i]*vl)*de(i+1) #Integral (v_i*u_mult+u_i*v_mult) d(electrode_i)
            self.A_lagr = assemble(lagrMult)


        # Integral 1/zi*(u-U_i)*(v-V_i) d(electrode_i)
        if not self.assembled:
            A_imp_0 = []
            for i in range(self.L): A_imp_0.append(assemble((u-un[i])*(v-vn[i])*de(i+1) ))
            self.A_imp_0 = A_imp_0
        A_imp = np.sum(self.A_imp_0*1/self.z)

        #Make my matriz to solve Ax=b.
        A=A_inner+A_imp+self.A_lagr 
        #We only do it only one time, if have mult. measurements, we reuse it.

        #Split w function in 3 parts, firt the function in H, the vector R^L, and the constant lagr
        w = Function(W) #Define a zero function based in W.
        u,U,u_lagr=w.split() 
        dm0 = W.sub(0).dofmap()
        dm1 = W.sub(1).dofmap()

        b0 = []
        for i in range(self.L): b0.append(assemble(vn[i]*(1/Intde[i])*de(i+1))) #Ax = sum (I_i*V_i)...
        #We integrate over electrode and divide by their size. If we don't make it, we get an error.

        A=scipy.sparse.csc_matrix(A.array())
        for j in range(l):
            I=I_all[j] if l!=1 else I_all #Is one measure or several?
            b = sum([b0[i]*I[i] for i in range(len(b0))]) #Make my b vector
            w = Function(W) #Define a zero function based in W.
            U_vec = w.vector()  #Return a vector. (x=U)
            #solve(A, U_vec, b) #Solve system AU=b. where A matrix and b vector.
            U_vec[:] = scipy.sparse.linalg.spsolve(A, b[:])

            #Append the result.
            u_aux=Function(V_FuncSpace)
            u_aux.vector()[:]= w.vector().vec()[dm0.dofs()]
            save_u.append(u_aux)
            save_U.append(w.vector().vec()[dm1.dofs()])

        self.sol_u, self.sol_U = save_u, save_U
        return self.sol_u, self.sol_U
    
    def sol_asarray(self):
        """
        Convert electrode potential results into an array and concatenate them.

        :returns: array -- A vector with concatenated potential values for all electrodes and measurements.

        :Example:
        >>> list_U0 = DirectProblem.sol_asarray()
        """
        list_U0=self.sol_U
        vec_U0=[]
        for i in range(len(list_U0)): vec_U0=np.concatenate((vec_U0, list_U0[i]), axis=0)
        return vec_U0
    
    def add_noise(self, noise_level=0, noise_type='uniform', seed=42):
        """
        Add noise to the potential values.

        :param noise_level: Noise level in percentage (between 0 and 1).
        :type noise_level: float
        :param noise_type: Type of noise to add ('uniform' or 'cauchy').
        :type noise_type: str.
        :param seed: Seed for the random number generator.
        :type seed: int

        :returns: array -- A vector with noised potential values for all electrodes and measurements.

        :Example:
        >>> list_U0_noised = DirectProblem.add_noise(noise_level=0.01, noise_type='uniform')
        """
        vec_U0=[]
        for i in range(len(self.sol_U)): vec_U0=np.concatenate((
            vec_U0, fn_addnoise(self.sol_U[i], noise_level, noise_type=noise_type, seed=seed)), axis=0)
        return vec_U0
    
    def select_potential(self, data, method=0):
        """
        Get a vector with the potential values and select a specific order.

        :param data: Potentials of all experiments.
        :type sol_index: array
        :param method: Method for selecting the potentials. 0 (do nothing) or 1 (select potentials).
        :type method: int

        :returns: array -- The selected potential values.

        :Example:
        >>> selected_U0 = DirectProblem.select_potential(list_U0, method=1)
        """
        L=self.L
        if method==1:
            if int(len(data)/L)>=L: raise Exception("This method only works with l<L") 
            data_save=[]
            j=0
            for i in range(np.size(data,0)):
                if i%L>=j:
                    data_save.append(data[i])
                if i%L==0: j+=1
            data_save=np.array(data_save)
            return data_save
        else: return data
    
    def verify_solution_graphs(self, gamma0, sol_index=0, method=1):
        """
        Plot boundary information to verify the solution.

        :param gamma0: Finite Element Function representing the electrical conductivity distribution.
        :type gamma0: :class:`dolfin.function.Function`
        :param sol_index: Index for the solution, ranging from 0 to l (number of measurements).
        :type sol_index: int
        :param method: Method for verification. 1: u+zi.gama.n.grad(u)=Ui, 2: boundary gamma.n.grad(u), 3: boundary gamma.n.grad(u) (only gaps).
        :type method: int

        :returns: array -- Plot boundary data.

        :Example:
        >>> data = DirectProblem.verify_solution_graphs(gamma0, sol_index=0, method=2)
        """
        mesh=self.mesh
        n = FacetNormal(mesh)
        u=self.sol_u[sol_index]
        de=self.de
        z=self.z

        VDG=FiniteElement('DG',mesh.ufl_cell(),0)
        Q=FunctionSpace(mesh,VDG)
        p, q = TrialFunction(Q), TestFunction(Q)
        M = assemble(inner(p, q)*ds)
        
        if method==1: L0 = inner(u+z[0]*inner(n,gamma0*grad(u)), q)*ds
        elif method==2: L0 = inner(inner(n,gamma0*grad(u)), q)*ds
        elif method==3: 
            M=assemble(inner(p, q)*de(0))
            L0 = inner(inner(n,gamma0*grad(u)), q)*de(0)
        
        b = assemble(L0)
        grad_u0 = Function(Q)
        x0 = grad_u0.vector()
        solve(M, x0, b)

        u_data =getBoundaryVertex(mesh, grad_u0);
        
        if method==1: name_='boundary u+zi.gama.n.grad(u)=Ui'
        elif method==2: name_='boundary gamma.n.grad(u)'
        elif method==3: name_='boundary gamma.n.grad(u) (only gaps)'
            
        data=plot_boundary(mesh, data=u_data, name=name_, line=0);
        return data

    def verify_solution_values(self, I_all, gamma0, sol_index=0, method=1):
        """
        Verify the solution values by comparing with the expected values.

        :param I_all: Current density in each electrode for each measurement.
        :type I_all: array or list of arrays
        :param gamma0: Finite Element Function representing the electrical conductivity distribution.
        :type gamma0: :class:`dolfin.function.Function`
        :param sol_index: Index for the solution, ranging from 0 to l (number of measurements).
        :type sol_index: int
        :param method: Method for verification. 1: Current values, 2: Average potential on electrodes.
        :type method: int

        :returns: None

        :Example:
        >>> DirectProblem.verify_solution_values(I_all, gamma0, sol_index=0, method=2)
        """
        set_log_level(50)
        mesh=self.mesh
        n = FacetNormal(mesh)
        u=self.sol_u[sol_index]
        list_U0=self.sol_U[sol_index]
        grad_u = grad(u)
        de=self.de
        z=self.z

        n = FacetNormal(mesh)
        sum0=0

        if not isinstance(I_all[0],float) or isinstance(I_all[0],int): I=I_all[0]
        else: I=I_all
        
        if method==1:
            print("Current values")
            for i in range(len(I)):
                integral=assemble(inner(n,gamma0*grad_u)*de(i+1))
                print("Calculated:", round(integral,4), "Expected:", I[i])
                sum0+=integral

            print("Soma das correntes calculada: ", sum0)
        elif method==2:
            #Average value potencial electrode.
            print("Potential values")
            for i in range(len(I)):
                integral1=assemble(inner(n,gamma0*grad_u)*de(i+1))*z[i]
                integral2=assemble(u*de(i+1))
                integral=(integral1+integral2)/self.ele_size[i]
                print("Calculated:", round(integral,5), "Expected:", round(list_U0[i],5))
        return
    
    



    