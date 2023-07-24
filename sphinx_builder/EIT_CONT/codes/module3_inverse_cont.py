from module2_forward import*

def etaFunc(mesh,cell_number): #Transforma a array com os valores na fronteira em uma função
    """Auxiliar function to do a characterisct function on an element. Value 1 on the element and 0 for the rest.
    
        :param mesh: Mesh.
        :type mesh: :class:`dolfin.cpp.mesh.Mesh`
        :param cell_number: Cell index
        :type cell_number: int
         
        """
    V=FiniteElement('DG',mesh.ufl_cell(),0) #Descontinous Galerkin degree 0
    Q=FunctionSpace(mesh,V) #Function Space based in DG and mesh
    eta=Function(Q)         #Null function 
    eta.vector()[cell_number]=1 #Eta is 1 in element with number=cell_number and zero otherwise.
    return eta

class InverseProblem(ForwardProblem):
    """Inverse Object EIT 2D 
    
    :param mesh: Any mesh from Fenics module. We recommend from :func:`MyMesh()`
    :type mesh: mesh
    :param V: FiniteElement Fenics object
    :type V: FiniteElement
    :param data: Vector with potencial in each vertex.
    :type data: array
    :param I_all: Current  for each measurement
    :type I_all: :func:`current_method()`
    
    :Example:
    
    .. code-block:: python
    
     VI=FiniteElement('CG',mesh_inverse.ufl_cell(),1) 
     InverseObject=InverseProblem(mesh_inverse, VI, u0_boundary, I_all)
     InverseObject.solve_inverse()
     gamma_k=InverseObject.gamma_k

    """
    
    def __init__(self, mesh, V, data, I_all):
        super().__init__(mesh)
        #"Basic definitions"
        self.V=FiniteElement('CG',mesh.ufl_cell(),1)  #Function Space CG degree 1 is necessary.
        self.I=I_all                   #Current pattern used in generated data.
        self.l=len(I_all)              #Number of measurements
        self.u0_boundary=data          #electrodes Potencial in array
        self.vertex=BoundaryMesh(mesh, 'exterior', order=True).entity_map(0).array()
        
        #"First guess and weight functions"
        self.Cellsgamma_k=np.ones(mesh.num_cells())*0.9           #First guess for Forwardproblem
        self.gamma_k=CellFunction(mesh, values=self.Cellsgamma_k) #Guess in cell function
        self.weight=np.ones(mesh.num_cells())        #Initial weight function
        self.innerstep_limit=50                      #Inner step limit while solve
        
        #"Solver configurations"
        self.weight_value=True  #Weight function in Jacobian matrix
        self.step_limit=30       #Step limit while solve
        self.min_v=0.05         #Minimal value in element for gamma_k
        
        #"Noise Configuration"
        self.noise_level=0      #Noise_level from data
        self.tau=1.01              #Tau for disprance principle
        
        #"Newton parameters"
        self.mu_i=0.9       #Mu initial (0,1]
        self.mumax=0.999    #Mu max
        self.nu=0.99        #Decrease last mu_n
        self.R=0.98         #Maximal decrease (%) for mu_n
        
        #"Inner parameters"
        self.inner_method='Landweber'  #Inner method for solve Newton
        
        self.land_a=1    #Step-size Landweber
        self.ME_reg=5E-4 #Regularization Minimal Error
        self.Tik_c0=1    #Regularization parameter Iterative Tikhonov
        self.Tik_q=0.95  #Regularization parameter Iterative Tikhonov
        self.LM_c0=1     #Regularization parameter Levenberg-Marquadt
        self.LM_q=0.95   #Regularization parameter Levenberg-Marquadt
        
        #"A priori information"
        self.gamma0=None #Exact Solution
        self.mesh0=None  #Mesh of exact solution
        
        #Creating a vector with all cell volumes. It's usefull for integrals in L2(Omega).
        cell_vec=[]
        for cell in cells(mesh):
            cell_vec.append(cell.volume())
        self.cell_vec=np.array(cell_vec)
        
        #Make a vector with boundary elements size that are chosen for the problem
        #This vector is used in normL2
        bmesh=BoundaryMesh(mesh, 'exterior', order=True) #Define boundarymesh
        bcell_vec=[]
        for bcell in cells(bmesh):  bcell_vec.append(bcell.volume()) #Save all boundary_elements size
        self.bcell_vec=np.tile(np.array(bcell_vec), self.l) #Adapting to l measurements.
        

        
        
    def solve_inverse(self):
        """Function that solves the inverse problem.
        
        :Example:
        
        >>> F_Problem=ForwardProblem(mesh_direct)
        >>> list_u0=F_Problem.solve_forward(VD, gamma0, I_all)

    """
        res_vec, error_vec=[], [] #To save about iterations
        self.innerstep_vec=[]     #Save inner_step newton
        mun_vec=[]                #Save mu in inner_step newton
        self.steps=0              #Save external step.

        ##############################################
        "First Forward solver"
        self.u = self.solve_forward(self.V, self.gamma_k, self.I)
        self.u_boundary=self.boundary_array() #Get boundary data and convert to array
        
        "First Save data"
        #Residue vector
        res_vec.append(np.linalg.norm(self.u0_boundary-self.u_boundary)/np.linalg.norm(self.u0_boundary)*100)
        self.innerstep_vec.append(int(0)) #Save number steps
        mun_vec.append(0)                 #Save number steps
        
        "Print information"
        if self.mesh0 is not None and self.gamma0 is not None:
            error_vec.append(self.error_gamma())    
            print("Error (%)=", error_vec[0], "Residuo (%)=", res_vec[0], " passo:", 0, "Inner step: ", 0)
        else:
            print("Residuo (%)=", res_vec[0], " passo:", 0, "Inner step: ", 0)
            
        ##############################################
        
        "Solver"
        #While discepancy or limit steps.
        while res_vec[self.steps]/100>=self.tau*self.noise_level and self.steps<=self.step_limit:

            "Derivative matrix calc"
            #If will be used LM or Tikhonov, we always have to calc. Jacobian.
            if (self.steps==0 and self.weight_value) or self.inner_method=='LM' or self.inner_method=='Tikhonov':
                Jacobiana_all=self.Jacobian_calc() #Derivative matrix calc
                
                #Create weight and add it.
                if self.weight_value: Jacobiana_all=self.weight_func(Jacobiana_all) 
                self.Jacobiana=Jacobiana_all
            else: Jacobiana_all=None

            "Inner iteration newton"
            sk, inner_step, mu=self.solve_innerNewton(Jacobiana_all)
            
            "Add sk in guess"
            self.Cellsgamma_k+=sk #Add a correction in each element
            
            #Don't have values less than c.
            self.Cellsgamma_k[self.Cellsgamma_k < self.min_v] = self.min_v
            self.gamma_k=CellFunction(self.mesh, values=self.Cellsgamma_k)  #Vector to function
            
            "Forward solver"
            self.u = self.solve_forward(self.V, self.gamma_k, self.I)
            self.u_boundary=self.boundary_array() #Get boundary data and convert to array
        
            "Saving data"
            res_vec.append(np.linalg.norm(self.u0_boundary-self.u_boundary)/np.linalg.norm(self.u0_boundary)*100)
            
            self.innerstep_vec.append(int(inner_step)) #Save number steps
            mun_vec.append(mu) #Save number steps
            
            self.steps+=1 #Next step
            
            if self.mesh0 is not None and self.gamma0 is not None:
                error_vec.append(self.error_gamma())
                print("Error (%)=", error_vec[self.steps], "Residuo (%)=", res_vec[self.steps],
                      " passo:", self.steps, "Inner step: ", inner_step)
            else: print("Residuo (%)=", res_vec[self.steps],
                      " passo:", self.steps, "Inner step: ", inner_step)
        
            #Vectors to memory object.
            self.res_vec=res_vec
            self.mun_vec=mun_vec
            self.error_vec=error_vec
        return
   
        
    def solve_innerNewton(self, Jacobiana_all):
        """Methods to solve inner step newton. Functions executed inside of :func:`solve_inverse()`. See set_InnerParameters() for more details.
            
    :param Jacobiana_all: Derivative Matrix generated by :func:`Jacobian_calc()`
    :type Jacobiana_all: Array ndim
    
    :returns:  (Array, int, float) -- Return a sk (Result of Inner Step to add in gamma_k), inner_step (Number of inner steps), mu (Regularization parameter used in the method).
            """
        b0=self.u0_boundary-self.u_boundary #Define vector b0 (Ask=b0)
        norm_b0=self.norm_funcL2(b0, 'dOmega')
        residuo=-b0       #Define res.
        norm_res=norm_b0  #Define norm_res first step.
        
        mu = self.newton_reg() #Calculate regularation parameter.
        inner_step=0
        
        sk=np.zeros(self.mesh.num_cells()) #s0 inicial do newton
        
        "------Landweber------"
        if self.inner_method=='Landweber':
            while norm_res>=mu*norm_b0 and inner_step<=self.innerstep_limit:
                sk+=-self.land_a*self.adj_dev_app_vec(residuo)

                residuo=-b0+self.dev_app_vec(sk)
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                #print(norm_res, mu*norm_b0, inner_step)
                
            
                
            "------Minimal Error------"
        elif self.inner_method=='ME':
            while norm_res>=mu*norm_b0 and inner_step<=self.innerstep_limit:
                sk_n=-self.adj_dev_app_vec(residuo)
                omega=(self.norm_funcL2(residuo, 'dOmega')**2/self.norm_funcL2(sk_n, 'Omega')**2)*self.ME_reg
                sk+=omega*sk_n
                
                residuo=-b0+self.dev_app_vec(sk)
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                #print(norm_res, mu*norm_b0, inner_step, omega)
        
            "------Conjugate-Gradient------"
        elif self.inner_method=='CG':
            while norm_res>=mu*norm_b0 and inner_step<=self.innerstep_limit:
                if inner_step==0:
                    rk=b0
                    ak=self.adj_dev_app_vec(rk)
                    pk=ak
                    ak_old=ak

                qk=self.dev_app_vec(pk)
                alphak=(self.norm_funcL2(ak_old, 'Omega')**2)/(self.norm_funcL2(qk, 'dOmega')**2)
                sk=sk+alphak*pk
                rk=rk-alphak*qk
                ak=self.adj_dev_app_vec(rk)
                betak=(self.norm_funcL2(ak, 'Omega')**2)/(self.norm_funcL2(ak_old, 'Omega')**2)
                pk=ak+betak*pk

                ak_old=ak
                residuo=-b0+self.dev_app_vec(sk)
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                #print(norm_res, mu*norm_b0, inner_step, alphak)
                #if alphak<=1: break

                
            "------Iterative Tikhonov------"
        elif self.inner_method=='Tikhonov' and inner_step<=self.innerstep_limit:
            while norm_res>=mu*norm_b0:
                alpha_k=self.Tik_c0*(self.Tik_q**inner_step)
                ADJ = Jacobiana_all.T
                square_m=ADJ@Jacobiana_all
                square_m+=alpha_k*np.identity(np.size(square_m, axis=0))
                sk=np.linalg.solve(square_m, ADJ.dot(b0)+alpha_k*sk)


                residuo=-b0+Jacobiana_all@sk
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                #print(norm_res, mu*norm_b0, inner_step)
                
            "------Levenberg-Marquadt------"
        elif self.inner_method=='LM' and inner_step<=self.innerstep_limit:
            while norm_res>=mu*norm_b0:
                alpha_k=self.LM_c0*(self.LM_q**inner_step)
                ADJ = Jacobiana_all.T
                square_m=ADJ@Jacobiana_all
                square_m+=alpha_k*np.identity(np.size(square_m, axis=0))
                sk=np.linalg.solve(square_m, ADJ.dot(b0))


                residuo=-b0+Jacobiana_all@sk
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                #print(norm_res, mu*norm_b0, inner_step)

        return sk, inner_step, mu

        
    def Jacobian_calc(self):
        """Calcuate derivative matrix. Function executed inside of :func:`solve_inverse()`.
        
        :returns:  (Array ndim) -- Return the derivative matrix.        
              """
        print("Calculando Jacobiana.")
        V=self.V
        mesh=self.mesh
        gamma=self.gamma_k
        R=FiniteElement('R',mesh.ufl_cell(),0)

        W=V*R
        W=FunctionSpace(mesh,W)

        bmesh=BoundaryMesh(mesh, 'exterior', order=True)
        indice_borda=bmesh.entity_map(0).array()
        linha=len(indice_borda) #Linha da matriz jacobiana
        
        coluna=mesh.num_cells() #Coluna da matriz jacobiana 
        

        
        (du,c)=TrialFunctions(W)
        (v,d)=TestFunction(W)
        lagrMult=(v*c+du*d)*ds
        a = gamma*inner(grad(du),grad(v))*dx + lagrMult
        A=assemble(a)
        
        for h in range(self.l):
            Jacobiana=np.zeros((linha,coluna)) #Armazena as derivadas em cada elemento    
            u=self.u[h]
            for cell in cells(mesh):    
                eta = etaFunc(mesh, cell.index())
                L=-eta*inner(grad(u),grad(v))*dx
                b = assemble(L)
                
                z = Function(W)
                U = z.vector()
                solve(A, U, b, 'cg', 'ilu')
                (du,c)=z.split(deepcopy=True)

                derivada=getBoundaryVertex(mesh,du)
                Jacobiana[:,cell.index()]=derivada
        
            Jacobiana=np.array(Jacobiana)    
            if h==0: Jacobiana_all=Jacobiana
            else: Jacobiana_all=np.concatenate((Jacobiana_all, Jacobiana), axis=0)

        print("Fim do cálculo da Jacobiana")
        
        return Jacobiana_all
    

    def adj_dev_app_vec(self, b0):
        mesh=self.mesh
        weight=self.weight
        vertex=self.vertex
        n_vert_boundary=len(vertex)
        gamma_k=self.gamma_k
        sol_u=self.sol_u

        ADJ=np.zeros(mesh.num_cells())

        for i in range(self.l):
            x_i, x_f = int(i*n_vert_boundary), int((i+1)*n_vert_boundary) #Para acertar as linhas
            sigma=self.Sigma(b0[x_i:x_f])
            psi=self.findpsi(gamma_k, sigma)
            ADJ+=self.calcADJ(psi, sol_u[i])
            
        ADJ=ADJ*1/weight 

        return ADJ

    def dev_app_vec(self, sk):
        mesh=self.mesh
        gamma_k=self.gamma_k
        V=self.V
        sol_u=self.sol_u
        vector_dev=sk

        R=FiniteElement('R',mesh.ufl_cell(),0)

        W=V*R
        W=FunctionSpace(mesh,W)

        bmesh=BoundaryMesh(mesh, 'exterior', order=True)
        indice_borda=bmesh.entity_map(0).array()
        derivada_sum=[]

        vector_dev=np.array(vector_dev)/np.array(self.weight)



        (du,c)=TrialFunctions(W)
        (v,d)=TestFunction(W)


        eta=CellFunction(mesh, values=vector_dev)
        lagrMult=(v*c+du*d)*ds(mesh)
        a = gamma_k*inner(grad(du),grad(v))*dx(mesh) + lagrMult
        A=assemble(a)
        
        for u in sol_u:    
            L=-eta*inner(grad(u),grad(v))*dx(mesh)

            b = assemble(L)
            w = Function(W)
            U = w.vector()
            solve(A, U, b, 'cg', 'ilu')
            (du_i,c)=w.split(deepcopy=True)

            derivada=getBoundaryVertex(mesh,du_i)
            derivada_sum = np.hstack((derivada_sum, derivada))

        return derivada_sum

        
        
    def weight_func(self, Jacobiana):
        """Determine the weights for Derivative Matrix and apply.
        
        :param Jacobiana: Derivative Matrix generated by :func:`Jacobian_calc()`
        :type Jacobiana: Array ndim
    
        :returns:  (Array ndim) -- Return the derivative matrix with weights.
        
        """
        self.weight=np.linalg.norm((Jacobiana.T*np.sqrt(self.bcell_vec)).T, axis=0)*(1/self.cell_vec)            
      
        Jacobiana=Jacobiana*1/self.weight
        return Jacobiana
    
    def newton_reg(self):
        """Determine the regulazation parameter for the Newton innerstep in :func:`solve_innerNewton()`."""
        #ref: https://doi.org/10.1137/040604029
        passo=self.steps
        innerstep_vec=self.innerstep_vec
        
        if passo>1:
            if innerstep_vec[passo-1]>=innerstep_vec[passo-2]: #If inner step increase
                mu_n=1-((innerstep_vec[passo-1]/innerstep_vec[passo])*(1-self.mu))
            else: mu_n=self.nu*self.mu #Remmember, nu<1.
            mu=self.mumax*np.amax([self.R*self.mu, mu_n]) #R limits how much the mu_n change.
        else: mu=self.mu_i #For the two first steps.
        
        self.mu=mu
        print("mu_n", mu)
        return mu

    def norm_funcL2(self, vector, domain):
        """This function is used in inner step of newton to calculate the norm's in L2 Omega and dOmega
         
        :param domain: 'Omega' or 'dOmega'.
        :type domain: str
        
        :returns:  (float) -- Return norm.
        
        """
        mesh=self.mesh
        weight=self.weight
        if domain=='Omega':
            norm=sum(weight*vector*vector*self.cell_vec)
        elif domain=="dOmega":
            norm=sum(vector*vector*self.bcell_vec)

        norm=sqrt(norm)
        return norm
    
    
    def error_gamma(self):
        """Percentual error in L2 of the exact and reached solution for gamma.
        To works need use :func:`set_answer()`."""
        V0=FiniteElement('DG',self.mesh.ufl_cell(),0)
        V1=FiniteElement('DG',self.mesh0.ufl_cell(),0)
        
        Q0=FunctionSpace(self.mesh0, V0)
        Q1=FunctionSpace(self.mesh, V1)
        
        GammaElement0=interpolate(self.gamma0, Q0) #Interpolate to mesh_forward
        GammaElement1=interpolate(self.gamma_k, Q1) #interpolate gamma_k to mesh_inverse
        GammaElement1=interpolate(GammaElement1, Q0)#Interpolate from mesh_inverse to mesh_forward
        
        error_L2 = errornorm(GammaElement0, GammaElement1, 'L2') #Error norm
        norm_gamma0 = norm(GammaElement0, 'L2')                  #Norm exact solution
        GammaError=error_L2/norm_gamma0*100                      #percentual.
        return GammaError
    
    def set_answer(self, gamma0, mesh0):
        """"Get and set the answer if we have it.
        It is usefull to determine the best solution reached.
        :func:`error_gamma()` will return you the percentual error in L2. 
        
        :param mesh0: Any mesh from Fenics module. We recommend from :func:`MyMesh()`
        :type mesh0: mesh
        :param gamma0: Finite Element Function
        :type gamma0: :func:`CellFunction()`
        
         :Example:

         >>> InverseObject=InverseProblem(mesh_inverse,  ele_pos,  z, list_U0_noised, I_all, l)
         >>> InverseObject.set_answer(gamma0, mesh_direct)
        
        """
        self.gamma0=gamma0
        self.mesh0=mesh0
        return
    
    def set_NewtonParameters(self,  **kwargs):
        """Newton Parameters
        
            Kwargs:
               * **mu_i** (float): Mu initial (0,1]
               * **mumax** (float): mumax (0,1]
               * **nu**    (float): Decrease last mu_n
               * **R**     (float): Minimal value for mu_n
        
            Default Parameters:
                >>> self.mu_i=0.9      
                >>> self.mumax=0.999   
                >>> self.nu=0.99       
                >>> self.R=0.9         
                
                :Example:

                >>> InverseObject=InverseProblem(mesh_inverse,  ele_pos,  z, list_U0_noised, I_all, l)
                >>> InverseObject.set_NewtonParameters(mu_i=0.90, mumax=0.999, nu=0.985, R=0.90)
        """        
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
        return
        
    def set_NoiseParameters(self, tau, noise_level):
        """Noise Parameters to stop with Discrepancy Principle.

            :param tau: Tau for disprance principle [0, \infty)
            :type tau: float
            :param noise_level: Noise_level(%) from data [0,1)
            :type noise_level: float

            Default Parameters:
                >>> self.tau=0      
                >>> self.noise_level=0   
                
            :Example:

            >>> InverseObject=InverseProblem(mesh_inverse,  ele_pos,  z, list_U0_noised, I_all, l)
            >>> InverseObject.set_NoiseParameters(tau=5, noise_level=0.01)
        """
        self.tau=tau
        self.noise_level=noise_level
        
    def set_firstguess(self, Cellsgamma_k):
        """ Default parameters for first guess in external step newton.
       
            :param Cellsgamma_k: We expect a vector that represents the value of gamma in your cell.
            :type Cellsgamma_k: array
            
            Default Parameters:
                >>> self.Cellsgamma_k=np.ones(self.mesh.num_cells())*0.9      
                
            :Example:

            >>> InverseObject=InverseProblem(mesh_inverse,  ele_pos,  z, list_U0_noised, I_all, l)
            >>> InverseObject.set_firstguess(np.ones(mesh_inverse.num_cells())*5)
            
        """
        self.Cellsgamma_k=Cellsgamma_k           #First guess for Forwardproblem
        self.gamma_k=CellFunction(self. mesh, values=self.Cellsgamma_k) #Guess in cell function """
        
    def set_solverconfig(self, **kwargs):
        """Solver config.
        
            Kwargs:
               * **weight_value** (bool): Weight function in Jacobian matrix
               * **step_limit** (float): Step limit while solve
               * **min_v**    (float): Minimal value in element for gamma_k             
        
            Default Parameters:       
               >>> self.weight_value=True  
               >>> self.step_limit=5       
               >>> self.min_v=0.05         
                
            :Example:

            >>> InverseObject=InverseProblem(mesh_inverse,  ele_pos,  z, list_U0_noised, I_all, l)
            >>> InverseObject.set_solverconfig(weight_value=True, step_limit=200, min_v=0.01)
            
        """       
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def set_InnerParameters(self, **kwargs):
        """Inner-step Newton Parameters
        
            Kwargs:
               * **inner_method** (str): Method to solver inner step newton. Options: 'Landweber', 'CG', 'ME', 'LM', 'Tikhonov'
               * **land_a**       (int): Step-size Landweber
               * **ME_reg**     (float): Minimal value in element for gamma_k             
               * **Tik_c0**     (float): Regularization parameter Iterative Tikhonov
               * **Tik_q**      (float): Regularization parameter Iterative Tikhonov
               * **LM_c0**     (float): Regularization parameter Levenberg-Marquadt
               * **LM_q**       (float): Regularization parameter Levenberg-Marquadt
        
            Default Parameters:
               >>> self.inner_method='Landweber'
               >>> self.land_a=1    
               >>> self.ME_reg=5E-4 
               >>> self.Tik_c0=1    
               >>> self.Tik_q=0.95  
               >>> self.LM_c0=1     
               >>> self.LM_q=0.95   

            :Example:

            >>> InverseObject=InverseProblem(mesh_inverse,  ele_pos,  z, list_U0_noised, I_all, l)
            >>> InverseObject.set_InnerParameters(inner_method='ME', ME_reg=5E-4)
            
        """
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            
    def Sigma(self, b0): #Transforma a array com os valores na fronteira em uma função
        "Array to values on the boundary, we use it in CalcADJ"
        mesh=self.mesh
        vertex=self.vertex
        sigma=np.array(b0)

        L=FiniteElement('CG',mesh.ufl_cell(),1)
        V=FunctionSpace(mesh,L)
        S=Function(V)
        ind=vertex_to_dof_map(V)
        S_vec=np.zeros(len(S.vector()[:]))
        S_vec[ind[vertex]]=sigma/self.bcell_vec[0]
        S.vector()[:]=S_vec
        #self.bcell_vec is the correction of Adjunt.
        return S

    def findpsi(self, gamma, sigma):
        "Gets Sigma and calculates psi, we use it in CalcADJ"
        mesh=self.mesh
        V=self.V

        R=FiniteElement('R',mesh.ufl_cell(),0)

        W=V*R
        W=FunctionSpace(mesh,W)

        (psi,c)=TrialFunctions(W)
        (v,d)=TestFunction(W)

        lagrMult=(v*c+psi*d)*ds

        a=gamma*inner(grad(psi),grad(v))*dx+lagrMult
        L=sigma*v*ds
        
        A=assemble(a)
        b = assemble(L)
        w = Function(W)
        U = w.vector()
        solve(A, U, b)

        #w=Function(W)
        #solve(a==L,w)
        (psi,c)=w.split(deepcopy=True)
        return psi

    def calcADJ(self, psi, u):
        """Adjunt matrix applied at a function."""
        mesh=self.mesh
        V=FiniteElement('DG',mesh.ufl_cell(),0) #Espaço das funções descontínuas DG grau 0
        Q=FunctionSpace(mesh,V)

        ADJ=project(-inner(grad(psi),grad(u)), Q).vector()[:]
        ADJ=np.array(ADJ)

        "Correction cell_volume"
        ADJ=ADJ*self.cell_vec

        return ADJ