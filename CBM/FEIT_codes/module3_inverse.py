from module2_forward import *
import copy

class InverseProblem(ForwardProblem):
    """
    Inverse Object EIT 2D.

    Class for solving inverse object electrical impedance tomography (EIT) problems in 2D. The inverse problem aims to reconstruct the electrical conductivity distribution within a domain based on measured electrical potentials at the boundary (electrodes).

    :param mesh: Any mesh from the Fenics module. We recommend using :func:`MyMesh()`.
    :type mesh: mesh
    :param data: Vector with the potentials measured at the electrodes or any other vector.
    :type data: array
    :param I_all: Current density in each electrode for each measurement. It can be a function generated using :func:`current_method()` or a list of arrays.
    :type I_all: :func:`current_method()` or list of arrays
    :param z: Vector of impedances in electrodes.
    :type z: array
    :param select_pot_method: Method to select the potential in the array. 0 - None, 1 - Select method. (Optional, Default: 0)
    :type select_pot_method: int

    :Example:
        
    Initialize the inverse problem and solve it:

    >>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, z)
    >>> InverseObject.solve_inverse()
    >>> gamma_k = InverseObject.gamma_k
    
    """
    def __init__(self, mesh, data, I_all, z=None, select_pot_method=0):
        super().__init__(mesh, z)
        #"Basic definitions"
        self.mesh=mesh
        self.V=FiniteElement('CG',mesh.ufl_cell(),1)  #Function Space CG degree 1 is necessary.
        self.Q_DG = FunctionSpace(self.mesh, "DG", 0)
        self.I=I_all          #Current pattern used in generated data.
        self.list_U0=data     #electrodes Potencial in array
        self.select_pot_method=select_pot_method  #Method select potencial in the array. 0 - None, 1- Select method
        #Verify if is a matrix or a simply vector
        self.I=np.array(self.I)
        if self.I.ndim==2:
            self.l=len(self.I)
        else: self.l=1

        #"First guess and weight functions"
        self.firstguess=np.ones(mesh.num_cells())           #First guess for Forwardproblem
        self.Cellsgamma_k=np.array(self.firstguess)                       #Solution in array.
        self.gamma_k = Function(self.Q_DG)
        self.gamma_k.vector()[:]=self.Cellsgamma_k
        self.weight=np.ones(mesh.num_cells())             #Initial weight function
        
        #"Solver configurations"
        self.verbose=False
        self.weight_value=False    #Are you going to use the weight function in the Jacobian matrix?
        self.step_limit=30        #Step limit while solve
        self.innerstep_limit=1000 #Inner step limit while solve
        self.min_v=1E-3           #Minimal value in element for gamma_k
        
        #"Noise Configuration"
        self.noise_level=0      #Noise_level from data (%) Ex: 0.01 = 1%
        self.tau=1.01           #Tau for disprance principle, tau>1
        
        #"Newton parameters"
        self.mu_i=0.9       #Mu initial (0,1]
        self.mumax=0.999    #Mu max
        self.nu=0.99        #Decrease last mu_n
        self.R=0.98         #Maximal decrease (%) for mu_n
        
        #"Inner parameters"
        self.inner_method='Landweber'  # Default inner method for solve Newton
        
        #Other Default parameters
        self.land_a=1    #Step-size Landweber
        self.ME_reg=5E-4 #Regularization Minimal Error
        self.Tik_c0=1    #Regularization parameter Iterative Tikhonov
        self.Tik_q=0.95  #Regularization parameter Iterative Tikhonov
        self.LM_c0=1     #Regularization parameter Levenberg-Marquadt
        self.LM_q=0.95   #Regularization parameter Levenberg-Marquadt
        
        #"A priori information"
        self.gamma0=None  #Exact Solution
        self.mesh0=None   #Mesh of exact solution
        
        #Creating a vector with all cell volumes. It's usefull for integrals in L2(Omega).
        cell_vec=[]
        for cell in cells(mesh):
            cell_vec.append(cell.volume())
        self.cell_vec=np.array(cell_vec)
        
        #Make a vector with electrodes size that are chosen for the problem
        #This vector is used in norm_L2(dOmega)
        self.size_elec_vec=self.select_potential(np.tile(self.ele_size, self.l), method=select_pot_method)
        
                #Banach Spaces parameters
        #Obs: Here we have several variables that we are working yet.
        self.Lp_space=2 #X, L_p space
        self.Lr_space=2 #Y, L_r space
    def solve_inverse(self):
        """Solve the inverse problem using a Newton-based method.

        This method is the main solver of the inverse problem. It iteratively solves the inverse EIT problem using a Newton-based method. The solution is stored in the `gamma_k` attribute.

        :Example:
        >>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, z)
        >>> InverseObject.solve_inverse()
        """
        "Creating basic informations."
        res_vec, error_vec=[], [] #To save about iterations
        self.innerstep_vec=[]     #Save inner_step newton
        mun_vec=[]                #Save mu in inner_step newton     
        gamma_all=np.array([])    #Saving all gamma_k
        self.steps=0              #Save external step.
        r=self.Lr_space #Data space L^r

        ##############################################
        "First Forward solver"
        self.list_u , self.list_U= self.solve_forward(self.V, self.I, self.gamma_k)
        self.list_U=np.array(self.list_U).flatten() #Convert to array
        self.list_U=self.select_potential(self.list_U, method=self.select_pot_method)
        b0 = self.list_U0-self.list_U #Define vector b0 (Ask=b0)
        
        "First Save data"
        #Residual vector
        res_vec.append(self.norm_funcL2(b0,           'dOmega', r)/
                       self.norm_funcL2(self.list_U0, 'dOmega', r)*100)
        self.innerstep_vec.append(int(0)) #Save inner number steps
        mun_vec.append(0)                 #Save number steps
        
        "Print information"
        if self.mesh0 is not None and self.gamma0 is not None and self.verbose:
            error_vec.append(self.error_gamma())    
            print("Error (%):", error_vec[0], "Residual (%):", res_vec[0], "step:", 0, "Inner step:", 0)
        elif self.verbose:
            print("Residual (%):", res_vec[0], "step:", 0, "Inner step:", 0)
            
        "Solver"
        ##############################################
        #While discepancy or limit steps.
        while res_vec[self.steps]/100>=self.tau*self.noise_level and self.steps<self.step_limit:
            
            "Inner iteration newton"
            Jacobiana_all=self.Jacobian_calc()        #Derivative matrix calc
            sk, inner_step, mu=self.solve_innerNewton(Jacobiana_all, b0)
            self.Cellsgamma_k+=sk #Add a correction in each element
            
            #Don't have values less than c.s
            self.Cellsgamma_k[self.Cellsgamma_k < self.min_v] = self.min_v
            self.gamma_k.vector()[:]=self.Cellsgamma_k
            
            
            "Forward solver"
            self.list_u , self.list_U= self.solve_forward(self.V, self.I, self.gamma_k)
            self.list_U=np.array(self.list_U).flatten() #Convert to array
            self.list_U=self.select_potential(self.list_U, method=self.select_pot_method)
            b0 = self.list_U0-self.list_U #Define vector b0 (Ask=b0)
            
            "Saving data"
            #Saving gamma.
            if self.steps==0: gamma_all=np.array(self.Cellsgamma_k)
            else:  gamma_all=np.vstack((gamma_all, (np.array(self.Cellsgamma_k))))
                
            #Append residuo
            res_vec.append(self.norm_funcL2(b0,           'dOmega', r)/
                           self.norm_funcL2(self.list_U0, 'dOmega', r)*100)
            
            self.innerstep_vec.append(int(inner_step)) #Save number steps
            mun_vec.append(mu)                         #Save number steps
            
            #If we have exact solution, save error.
            if self.mesh0 is not None and self.gamma0 is not None:
                error_vec.append(self.error_gamma())
                
            self.steps+=1 #Next step.
            
            #Print information
            if self.mesh0 is not None and self.gamma0 is not None and self.verbose:
                print("Error (%):", np.round(error_vec[self.steps],6), "Residual (%):", np.round(res_vec[self.steps], 6),
                      "step:", self.steps, "Inner step:", inner_step, "mu_n:", np.round(self.mu, 6))
            elif self.verbose: print("Residual (%):", np.round(res_vec[self.steps],6),
                      "step:", self.steps, "Inner step:", inner_step, "mu_n:", np.round(self.mu, 6))
            ####################
            #Vectors to memory object.
            self.gamma_all=np.copy(gamma_all)
            self.res_vec=res_vec
            self.mun_vec=mun_vec
            self.error_vec=error_vec
            
        #############End-While############################
        return
   
        
    def solve_innerNewton(self, Jacobiana_all,b0):
        """Solve the inner step of the Newton method.

        This method is used to solve the inner step of the Newton method. It iteratively updates the solution `gamma_k` using different regularization methods.

        :param Jacobiana_all: Derivative Matrix generated by :func:`Jacobian_calc()`.
        :type Jacobiana_all: ndarray
        :param b0: Vector containing the difference between the measured potentials and the potentials calculated with the current guess.
        :type b0: array

        :returns: (Array, int, float) -- Returns `sk` (the result of the inner step to add to `gamma_k`), `inner_step` (number of inner steps taken), `mu` (regularization parameter used in the method).
        """        
        #If weight True and step=0, determine the weight.
        if self.weight_value and self.steps==0: self.weight_func(Jacobiana_all) 
            
        ADJ=(Jacobiana_all*1/self.weight).T #Add weight.
        norm_b0=self.norm_funcL2(b0, 'dOmega', self.Lr_space)
        residuo=-b0            #Define res.
        norm_res=norm_b0       #Define norm_res first step.
        
        mu = self.newton_reg() #Calculate regularation parameter.
        inner_step=0
        
        sk=np.zeros(self.mesh.num_cells()) #s0 inicial do newton
        
        "------Landweber------"
        if self.inner_method=='Landweber':
            while norm_res>=mu*norm_b0 and inner_step<self.innerstep_limit:
                sk+=-self.land_a*ADJ@residuo

                residuo=-b0+Jacobiana_all@sk
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                
            "------Minimal Error------"
        elif self.inner_method=='ME' :
            while norm_res>=mu*norm_b0 and inner_step<self.innerstep_limit:
                sk_n=-ADJ@residuo
                omega=self.norm_funcL2(residuo, 'dOmega')**2/self.norm_funcL2(sk_n, 'Omega')**2*self.ME_reg
                sk+=omega*sk_n

                residuo=-b0+Jacobiana_all@sk
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
        
            "------Conjugate-Gradient------"
        elif self.inner_method=='CG' :
            while norm_res>=mu*norm_b0 and inner_step<self.innerstep_limit:
                if inner_step==0:
                    rk=b0
                    ak=ADJ@rk
                    pk=ak
                    ak_old=ak                    

                qk=Jacobiana_all@pk
                alphak=(self.norm_funcL2(ak_old, 'Omega')**2)/(self.norm_funcL2(qk, 'dOmega')**2)
                sk=sk+alphak*pk
                rk=rk-alphak*qk
                ak=ADJ@rk
                betak=(self.norm_funcL2(ak, 'Omega')**2)/(self.norm_funcL2(ak_old, 'Omega')**2)
                pk=ak+betak*pk

                ak_old=ak
                residuo=-b0+Jacobiana_all@sk
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                

                
            "------Iterative Tikhonov------"
        elif self.inner_method=='Tikhonov':
            square_m0=ADJ@Jacobiana_all
            while norm_res>=mu*norm_b0 and inner_step<self.innerstep_limit:
                alpha_k=self.Tik_c0*(self.Tik_q**inner_step)
                square_m=square_m0+alpha_k*np.identity(np.size(square_m0, axis=0))
                sk+=np.linalg.solve(square_m, ADJ.dot(b0-Jacobiana_all@sk)) #Verificar com fab


                residuo=-b0+Jacobiana_all@sk
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1
                
            "------Levenberg-Marquadt------"
        elif self.inner_method=='LM':
            square_m0=ADJ@Jacobiana_all
            while norm_res>=mu*norm_b0 and inner_step<self.innerstep_limit:
                alpha_k=self.LM_c0*(self.LM_q**inner_step)
                square_m=square_m0+alpha_k*np.identity(np.size(square_m0, axis=0))
                sk=np.linalg.solve(square_m, ADJ.dot(b0))

                residuo=-b0+Jacobiana_all@sk
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                inner_step+=1   
                

        return sk, inner_step, mu
        
    def Jacobian_calc(self):
        """Calculate the derivative matrix (Jacobian).

        This method calculates the derivative matrix (Jacobian) required for the inverse EIT problem.

        :returns: (ndarray) -- Returns the derivative matrix.
        """
        BU_save=[] #Save potential electrodes
        bu_save=[] #Save potential domain
        list_u=self.list_u           #Get potential domain list
        
        
        
        #Construction new current pattern for Jacobian calc.
        #Ref: https://fabiomargotti.paginas.ufsc.br/files/2017/12/Margotti_Fabio-3.pdf chap 5.2.1
        I2_all=[]  #We will use to construct a smart way to calc. Jacobian
        for i in range(self.L):
            #I2_i=1 at electrode i and zero otherwise
            I2=np.zeros(self.L)
            I2[i]=1
            I2_all.append(I2)
        
        bu,BU=self.solve_forward(self.V, I2_all, self.gamma_k)
        bu_save=bu
        
        
        #separating electrodes data
        select_data=np.tile(range(0,self.L), self.l)
        select_data=self.select_potential(select_data, method=self.select_pot_method)
        select_data = np.split(select_data, np.where(select_data[:-1] == self.L-1)[0]+1)
        
        Q_DG = VectorFunctionSpace(self.mesh, "DG", 0, dim=2)
        list_grad_u = [project(grad(u), Q_DG).vector()[:].reshape(-1, 2) for u in list_u]
        list_grad_bu = [project(grad(bu), Q_DG).vector()[:].reshape(-1, 2) for bu in bu_save]      

        for h in range(self.l): #For each experiment
            derivada=[]
            for j in select_data[h]: #for each electrode
                derivada.append(-1*np.sum(list_grad_bu[j]*list_grad_u[h], axis=1)) #Get the function value in eache element
                
            Jacobiana=derivada*self.cell_vec #Matrix * Volume_cell
            if h==0: Jacobiana_all=Jacobiana #Append all jacs.
            else: Jacobiana_all=np.concatenate((Jacobiana_all, Jacobiana), axis=0)
                
        return Jacobiana_all
    
    def weight_func(self, Jacobiana):
        """Determine the weights for the derivative matrix and apply them.

        This method determines the weights for the derivative matrix and applies them. The weights are used to improve the convergence of the Newton-based method.

        :param Jacobiana: Derivative Matrix generated by :func:`Jacobian_calc()`.
        :type Jacobiana: ndarray

        :returns: (ndarray) -- Returns the derivative matrix with weights.
        """
        p=self.Lp_space
        #norm(Jacobian_line)*1/vol_cell_k*1/gamma_cell_k
        self.weight=np.linalg.norm(Jacobiana, ord=p, axis=0)*(1/self.cell_vec)*(1/self.Cellsgamma_k)
        return self.weight
    
    def newton_reg(self):
        """Determine the regularization parameter for the Newton inner step.

        This method determines the regularization parameter used in the Newton inner step, considering the Newton method's progress.

        :returns: (float) -- Returns the regularization parameter `mu`.
        """
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
        return mu
    
    def norm_funcL2(self, vector, domain, p=2):
        """Calculate the L2 norm of the vector.

        This method calculates the L2 norm of the input vector for the given domain.

        :param vector: Input vector.
        :type vector: array
        :param domain: The domain for which the norm is calculated. It can be 'Omega' or 'dOmega'.
        :type domain: str
        :param p: Power of the norm. (Optional, Default: 2)
        :type p: int

        :returns: (float) -- Returns the L2 norm of the vector for the specified domain.
        """
        mesh=self.mesh
        weight=self.weight
        if domain=='Omega':
            norm=sum(abs(vector)**p*self.cell_vec*weight)
        elif domain=="dOmega":
            norm=sum(abs(vector)**p*self.size_elec_vec)

        norm=norm**(1/p)
        return norm
    
    def error_gamma(self):
        """Calculate the percentual error in L2 of the exact and reached solution for gamma.

        This method calculates the percentage error in the L2 norm between the exact and the reached solution for gamma. To use this method, the exact solution should be set using :func:`set_answer()`.

        :returns: (float) -- Returns the percentage error in the L2 norm.
        """
        V0=FiniteElement('DG',self.mesh.ufl_cell(),0)
        V1=FiniteElement('DG',self.mesh0.ufl_cell(),0)
        
        Q0=FunctionSpace(self.mesh0, V0)
        Q1=FunctionSpace(self.mesh, V1)
        
        GammaElement0=interpolate(self.gamma0, Q0) #Interpolate to mesh_refined
        GammaElement1=interpolate(self.gamma_k, Q1) #interpolate gamma_k to mesh_inverse
        GammaElement1=interpolate(GammaElement1, Q0)#Interpolate from mesh_inverse to mesh_refined
        
        error_L2 = errornorm(GammaElement0, GammaElement1, 'L2') #Error norm
        norm_gamma0 = norm(GammaElement0, 'L2')                  #Norm exact solution
        GammaError=error_L2/norm_gamma0*100                      #percentual.
        
        if self.steps<=1:
            cell_vec0=[]
            for cell in cells(self.mesh0): cell_vec0.append(cell.volume())
            self.cell_vec0=np.array(cell_vec0)        
        
        #A way to calculated the error in a different norm.
        if self.Lp_space!=2:
            #Error norm
            error_Lp=sum(abs((np.array(GammaElement1.vector())-np.array(GammaElement0.vector())))**self.Lp_space*self.cell_vec0 )
            error_Lp=error_Lp**(1/self.Lp_space)
            #Exact norm
            norm_gamma0_Lp=sum(abs((np.array(np.array(GammaElement0.vector())))**self.Lp_space*self.cell_vec0 ))
            norm_gamma0_Lp=norm_gamma0_Lp**(1/self.Lp_space)
            
            GammaError_Lp= error_Lp/norm_gamma0_Lp*100 #Percentual
        #print(GammaError_Lp)
        
        return GammaError
    
    def set_answer(self, gamma0, mesh0):
        """
        Set the exact solution for gamma.

        This method sets the exact solution (gamma0) and its corresponding mesh (mesh0) to be used for comparison and error calculation. This is useful to determine the best solution reached.

        :param gamma0: Finite Element Function representing the exact solution for gamma.
        :type gamma0: Function
        :param mesh0: Mesh
        :type mesh0:  :func:`MyMesh`

        :Example:
        >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, z)
        >>> InverseObject.set_answer(gamma0, mesh_refined)

        """
        self.gamma0=gamma0
        self.mesh0=mesh0
        return
    
    def set_NewtonParameters(self,  **kwargs):
        """Set Newton Parameters for the inverse problem.

        Kwargs:
            * **mu_i** (float): Initial value for mu (0, 1].
            * **mumax** (float): Maximum value for mu (0, 1].
            * **nu** (float): Factor to decrease the last mu_n.
            * **R** (float): Minimal value for mu_n.

        Default Parameters:
            >>> self.mu_i = 0.9
            >>> self.mumax = 0.999
            >>> self.nu = 0.99
            >>> self.R = 0.9

        :Example:
        >>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
        >>> InverseObject.set_NewtonParameters(mu_i=0.90, mumax=0.999, nu=0.985, R=0.90)
        """
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
        return
        
    def set_NoiseParameters(self, tau, noise_level):
        """
        Set Noise Parameters for stopping with the Discrepancy Principle.

        :param tau: Tau value for the discrepancy principle [0, ∞).
        :type tau: float
        :param noise_level: Noise level (%) in the data [0, 1).
        :type noise_level: float

        Default Parameters:
            >>> self.tau = 0
            >>> self.noise_level = 0

        :Example:
        >>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
        >>> InverseObject.set_NoiseParameters(tau=5, noise_level=0.01)
        """
        self.tau=tau
        self.noise_level=noise_level
        
    def set_firstguess(self, Cellsgamma_k):
        """Set default parameters for the first guess in the external step Newton.

        :param Cellsgamma_k: A vector representing the initial values of gamma in each cell.
        :type Cellsgamma_k: array

        Default Parameters:
            >>> self.Cellsgamma_k = np.ones(self.mesh.num_cells()) * 0.9

        :Example:
        >>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
        >>> InverseObject.set_firstguess(np.ones(mesh_inverse.num_cells()) * 5)
        """
        self.firstguess=Cellsgamma_k           #First guess for Forwardproblem
        self.Cellsgamma_k=self.firstguess           
        self.gamma_k.vector()[:]=self.Cellsgamma_k #Guess in cell function """
        
    def set_solverconfig(self, **kwargs):
        """Set Solver configuration for the inverse problem.

        Kwargs:
            * **weight_value** (bool): Use a weight function in the Jacobian matrix.
            * **step_limit** (float): Step limit while solving.
            * **min_v** (float): Minimal value in an element for gamma_k.

        Default Parameters:
            >>> self.weight_value = True
            >>> self.step_limit = 5
            >>> self.min_v = 0.05

        :Example:
        >>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
        >>> InverseObject.set_solverconfig(weight_value=True, step_limit=200, min_v=0.01)
        """    
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

    def set_InnerParameters(self, **kwargs):
        """Set Inner-step Newton Parameters for the inverse problem.

        Kwargs:
            * **inner_method** (str): Method to solve the inner step Newton. Options: 'Landweber', 'CG', 'ME', 'LM', 'Tikhonov'.
            * **land_a** (int): Step-size for Landweber method.
            * **ME_reg** (float): Minimal value in an element for gamma_k.
            * **Tik_c0** (float): Regularization parameter for Iterative Tikhonov.
            * **Tik_q** (float): Regularization parameter for Iterative Tikhonov.
            * **LM_c0** (float): Regularization parameter for Levenberg-Marquardt.
            * **LM_q** (float): Regularization parameter for Levenberg-Marquardt.

        Default Parameters:
            >>> self.inner_method = 'Landweber'
            >>> self.land_a = 1
            >>> self.ME_reg = 5E-4
            >>> self.Tik_c0 = 1
            >>> self.Tik_q = 0.95
            >>> self.LM_c0 = 1
            >>> self.LM_q = 0.95

        :Example:
        >>> InverseObject = InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
        >>> InverseObject.set_InnerParameters(inner_method='ME', ME_reg=5E-4)
        """
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])