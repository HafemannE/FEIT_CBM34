from .module2_forward import *
from scipy.sparse import csr_matrix
from numba import njit
import copy

class InverseProblem(ForwardProblem):
    """Inverse Object EIT 2D 
    
    :param mesh: Any mesh from Fenics module. We recommend from :func:`MyMesh()`
    :type mesh: mesh
    :param z:  Vector of impedances in electrodes
    :type z: array
    :param data: Vector with potencial in electrodes or any other vector.
    :type data: array
    :param I_all: Current density in each electrode for each measurement
    :type I_all: :func:`current_method()` or list of arrays
    
    :Example:
    
    >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, z)
    >>> InverseObject.solve_inverse()
    >>> gamma_k=InverseObject.gamma_k

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
        self.weight_value=True    #Are you going to use the weight function in the Jacobian matrix?
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
        """Function that solves the inverse problem.
        
        :Example:
        
        >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, z, select_pot_method=0)
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
        #Residue vector
        res_vec.append(self.norm_funcL2(b0,           'dOmega', r)/
                       self.norm_funcL2(self.list_U0, 'dOmega', r)*100)
        self.innerstep_vec.append(int(0)) #Save inner number steps
        mun_vec.append(0)                 #Save number steps
        
        "Print information"
        if self.mesh0 is not None and self.gamma0 is not None:
            error_vec.append(self.error_gamma())    
            print("Error (%)=", error_vec[0], "Residue (%)=", res_vec[0], " step:", 0, "Inner step: ", 0)
        else:
            print("Residue (%)=", res_vec[0], " step:", 0, "Inner step: ", 0)
            
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
            if self.mesh0 is not None and self.gamma0 is not None:
                print("Error (%)=", error_vec[self.steps], "Residue (%)=", res_vec[self.steps],
                      " step:", self.steps, "Inner step: ", inner_step)
            else: print("Residue (%)=", res_vec[self.steps],
                      " step:", self.steps, "Inner step: ", inner_step)
            ####################
            #Vectors to memory object.
            self.gamma_all=np.copy(gamma_all)
            self.res_vec=res_vec
            self.mun_vec=mun_vec
            self.error_vec=error_vec
            
        #############End-While############################
        return
   
        
    def solve_innerNewton(self, Jacobiana_all,b0):
        """Methods to solve inner step newton. Functions executed inside of :func:`solve_inverse()`. See set_InnerParameters() for more details.
            
    :param Jacobiana_all: Derivative Matrix generated by :func:`Jacobian_calc()`
    :type Jacobiana_all: Array ndim
    
    :returns:  (Array, int, float) -- Return a sk (Result of Inner Step to add in gamma_k), inner_step (Number of inner steps), mu (Regularization parameter used in the method).
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
        "------Total Variation------"
        if self.inner_method=='LMrrTV':
            if self.steps == 0:
                self.norm_U0=self.norm_funcL2(self.list_U0, 'dOmega')
                self.delta=self.noise_level*self.norm_U0 
                self.list_alpha=[]
                self.lin_res=[]
                self.list_ck, self.list_dk=[],[]
                self.list_ck2, self.list_dk2=[],[]
                self.grad_vec=[]
                self.intervalstep_vec=[]
                self.sk_save=[]
                self.hk_save=[]
                self.Jacob_save=[]

                #Save Section   
            
            #Basic Parameters Ranged Relaxed
            norm_U0 = self.norm_U0
            eta = self.LMrrTV_eta
            ph, phh = self.LMrrTV_ph, self.LMrrTV_phh 
            a1, a2 = self.LMrrTV_a1, self.LMrrTV_a2
            delta = self.delta
            alpha_k=self.LMrrTV_alpha
            phInner, phhInner = self.LMrrTV_phInner, self.LMrrTV_phhInner 

            #Duality Operator
            J_r = lambda f, r: np.abs(f)**(r-1)*np.sign(f)
            
            #Defining Parameters
            reduc=self.LMrrTV_reduc #0.1
            p_space=self.LMrrTV_p #1.1
            mu = self.LMrrTV_mu # 1 =Tikhonov, 0 = TV
            tol=self.LMrrTV_tolLm # 1E10
            beta=self.LMrrTV_beta # 1E10
            method=self.LMrrTV_method # newton
            grad_step=self.LMrrTV_grad_step #grad step
            
            #Preparation Vars.
            Lm = self.LMrrTV_L
            ADJLm = csr_matrix((self.LMrrTV_L).T)

            A=Jacobiana_all
            ADJdotA=ADJ@A
            I=np.identity(len(ADJdotA))
            
            #Interval definition
            ck = (1-ph)*((1+eta)*delta+eta*norm_b0)+ph*norm_b0
            dk = (1-phh)*((1+eta)*delta+eta*norm_b0)+phh*norm_b0

            #If you need otimize in a lower interval.
            #if phInner=1, then ck2=ck
            ck2=phInner*ck+(1-phInner)*dk 
            dk2=phhInner*ck+(1-phhInner)*dk


            
            #Iniatiate range-relaxed
            interval_step=0
            while (norm_res>=dk or norm_res<=ck):
                #Initiate minimization functional by newton method
                inner_step=0
                interval_step+=1
                
                grad_list=[]
                
                
                #First derivative
                xk= np.copy(self.Cellsgamma_k)
                x=np.copy(xk)
                residuo=-b0
                
                Lmdotx=Lm@x
                Lmdotxk=Lm@xk
                Ek_const=Lmdotxk/(np.sqrt((Lmdotxk)**2+beta))
                Ek = (Lmdotx/(np.sqrt((Lmdotx)**2+beta)))-Ek_const
                grad=alpha_k*ADJ@residuo+((1-mu)*(ADJLm@Ek)+mu*(x-xk))
                
                grad_norm0=self.norm_funcL2(grad, 'Omega')
                grad_norm=grad_norm0
                grad_list.append(grad_norm)
                
                print_num=1000
                
                @njit
                def Ek_calc(x, Lmdotx, beta, Ek_const):
                    return Lmdotx/(np.sqrt((Lmdotx)**2+beta))-Ek_const
                
                @njit
                def grad_calc(x,xk, A, b0, ADJ, residuo, ADJLm_Ek, mu, alpha_k):
                    sk=x-xk
                    residuo=-b0+A@sk                   
                    grad=alpha_k*ADJ@residuo+((1-mu)*(ADJLm_Ek)+mu*sk)
                    return grad, residuo, sk
                while grad_norm >=grad_norm0*reduc and inner_step<self.innerstep_limit:               
                    #if inner_step==0:
                    #    g = lambda xi: (1-mu)*np.sum( np.sqrt(((Lm@xi)**2+beta)))+1/2*mu*self.norm_funcL2(xi, 'Omega')**2
                    #    gradFxk = (1-mu)*(ADJLm@Ek_const)+mu*xk
                    # = np.dot( gradFxk, (x-xk)*self.cell_vec)

                    norm_res=self.norm_funcL2(residuo, 'dOmega')
                    omega=1/(grad_norm0)
                    #omega=(1/grad_norm)
                    x+=-grad_step*omega*grad
                    
                    Lmdotx=Lm@x
                    #Ek = Lmdotx/(np.sqrt((Lmdotx)**2+beta))-Ek_const
                    Ek=Ek_calc(x, Lmdotx, beta, Ek_const)
                    ADJLm_Ek=ADJLm@Ek
                    
                    #sk=x-xk
                    grad, residuo, sk=grad_calc(x,xk, A, b0, ADJ, residuo, ADJLm_Ek, mu, alpha_k)                 
                    #grad=alpha_k*ADJ@residuo+((1-mu)*(ADJLm@Ek)+mu*sk)
                    grad_norm=self.norm_funcL2(grad, 'Omega')
                    grad_list.append(grad_norm)
                    
                    #Prints
                    if inner_step%print_num==0:
                        print("grad0={:.4f}, grad={:.4f}, alpha={}".format(grad_norm0, grad_norm,  alpha_k))         
                    
                    inner_step+=1

                    #-----   
                    #Verify if is inside of the interval
                
                norm_res = self.norm_funcL2(residuo, 'dOmega')
                if self.LMrrTV_alpha_method=="constant":
                    break
                else:
                    if norm_res>=dk:
                        alpha_k=alpha_k*a2 #a2<1
                    elif norm_res<=ck: 
                        alpha_k=alpha_k*a1 #a1>1
                    print("ck={:.3f}, res={:.3f}, dk={:.3f}".format(ck, norm_res, dk))
            
            #Saving Data
            self.list_alpha.append(alpha_k)   
            self.list_ck.append(ck)  
            self.list_ck2.append(ck2)  
            self.list_dk.append(dk)  
            self.list_dk2.append(dk2)
            self.lin_res.append(norm_res)
            self.grad_vec.append(grad_list)
            self.innerstep_vec.append(inner_step)
            self.intervalstep_vec.append(interval_step)
            self.sk_save.append(sk)
            self.hk_save.append(A@sk)
            self.Jacob_save.append(A)
            

            #Correcting for the next step.
            if self.LMrrTV_alpha_method!="constant":
                if norm_res>=dk2:
                    alpha_k=alpha_k*a2 #a2<1
                elif norm_res<=ck2: 
                    alpha_k=alpha_k*a1 #a1>1
                
            #Att alpha
            self.LMrrTV_alpha=alpha_k        

        return sk, inner_step, mu
        
    def Jacobian_calc(self):
        """Calcuate derivative matrix. Function executed inside of :func:`solve_inverse()`.
        
        :returns:  (Array ndim) -- Return the derivative matrix.        
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
        """Determine the weights for Derivative Matrix and apply.
        
        :param Jacobiana: Derivative Matrix generated by :func:`Jacobian_calc()`
        :type Jacobiana: Array ndim
    
        :returns:  (Array ndim) -- Return the derivative matrix with weights.
        
        """
        p=self.Lp_space
        #norm(Jacobian_line)*1/vol_cell_k*1/gamma_cell_k
        self.weight=np.linalg.norm(Jacobiana, ord=p, axis=0)*(1/self.cell_vec)*(1/self.Cellsgamma_k)
        return self.weight
    
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
    
    def norm_funcL2(self, vector, domain, p=2):
        """This function is used in inner step of newton to calculate the norm's in L2 Omega and dOmega
         
        :param domain: 'Omega' or 'dOmega'.
        :type domain: str
        
        :returns:  (float) -- Return norm.
        
        """
        mesh=self.mesh
        weight=self.weight
        if domain=='Omega':
            norm=np.sum(np.abs(vector)**p*self.cell_vec*weight)
        elif domain=="dOmega":
            norm=np.sum(np.abs(vector)**p*self.size_elec_vec)

        norm=norm**(1/p)
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
        
        if self.steps<=1:
            cell_vec0=[]
            for cell in cells(self.mesh0): cell_vec0.append(cell.volume())
            self.cell_vec0=np.array(cell_vec0)        
        
        #A way to calculated the error in a different norm.
        if self.Lp_space!=2:
            #Error norm
            error_Lp=np.sum(np.abs((np.array(GammaElement1.vector())-np.array(GammaElement0.vector())))**self.Lp_space*self.cell_vec0 )
            error_Lp=error_Lp**(1/self.Lp_space)
            #Exact norm
            norm_gamma0_Lp=np.sum(np.abs((np.array(np.array(GammaElement0.vector())))**self.Lp_space*self.cell_vec0 ))
            norm_gamma0_Lp=norm_gamma0_Lp**(1/self.Lp_space)
            
            GammaError_Lp= error_Lp/norm_gamma0_Lp*100 #Percentual
        #print(GammaError_Lp)
        
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

         >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, z)
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

                >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
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

            >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
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

            >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
            >>> InverseObject.set_firstguess(np.ones(mesh_inverse.num_cells())*5)
            
        """
        self.firstguess=Cellsgamma_k           #First guess for Forwardproblem
        self.Cellsgamma_k=self.firstguess           
        self.gamma_k.vector()[:]=self.Cellsgamma_k #Guess in cell function """
        
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

            >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
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

            >>> InverseObject=InverseProblem(mesh_inverse, list_U0_noised, I_all, l, z)
            >>> InverseObject.set_InnerParameters(inner_method='ME', ME_reg=5E-4)i
            
        """
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])