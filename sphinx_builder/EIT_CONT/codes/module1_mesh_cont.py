import numpy as np
from mshr import*
from fenics import*

#If you wanna listening something I recommend: https://www.youtube.com/watch?v=lVFt__nrRxY
#Bachianas Brasileiras Nº 4 - Villa Lobos.

def MyMesh(r, n, n_vertex):
    """Function that generate the mesh based in number of vertex in boundary. It is extremely important that
    forward mesh and inverse mesh have vertex in commum, so this routine generate both, mesh_direct is a refiniment from mesh_inverse.
    
    :param r: Circle Radius
    :type r: float.
    :param n: Refinament parameter
    :type n: int.
    :param n_vertex: Vertices number in boundary
    :type n_vertex: int
    
    :return: :class:`dolfin.cpp.mesh.Mesh`

    :Example:
            >>> mesh_inverse, mesh_forward=MyMesh(r=1, n=8, n_vertex=100)
            
    .. image:: codes/mesh.png
      :scale: 75 %

    """
    ###Points Generator###
    points=[] #To save points
    for theta in np.linspace(0, 2*pi, n_vertex):
        a=[Point((cos(theta)*r,sin(theta)*r))]     #Defining Point object from Fenics.
        points=np.concatenate((points,a), axis=0)  #Grouping points.


    domain = Polygon(points)         #Function creates a polygon with the points.
    mesh = generate_mesh(domain,n)   #Get the polygon and generate the mesh

    mesh2 = refine(mesh)   #Get the mesh_inverse and refine it.
    mesh.radius=r  #Just add info. in the object.
    mesh2.radius=r #Just add info. in the object.

    return mesh, mesh2 #Mesh inverse and mesh forward.

def getBoundaryVertex(mesh,u):
    """ Functions that calculate the values of a function on boundary and return a array with that.
    
    :param mesh: Mesh where u is defined.
    :type mesh: :class:`dolfin.cpp.mesh.Mesh`
    :param u: Function that you want compute vertex values on the boundary.
    :type u: :class:`dolfin.cpp.mesh.Function`
    
    :return: array

    :Example:
    
    >>> u_boundary=getBoundaryVertex(mesh,u)
    
    """
    bmesh=BoundaryMesh(mesh, 'exterior', order=True) 
    indice_borda=bmesh.entity_map(0).array() #Get vertex indice (Boundary)
    
    u_bvertex=u.compute_vertex_values()[indice_borda] #Save all u(vertex).
    return u_bvertex

def getBoundaryVertexTwoMesh(mesh_inverse, mesh_forward, u, u0):#Fina, grossa, função.
    """ Functions that calculate the values of two function on the boundary and select the vertex in commum, then return an array with it.
    
    :param mesh_inverse: Coarsed mesh where Function u is defined.
    :type mesh_inverse: :class:`dolfin.cpp.mesh.Mesh`
    :param mesh_forward: Refined mesh where Function u0 is defined.
    :type mesh_forward: :class:`dolfin.cpp.mesh.Mesh`
    :param u: Function that you want compute vertex values on the boundary.`
    :type u: :class:`dolfin.cpp.mesh.Function`
    :param u0: Function that you want compute vertex values on the boundary.`
    :type u0: :class:`dolfin.cpp.mesh.Function`
    
    :return: (array) u_boundary, u0_boundary, vertex_index.

    :Example:
            >>> u_boundary, u0_boundary, vertex_index=getBoundaryVertexTwoMesh(mesh_inverse, mesh_direct, u, u0)
    """
    u_bvertex=[]    #Save u(vertex) mesh_inverse
    u_b2vertex=[]   #Save u(vertex) mesh_forward
    vertexnumber=[] #Save mesh index
    
  
    bmesh_inverse=BoundaryMesh(mesh_inverse, 'exterior', order=True)
    bmesh_forward=BoundaryMesh(mesh_forward, 'exterior', order=True)

    index_bond_inverse=bmesh_inverse.entity_map(0).array()
    index_bond_forward=bmesh_forward.entity_map(0).array()

    for ind_inv in index_bond_inverse:    #For each  index vertex in bmesh_inv.
        for ind_dir in index_bond_forward: #For each  index vertex in bmesh_forward.
            vertex1 = Vertex(mesh_inverse, ind_inv) #Map between index and vertex position (inverse).
            vertex2 = Vertex(mesh_forward, ind_dir) #Map between index and vertex position (forward).
            if vertex1.x(0)==vertex2.x(0) and vertex1.x(1)==vertex2.x(1): #If they have x_i=x_f and y_f=y_f.
                vertexnumber.append(ind_inv) #Save vertex number index for inverse.
                break

    for ind in vertexnumber:
        vertex = Vertex(mesh_inverse, ind) #Map between index and vertex position (inverse).
        u_bvertex.append(u(vertex.point())) #Append to ub_inverse
        u_b2vertex.append(u0(vertex.point())) #Append to ub_forward
                
    
    u_bvertex=np.array(u_bvertex)
    u_b2vertex=np.array(u_b2vertex)

    return u_bvertex, u_b2vertex, vertexnumber