###########################################################################################################################################################################################################################
# Unified Fenics code: 
# # Contact Umar Khayaz (ukhayaz3@gatech.edu), Aditya Kumar (akumar355@gatech.edu) for questions.
###########################################################################################################################################################################################################################

from dolfin import *
import numpy as np
import time
import sys
import csv
from datetime import datetime
#  Determine the start time for the analysis
startTime = datetime.now()


strength_surface_choice = int(sys.argv[1])            # 1 for DP, 2 for MC

if strength_surface_choice==1:
    folder_prefix = '_DP'
elif strength_surface_choice==2:
    folder_prefix = '_MC'
else:
    raise ValueError("Invalid choice for the strength surface. Choose 1 for Ducker Prager, or 2 for Mohr Columb criterion.")

#For DP: 1: sts sss, 2:sts sbs, 3:sts shs; For MC: 1: sts sss, 2: sts scs, 3: sss sbs
calibrationchoice = int(sys.argv[2]) 
I1_correction = 1 #1 if you want to correct for negative I1 in phase field SS
#1 if you want to use shs_DP in Kamarei Delta_eps, 2 if you want to use the corresponding shs
shs_delta_choice = int(sys.argv[3])


# Material properties
E, nu = 9800, 0.13      #Young's modulus and Poisson's ratio
Gc= 0.091125            #Critical energy release rate
sts, scs= 27, 77        #Tensile strength and compressive strength


lch=3*Gc*E/8/(sts**2)       #Irwin characteristic length
eps= float(sys.argv[4])     #The regularization length
h = float(sys.argv[5])

T = int(sys.argv[5])
Totalsteps= int(sys.argv[6])
loadfactor = float(sys.argv[7])                      #multiplier on load compared to expected critical stress
#Critical value of phase field parameter to break code


z_penalty_factor = float(sys.argv[8])                #multiplier to enforce strictness of z irreversibility
eta=float(sys.argv[9])                               #non-zero minimum numerical stiffness coefficient
stag_iter_max = int(sys.argv[10])

linear_solver_for_u = int(sys.argv[11])
non_linear_solver_choice =  int(sys.argv[12])        # solver choice for non linear solver inside snes
paraview_at_step = int(sys.argv[13])
disp = int(sys.argv[14])                             # maximum displacement # In case Mode II, Mode III, 
problem_dim = int(sys.argv[15])                      # Change to 3 for a 3D problem

# Problem description
comm = MPI.comm_world 
comm_rank = MPI.rank(comm)

#Geometry of the single edge notch geometry
ac = 0.01                                             #notch length
W, L = 100, 150                                       #making use of symmetry, the geometry is 6x40, see last example in JMPS 2020
Lz = 1
CrackZ = 10


# ___________________________________________________________________________________________________________________________________________________________


# Criterion choice for phase_model == 2
vonMises_or_Rankine = int(sys.argv[16])  # 1 for von Mises, 2 for Rankine

if vonMises_or_Rankine == 1:
    folder_prefix += '_vonMises'
elif vonMises_or_Rankine == 2:
    folder_prefix += 'Rankine'
else: 
    raise ValueError("Invalid criterion choice. Choose 1 for vonMises criterion, or 2 for Rankine criterion.")


# Model choices
phase_model = int(sys.argv[17])  # 1, 2, 3, or 4

if phase_model == 1:
    folder_prefix += '_Nucleation'
elif phase_model == 2:
    folder_prefix += '_PF_CZM'
elif phase_model == 3:
    folder_prefix += '_Variational'
elif phase_model == 4:
    folder_prefix += '_Miehe_2015'
else:
    raise ValueError("Invalid Phase Field Model choice. Choose 1 for Nucleation, 2 for PF_CZM, 3 for Variational, or 4 for Miehe_2015")

variational_model = int(sys.argv[18])  # 1 for Star Convex, 2 for Vol-Dev
stress_state_choice = int(sys.argv[19])  # 1 for Plane Stress, 2 for Plane Strain, 3 for 3D

# Choose Elasticity Params if Plain Stress, Plain Strain, or 3D
if stress_state_choice == 1:
    folder_prefix += '_PlaneStress'
elif stress_state_choice == 2:
    folder_prefix += '_PlaneStrain'
elif stress_state_choice == 3:
    folder_prefix += '_3D'
else:
    raise ValueError("Invalid stress state choice. Choose 1 for Plane Stress, 2 for Plane Strain, or 3 for 3D.")


# Read boundary condition choice
markBC_choice = int(sys.argv[20])  # 1 for Surfing, 2 for Mode_I, etc.

# Read mesh choice and notch angle choice
mesh_choice = int(sys.argv[21])  # 1 for Surfing, 2 for Mode_I, etc.
notch_angle_choice = int(sys.argv[22])  # 1 for 45 degree, 2 for 30 degree   # only for 4-P bending test

# Read the DirichletBC_choice argument
DirichletBC_choice = int(sys.argv[23])

# Read the problem type choice argument
problem_type = int(sys.argv[24])

# ___________________________________________________________________________________________________________________________________________________________
# Initialize an empty mesh object
mesh = Mesh()

meshname = "crack.xdmf"
facet_meshname = "facet_crack.xdmf"

# Read the .xdmf  file data into mesh object
with XDMFFile(meshname) as infile:
    infile.read(mesh)
    
# Extract initial mesh coords
x = SpatialCoordinate(mesh)


# Read the subdomain data stored in the *.xdmf file
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(meshname) as infile:
    infile.read(mvc, "name_to_read")

mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Read mesh boundary facets.
# Specify parameters of 1-D mesh function object
mvc_1d = MeshValueCollection("size_t", mesh, 1)

# Read the   *.xdmf  data into mesh function object
with XDMFFile(facet_meshname) as infile:
    infile.read(mvc_1d, "name_to_read")

# Store boundary facets data
facets = cpp.mesh.MeshFunctionSizet(mesh, mvc_1d)

ds = Measure('ds', domain=mesh, subdomain_data=facets)
# ___________________________________________________________________________________________________________________________________________________________


if mesh_choice == 4:                                 # Mode III
    mesh = Mesh("crack.xml")
    domain2 = CompiledSubDomain("x[0]>12 && x[0]<19 && x[1]<(16) && x[1]>(9)")
    for _ in range(5):
        d_markers = MeshFunction("bool", mesh, 2, False)
        domain2.mark(d_markers, True)
        mesh = refine(mesh, d_markers, True)

elif mesh_choice == 7:                                 # 4-Point Bending
    mesh = Mesh("crack.xml")
    if notch_angle_choice == 1:                                                                         # 45-degree notch
        domain2 = CompiledSubDomain("x[0]>x[2]-1 && x[0]<x[2]+1 && x[1]<8 && x[1]>5.5")
    elif notch_angle_choice == 2:                                                                        # 30-degree notch
        domain2 = CompiledSubDomain("x[0]>x[2]/sqrt(3)-1 && x[0]<x[2]/sqrt(3)+1 && x[1]<8 && x[1]>5.5")
    
    for _ in range(6):
        d_markers = MeshFunction("bool", mesh, 2, False)
        domain2.mark(d_markers, True)
        mesh = refine(mesh, d_markers, True)


n = FacetNormal(mesh)
m = as_vector([n[1], -n[0]])
# ___________________________________________________________________________________________________________________________________________________________

righttop = CompiledSubDomain("abs(x[0]-W)<1e-4 && abs(x[1]-L/2)<1e-4")
cracktip = CompiledSubDomain("abs(x[1]-0.0)<1e-4 && x[0]<ac+h && x[0]>ac-2*eps", ac=ac, h=h, eps=eps)
outer = CompiledSubDomain("x[1]>L/10", L=L)
leftbot = CompiledSubDomain("abs(x[1]-side)<1e-4 && abs(x[0]-side)<1e-4", side=-W)



if markBC_choice == 7:                                # 4-Point Bending
    if notch_angle_choice == 45:
        left = CompiledSubDomain("abs(x[0]-side)<1e-4", side=-W/2, tol=1e-4)
        right = CompiledSubDomain("abs(x[0]-side)<1e-4", side=W/2, tol=1e-4)
        Pleft = CompiledSubDomain("x[0]>(-20.25) && x[0]<(-20.0) && abs(x[1]-L)<1e-4", L=L)
        Pright = CompiledSubDomain("x[0]>(20) && x[0]<(20.25) && abs(x[1]-L)<1e-4", L=L)
        Rleft = CompiledSubDomain("x[0]>(-36.125) && x[0]<(-35.875) && abs(x[1]-L)<1e-4", L=0)
        Rright = CompiledSubDomain("x[0]>(35.875) && x[0]<(36.125) && abs(x[1]-L)<1e-4", L=0)
        outer = CompiledSubDomain("abs(x[0])>7", W=W, L=L, Lz=Lz)
    
    elif notch_angle_choice == 30:
        left = CompiledSubDomain("abs(x[0]-side)<1e-4", side=-W/2, tol=1e-4)
        right = CompiledSubDomain("abs(x[0]-side)<1e-4", side=W/2, tol=1e-4)
        Pleft = CompiledSubDomain("x[0]>(-20.25) && x[0]<(-20.0) && abs(x[1]-Ly)<1e-4", L=L)
        Pright = CompiledSubDomain("x[0]>(20) && x[0]<(20.25) && abs(x[1]-Ly)<1e-4", L=L)
        Rleft = CompiledSubDomain("x[0]>(-36.125) && x[0]<(-35.875) && abs(x[1]-Ly)<1e-4", L=0)
        Rright = CompiledSubDomain("x[0]>(35.875) && x[0]<(36.125) && abs(x[1]-Ly)<1e-4", L=0)
        outer = CompiledSubDomain("abs(x[0])>7", W=W, L=L, Lz=Lz)



# ___________________________________________________________________________________________________________________________________________________________


#Don't need to change anything from here on.
########################################################

set_log_level(40)  #Error level=40, warning level=30
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
            "eliminate_zeros": True, \
            "precompute_basis_const": True, \
            "precompute_ip_const": True}


# ___________________________________________________________________________________________________________________________________________________________
# Define function space
V = VectorFunctionSpace(mesh, "CG", 1)   #Function space for u
Y = FunctionSpace(mesh, "CG", 1)         #Function space for z



if DirichletBC_choice == 1:                                   # Surfing

    c = Expression("K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*cos(atan2(x[1],(x[0]-V*(t+0.1)))/2)", degree=4, t=0, V=20, K1=30, mu=4336.28, kap=2.54)
    r = Expression("K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*sin(atan2(x[1],(x[0]-V*(t+0.1)))/2)", degree=4, t=0, V=20, K1=30, mu=4336.28, kap=2.54)
    c0 = Expression("(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*cos(atan2(x[1],(x[0]-V*(t+0.1)))/2))-(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(tau+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(tau+0.1)))))*cos(atan2(x[1],(x[0]-V*(tau+0.1)))/2))", degree=4, t=0, tau=0, V=20, K1=30, mu=4336.28, kap=2.54)
    r0 = Expression("(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*sin(atan2(x[1],(x[0]-V*(t+0.1)))/2))-(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(tau+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(tau+0.1)))))*sin(atan2(x[1],(x[0]-V*(tau+0.1)))/2))", degree=4, t=0, tau=0, V=20, K1=30, mu=4336.28, kap=2.54)

    # Boundary conditions setup
    bc_rt= DirichletBC(V.sub(0), Constant(0.0), righttop, method='pointwise')
    bc_bot = DirichletBC(V.sub(1), r, facets, 22)
    bc_top = DirichletBC(V.sub(1), r, facets, 21)
    bcs = [bc_rt, bc_bot, bc_top]

    cz=Constant(1.0)
    bcb_z = DirichletBC(Y, cz, facets, 22)
    bct_z = DirichletBC(Y, cz, facets, 21)
    cz2=Constant(0.0)
    bcc_z = DirichletBC(Y, cz2, facets, 25)
    bcs_z=[bcb_z, bct_z]

    bcb_du=DirichletBC(V.sub(1),Constant(0.0), facets, 22)
    bct_du=DirichletBC(V.sub(1),Constant(0.0),facets, 21)
    bcs_du = [bc_rt, bcb_du, bct_du]

elif DirichletBC_choice == 2:                                 # Mode I
    c = Expression("t*0.0", degree=1, t=0)
    bc_rt = DirichletBC(V.sub(0), Constant(0.0), righttop, method='pointwise')
    bc_bot = DirichletBC(V.sub(1), c, facets, 22)
    bcs = [bc_rt, bc_bot]
    
    cz = Constant(1.0)
    bcb_z = DirichletBC(Y, cz, facets, 22)
    bct_z = DirichletBC(Y, cz, facets, 21)
    cz2=Constant(0.0)
    bct_z2 = DirichletBC(Y, cz2, cracktip)
    bcs_z=[bcb_z, bct_z]

    # Define Neumann boundary conditions
    sigma_critical_crack = sqrt(E*Gc/np.pi/ac)/((0.752+2.02*(ac/W)+0.37*(1-np.sin(np.pi*ac/2/W))**3)*(sqrt(2*W/np.pi/ac*np.tan(np.pi*ac/2/W)))/(np.cos(np.pi*ac/2/W)))
    if sigma_critical_crack>sts:
        sigma_critical = sts
    else:
        sigma_critical = sigma_critical_crack

    sigma_external = loadfactor*sigma_critical

    Tf = Expression(("t*0.0", "t*sigma"), degree=1, t=0, sigma=sigma_external)
    

elif DirichletBC_choice == 3:                                 # Mode II
    c = Expression(("d_u * t", "t*0.0"), degree=1, t=0, d_u=disp)

    bc_bot = DirichletBC(V, Constant(0.0,0.0), facets, 22)
    bc_top = DirichletBC(V, c, facets, 21)
    bcs = [bc_bot, bc_top]

    cz = Constant(1.0)
    bcb_z = DirichletBC(Y, cz, facets, 22)
    bct_z = DirichletBC(Y, cz, facets, 21)
    cz2=Constant(0.0)
    bct_z2 = DirichletBC(Y, cz2, cracktip)
    bcs_z=[bcb_z, bct_z]

elif DirichletBC_choice == 4:                                 # Mode III

    r = Expression(("t*0.0", "t*0.0", "t*disp"), degree=1, t=0, disp=disp)
    r0 = Expression(("t*0.0", "t*0.0", "(t-tau)*disp"), degree=1, t=0, tau=0, disp=disp)
    c = Expression(("t*0.0", "t*0.0", "t*0.0"), degree=1, t=0)

    bc_top = DirichletBC(V, r, facets, 21)
    bc_bot = DirichletBC(V, c, facets, 22)
    bcs = [bc_bot, bc_top]

    bc_top0 = DirichletBC(V.sub(2), r0, facets, 21)
    bcs_du0 = [bc_top0, bc_bot]

    bc_top1 = DirichletBC(V, c, facets, 21)
    bcs_du = [bc_top1, bc_bot]

    cz = Constant(1.0)
    cz2 = Constant(0.0)
    bct_z = DirichletBC(Y, cz, outer)
    bct_z2 = DirichletBC(Y, cz2, cracktip)
    bcs_z = [bct_z]

elif DirichletBC_choice == 5:                                 # Biaxial
    c = Expression("t*0.0", degree=1, t=0)

    bc_left = DirichletBC(V.sub(0), c, facets, 23)
    bc_bot = DirichletBC(V.sub(1), c, facets, 22)
    bcs = [bc_left, bc_bot]

    cz = Constant(1.0)
    bct_z = DirichletBC(Y, cz, outer)
    cz2 = Constant(0.0)
    bct_z2 = DirichletBC(Y, cz2, cracktip)
    bcs_z = []

    sigma_external = 1.05 * sqrt(E * Gc / np.pi / ac)
    Tf = Expression("t*sigma", degree=1, t=0, sigma=sigma_external)

elif DirichletBC_choice == 6:                                                             # Pure Shear
    c = Expression(("t*0.0", "t*0.0"), degree=1, t=0)
    bc_lb = DirichletBC(V, c, leftbot, method='pointwise')
    bcs = [bc_lb]

    cz = Constant(1.0)
    bct_z = DirichletBC(Y, cz, outer)
    cz2 = Constant(0.0)
    bct_z2 = DirichletBC(Y, cz2, cracktip)
    bcs_z = []

    sigma_external = 1.1 * sqrt(E * Gc / np.pi / ac)
    Tf = Expression("t*sigma", degree=1, t=0, sigma=sigma_external)

elif DirichletBC_choice == 7:                                 # 4-P Bending

    cr=Expression(("t*0.0", "t*0.0", "t*0.0"),degree=1,t=0)
    crr=Expression("t*0.0",degree=1,t=0)
    cl=Expression("-du*t", degree=1, t=0, du=disp)

    bcrl = DirichletBC(V, cr, Rleft)                     # Left Reaction Restraint
    bcrry = DirichletBC(V.sub(1), crr, Rright)           # Right Reaction Restraint
    bcrrz = DirichletBC(V.sub(2), crr, Rright)           # Right Reaction Restraint
    bcll = DirichletBC(V.sub(1), cl, Pleft)              # Left Loading Point
    bclr = DirichletBC(V.sub(1), cl, Pright)             # Right Loading Point
    bcs = [bcrl, bcrry, bcrrz, bcll, bclr]

    cz=Constant(1.0)
    bct_z = DirichletBC(Y, cz, outer)
    cz2=Constant(0.0)
    bct_z2 = DirichletBC(Y, cz2, cracktip)
    bcs_z=[bct_z]   


# ___________________________________________________________________________________________________________________________________________________________


# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
u_inc = Function(V)
dz = TrialFunction(Y)            # Incremental phase field
y  = TestFunction(Y)             # Test function
z  = Function(Y)                 # Phase field from previous iteration
z_inc = Function(Y)
d = u.geometric_dimension()


# ___________________________________________________________________________________________________________________________________________________________


# Common initialization for displacement field (u) and phase field (z)
# Set problem dimension: 2 for 2D, 3 for 3D


# Set the dimension-specific constant for displacement initialization.
u_dim = (0.0, 0.0) if problem_dim == 2 else (0.0, 0.0, 0.0)

# ---------------------------
# Common initialization for displacement (u) and phase field (z)
# ---------------------------
u_init = Constant(u_dim)
u.interpolate(u_init)
for bc in bcs:
    bc.apply(u.vector())

z_init = Constant(1.0)
z.interpolate(z_init)
for bc in bcs_z:
    bc.apply(z.vector())

z_ub = Function(Y)
z_ub.interpolate(Constant(1.0))
z_lb = Function(Y)
z_lb.interpolate(Constant(-0.0))

u_prev = Function(V)
assign(u_prev, u)
z_prev = Function(Y)
assign(z_prev, z)

# ---------------------------
# Helper function: Extract DOFs on boundary
# ---------------------------
def extract_dofs_boundary(V, facets, bsubd):
    # Use a constant vector of ones whose length depends on the problem dimension
    bc_val = Constant((1, 1)) if problem_dim == 2 else Constant((1, 1, 1))
    # For 3D, use method='pointwise'; for 2D the default is sufficient.
    label_bc = DirichletBC(V, bc_val, facets, bsubd)
    label = Function(V)
    label_bc.apply(label.vector())
    return np.where(label.vector() == 1)[0]

# ---------------------------
# Helper function: Evaluate a field at a given point
# ---------------------------
def evaluate_function(u, x):
    comm = u.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return u(*x)
    cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
    u_eval = u(*x) if distance < DOLFIN_EPS else None
    comm = mesh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)
    if comm.rank == 0:
        global_u_evals = np.array([y for y in computed_u if y is not None], dtype=np.double)
        # Ensure consistency across processes.
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)
        computed_u = global_u_evals[0]
    else:
        computed_u = None
    computed_u = comm.bcast(computed_u, root=0)
    return computed_u





# Condition-specific code
if problem_type == 1:                                                  # Surfing problem
    top_dofs = extract_dofs_boundary(V, facets, 21)
    y_dofs_top = top_dofs[1::d]

elif problem_type == 2:                                                # Mode I
    top_dofs = extract_dofs_boundary(V, facets, 21)
    y_dofs_top = top_dofs[1::d]

elif problem_type == 3:                                                # Mode II
    top_dofs = extract_dofs_boundary(V, facets, 21)
    y_dofs_top = top_dofs[0::d]

elif problem_type == 4:                                                # Mode III
    top_dofs = extract_dofs_boundary(V, facets, 21)
    y_dofs_top = top_dofs[2::d]

elif problem_type == 5:                                                # Biaxial
    top_dofs = extract_dofs_boundary(V, facets, 24)
    y_dofs_top = top_dofs[1::d]

elif problem_type == 6:                                                # Pure Shear
    top_dofs = extract_dofs_boundary(V, facets, 21)
    y_dofs_top = top_dofs[0::d]

elif problem_type == 7:                                                 # 4-P Bending
    Rleft_dofs = extract_dofs_boundary(V, Rleft)
    y_dofs_Rleft = Rleft_dofs[1::d]

    Rright_dofs = extract_dofs_boundary(V, Rright)
    y_dofs_Rright = Rright_dofs[1::d]

    Pleft_dofs = extract_dofs_boundary(V, Pleft)
    y_dofs_Pleft = Pleft_dofs[1::d]

    Pright_dofs = extract_dofs_boundary(V, Pright)
    y_dofs_Pright = Pright_dofs[1::d]


# ___________________________________________________________________________________________________________________________________________________________


# Elasticity parameters
mu, lmbda, kappa = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu))), Constant(E/(3*(1 - 2*nu))) 

def energy(v, mode="plane_stress"):
    """Computes the strain energy density based on mode."""
    eps = sym(grad(v))
    if mode == "plane_stress":
        return mu * (inner(eps, eps) + ((nu / (1 - nu))**2) * (tr(eps))**2) + 0.5 * lmbda * (tr(eps) * (1 - 2 * nu) / (1 - nu))**2
    elif mode == "plane_strain":
        return mu * inner(eps, eps) + 0.5 * lmbda * (tr(eps))**2
    elif mode == "3D":
        return mu * inner(eps, eps) + 0.5 * lmbda * (tr(eps))**2
    else:
        raise ValueError("Invalid mode. Choose 'plane_stress', 'plane_strain', or '3D'.")

def epsilon(v):
    """Computes the symmetric strain tensor."""
    return sym(grad(v))

def epsilon_dev(v, mode="plane_stress"):
    """Computes the deviatoric strain tensor."""
    eps = epsilon(v)
    if mode == "plane_stress":
        return eps - (1 / 3) * tr(eps) * (1 - 2 * nu) / (1 - nu) * Identity(len(v))
    else:
        return eps - (1 / 3) * tr(eps) * Identity(len(v))

def sigma(v, mode="plane_stress"):
    """Computes the stress tensor."""
    eps = sym(grad(v))
    if mode == "plane_stress":
        return 2.0 * mu * eps + lmbda * tr(eps) * (1 - 2 * nu) / (1 - nu) * Identity(len(v))
    elif mode in ["plane_strain", "3D"]:
        return 2.0 * mu * eps + lmbda * tr(eps) * Identity(len(v))
    else:
        raise ValueError("Invalid mode. Choose 'plane_stress', 'plane_strain', or '3D'.")

def sigmavm(sig, v, mode="plane_stress"):
    """Computes the von Mises stress."""
    dev_stress = sig - (1 / 3) * tr(sig) * Identity(len(v))
    if mode == "plane_stress":
        return sqrt(0.5 * (inner(dev_stress, dev_stress) + (1 / 9) * tr(sig)**2))
    elif mode == "plane_strain":
        return sqrt(0.5 * (inner(dev_stress, dev_stress) + ((2 * nu / 3 - 1 / 3)**2) * tr(sig)**2))
    elif mode == "3D":
        return sqrt(0.5 * inner(dev_stress, dev_stress))
    else:
        raise ValueError("Invalid mode. Choose 'plane_stress', 'plane_strain', or '3D'.")



def safe_sqrt(x):
    # A safe square root that avoids negative arguments
    return sqrt(x + 1.0e-16)

def compute_eigenvalues(A, dim=None):
    
    if dim == 2:
        # 2D case: closed-form solution for a 2x2 tensor
        eig1 = (tr(A) + sqrt(abs(tr(A)**2 - 4 * det(A)))) / 2
        eig2 = (tr(A) - sqrt(abs(tr(A)**2 - 4 * det(A)))) / 2 
        return eig1, eig2

    elif dim == 3:
        # 3D case: analytical solution based on invariants.
        Id = Identity(3)
        I1 = tr(A)
        I2 = (tr(A)**2 - tr(A*A)) / 2.0  # second invariant (not used here)
        I3 = det(A)                     # third invariant
        
        # Parameters for eigenvalue computation
        d_par = I1 / 3.0
        e_par = safe_sqrt(tr((A - d_par*Id)*(A - d_par*Id)) / 6.0)
        
        # Define f_par carefully to avoid division by zero
        zero = 0 * Id
        f_par_expr = (1.0 / e_par) * (A - d_par * Id)
        f_par = conditional(eq(e_par, 0), zero, f_par_expr)
        
        # Compute the argument of the acos, and bound it to [-1, 1]
        g_par0 = det(f_par) / 2.0
        tol = 3.e-16  # tolerance to avoid acos issues at the boundaries
        g_par1 = conditional(ge(g_par0, 1.0 - tol), 1.0 - tol, g_par0)
        g_par = conditional(le(g_par1, -1.0 + tol), -1.0 + tol, g_par1)
        
        # Angle for the eigenvalue formulas
        h_par = acos(g_par) / 3.0
        
        # Compute the eigenvalues (ordered such that eig1 >= eig2 >= eig3)
        eig3 = d_par + 2.0 * e_par * cos(h_par + 2.0 * np.pi / 3.0)
        eig2 = d_par + 2.0 * e_par * cos(h_par + 4.0 * np.pi / 3.0)
        eig1 = d_par + 2.0 * e_par * cos(h_par + 6.0 * np.pi / 3.0)
        return eig1, eig2, eig3

    else:
        raise ValueError("Dimension not supported. Only 2D and 3D cases are implemented.")




# ___________________________________________________________________________________________________________________________________________________________

'''
# Choose Elasticity Params if Plain Stress, Plain Strain, or 3D
if stress_state_choice == 1:
    folder_prefix += '_PlaneStress'
elif stress_state_choice == 2:
    folder_prefix += '_PlaneStrain'
elif stress_state_choice == 3:
    folder_prefix += '_3D'
else:
    raise ValueError("Invalid stress state choice. Choose 1 for Plane Stress, 2 for Plane Strain, or 3 for 3D.") 
    
'''


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 1. Kumar JMPS2020        
# ---------------------------------------------------------------------------------------------------------------------------------------------
if phase_model == 1:
    
    shs_DP = (2*scs*sts)/(3*(scs-sts))
    shs_MC = (scs*sts)/(scs-sts)
    
    if(shs_delta_choice==2 and strength_surface_choice==2):
        shs_delta = shs_MC        
    else:
        shs_delta = shs_DP    
    
    Wts = 0.5*(sts**2)/E
    if strength_surface_choice == 1:
        delta = (1+3*h/(8*eps))**(-2) * ((sts+(1+2*sqrt(3))*shs_delta)/((8+3*sqrt(3))*shs_delta)) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1) * (2/5)
    elif strength_surface_choice == 2:
        delta = (1+3*h/(8*eps))**(-2) * (5/(3+8*sqrt(3))) * 3*Gc/(16*Wts*eps) + (1+3*h/(8*eps))**(-1) * (9/(8*sqrt(3)))
    
    alpha_tc = sts/scs
    alpha_ct = 1/alpha_tc
    Wcs = (alpha_ct**2)*Wts
    
    omega_eps = (3*Gc*delta)/8/eps
    
    #Chockalingam JAM 2025
    if(strength_surface_choice==1):
        shs = (2*scs*sts)/(3*(scs-sts))
        Whs = 0.5*(shs**2)/kappa
        alpha_th = sts/shs
        alpha_tb = 1.5 - 0.5*alpha_tc 
        sbs = sts/alpha_tb
        alpha_bt = 1/alpha_tb
        alpha_ts = 0.5*sqrt(3)*(1+alpha_tc)
        alpha_st = 1/alpha_ts
        sss = sts/alpha_ts
        Wbs = ((alpha_bt**2)/6)*(sts**2)*(1/mu + 4/3/kappa)
        Wss = 0.5*(sts**2.0)*((alpha_st**2.0)/mu)
        if(calibrationchoice==1): #sts sss
            beta1_eps = (1-alpha_ts/sqrt(3)) - (2/omega_eps)*(Wts - alpha_ts*Wss/sqrt(3))
            beta2_eps = alpha_ts*(1-2*Wss/omega_eps)
            fname_prefix = 'DP_sts_sss'
        elif(calibrationchoice==2): #sts sbs
            beta1_eps = (alpha_tb-1) - (2/omega_eps)*(alpha_tb*Wbs - Wts)
            beta2_eps = sqrt(3)*(2 - alpha_tb) -(2*sqrt(3)/omega_eps)*(2*Wts - alpha_tb*Wbs)
            fname_prefix = 'DP_sts_sbs'
        elif(calibrationchoice==3): #sts shs
            beta1_eps = alpha_th/3 - (2*alpha_th/3)*(Whs/omega_eps)
            beta2_eps = sqrt(3) - alpha_th/sqrt(3) - 2*sqrt(3)*(Wts/omega_eps) + (2*alpha_th/sqrt(3))*(Whs/omega_eps)
            fname_prefix = 'DP_sts_shs'        
    elif(strength_surface_choice==2):     
        alpha_bt = 1
        Wbs = ((alpha_bt**2)/6)*(sts**2)*(1/mu + 4/3/kappa)
        alpha_ts = 1 + alpha_tc
        alpha_st = 1/alpha_ts
        Wss = 0.5*(sts**2.0)*((alpha_st**2.0)/mu)
        if(calibrationchoice==1): #sts sss     
            beta1_eps = 1 - 2*(Wts/omega_eps)
            beta2_eps = (1 - alpha_ts) + alpha_ts*(2*Wss/omega_eps) - 2*(Wts/omega_eps)
            fname_prefix = 'MC_sts_sss'
        elif(calibrationchoice==2): #sts scs                
            beta1_eps = 1 - 2*(Wts/omega_eps)
            beta2_eps = -alpha_tc*(1- 2*(Wcs/omega_eps))
            fname_prefix = 'MC_sts_scs'
        elif(calibrationchoice==3): #sss sbs
            beta1_eps = 1 - 2*(Wbs/omega_eps)
            beta2_eps = (1 - alpha_ts) + alpha_ts*(2*Wss/omega_eps) - 2*(Wbs/omega_eps)
            fname_prefix = 'MC_sss_sbs'

    beta1 =  -beta1_eps*(omega_eps)/sts
    beta2  = -beta2_eps*(omega_eps)/sts

    pen=1000*(3*Gc/8/eps) * conditional(lt(delta, 1), 1, delta)
    
    # Stored strain energy density (compressible L-P model)
    psi1 =(z**2+eta)*(energy(u))    
    psi11=energy(u)
    stress=(z**2+eta)*sigma(u)
    
    # Total potential energy
    if problem_type == 1 or 3 or 4 or 7:
        Pi = psi1*dx

    if problem_type == 2:
        Pi = psi1*dx - dot(Tf, u)*ds(21)

    elif problem_type == 5:
        Pi = psi1*dx - dot(Tf*n, u)*ds(24)

    elif problem_type == 6: 
        Pi = psi1*dx - (dot(-Tf*m, u)*ds(23) + dot(-Tf*m, u)*ds(24) + dot(Tf*m, u)*ds(21) + dot(Tf*m, u)*ds(22))
    

    I1_d = (z**2)*tr(sigma(u))
    if strength_surface_choice==1:
        I1_d = (z**2)*tr(sigma(u))
        SQJ2_d = (z**2)*sigmavm(sigma(u),u)
        ce = beta2*SQJ2_d + beta1*I1_d

    elif strength_surface_choice == 2:
        # Determine the tensor dimension from σ(u)
        dim = sigma(u).ufl_shape()[0]
        
        if stress_state_choice == 1:                                                                # Plain Stress
            # For a 2D problem: compute the two eigenvalues.
            eig1_val, eig2_val = compute_eigenvalues(sigma(u), dim=2)
            sigma_p1 = eig1_val
            sigma_p2 = eig2_val

        if stress_state_choice == 2:                                                                # Plain Stress
            # For a 2D problem: compute the two eigenvalues.
            eig1_val, eig2_val = compute_eigenvalues(sigma(u), dim=2)
            eig3_val = nu * tr(sigma(u))
            sigma_p1 = max(eig1_val, eig2_val, eig3_val)
            sigma_p2 = min(eig1_val, eig2_val, eig3_val)

        elif stress_state_choice == 3:                                                              # 3D 
            # For a 3D problem: compute the three eigenvalues.
            eig1_val, eig2_val, eig3_val = compute_eigenvalues(sigma(u), dim=3)
            sigma_p1 = eig1_val   # largest eigenvalue (tensile candidate)
            sigma_p2 = eig3_val   # smallest eigenvalue (compressive candidate)
        
        # Only consider positive part for tensile and negative part for compressive response.
        sigma_max_d = (z**2) * conditional(gt(sigma_p1, 0), sigma_p1, 0)
        sigma_min_d = (z**2) * conditional(lt(sigma_p2, 0), sigma_p2, 0)
        
        # Combine contributions with the calibration coefficients.
        ce = beta1 * sigma_max_d + beta2 * sigma_min_d





    if I1_correction==1:
        ce = ce + z*(1-sqrt(I1_d**2)/I1_d)*psi11  

    
    # Compute first variation of Pi (directional derivative about u in the direction of v)
    R = derivative(Pi, u, v)

    # Compute Jacobian of R
    Jac = derivative(R, u, du)

    #To use later for memory allocation for these tensors
    A=PETScMatrix()
    b=PETScVector()

    #Balance of configurational forces PDE
    Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
    Wv2=conditional(le(z, 0.05), 1, 0)*z_penalty_factor*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
    R_z = y*2*z*(psi11)*dx - y*(ce)*dx + 3*delta*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) +  derivative(Wv2,z,y)   # Complete Model
    
    # Compute Jacobian of R_z
    Jac_z = derivative(R_z, z, dz)



# -----------------------------------------------------------------------------
# 2. PF_CZM 2017 - A Unified Phase Field Theory (Dr. Y. Wu)
# -----------------------------------------------------------------------------

elif phase_model == 2:

    pen = 1000 * Gc / eps
    
    pho_c = scs / sts
    a1 = 4 * E * Gc / (pi * eps * sts**2)
    a2, a3 = -0.5, 0.0
    
    w_z = z**2 / (z**2 + a1 * (1 - z) * (1 + a2 * (1 - z) * (1 + a3 * (1 - z))))
    stress = (w_z + eta) * sigma(u)
    
    J2 = sigmavm(sigma(u), u)**2
    
    # Condition to choose between von Mises and Rankine criterion
    if vonMises_or_Rankine == 1:
        # von Mises criterion (Wu 2020)
        sigma_eq = (1 / (2 * pho_c)) * ((pho_c - 1) * tr(sigma(u)) + sqrt((pho_c - 1) ** 2 * (tr(sigma(u))) ** 2 + 12 * pho_c * J2))
        folder_prefix += '_vonMises'
    elif vonMises_or_Rankine == 2:
        # Rankine criterion
        dim = sigma(u).ufl_shape()[0]
        if stress_state_choice == 1:                                                                                     # Plain Stress
            # For a 2D problem: compute the two eigenvalues and take the largest.
            eig1_val, _ = compute_eigenvalues(sigma(u), dim=2)
            sigma_eq = (eig1_val + abs(eig1_val)) / 2.0

        if stress_state_choice == 2:                                                                                # Plain Strain case
            # For a 2D problem: compute the two eigenvalues and take the largest.
            eig1_val, _ = compute_eigenvalues(sigma(u), dim=2)
            eig3_val = nu * tr(sigma(u))
            eig_max = max(eig1_val, eig3_val)
            sigma_eq = (eig_max + abs(eig_max)) / 2.0

        elif stress_state_choice == 3:                                                                                   # 3D
            # For a 3D problem: compute the three eigenvalues and take the largest.
            eig1_val, eig2_val, eig3_val = compute_eigenvalues(sigma(u), dim=3)
            sigma_eq = (eig1_val + abs(eig1_val)) / 2.0
        folder_prefix += '_Rankine'

    
    y_wu = sigma_eq**2 / (2 * E)
    Y_wu = -(a1 * z * (2 + 2 * a2 * (1 - z) - z)) / (z**2 + a1 * (1 - z) + a1 * a2 * (1 - z)**2)**2 * y_wu
    
    # Stored strain energy density
    psi1 = (w_z + eta) * energy(u)
    psi11 = energy(u)
    
     # Total potential energy
    if problem_type == 1 or 3 or 4 or 7:
        Pi = psi1*dx

    if problem_type == 2:
        Pi = psi1*dx - dot(Tf, u)*ds(21)

    elif problem_type == 5:
        Pi = psi1*dx - dot(Tf*n, u)*ds(24)

    elif problem_type == 6: 
        Pi = psi1*dx - (dot(-Tf*m, u)*ds(23) + dot(-Tf*m, u)*ds(24) + dot(Tf*m, u)*ds(21) + dot(Tf*m, u)*ds(22))
    
    # Compute first variation of Pi (directional derivative about u in the direction of v)
    R = derivative(Pi, u, v)
    
    # Compute Jacobian of R
    Jac = derivative(R, u, du)
    
    # Memory allocation for tensors
    A = PETScMatrix()
    b = PETScVector()
    
    # Balance of configurational forces PDE
    Wv = pen / 2 * ((abs(z) - z)**2 + (abs(1 - z) - (1 - z))**2) * dx
    Wv2 = conditional(le(z, 0.05), 1, 0) * 40 * pen / 2 * (1 / 4 * (abs(z_prev - z) - (z_prev - z))**2) * dx
    R_z = -Y_wu * y * dx + (1 / (1 + h / (pi * eps))) * Gc / pi * (y * (-2 * z) / eps + 2 * eps * inner(grad(z), grad(y))) * dx + derivative(Wv, z, y) + derivative(Wv2, z, y)
    
    # Compute Jacobian of R_z
    Jac_z = derivative(R_z, z, dz)

# ---------------------------------------------------------------------------------------------------------------------------------------------
# 3. Star Convex Model 2023		Model on Enegy Decomposition under Multi Axial Stress States  # Dr. Lorenzis
# ---------------------------------------------------------------------------------------------------------------------------------------------
elif phase_model == 3:
    # Set the variational model and update folder_prefix and gamma accordingly
    if variational_model == 1:
        folder_prefix += '_StarConvex'
        gamma = (3 * (sts / scs)**2 - 2 * (nu + 1)) / (2 * nu - 1)  # Star Convex model
    elif variational_model == 2:
        folder_prefix += '_VolDev'
        gamma = 0.0  # Vol-Dev model
    elif variational_model == 3:
        folder_prefix += '_Spec_Split'
    else:
        raise ValueError("Invalid variational_model choice. Choose 1 (StarConvex), 2 (VolDev) or 3 (Spec_Split).")


    # For models 1 and 2, use the same energy split functions (with gamma)
    if variational_model in [1, 2]:
        def energy_pos(v):
            strain = epsilon(v)
            strain_dev = epsilon_dev(v)
            # For plain stress use a modified trace; for plane strain and 3D use the standard trace.
            trstrain = tr(strain) * (1 - 2 * nu) / (1 - nu) if stress_state_choice == 1 else tr(strain)
            tr_eps_plus = 0.5 * (trstrain + abs(trstrain))
            tr_eps_neg  = 0.5 * (trstrain - abs(trstrain))
            # Use different contributions for plain stress versus plane strain/3D
            if stress_state_choice == 1:  # Plain Stress
                energy_density = 0.5 * kappa * (tr_eps_plus)**2 + mu * (inner(strain_dev, strain_dev) + ((1+nu)/(3*(nu-1))*tr(strain))**2) - gamma * 0.5 * kappa * (tr_eps_neg)**2
            else:  # Plane Strain and 3D
                energy_density = 0.5 * kappa * (tr_eps_plus)**2 + mu * inner(strain_dev, strain_dev) - gamma * 0.5 * kappa * (tr_eps_neg)**2

            return energy_density

        def energy_neg(v):
            strain = epsilon(v)
            trstrain = tr(strain) * (1 - 2 * nu) / (1 - nu) if stress_state_choice == 1 else tr(strain)
            tr_eps_neg = 0.5 * (trstrain - abs(trstrain))
            energy_density = (1 + gamma) * 0.5 * kappa * (tr_eps_neg)**2
            return energy_density

    elif variational_model == 3:
        
        # Helper functions for Spec_Split energy
        def spec_split_energy_pos(trstrain, eigenvals):
            tr_eps_plus = 0.5 * (trstrain + abs(trstrain))
            _sum = sum((0.5 * (eig + abs(eig)))**2 for eig in eigenvals)
            return 0.5 * lmbda * tr_eps_plus**2 + mu *_sum

        def spec_split_energy_neg(trstrain, eigenvals):
            tr_eps_neg = 0.5 * (trstrain - abs(trstrain))
            _sum = sum((0.5 * (eig - abs(eig)))**2 for eig in eigenvals)
            return 0.5 * lmbda * tr_eps_neg**2 + mu * _sum

        # Compute common strain quantities
        strain = epsilon(v)
        trstrain = tr(strain) * (1 - 2*nu)/(1-nu) if stress_state_choice == 1 else tr(strain)
        
        # Determine eigenvalues based on the stress state and problem dimension.
        # Here we assume that:
        #   - For plain stress (stress_state_choice==1): we use a 2D eigenvalue calculation and take the out‐of‐plane component from strain[2,2].
        #   - For plain strain (stress_state_choice==2): we use the 2D eigenvalue calculation.
        #   - Otherwise (3D case): we use the 3D eigenvalue calculation.
        if stress_state_choice == 1:
            eig1_val, eig2_val = compute_eigenvalues(strain, dim=2)
            eig3_val = -nu/E * tr(sigma(u))                                    # Out-of-plane stress component
            eigenvals = [eig1_val, eig2_val, eig3_val]
        elif stress_state_choice == 2:
            eigenvals = list(compute_eigenvalues(strain, dim=2))
        elif sigma(u).ufl_shape()[0] == 3:
            eigenvals = list(compute_eigenvalues(strain, dim=3))
        else:
            raise ValueError("Dimension not supported. dim must be 2 or 3.")
        
        def energy_pos(v):
            return spec_split_energy_pos(trstrain, eigenvals)

        def energy_neg(v):
            return spec_split_energy_neg(trstrain, eigenvals)

    else:
        raise ValueError("Invalid variational model selected.")

    pen = 1000 * Gc / eps

    psi1 = (z ** 2 + eta) * energy_pos(u) + energy_neg(u)
    psi11 = energy_pos(u)
    stress = (z ** 2 + eta) * sigma(u)

    # Total potential energy
    if problem_type == 1 or 3 or 4 or 7:
        Pi = psi1*dx

    if problem_type == 2:
        Pi = psi1*dx - dot(Tf, u)*ds(21)

    elif problem_type == 5:
        Pi = psi1*dx - dot(Tf*n, u)*ds(24)

    elif problem_type == 6: 
        Pi = psi1*dx - (dot(-Tf*m, u)*ds(23) + dot(-Tf*m, u)*ds(24) + dot(Tf*m, u)*ds(21) + dot(Tf*m, u)*ds(22))


    R = derivative(Pi, u, v)
    Jac = derivative(R, u, du)

    # Balance of configurational forces PDE
    Wv = pen / 2 * ((abs(z) - z) ** 2 + (abs(1 - z) - (1 - z)) ** 2) * dx
    Wv2 = 40 * pen / 2 * (1 / 4 * (abs(z_prev - z) - (z_prev - z)) ** 2) * dx

    R_z = y * 2 * z * psi11 * dx + (1 / (1 + 3 * h / (8 * eps))) * 3 * Gc / 8 * (y * (-1) / eps + 2 * eps * inner(grad(z), grad(y))) * dx \
          + derivative(Wv, z, y) + derivative(Wv2, z, y)

    Jac_z = derivative(R_z, z, dz)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# 2. Miehe et. al 2015
# ---------------------------------------------------------------------------------------------------------------------------------------------

elif phase_model == 4:
    pen = 1000 * Gc / eps
    stress = (z**2 + eta) * sigma(u)
    psi_c = sts**2 / (2 * E)
    
    # Determine the dimension of the stress tensor σ(u)
    dim = sigma(u).ufl_shape()[0]
    
    if stress_state_choice == 1:                                                                                                    # Plain Stress case
        # For a 2D problem: compute the two eigenvalues.
        eig1_val, eig2_val = compute_eigenvalues(sigma(u), dim=2)
        eig1_ = (eig1_val + abs(eig1_val)) / 2.0
        eig2_ = (eig2_val + abs(eig2_val)) / 2.0
        # Compute D_d using the two eigenvalues
        D_d = 0.5 * ((((eig1_**2 + eig2_**2) / sts**2) - 1) + abs(((eig1_**2 + eig2_**2) / sts**2) - 1)) / 2.0

    if stress_state_choice == 2:                                                                                                    # Plain Strain case
        # For a 2D problem: compute the two eigenvalues.
        eig1_val, eig2_val = compute_eigenvalues(sigma(u), dim=2)
        eig1_ = (eig1_val + abs(eig1_val)) / 2.0
        eig2_ = (eig2_val + abs(eig2_val)) / 2.0
        eig3_ = nu * tr(sigma(u))
        # Compute D_d using the three eigenvalues
        D_d = 0.5 * ((((eig1_**2 + eig2_**2 + eig3_**2) / sts**2) - 1) + abs(((eig1_**2 + eig2_**2 + eig3_**2) / sts**2) - 1)) / 2.0
        
    elif stress_state_choice == 3:                                                                                                   # 3D case
        # For a 3D problem: compute the three eigenvalues.
        eig1_val, eig2_val, eig3_val = compute_eigenvalues(sigma(u), dim=3)
        eig1_ = (eig1_val + abs(eig1_val)) / 2.0
        eig2_ = (eig2_val + abs(eig2_val)) / 2.0
        eig3_ = (eig3_val + abs(eig3_val)) / 2.0
        # Compute D_d using the three eigenvalues.
        D_d = 0.5 * ((((eig1_**2 + eig2_**2 + eig3_**2) / sts**2) - 1) + abs(((eig1_**2 + eig2_**2 + eig3_**2) / sts**2) - 1)) / 2.0
    
    # Stored strain energy density
    psi1 = (z**2 + eta) * energy(u)
    psi11 = energy(u)
    
    # Total potential energy
    if problem_type == 1 or 3 or 4 or 7:
        Pi = psi1*dx

    if problem_type == 2:
        Pi = psi1*dx - dot(Tf, u)*ds(21)

    elif problem_type == 5:
        Pi = psi1*dx - dot(Tf*n, u)*ds(24)

    elif problem_type == 6: 
        Pi = psi1*dx - (dot(-Tf*m, u)*ds(23) + dot(-Tf*m, u)*ds(24) + dot(Tf*m, u)*ds(21) + dot(Tf*m, u)*ds(22))
    
    # Compute the first variation of Pi (directional derivative about u in the direction of v)
    R = derivative(Pi, u, v)
    
    # Compute the Jacobian of R
    Jac = derivative(R, u, du)
    
    # Allocate PETSc matrices/vectors for later use
    A = PETScMatrix()
    b = PETScVector()
    
    # Balance of configurational forces PDE
    Wv = pen/2 * (((abs(z) - z)**2) + ((abs(1 - z) - (1 - z))**2)) * dx
    Wv2 = conditional(le(z, 0.05), 1, 0) * 40 * pen/2 * (1/4 * (abs(z_prev - z) - (z_prev - z))**2) * dx
    R_z = z * D_d * y * dx + (y * (z - 1) + eps**2 * inner(grad(z), grad(y))) * dx  # + derivative(Wv, z, y) + derivative(Wv2, z, y)
    
    # Compute the Jacobian of R_z
    Jac_z = derivative(R_z, z, dz)


# ___________________________________________________________________________________________________________________________________________________________


# Choose preconditioner based on problem type
if problem_dim == 3:
    precond = "petsc_amg"
else:
    precond = "amg"

# Set nonlinear solver parameters based on the solver choice
if non_linear_solver_choice == 1:
    snes_solver_parameters = {
        "nonlinear_solver": "snes",
        "snes_solver": {
            "linear_solver": "cg",
            "preconditioner": precond,
            "maximum_iterations": 10,
            "report": True,
            "error_on_nonconvergence": False
        }
    }
elif non_linear_solver_choice == 2:
    # Create a Krylov solver and set its parameters
    solver_u = KrylovSolver('cg', precond)
    max_iterations = 200
    solver_u.parameters["maximum_iterations"] = max_iterations
    solver_u.parameters["error_on_nonconvergence"] = False

    snes_solver_parameters = {
        "nonlinear_solver": "snes",
        "snes_solver": {
            "linear_solver": "cg",
            "preconditioner": precond,
            "maximum_iterations": 10,
            "absolute_tolerance": 1e-8,
            "report": True,
            "error_on_nonconvergence": False
        }
    }
elif non_linear_solver_choice == 3:
    snes_solver_parameters = {
        "nonlinear_solver": "snes",
        "snes_solver": {
            "linear_solver": "mumps",
            "maximum_iterations": 20,
            "report": True,
            "error_on_nonconvergence": False
        }
    }

# Set linear solver parameters if required
if linear_solver_for_u == 1:
    linear_solver_parameters = {"monitor_convergence": True}


#time-stepping parameters

if problem_type == 7: 
    stepsize=5e-3/disp
    Totalsteps=int(1/stepsize)
    startstepsize=1/Totalsteps
    stepsize=startstepsize
    t=stepsize
    step=1
    rtol=1e-9

else: 
    startstepsize=1/Totalsteps
    stepsize=startstepsize
    t=stepsize
    step=1
    rtol=1e-9
    tau=0

# other time stepping parameters
samesizecount = 1
terminate = 0
stag_flag = 0
gfcount = 0
nrcount = 0
minstepsize = startstepsize/10000
maxstepsize = startstepsize



while t-stepsize < T:

    if comm_rank==0:
        print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)
        sys.stdout.flush()

    if problem_type == 1:                                                            # Surfing
        c.t=t; c0.t=t; r.t=t; r0.t=t; c0.tau=tau; r0.tau=tau; 

    elif problem_type == 2 or 5:                                                     # Mode I, Biaxial
        c.t=t; Tf.t=t; 
    
    elif problem_type == 3:                                                          # Mode II
        c.t=t; 
    
    elif problem_type == 4:                                                          # Mode III
        r.t=t; r0.t=t; r0.tau=tau; 

    elif problem_type == 6:                                                          # Pure Shear
        Tf.t=t; 
    
    elif problem_type == 7:                                                          # 4-P Bending
        cl.t=t; 

        if t>0.5:
            stepsize == 5e-4 / disp



    stag_iter=1
    rnorm_stag=1
    terminate = 0
    stag_flag = 0


    while stag_iter<stag_iter_max and rnorm_stag > 1e-8:                
        start_time=time.time()
        ##############################################################
        #First PDE
        ##############################################################        

        if linear_solver_for_u==1:
            a_u = inner(stress, epsilon(v))*dx
            L_u = inner(Tf, v)*ds(1)
            Jac_u, Res_u = assemble_system(a_u, L_u, bcs)

            # Create solver instance
            linearsolver = KrylovSolver("cg", precond)
            # Set solver parameters
            linearsolver.parameters.update(linear_solver_parameters)
            linearsolver.solve(Jac_u, u.vector(), Res_u)
            converged_u = True


        if non_linear_solver_choice==1 or 3:
            Problem_u = NonlinearVariationalProblem(R, u, bcs, J=Jac)
            solver_u  = NonlinearVariationalSolver(Problem_u)
            solver_u.parameters.update(snes_solver_parameters)   
            (iter, converged_u) = solver_u.solve()	

        else:    
            nIter = 0
            rnorm = 10000.0
            rnorm_prev=10000.0
            while nIter < 10:
                nIter += 1
                
                if nIter==1 and stag_iter==1:
                    A, b = assemble_system(Jac, -R, bcs_du0)
                else:
                    A, b = assemble_system(Jac, -R, bcs_du)
                
                rnorm=b.norm('l2')
                
                if comm_rank==0:
                    print('Iteration number= %d' %nIter,  'Residual= %e' %rnorm)
				
                if rnorm < rtol:             #residual check
                    break
                rnorm_prev=rnorm

                converged = solver_u.solve(A, u_inc.vector(), b);
			
                if comm_rank==0:
                    print(converged)
			
                u.vector().axpy(1, u_inc.vector())  	


		##############################################################
		#Second PDE
		##############################################################
        Problem_z = NonlinearVariationalProblem(R_z, z, bcs_z, J=Jac_z)
        solver_z  = NonlinearVariationalSolver(Problem_z)
        solver_z.parameters.update(snes_solver_parameters)
        (iter, converged) = solver_z.solve() 


        min_z = z.vector().min(); 
        zmin = MPI.min(comm, min_z)
        if comm_rank==0:
            print(zmin)
            
        if comm_rank==0:
            print("--- %s seconds ---" % (time.time() - start_time))


        ###############################################################
        #Residual check for stag loop
        ###############################################################
        b=assemble(-R, tensor=b)
        fint=b.copy() #assign(fint,b) 
        for bc in bcs:
            bc.apply(b)
        rnorm_stag=b.norm('l2')	
        if comm_rank==0:
            print('Stag Iteration no= %d' %stag_iter,  'Residual= %e' %rnorm_stag)
        stag_iter+=1 

        if not converged or rnorm_stag > 1e2:
            terminate = 1
            break


    
                
    ######################################################################
    #Post-Processing
    ######################################################################
    if terminate == 1:
        assign(u, u_prev)
        assign(z, z_prev)
    else:
        assign(u_prev, u)
        assign(z_prev, z)


    # If maximum stag iterations reached, flag stag_flag and exit time stepping
    if stag_iter == stag_iter_max-1:
        stag_flag = 1
        break

    assign(u, u_prev)
    assign(z, z_prev)



    # ___________________________________________________________________________________________________________________________________________________________


    # Common post-processing steps
    def calculate_reaction(fint, y_dofs_top):
        return MPI.sum(comm, sum(fint[y_dofs_top]))

    def evaluate_z_at_point(z, point):
        return evaluate_function(z, point)

    def write_to_file(filename, data, mode='a'):
        with open(filename, mode) as rfile:
            rfile.write("{} {} {}\n".format(*data))

    def save_xdmf(ac, step, u, z, t):
        file_results = XDMFFile("Paraview_ac=" + str(ac) + "/SENT_" + str(step) + ".xdmf")
        file_results.parameters["flush_output"] = True
        file_results.parameters["functions_share_mesh"] = True
        u.rename("u", "displacement field")
        z.rename("z", "phase field")
        file_results.write(u, t)
        file_results.write(z, t)

    # Common variables
    comm = MPI.comm_world
    comm_rank = MPI.rank(comm)
    step = 0  # Assuming step is defined elsewhere

    # Post-processing based on problem_type
    if problem_type == 1:                                                                      # Surfing problem
        Fx = calculate_reaction(fint, y_dofs_top)
        JI1 = (psi1 * n[0] - dot(dot(stress, n), u.dx(0))) * ds(24)
        JI2 = (-dot(dot(stress, n), u.dx(0))) * ds(21)
        Jintegral = 2*(assemble(JI1)+assemble(JI2))

        if comm_rank == 0:
            write_to_file('Surfing_ce.txt', [t, Jintegral])

        if step % paraview_at_step == 0:
            save_xdmf("FullModel/Surfing_ce", u, z, t, step)

    elif problem_type == 2:                                                                    # Mode I
        Fx = calculate_reaction(fint, y_dofs_top, MPI.COMM_WORLD)
        z_x = evaluate_z_at_point(z, (ac + eps, 0.0))

        if comm_rank == 0:
            print(Fx)
            print(z_x)
            write_to_file('Alumina_SENT_AT2.txt', [t, zmin, z_x])

        if step % paraview_at_step == 0:
            save_xdmf(ac, step, u, z, t)

        if z_x < 0.1:
            t1 = t
            break

        if t > 0.997:
            t1 = 1

    elif problem_type == 3:                                                                    # Mode II
        Fx = calculate_reaction(fint, y_dofs_top)
        z_x = evaluate_z_at_point(z, (23 + eps, -12.5))        # 23, -12.5 are inputs, define as var

        if comm_rank == 0:
            print(Fx)
            print(z_x)
            write_to_file('SENT_Shear.txt', [t, zmin, z_x])
            write_to_file('Traction.txt', [t, Fx])

        if step % paraview_at_step == 0:
            save_xdmf("Paraview/SENT", u, z, t, step)

        if z_x < 0.01:
            break

    elif problem_type == 4:                                                                    # Mode III
        Fx = calculate_reaction(fint, y_dofs_top)

        if comm_rank == 0:
            printdata = [t, t * disp, Fx]
            with open('Results.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(printdata)

        if step % paraview_at_step == 0:
            save_xdmf("paraview/pararesult", u, z, t, step)

    elif problem_type == 5:                                                                    # Biaxial
        Fx = calculate_reaction(fint, y_dofs_top)
        z_x = evaluate_z_at_point(z, (ac + eps, 0.0))

        if comm_rank == 0:
            print(Fx)
            print(z_x)
            write_to_file('Graphite_Biaxial.txt', [t, zmin, z_x])

        if step % paraview_at_step == 0:
            save_xdmf(f"Paraview_ac={2 * ac}/SENT", u, z, t, step)

        if z_x < 0.05:
            t1 = t
            break

        if t > 0.997:
            t1 = 1

    elif problem_type == 6:                                                                    # Pure Shear
        Fx = calculate_reaction(fint, y_dofs_top)
        z_x = [evaluate_z_at_point(z, (ac + 2 * eps * np.cos(angle), 2 * eps * np.sin(angle))) for angle in np.linspace(5 / 4 * np.pi, 2 * np.pi, 100)]

        if comm_rank == 0:
            print(Fx)
            print(z_x)
            write_to_file('SENT_Shear.txt', [t, zmin, min(z_x)])
            write_to_file('Traction.txt', [t, Fx])

        if step % paraview_at_step == 0:
            save_xdmf("Paraview/SENT", u, z, t, step)

        if min(z_x) < 0.01:
            t1 = t
            break

        if t > 0.997:
            t1 = 1

    elif problem_type == 7:                                                                    # 4-P Bending
        Fx_Rleft = calculate_reaction(fint, y_dofs_Rleft)
        Fx_Rright = calculate_reaction(fint, y_dofs_Rright)
        Fx_Pleft = calculate_reaction(fint, y_dofs_Pleft)
        Fx_Pright = calculate_reaction(fint, y_dofs_Pright)

        z_x = evaluate_z_at_point(z, (0.0, 7.0, 0.0))

        if comm_rank == 0:
            print(Fx_Rleft)
            print(Fx_Rright)
            print(Fx_Pleft)
            print(Fx_Pright)
            print(z_x)
            write_to_file('Graphite_Biaxial.txt', [t, zmin, z_x])
            write_to_file('Traction.txt', [t, Fx_Rleft, Fx_Rright, Fx_Pleft, Fx_Pright])

        if step % paraview_at_step == 0:
            save_xdmf("Paraview/4P", u, z, t, step)


    # ___________________________________________________________________________________________________________________________________________________________


    # Time-stepping update (combined)
    ######################################################################
    if terminate:
        # If termination flagged, reduce the time step if possible
        if stepsize > minstepsize:
            t = t - stepsize
            stepsize /= 2
            t = t + stepsize
            samesizecount = 1
        else:
            break
    else:
        # If no termination, either repeat the same step or increase the step size
        if samesizecount < 2:
            step += 1
            if t + stepsize <= T:
                samesizecount += 1
                t += stepsize
            else:
                samesizecount = 1
                stepsize = T - t
                t += stepsize
        else:
            step += 1
            if stepsize * 2 <= maxstepsize and t + stepsize * 2 <= T:
                stepsize *= 2
                t += stepsize
            elif stepsize * 2 > maxstepsize and t + maxstepsize <= T:
                stepsize = maxstepsize
                t += stepsize
            else:
                stepsize = T - t
                t += stepsize
            samesizecount = 1


    if stag_flag:
        # If termination flagged, reduce the time step for the next update.
        if stepsize > minstepsize:
            stepsize /= 2
            samesizecount = 1
            t += stepsize
        else:
            break
    else:
        # If no termination, either repeat the same step or increase the step size.
        if samesizecount < 2:
            step += 1
            if t + stepsize <= T:
                samesizecount += 1
                t += stepsize
            else:
                samesizecount = 1
                stepsize = T - t
                t += stepsize
        else:
            step += 1
            if stepsize * 2 <= maxstepsize and t + stepsize * 2 <= T:
                stepsize *= 2
                t += stepsize
            elif stepsize * 2 > maxstepsize and t + maxstepsize <= T:
                stepsize = maxstepsize
                t += stepsize
            else:
                stepsize = T - t
                t += stepsize
            samesizecount = 1       


    

def write_and_print_critical_stress(sigma_critical, notch_length, filename='Critical_stress.txt'):
    if comm_rank == 0:
        with open(filename, 'a') as rfile:
            rfile.write(f'Critical stress = {sigma_critical:.4f} (MPa), at Notch Length = {notch_length:.2f} mm\n')
        print(f'Critical stress = {sigma_critical:.4f} (MPa), at Notch Length = {notch_length:.2f} mm')

# Common variables
sigma_critical = t1 * sigma_external

# Condition-specific code
if problem_type == 2:                                                     # Mode I
    write_and_print_critical_stress(sigma_critical, ac)

elif problem_type == 5:                                                   # Biaxial
    write_and_print_critical_stress(sigma_critical, 2 * ac)

elif problem_type == 6:                                                   # Pure Shear
    write_and_print_critical_stress(sigma_critical, 2 * ac)


#######################################################end of all loops
if comm_rank==0:	
	print("--- %s seconds ---" % (time.time() - start_time))


if comm_rank == 0: 
    endTime = datetime.now()
    elapseTime = endTime - startTime
    print("------------------------------------")
    print("Elapsed real time:  {}".format(elapseTime))
    print("------------------------------------")