import cvxpy as cp
import numpy as np

def optimise(**kwargs):
    '''
    Author: Taimoor Sohail (2022)
    This function takes matrices of tracers, volumes, weights, and constraints, 
    and produces an optimal transport estimate (g_ij) based on these constraints and weights.

    Inputs:

    volumes: A [2 x N] array of volumes/masses corresponding to the early and late watermasses
    tracers: A [2 x M x N]  array of tracers, where N is the number of watermasses, M is the number of distinct tracers, and 2 corresponds to the early and late watermasses
    For just T and S, M = 2. Other tracers such as carbon may be added to this matrix.
    cons_matrix: A [N X N] matrix defining the connectivity from one 'N' watermass to any other 'N' watermass. 
    The elements in this matrix must be between 0 (no connection) and 1 (fully connected).
    trans: Set of constraints on inter-basin transport (e.g., we can fix ITF transport to be 15 Sv). Threshold must be provided.
    Asection: Matrix which defines the section areas across each basin mask. Threshold must be provided.
    weights: An [M x N] matrix defining any tracer-specific weights to scale the transports by watermass, 
    for instance, outcrop surface area, or a T/S scaling factor. 
    hard_area: A Way to deal with zero surface outcrop water masses that isn't factored into the weights above - we add a hard constraint that T_1,j*V_1,j = sum(g_ij*T_0i)
    Note - The optimiser uses the MOSEK solver, and advanced optimisation software that requires a (free) license. You MUST install MOSEK to use the function. 
    Outputs:

    g_ij: A transport matrix of size [N x N] which represents the transport from one watermass to another. 
    Mixing: A matrix comprising the change in tracer due to mixing from t1 to t2, of size [M x N]
    Adjustment: A matrix comprising the change in tracer due to adjustment from t1 to t2, of size [M x N]
    '''
    
    ## Break down kwarg inputs
    A_exists = False
    trans_exists = False
    hard_A_cons = False
    names = list(kwargs.keys())
    for i in range(np.array(names).size):
        if names[i] == 'volumes':
            volumes = np.array(list(kwargs.values())[i])
        if names[i] == 'tracers':
            tracers = np.array(list(kwargs.values())[i])
        if names[i] == 'cons_matrix':
            cons_matrix = np.array(list(kwargs.values())[i])
        if names[i] == 'trans':
            trans_list = np.array(list(kwargs.values())[i],dtype=object)
            trans = trans_list[0]
            trans_val = trans_list[1]
            trans_exists = True
        if names[i] == 'weights':
            weights = np.array(list(kwargs.values())[i])
        if names[i] == 'Asection':
            Asection_list = np.array(list(kwargs.values())[i])
            Asection = Asection_list[0]
            threshold = Asection_list[1]
            A_exists = True
        if names[i] == 'hard_area':
            area_hard = np.array(list(kwargs.values())[i])
            hard_A_cons = True

    ## Define matrices for the linear optimisation
    
    N = volumes.shape[-1]
    M = tracers.shape[1]

    nofaces = np.count_nonzero(cons_matrix)
    if trans_exists:
        trans_full = np.zeros((int(nofaces),trans.shape[2]))
    if A_exists:
        Asection_full = np.zeros(int(nofaces))

    C1_connec=np.zeros((N,int(nofaces)))
    C2_connec=np.zeros((N,int(nofaces)))
    # Also make T and S matrix with the T(k,i) the temp of the ith early WM
    Tmatrix=np.zeros((N,int(nofaces)))
    Smatrix=np.zeros((N,int(nofaces)))
    if M>2:
        trac_matrix = np.zeros((M-2,N,int(nofaces)))
    ix=0
    for i in (range(N)):
        for j in range(N):
            if cons_matrix[i,j]>0:
                C1_connec[i,ix] = cons_matrix[i,j] # vertex ix connects from WM i
                C2_connec[j,ix] = cons_matrix[i,j] # vertex ix connects to WM j
                if trans_exists:
                    if trans.shape[2]>1:
                        for k in range(trans.shape[2]):
                            trans_full[ix,k] = trans[i,j,k]
                    else:
                        trans_full[ix] = trans[i,j]
                if A_exists:
                    Asection_full[ix] = Asection[i,j]
                Tmatrix[j,ix] = tracers[0,1,i] #vertex ix brings temp of WM i to WM j
                Smatrix[j,ix] = tracers[0,0,i] #vertex ix brings temp of WM i to WM j
                if M>2:
                    trac_matrix[:,j,ix] = tracers[0,2:,i] #vertex ix brings temp of WM i to WM j
                ix=ix+1

    
    Cmatrix = trac_matrix[0,:,:]
    
    C = np.concatenate((C1_connec,C2_connec),axis=0)

    d = np.concatenate((volumes[0,:],volumes[1,:]),axis=0)

    A_T = np.zeros_like(Tmatrix)
    A_S = np.zeros_like(Tmatrix)
    A_C = np.zeros_like(trac_matrix[0,:,:])
    for i in range(int(nofaces)):
        A_T[:,i] = Tmatrix[:,i]*weights[1,:]
        A_S[:,i] = Smatrix[:,i]*weights[0,:]
        A_C[:,i] = Cmatrix[:,i]*weights[2,:]

    A = np.concatenate((A_T,A_S,A_C),axis=0)
    #A = np.concatenate((A_T,A_S),axis=0)
    
    b = np.concatenate((volumes[1,:]*tracers[1,1,:]*weights[1,:],\
                    volumes[1,:]*tracers[1,0,:]*weights[0,:],\
                       volumes[1,:]*tracers[1,2,:]*weights[2,:]), axis=0)
    b[np.isnan(b)]=0

    if hard_A_cons:
        A_T2 = np.zeros_like(Tmatrix)
        A_S2 = np.zeros_like(Tmatrix)
        A_C2 = np.zeros_like(Cmatrix)
        for i in range(int(nofaces)):
            A_T2[:,i] = Tmatrix[:,i]*area_hard[1,:]
            A_S2[:,i] = Smatrix[:,i]*area_hard[0,:]
            A_C2[:,i] = Cmatrix[:,i]*area_hard[2,:]
        
        A2 = np.concatenate((A_T2,A_S2,A_C2),axis=0)

        b2 = np.concatenate((volumes[1,:]*tracers[1,1,:]*area_hard[1,:],\
                volumes[1,:]*tracers[1,0,:]*area_hard[0,:],\
                            volumes[1,:]*tracers[1,2,:]*area_hard[2,:]), axis=0)
        b2[np.isnan(b)]=0

    ## Invoke solver to calculate transports
    u = A.shape[1]

    x = cp.Variable(u)

    cost = cp.sum_squares(A@x-b)

    constraints = [C@x==d, x>=0]
    #constraints = [C@x-d<=10**(-2), x>=0] # NSM 11/7/23
    #constraints.append(C@x-d >= -10**(-2))

    if trans_exists==True:
        print('Using ' +str(trans.shape[2])+ ' transport constraints')
        if trans.shape[2]>1:
            for k in range(trans.shape[2]):
                constraints.append(cp.sum(x*trans_full[:,k].flatten())==trans_val[k])
                #constraints.append(cp.abs(cp.sum(x*trans_full[:,k].flatten())-trans_val[k])<=0.1) # NSM 11/7/23
        else:
            constraints.append(cp.sum(x*trans_full.flatten())==trans_val)
    if A_exists==True:
        constraints.append(x/Asection_full.flatten()<=threshold)
    if hard_A_cons==True:
        constraints.append(A2@x==b2)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    # The optimal objective value is returned by prob.solve()`.
    # OSQP, ECOS, ECOS_BB, MOSEK, CBC, CVXOPT, NAG, GUROBI, and SCS
    result = prob.solve(verbose=True, solver=cp.MOSEK)
    
    
    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)
    for variable in prob.variables():
        print("Variable %s: value %s" % (variable.name(), variable.value))

    # The optimal value for x is stored in `x.value`.
    g_ij = x.value

    ## Convert g_ij from a long [1 x N^2] matrix to an [N x N] matrix

    G = np.zeros((N,N))
    ix=0
    for i in (range(N)):
        for j in range(N):
            if cons_matrix[i,j]>0:
                G[i,j] = g_ij[ix]
                ix=ix+1   

    # This is the temperature and salinity the late water masses acheive by mixing the early water masses
    Tmixed = np.matmul(Tmatrix,g_ij)/volumes[1,:]
    Tmixed[~np.isfinite(Tmixed)]= np.nan
    Tmixed[Tmixed>100] = np.nan
    Smixed = np.matmul(Smatrix,g_ij)/volumes[1,:]
    Smixed[~np.isfinite(Smixed)]= np.nan
    Smixed[Smixed>10**4] = np.nan
    if M>2:
        trac_mixed = np.zeros((M-2,Smixed.size))
        for i in range(M-2):
            trac_mixed[i,:] = np.matmul(trac_matrix[i,:,:],g_ij)/volumes[1,:]
            trac_mixed[~np.isfinite(trac_mixed)]= np.nan
            trac_mixed[trac_mixed>10**4] = np.nan


    # Now the necessary heat and salt adjustment is simply the difference
    # between this and what we actually get
    T_Av_adj = (tracers[1,1,:]-Tmixed)
    T_Av_adj[np.isnan(T_Av_adj)] = 0

    S_Av_adj = (tracers[1,0,:]-Smixed)
    S_Av_adj[np.isnan(S_Av_adj)]= 0
    if M>2:
        trac_Av_adj = (tracers[1,2:,:]-trac_mixed)
        trac_Av_adj[np.isnan(trac_Av_adj)]= 0

    Smixed[~np.isfinite(Smixed)]=0
    Tmixed[~np.isfinite(Tmixed)]=0
    if M>2:
        trac_mixed[~np.isfinite(trac_mixed)]=0

    dTmix = np.matmul(G,Tmixed)/volumes[0,:]-tracers[0,1,:]
    dSmix = np.matmul(G,Smixed)/volumes[0,:]-tracers[0,0,:]
    if M>2:
        dtrac_mix = np.zeros((M-2,Smixed.size))
        for i in range(M-2):
            dtrac_mix[i,:] = np.matmul(G,trac_mixed[i,:])/volumes[0,:]-tracers[0,i+2,:]

    if M>2:
        Mix_matrix = np.vstack((dSmix, dTmix, dtrac_mix))
        Adj_matrix = np.vstack((S_Av_adj, T_Av_adj, trac_Av_adj))
    else:
        Mix_matrix = np.vstack((dSmix, dTmix))
        Adj_matrix = np.vstack((S_Av_adj, T_Av_adj))

    return {'g_ij':G, 'G':g_ij, 'Mixing': Mix_matrix, 'Adjustment': Adj_matrix}