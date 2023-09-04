# Files which contains all the functions needs in the iterative Kalman filter using the EM algorithm

import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv 
from sklearn.decomposition import PCA


"""
Function returning the list of indices without observations
Input arguments :
    - y : vector of observations
    - t : time iteration
    - p : size of the observation vector
Output argument :
    - ind : list of indices without observation (ie where wa have a nan)
"""
def lisind(y,t,p):
    ind=[]
    for i in range(p):
        if np.isnan(y[t,i])==True:
            ind.append(i)
    return ind


"""
Function returning the observation operator matrix H
Input arguments :
    - ind : list of indices without observation
    - H : observation operator matrix 
Output Argument :
    - Ht :  observation operator matrix at time iteration t 
"""
def matopobs(ind,H):
    Ht=np.delete(H,ind,0)
    return Ht


"""
Function returning the covariance matrix of observations
Input Arguments :
    - R : covariance matrix of observations at the iteration t 
    - ind : list of indices without observations
Output argument :
    - R : Rt covariance matrix at the iteration t (without indices where we haven't observations)
"""
def matcovR(R,ind):
    Rt=np.delete(R,ind,0)
    Rt=np.delete(Rt,ind,1)
    return Rt


"""
Function returning the new matrix R using the EM algorithm
Input arguments :
    - y : vector of observations
    - xs : reconstructed currents (get by the Kalman smoother)
    - Ps : covariance matrix of reconstructed currents (get by the Kalman smoother)
    - p : size of the observation vector
    - H : observation operator matrix 
Output argument :
    - R : covariance matrix modified (by equation of the paper of Shumway and Stoffer on the EM algorithm )
"""
def matREM(y,xs,Ps,p,H):
    T=y.shape[0]
    R=np.zeros((p,p))
    CO=np.zeros((p,p))
    for t in range(1,T):
        ind=lisind(y,t,p)
        Ht=matopobs(ind,H)
        newy=y[t,:]
        newy=np.delete(newy,ind)
        Rt=(np.transpose(np.array([newy]))-Ht@np.transpose(np.array([xs[t,:]])))@np.transpose(np.transpose(np.array([newy]))-Ht@np.transpose(np.array([xs[t,:]])))+Ht@Ps[t,:,:]@np.transpose(Ht)
        for i in range(len(ind)):
            Rt=np.insert(Rt,ind[i],0,0)
            Rt=np.insert(Rt,ind[i],0,1)
            CO[ind[i],:]-=1
            CO[:,ind[i]]-=1
        for j in range(len(ind)):
            for k in range(len(ind)):
                CO[ind[j],ind[k]]+=1
        R=R+Rt

    CO+=(T-1)
    R=R/CO
    return R


"""
function which calculate the great circle distance between two points on the Earth, specified as (lon,lat)
where lon and lat are in degrees
Input arguments :
    - point1 : A point on the earth given by point1=[lon,lat]
    - point2 : same as point1
Output argument:
    distance between the two points in km
"""
#Mean radius of the Earth( in km)
EARTH_RADIUS=6371.009
def haversine(point1,point2):
    #Convert decimal degrees to radians
    lon1,lat1=[np.radians(x) for x in point1]
    lon2,lat2=[np.radians(x) for x in point2]

    #haversine formula
    dlon=lon2-lon1
    dlat=lat2-lat1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_RADIUS*np.arcsin(np.sqrt(a))

"""
Function which return initial conditions of all the matrix and vector necessary to apply the Kalman filter.
This function return initial conditions according to a PCA is made or not 
We apply the PCA in this function du to the fact that we only need of the base change matrix 
Input  arguments :
    - opt : integer which permitt to know if a PCA is apply on the data (0:no PCA)
    - p : size of observations vector 
    - k : size of state vector with an PCA
    - LONGdi : matrix which contain all the longitude when no PCA is apply 
    - LATdi : matrix which contain all the latitude when no PCA is apply 
    - y : vector of observations 
    - var_y : vector of variances associated of observations
Output arguments:
    - M : model operator matrix
    - H : observation operator matrix
    - Q : state vector covariance matrix
    - R : covariance matrix of observations
    - x0 : initial condition of the state vector
    - P0 : initial condition of the covaraince matrix associated of x
"""
def IC(opt,p,k,LONGdi,LATdi,y,var_y):
    T=var_y.shape[0] 
    # Calcul of R ( it is the same n the two case)
    R=np.zeros((T,p,p))
    for t in range(T):
        for i in range(p):
            R[t,i,i]=var_y[t,i]
    # When a PCA is not apply on the data
    if opt==0:
        # Matrix M and H
        #H=np.eye(p)
        M=np.eye(p)
        # Matrix of distance to calculate Q
        sigsqu=0.01
        lamb=100
        pix=int((p/2))
        longdi=LONGdi.flatten()
        latdi=LATdi.flatten()
        dist=np.zeros((pix,pix))
        for i in range(pix):
            for j in range(0,i):
                dist[i,j]=haversine([longdi[i],latdi[i]],[longdi[j],latdi[j]])
        dist=dist+np.transpose(dist)
        # Matrix Q
        Q=np.zeros((p,p))
        Q[0:pix,0:pix]=sigsqu*np.exp(-dist/lamb)
        Q[pix:,pix:]=sigsqu*np.exp(-dist/lamb)
        # Initial conditions x0 and P0
        x0=np.zeros(p)
        P0=np.eye(p)
    
    #When we apply a PCA on our data 
    else :
        # Matrix M
        M=np.eye(k)
        # Matrix Q
        Q=np.eye(k)
        # Initial conditions x0 and P0
        x0=np.zeros(k)
        P0=np.eye(k)
    
    return M,Q,R,x0,P0
    
"""
Function which apply the Kalman filter
Imput arguments:
    - y : vector of observations
    - var_y : vector of variances associated of observations
    - x0 : Initial condition of the state vector (X0=xb in literature )
    - P0 : Initial condition of the covariance matrix (P0=B in literature)
    - M : model operator matrix
    - Q :  state vectore covariance matrix
    - R : covariance matrix of observations
    - H : observation operator matrix
Output arguments :
    - xf : propagated state vector
    - Pf : covariance matrix of xf
    - xa : filtered state vector
    - Pa : covariance matrix of xa
    - loglik : list of the likelihood at each iteration 
"""
def Kalman_filter(y,var_y,x0,P0,M,Q,R,H):
    # shapes
    r=len(x0)    # 
    p=y.shape[1] # 
    T=y.shape[0]

    # Initialization
    xf=np.zeros((T,r))
    Pf=np.zeros((T,r,r))
    xa=np.zeros((T,r))
    Pa=np.zeros((T,r,r))
    loglik=np.zeros(T)

    xa[0,:]=x0
    Pa[0,:,:]=P0

    # Apply kalman filter
    for t in range(1,T):
        indobs=lisind(y,t,p)
        Ht=matopobs(indobs,H)

        indcov=lisind(var_y,t,p)
        Rt=matcovR(R[t,:,:],indcov)
        
        newy=y[t,:]
        newy=np.delete(newy,indobs)

        # Prediction step
        xf[t,:]=M@xa[t-1,:]
        Pf[t,:,:]=M@Pa[t-1,:,:]@np.transpose(M)+Q

        if len(newy)>0:
            #Kalman gain
            K=Pf[t,:,:]@np.transpose(Ht)@inv(Ht@Pf[t,:,:]@np.transpose(Ht)+Rt)

            # update step
            xa[t,:]=xf[t,:]+K@(newy-Ht@xf[t,:])
            Pa[t,:,:]=(np.eye(r)-K@Ht)@Pf[t,:,:]
        else:
            xa[t,:]=xf[t,:]
            Pa[t,:,:]=Pf[t,:,:]
        
        #Loglikelihood
        loglik[t]=--0.5*(np.transpose(newy-Ht@xf[t,:])@inv(Ht@Pf[t,:,:]@np.transpose(Ht)+Rt)@(newy-Ht@xf[t,:]))
    
    return xf,Pf,xa,Pa,loglik


"""
Function which apply the Kalman smoother
Imput arguments:
    - y : vector of observations
    - var_y : vector of variances associated of observations
    - x0 : Initial condition of the state vector (X0=xb in literature )
    - P0 : Initial condition of the covariance matrix (P0=B in literature)
    - M : model operator matrix
    - Q :  state vectore covariance matrice
    - R : covariance matrix of observations
    - H : observation operator matrix
Output arguments :
    - xf : propagated state vector
    - Pf : covariance matrix of xf
    - xa : filtered state vector
    - Pa : covariance matrix of xa
    - xs : smooth state vector
    - Ps : covariance matrix of xs
    - loglik : list of the likelihood at each iteration
    - Ps_lag : smooth lagged error covariance matrix 
"""
def Kalman_Smoother(y,var_y,x0,P0,M,Q,R,H):
    
    #shapes
    r=len(x0)
    p=y.shape[1]
    T=y.shape[0]

    # Initialization
    xs=np.zeros((T,r))
    Ps=np.zeros((T,r,r))
    Ps_lag=np.zeros((T-1,r,r))

    # Apply Kalman filter
    xf,Pf,xa,Pa,loglik=Kalman_filter(y,var_y,x0,P0,M,Q,R,H)

    # Apply the Kalmna Smoother
    for t in range(T-1,-1,-1):
        if t==T-1:
            xs[t,:]=xa[t,:]
            Ps[t,:,:]=Pa[t,:,:]
        else:
            Ks=Pa[t,:,:]@np.transpose(M)@inv(Pf[t+1,:,:])
            xs[t,:]=xa[t,:]+Ks@(xs[t+1,:]-xf[t+1,:])
            Ps[t,:]=Pa[t,:,:]+Ks@(Ps[t+1,:,:]-Pf[t+1,:,:])@np.transpose(Ks)
    
    #Calcul of Ps_lag
    for t in range(0,T-1):
        A=np.zeros((r,r))
        # need to calculate A=(I-KH)MPa(t-1)
        indobs=lisind(y,t+1,p)
        Ht=matopobs(indobs,H)
        indcov=lisind(var_y,t+1,p)
        Rt=matcovR(R[t+1,:,:],indcov)
        K=Pf[t+1,:,:]@np.transpose(Ht)@inv(Ht@Pf[t+1,:,:]@np.transpose(Ht)+Rt)

        newy=y[t,:]
        newy=np.delete(newy,indobs)

        if len(newy)>0:
            # Calcul of A
            A=(np.eye(r)-K@Ht)@M@Pa[t,:,:]
        
        # Calcul of B
        B=(Ps[t+1,:,:]-Pa[t+1,:,:])@pinv(Pa[t+1,:,:])

        # Calcul of Ps_lag
        Ps_lag[t,:,:]=A+B@A
    
    return xf,Pf,xa,Pa,xs,Ps,loglik,Ps_lag



"""
Function which return result of iterative kalman filter using the EM algorithm in the 
paper of Shumway and Stoffer 
Input arguments :
    - y : vector of observations
    - var_y : vector of variances associated of observations
    - x0 : Initial condition of the state vector (X0=xb in literature )
    - P0 : Initial condition of the covariance matrix (P0=B in literature)
    - M : model operator matrix
    - Q :  state vectore covariance matrice
    - R : covariance matrix of observations
    - H : observation operator matrix
    - N : number of iteration 
Output arguments
    -slk : list which contain the sum of all loglikelihood of each iteration N 
    - M : model operator matrix optimized
    - Q : covariance matrix of state model optimized 
    - R : covariance matrix of observations optimized
    - xs : smooth state vector
    - Ps : covariance matrix of xs 
"""
def Kalman_EM(y,var_y,x0,P0,M,Q,R,H,N):
    #shapes
    r=len(x0)
    p=y.shape[1]
    T=y.shape[0]
    slk=[]
    
    for k in range(N):
        xf,Pf,xa,Pa,xs,Ps,loglik,Ps_lag=Kalman_Smoother(y,var_y,x0,P0,M,Q,R,H)
        slk.append(np.sum(loglik))
        
        x0=xs[0,:]
        P0=Ps[0,:,:]

        A=np.zeros((r,r))
        for t in range(T-1):
            A+=Ps[t,:,:]+np.transpose(np.array([xs[t,:]]))@np.array([xs[t,:]])
        B=np.zeros((r,r))
        for t in range(T-1):
            B+=Ps_lag[t,:,:]+np.transpose(np.array([xs[t+1,:]]))@np.array([xs[t,:]])
        C=np.zeros((r,r))
        for t in range(T-1):
            C+=Ps[t+1,:,:]+np.transpose(np.array([xs[t+1,:]]))@np.array([xs[t+1,:]])
        
        M=B@inv(A)
        Q=(C-M@np.transpose(B))/(T-1)
        R[0,:,:]=matREM(y,xs,Ps,p,H)
        R[:,:,:]=R[0,:,:]
    
    return slk,M,Q,R,xs,Ps
