from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import linalg as LA
import gc
import csv

############################## Nystrom-Sprectral Clustering ########################################
def SpectralNystrom(X,g):
    # cols of X: 0-2: x, y, z, 3: count of labels
    nXx=np.shape(X)[0] 
    std_dev=10
    knn=40
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm="kd_tree").fit(np.array(X[:,:2]))
    distances, indices = nbrs.kneighbors(np.array(X[:,:2]))  # dis:

    distances0=np.zeros((knn,))
    for jj in range(nXx):  # voxel 0 - 2600
        l1=X[jj,2]  # l1: v0's height
        l2=X[indices[jj,:],2]  # neighbors of v0 heights
        deta_l=abs(l1-l2)  # height difference between v0 and its 40 neighbors
        distances0=np.vstack((distances0,deta_l))
    distances0=distances0[1:(nXx+1),:]

    # distances means 2d xy distance, distances0 is the height(z) difference
    # similarity graph，
    for j in range(knn):
        # the equation has two parts: Dxy^2 * n_i * n_j, where X[:, 3], X[indices[:,j],3] is n_i and n_j
        # the other part is Dz^2 * n_i * n_j
        # the weight number n_i is the number of points of the target voxel, n_j is that of the jth NN voxel (j = 0-39)
         distances[:, j] = distances[:, j]*distances[:, j]*X[:, 3]*X[indices[:, j], 3]
         distances0[:, j] = distances0[:, j]*distances0[:, j]*X[:, 3]*X[indices[:, j], 3]
     
    distances = np.exp(-distances/std_dev)*np.exp(-distances0/std_dev)
    del X
    gc.collect()

    #******************************* KNNS sampling *******************************************#
    # Remember that
    # <distances> are the VALUES of similarity for each voxel and its NNs
    # <indices> are actually  row number, or voxel number of <distances>
    sumw=np.sum(distances,axis=1)  # summation of the similarities of all neighbor
    sumw=np.vstack((sumw,[0 for i in range(nXx)])).T  # 2nd column of sumw acts as a label with default value 0
    idsumw=np.lexsort([-sumw[:,0]])  # indices in descending order
    X1,X2,AA,BB=[],[],[],[]  # x1: sampling index (voxel number), x2 remaining index (voxel number)
    """
    once a voxel was sampled into X1 (the samples), its knns was put into X2 (the remains),  
    see that in Pang 2021 Paper, this voxel number turn to Blue.
    """
    iter_count = 0
    for ii in range(nXx):
        i=idsumw[ii]  # i starts from the voxel number with the largest similarity
        if sumw[i,1]==0:
            iter_count += 1
            X1.extend([i])  # X1 was only updated here
            sumw[i,1]=-1  # -1 means this voxel has been put into the sample or the remain
            sim1=[0 for k1 in range(len(X1))]  # sim1: sample similarity, append to AA
            sim2=[0 for k2 in range(len(X2)+knn)]  # sim2: remain similarity, append to BB
            num_new_remain = 0
            for j in range(knn):  # check the every NN, see if NN was 1) in the sample, 2) in the remain, or 3) neither
                if indices[i, j] in X1:
                    # this voxel has been put in the sample,
                    id1 = X1.index(indices[i, j])
                    sim1[id1] = distances[i, j]  # similarity was assigned at where X1 == voxel(i, j)
                    sim2.pop()  # remove the last elements and length -1
                elif indices[i, j] in X2:  # NN was in the remain
                    id2 = X2.index(indices[i, j])
                    sim2[id2] = distances[i, j]  # sim2[where x2 = index[i, j]]
                    sim2.pop()  # remove the last element
                else:
                    num_new_remain += 1
                    X2.extend([indices[i, j]])
                    sim2[len(X2)-1] = distances[i, j]
                    sumw[indices[i, j], 1] = -1
            AA.append(sim1)
            BB.append(sim2)
    # size of AA is 124
    del distances, indices
    gc.collect()
    # make AA BB as square matrix
    samples=len(X1)
    remains=len(X2)
    A=np.eye(samples)
    B=np.zeros((samples,remains))  
    for i in range(samples):
        A[i,:(i+1)]=AA[i]
        B[i,:len(BB[i])]=BB[i]
    del AA,BB
    gc.collect()
    
    #********************************** Eigendecomposition  *************************************#
    idx=np.hstack((X1,X2))
    sumw=sumw[idx,0]
    d=np.power(sumw,-0.5)   #
    dd=np.dot(d.reshape((len(d),1)),d.reshape((1,len(d))))  # dii = sum()
    A=A*dd[:samples,:samples]  # Normalize sim values of the sample matrix
    B=B*dd[:samples,samples:]
    detA=LA.det(np.sqrt(A))   
    if detA>0:  # get inv: Asi = A^(-1/2)
        Asi=LA.inv(np.sqrt(A))   
    else:
        Asi=LA.pinv(np.sqrt(A))   
    Q=A+np.dot(np.dot(np.dot(Asi,B),B.T),Asi)   # S = A + A^(-1/2) * BB^T * A^(-1/2)
    eigvals, eigvecs = LA.eig(Q)   
    Lamda=np.diag(np.power(eigvals, -0.5))  # diagonalize Q, or S, get sqrt Lamda
    V=np.dot(np.dot(np.dot(np.vstack((A,B.T)),Asi),eigvecs),Lamda)

    #***************************** Clustering segementation *************************************#
    eeigval=sorted(eigvals)
    if g==1 or len(eeigval)-1<g:
        g=len(eeigval)-1
    eeigval=np.array(eeigval)
    g1=int(2*g/4)   
    gap=eeigval[(g1+1):(g+1)]-eeigval[g1:g] 
    sk0=np.argsort(-gap)[0]
    sk=sk0+g1 
    idsk = np.argsort(-eigvals)[:sk]  
    k_biggest_eigenvectors = normalize(np.real(V[:, idsk])) 
    labels=KMeans(n_clusters=int(sk)).fit_predict(k_biggest_eigenvectors)
    sk=len(np.unique(labels))
    return sk, idx, labels


################################## Individual tree parameters ###########################################
def Parameter(C,labels,total):
    indices = np.argsort(labels)
    labels=labels[indices]
    C=C[indices, :]
    subid=[]
    for i in range(total):
        subid.extend([labels.tolist().index(i)])  
    bio=[]
    for j in range(total):
        if j==(total-1):
            final_subX=C[subid[j]:,:]
        else:
            final_subX=C[subid[j]:subid[j+1],:]
        index_xmin=np.lexsort([final_subX[:,0]])[0]  
        index_xmax=np.lexsort([-final_subX[:,0]])[0] 
        index_ymin=np.lexsort([final_subX[:,1]])[0]  
        index_ymax=np.lexsort([-final_subX[:,1]])[0] 
        index_zmax=np.lexsort([-final_subX[:,2]])[0] 
        x=final_subX[index_zmax,0]
        y=final_subX[index_zmax,1]
        xmin=final_subX[index_xmin,0]
        xmax=final_subX[index_xmax,0]
        ymin=final_subX[index_ymin,1]
        ymax=final_subX[index_ymax,1]
        bio.append([j,x,y,(xmax-xmin+ymax-ymin)/4,final_subX[index_zmax,2]]) 

    return bio
    

########################### Voxelization & calling NystromSpectralClustering #############################
def VoxelNystromSC(P,xid,gap,path):
    zd=6
    a=np.array([1,1,zd])
    P=P/a
    nP=np.shape(P)[0]  
    x0=P[np.lexsort([P[:,0]])[0],0]  # min value
    x1=P[np.lexsort([-P[:,0]])[0],0]  # max value
    y0=P[np.lexsort([P[:,1]])[0],1]
    y1=P[np.lexsort([-P[:,1]])[0],1]
    den0=round(nP/((x1-x0)*(y1-y0)))  # density of point cloud
    bandwidth = estimate_bandwidth(P, quantile=den0/nP)
    # returns N cluster centers from M points, where N = 2601, M = 25542
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(P)
    labels = ms.labels_  
    
    cluster_centers = ms.cluster_centers_
    # stack xyz of voxel with number of points that each voxel contains, 2601 voxels in total.
    cluster_centers=np.hstack((cluster_centers,[[labels.tolist().count(i)] for i in range(np.shape(cluster_centers)[0])]))

    [k,mlabels,Slabels]=SpectralNystrom(cluster_centers,gap)  
    idclu=np.argsort(mlabels)  
    Slabels=Slabels[idclu]  
    for i in range(nP):
        j=labels[i]
        labels[i]=Slabels[j]
    del cluster_centers,Slabels,mlabels
    gc.collect()
    P=P*a 
    PP=np.column_stack((P,labels))
    PP0=PP[np.lexsort(PP.T)]
    SSbio=Parameter(PP0[:,:3],PP0[:,3],k)  
    SSbio=np.array(SSbio)
   
    out1=open(path+"\\results\\Data_seg_%s.csv"%(xid),'w',newline='\n')
    csv_write1=csv.writer(out1,dialect='excel')
    csv_write1.writerow(('x','y','z','label'))
    for i in range(np.shape(PP0)[0]):
        csv_write1.writerow((PP0[i,0],PP0[i,1],PP0[i,2],PP0[i,3]))
       
    out2=open(path+"\\results\\Parameter_%s.csv"%(xid),'w',newline='\n')
    csv_write2=csv.writer(out2,dialect='excel')
    csv_write2.writerow(('TreeID','Position_X','Position_Y','Crown','Height'))
    for i in range(np.shape(SSbio)[0]):
        csv_write2.writerow((SSbio[i,0], SSbio[i,1], SSbio[i,2], SSbio[i,3], SSbio[i,4]))



   
    
