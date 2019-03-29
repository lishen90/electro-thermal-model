# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:54:14 2019

@author: sli8
"""
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import numpy as np
import math as m
import matplotlib.pyplot as plt
from mayavi import mlab


#########################################################   
################## function for matrix1 #################
#########################################################
def fun_matrix1():
    node=np.arange(1, ntotal+1)
    an_unit=np.linspace(1,nx,nx,dtype=int)     #repeating node number for an
    ra_unit=np.linspace(1,nra,nra,dtype=int)   #repeating node number for ra
    ax_unit=np.linspace(1,ny,ny,dtype=int)     #repeating node number for ax
    an=np.tile(an_unit,nra*ny) 
    ra=np.repeat(ra_unit,nx); ra=np.tile(ra,ny)
    ax=np.repeat(ax_unit,nx*nra)        

    lap=(ra-1)//((nRC+2)*2)+1
    theta=2*m.pi*(lap-1)+2*m.pi*(an-1)/nx
    mat_unit=np.array([1,3,2,4])                      #repeating node number for mat
    mat_tmp1=np.repeat(mat_unit,[nx,ne*nx,nx,ne*nx])  #within one lap Al,El_b,Cu,El_r [1,3,2,4]  ax=1
    mat_tmp2=np.tile(mat_tmp1,nlap)                   #within two laps Al,El_b,Cu,El_r [1,3,2,4,1,3,2,4]  ax=1
    mat_tmp3=mat_tmp2[:(nra*nx)]                      #within two laps after trimming Al,El_b,Cu,El_r [1,3,2,4,1,3,2]  ax=1
    mat=np.tile(mat_tmp3,ny)

    ind1=np.where(mat==1); ind2=np.where(mat==2); ind3=np.where(mat==3); ind4=np.where(mat==4)   #for case of 192 nodes, ind from 0 to 191

    xi=np.zeros(ntotal)
    xi[ind1]=spiralfrom0(b01,theta[ind1])    
    xi[ind2]=spiralfrom0(b02,theta[ind2])
    xi[ind3]=spiralfrom0(b03,theta[ind3])
    xi[ind4]=spiralfrom0(b04,theta[ind4])
 
    zi=np.zeros(ntotal)
    zi[ind1]=a0*theta[ind1]+b01
    zi[ind2]=a0*theta[ind2]+b02
    zi[ind3]=a0*theta[ind3]+b03
    zi[ind4]=a0*theta[ind4]+b04

    if ny==1:
        yi=np.zeros(ntotal)
    else:
        yi=(ax-1)*LG/(ny-1)

    node1=node.astype(float)
#    for i in (node-1):                                     #for node neighbors in thermal model 
#        if (ra[i]%(ne+1)!=1) and (ra[i]%(ne+1)!=2):
#            node1[i]=None

    jx1=np.zeros(ntotal,dtype=float)            #initialize left-neighbor node number in x direction
    jx2=np.zeros(ntotal,dtype=float)            #initialize right-neighbor node number in x direction
    jy1=np.zeros(ntotal,dtype=float)            #initialize up-neighbor node number in y direction
    jy2=np.zeros(ntotal,dtype=float)            #initialize down-neighbor node number in y direction
    jz1=np.zeros(ntotal,dtype=float)            #initialize inner-neighbor node number in z direction
    jz2=np.zeros(ntotal,dtype=float)            #initialize outer-neighbor node number in z direction
    

    ############################# neighbor node number #################################
    for i in (node-1):
        if nx==1:                                     #for lumped model(nx=1), all nodes in line, no left neighbor
            jx1[i]=None        
        elif an[i]==1:
            if node1[i] <= (ne*2+2)*nx:                               
                jx1[i]=None                           #for node[i] with an==1, starting mndoes, no left-neighbor number jx1[i]
            else:
                jx1[i]=node1[i] - ((ne*2+2)*nx-nx+1)  
        else:
            jx1[i]=node1[i]-1                         #for node[i], left-neighbor number jx1[i] is node[i]-1

    for i in (node-1):
        if nx==1:                                     #for lumped model(nx=1), all nodes in line, no right neighbor
            jx2[i]=None
        elif an[i]==nx:
            if node1[i] >= nx+((2*nlap-1)*ne+2*nlap-1)*nx: #for node[i] with an==4, ending nodes, no right-neighbor number jx2[i]
                jx2[i]=None
            else:
                jx2[i]=node1[i] + ((ne*2+2)*nx-nx+1)            
        else:
            jx2[i]=node1[i]+1                         #for node[i], right-neighbor number jx2[i] is node[i]+1

    for i in (node-1):
        if ax[i]==1:
            jy1[i]=None                               #for node[i] in the top layer, no up-neighbor number jy1[i] 
        else:
            jy1[i]=node1[i]-nx*nra                    
    for i in (node-1):
        if ax[i]==ny:
            jy2[i]=None                               #for node[i] in the bottom layer, no down-neighbor number jy2[i]
        else:
            jy2[i]=node1[i]+nx*nra
    
    for i in (node-1):
        if ra[i]==1:
            jz1[i]=None                               #for node[i] in the innermost lap, no inner-neighbor number jz1[i]
        else:
            jz1[i]=node1[i]-nx
    for i in (node-1):
        if ra[i]==nra:
            jz2[i]=None
        else:
            jz2[i]=node1[i]+nx                        #for node[i] in the outermost lap, no outer-neighbor number jz2[i]

    ############################# volume #################################
    V_ele_lib=np.zeros([2*nlap-1,nx]) #volume library, for example nx=4,ny=3,nlap=2 there are 10 kinds of volume, which is a library
    for i0 in np.arange(2*nlap-1):    #i0: 0index of rows in library array, for example, 0,1,2
        for j0 in np.arange(nx):      #j0: 0index 0f column in library array, for example, 0,1,2,3
            if not( (i0==2*nlap-3 and j0==nx-1) or (i0==2*nlap-2 and j0==nx-1) ):  #There shouldn't be last two volumes (ending volumes in each layer); if not the two volumes case, do the following       
                node_i0=i0*(nx*(nRC+2))+j0  #node_i0: 0index of node, described in i0, j0
                if i0//2==0 and an[node_i0]!=nx:            #if ELb case, use b03; if ending node in angular direction, for example in nx=4,ny=3,nlap=2 case, node 4 connects node 41
                    V_ele_lib[i0,j0]=( spiralfrom0(b03,theta[node_i0+1])-spiralfrom0(b03,theta[node_i0]) ) * (LG/(ny-1)) * delta_El
                elif i0//2==0 and an[node_i0]==nx:          #if ELb case, use b03; if other nodes, for example node 1,2,3
                    V_ele_lib[i0,j0]=( spiralfrom0(b03,theta[node_i0+nx*2*(nRC+2)-nx+1])-spiralfrom0(b03,theta[node_i0]) ) * (LG/(ny-1)) * delta_El
                elif i0//2==1 and an[node_i0]!=nx:          #if ELr case, use b04
                    V_ele_lib[i0,j0]=( spiralfrom0(b04,theta[node_i0+1])-spiralfrom0(b04,theta[node_i0]) ) * (LG/(ny-1)) * delta_El
                else:
                    V_ele_lib[i0,j0]=( spiralfrom0(b04,theta[node_i0+nx*2*(nRC+2)-nx+1])-spiralfrom0(b04,theta[node_i0]) ) * (LG/(ny-1)) * delta_El
            else:
                V_ele_lib[i0,j0]=None    
    #the above gets the volume library
    V_ele=np.zeros(ntotal)
    for i0 in node-1:
        if mat[i0]<=2 and not(ra[i0]==nra) and not(an[i0] == nx and ra[i0] >= nra-2*(nRC+2) ) :   #the node is on CC, not outermost node 61,62,63,64, not node 24, 44
            i00=ra[i0]//(nRC+2); j00=an[i0]-1    #i00 is the row 0index, for example, 0,1,2; j00 is the column 0index, for example, 0,1,2,3
            V_ele[i0]=V_ele_lib[i00,j00]
        else:
            V_ele[i0]=None
        ############################################################## to be deleted, this section is for assigning the node 24,44 volume the same as node 23,43
        if an[i0] == nx and ra[i0] >= nra-2*(nRC+2):   #if node 24,44
            V_ele[i0]=V_ele[i0-1]
        ##############################################################
                
    
    return node, ax, ra, an, lap, theta, mat, xi, yi, zi, jx1, jx2, jy1, jy2, jz1, jz2, V_ele 
#####################################################################################
#####################################################################################    
#########################################################   
############ function for node visualization ############
#########################################################
def fun_vln_node(yi, zi, theta, node):
    X=-zi*np.sin(theta); Y=zi*np.cos(theta); Z=LG-yi        #Cartesian location of node
    
    points = mlab.points3d(X, Y, Z, colormap="flag", scale_factor=0.2)    #ploting node
    mlab.points3d(0, 0, 0, scale_factor=0.5); mlab.text3d(0, 0, 0, 'Origin', scale=0.3)                              #ploting origin
    colors = np.zeros(ntotal)
    
    points.glyph.scale_mode = 'scale_by_vector'
    points.mlab_source.dataset.point_data.scalars = colors
    colors[np.where(mat==1)]=0.0157; colors[np.where(mat==2)]=0.0052; colors[np.where(mat==3)]=0.2251; colors[np.where(mat==4)]=0.8848  #values are grey, orange, blue and red from colormap "flag"
    node_plot=node.astype('str')
    for i in (node-1):
        mlab.text3d(X[i], Y[i], Z[i], node_plot[i], scale=0.2)
    mlab.show()
#####################################################################################
#####################################################################################
#########################################################   
######## function for spiral length from origin #########
#########################################################
#input 1. b0 from spiral equation (b0 can be b01-Al, b02-Cu, b03-Elb, b04-Elr) and 2. theta, output spiral length starting from original
def spiralfrom0(b0_local,theta_local):
    L = (a0*theta_local+b0_local)*np.sqrt(a0**2+b0_local**2+2*a0*b0_local*theta_local+a0**2*theta_local**2)/2/a0 \
         + a0/2*np.log(theta_local+b0_local/a0+np.sqrt((theta_local+b0_local/a0)**2+1)) \
         -np.sqrt(a0**2+b0_local**2)*b0_local/2/a0- a0/2*np.log(b0_local/a0+np.sqrt((b0_local/a0)**2+1))
    return L     
#####################################################################################
#####################################################################################   
#########################################################   
#########       function for element volume      #########
##########################################################
##output array V_ele
#def fun_V_ele():
#    V_ele=np.zeros([2*nlap-1,nx])
#    for i0 in np.arange(2*nlap-1):
#        for j0 in np.arange(nx):
#            V_ele[i0,j0]=spiralfrom0(b03,)
#            
#    return V_ele     
######################################################################################
######################################################################################     
#########################################################   
################## function for MatrixC #################
#########################################################
def fun_matrixC():
    global nECN, R0_pair, RC_pair, Ei_pair
    MatrixC1=np.zeros([ntotal,ntotal])
    temp1=np.arange(0,nRC+2); temp2=np.roll(temp1,-1); temp3=np.append(temp1,temp2)
    ECN_pair_lib=np.reshape(temp3,(2,nRC+2)).T    #for 3 RCs, node link cases: array([[0,1],[1,2],[2,3],[3,4],[4,0]]) for R0, R1, R2, R3 and Ei respectively
    nECN=(2*nlap-1)*nx*ny   #number of lumped ECN, 36     
    R0_pair=np.zeros([nECN,3]); counter0=0             #R0 case: two linking nodes (0index). For example, [[0,4,R0],[1,5,R0],[2,6,R0],[3,7,R0],[20,25,R0]...]
                                                       #means node1,5 is R0; node2,6 is R0; node3,7 is R0; node4,8 is R0; node21,26 is R0
                                                       #Note: first number is no larger than second number: [node1,node2,R0], node1 <= node2
    RC_pair=np.zeros([nRC*nECN,4]); counter1=0         #RC case: two linking nodes (0index). For example, [[4,8,R1,C1],[5,9,R1,C1],[6,10,R1,C1],[7,11,R1,C1],[8,12,R2,C2]...]
                                                       #means node5,9 is R1,C1; node6,10 is R1,C1; node7,11 is R1,C1; node8,12 is R1,C1; node9,13 is R2,C2   
                                                       #Note: first number is no larger than second number: [node1,node2,R1,C1], node1 <= node2                                             
    Ei_pair=np.zeros([nECN,3]); counter2=0             #Ei case: two linking nodes (0index). For example, [[16,20,Ei],[17,21,Ei],[18,22,Ei],[19,23,Ei],[36,40,Ei]...]
                                                       #means node17,21 is Ei; node18,22 is Ei; node19,23 is Ei; node20,24 is Ei; node37,41 is Ei
                                                       #Note: first number is no larger than second number: [node1,node2,Ei], node1 <= node2
    #######################getting MatrixC1#######################
    for i0 in (node-1):      
        for j0 in (node-1):
            isneighbor = ((node[i0]==jx1[j0]) or (node[i0]==jx2[j0]) or (node[i0]==jy1[j0]) or (node[i0]==jy2[j0]) or (node[i0]==jz1[j0]) or (node[i0]==jz2[j0]))

            if i0<j0 and isneighbor:        # for elements in upper triangle and not zero
                if mat[i0]==mat[j0] and (mat[i0]==1 or mat[i0]==2):    #i. RAl or RCu case           
                    if mat[i0]==1:
                        b=b01; delta=delta_Al; Conductivity=Conductivity_Al
                    if mat[i0]==2:
                        b=b02; delta=delta_Cu; Conductivity=Conductivity_Cu
                    
                    if yi[i0]==yi[j0]:                     #1. horizontal resistance     yi[j0]>yi[i0] because j0 is larger than i0
                        L=spiralfrom0(b,theta[j0]) - spiralfrom0(b,theta[i0])    
                        A=delta*LG/(ny-1)
                    else:                                  #2. vertical resistance     yi[j0]>yi[i0] because j0 is larger than i0
                        L=LG/(ny-1)
                        if an[i0] != nx:
                            A=delta*(spiralfrom0(b,theta[i0+1]) - spiralfrom0(b,theta[i0]))
                        else:
                            if ra[i0] <= nra-2*(nRC+2):    #for vertical resistance of node4,68 and node24,88,
                                A=delta*(spiralfrom0(b,theta[i0+nx*2*(nRC+2)-nx+1]) - spiralfrom0(b,theta[i0]))
                            else:                          #for vertical resistance of node44,108 and node64,128
                                A=delta*(spiralfrom0(b,theta[i0]) - spiralfrom0(b,theta[i0-1]))
                    R=L/A/Conductivity                    
                    #R=1      #this is for debug, later this should be removed: to let the CC have the same resistance 1, irrespective of CC volume difference                                             
                    MatrixC1[i0,j0]=1/R
                    
                elif an[i0]==an[j0] and ax[i0]==ax[j0]:                                                  #ii. Elb and Elr case
                    temp4=(ra[i0]-1)%(nRC+2); temp5=(ra[j0]-1)%(nRC+2)            #no. in lumped RCs. For example, temp3 and temp4 can be 0,1,2,3,4                                 
                    indRC=np.where((ECN_pair_lib==[temp4,temp5]).all(1))[0]             #find the row index of link case [temp1,temp2] in ECN_pair_lib    np.where((ECN_pair_lib==[temp4,temp5]).all(1)) returns tuple, so add [0] in the end
                    if indRC==0:        #1. case of R0
                        R=R0_LUT[0,0,0]/V_ele[i0]    #R0_LUT unit is (Ω·mm3)
                        MatrixC1[i0,j0]=1/R
                        R0_pair[counter0,0]=i0; R0_pair[counter0,1]=j0; R0_pair[counter0,2]=1/R; counter0=counter0+1
                    elif indRC==nRC+1:  #2. case of Ei
                        E=Ei_LUT[0]
                        Ei_pair[counter2,0]=i0; Ei_pair[counter2,1]=j0; Ei_pair[counter2,2]=E; counter2=counter2+1          #for later use in computing RectangleC2                        
                    else:               #3. case of R123 and C123            
                        Ri=Ri_LUT[indRC-1,0,0]/V_ele[i0-indRC*nx]; Ci=Ci_LUT[indRC-1,0,0] * V_ele[i0-indRC*nx]  #Ri_LUT unit is (Ω·mm3), Ci_LUT unit is (C/V/mm3)
                        MatrixC1[i0,j0]=1/Ri+Ci/dt  #Ri_LUT[indRC,0,0], Ci_LUT[indRC-1,0,0] for temperatory
                        #MatrixC[i0,j0]=1/Ri_LUT[indRC,indT,indSoC]+Ci_LUT[indRC-1,indT,indSoC]/dt  #Ri_LUT is (4,nT,nSoC), a 3 dimensional array
                        RC_pair[counter1,0]=i0; RC_pair[counter1,1]=j0; RC_pair[counter1,2]=Ri; RC_pair[counter1,3]=Ci; counter1=counter1+1                      

    MatrixC1=np.triu(MatrixC1,1).T + MatrixC1              #upper and lower
    np.fill_diagonal(MatrixC1, -np.sum(MatrixC1,axis=1))  #upper and diagonal
 
    #######################getting RectangleC2#######################       
    RectangleC2=np.zeros([nECN,ntotal+nECN])    
    for ii0 in np.arange(0,nECN):      #for row loop in RectangleC2     row number from small to large, lumped ECN from inner to outer, from top to bottom
        temp6=Ei_pair[ii0,0].astype(int); temp7=Ei_pair[ii0,1].astype(int)          #node 0-index in Ei_pair                   #note that i0 < j0        
        if mat[temp6]==3:              #1. case of Elb        
            RectangleC2[ii0,temp6]=1          #if the left node from the Ei pair is blue, the node is positive, put 1 in RecangleC2
            RectangleC2[ii0,temp7]=-1
        if mat[temp6]==4:              #2. case of Elr
            RectangleC2[ii0,temp6]=-1         #if the left node from the Ei pair is red, the node is negative, put -1 in RecangleC2
            RectangleC2[ii0,temp7]=1
    #######################  forming MatrixC  #######################
    temp8_downleft=np.hsplit(RectangleC2,[ntotal])[0]     #split RectangleC2 to form downleft array (36,192)
    temp9_left=np.vstack((MatrixC1,temp8_downleft))       #stack MatrixC1 and downleft array to form left array of MatrixC
    MatrixC=np.hstack((temp9_left,RectangleC2.T))
    MatrixC=np.delete(MatrixC,node_negative_0ind,0); MatrixC=np.delete(MatrixC,node_negative_0ind,1)   #delete row and column of negative node in MatrixC
    return MatrixC

#########################################################   
##################    function for I    #################
#########################################################
def fun_I():
    I_up=np.zeros([ntotal])
    for i0 in np.arange(nRC*nECN): #loop for all RC linking pairs
        i00=RC_pair[i0,0].astype(int); j00=RC_pair[i0,1].astype(int); Ci=RC_pair[i0,3]   #RC case: two linking nodes (0index) and their capacitance Ci
        I_up[i00]=I_up[i00]+(U1[j00]-U1[i00])*Ci/dt
        I_up[j00]=I_up[j00]+(U1[i00]-U1[j00])*Ci/dt
    I_down=Ei_pair[:,2]     #OCV terms in down part of I vector, getting directly from Ei_pair
    I=np.append(I_up,I_down)
    I[node_positive_0ind]=I[node_positive_0ind]+I_ext     #I_ext added for positive node
    I[node_negative_0ind]=I[node_negative_0ind]-I_ext     #I_ext added for negative node
    I=np.delete(I,node_negative_0ind)                     #delete element of negative node in VectorI
    return I
        
    


#########################################################   
##################    function for U0   #################
#########################################################
def fun_Uini():
    Uini=np.zeros([ntotal,1])
    for i0 in (node-1):
        if mat[i0]==1:        #Al
            Uini[i0]=E_cell
        if mat[i0]==2:        #Cu
            Uini[i0]=0
        if mat[i0]==3:        #Elb
            Uini[i0]=E_cell
        if mat[i0]==4:        #Elr
            Uini[i0]=0
    return Uini
#####################################################################################
#####################################################################################     





###     User inputs     ###                           

LG=65;                                            #axial length of jelly roll
#delta_Cu=0.62; delta_Al=0.78; delta_El=2.43       #thickness: Copper, Aluminium and Electrode
delta_Cu=0.021; delta_Al=0.021; delta_El=0.091       #thickness: Copper, Aluminium and Electrode
a0=(2*delta_El+delta_Cu+delta_Al)/2/m.pi; b0=0                                      #spiral geometry for center r=a0θ+b0       b0>=(3/2)delta_Cu or Al or Sep
#δAl+δCu+2δEl=2πa0   ∴δAl,δCu,δEl < 2πa0  These values are randomly set at present and need to be changed later
nx=4; ny=3; nlap=10                                #number of angular nodes, axial nodes and laps     initially nx=4, ny=3, nlap=2, nRC=3
nRC=0                                             #number of RC pairs
CC='No'                                              #current collector resistance consideration: 0 for no, 1 for yes
if ny==1:                      
    if nx==1:     #lumped model
        LG=0
    else:         #1-layer lumped model, ny=1, nx>1
        print('error: nx>1,ny=1 should not happen')
        raise Exception('exit')                   #exit the script
if nx==1 and not(ny==1 and nlap==1):
        print('error: nx=1 but not lumped is not ready. The code need further test')
        raise Exception('exit')                   #exit the script
#if lumped model, put nx=1, ny=1. nlap=1 and LG=0
 
E_cell=4.2                                        #cell voltage
Conductivity_Al=6.0e4; Conductivity_Cu=3.5e4        #electrical conductivity
Lamda_El=1; rou_El=1; c_El=1                      #thermal conductivity, density and heat capacity
Lamda_Cu=1; rou_Cu=1; c_Cu=1
Lamda_Al=1; rou_Al=1; c_Al=1
nT=3; nSoC=10                   #temperature numbers and SoC numbers in LUT
R0_LUT=0.0163*100*140*42*0.091*np.ones([1,nT,nSoC]); Ri_LUT=0.0020*100*140*42*0.091*np.ones([nRC,nT,nSoC]); Ci_LUT=5e3/100/140/42/0.091*np.ones([nRC,nT,nSoC]); Ei_LUT=E_cell*np.ones(nSoC)    #LUT for R, C and Ei
dt=1; nt=5
### parameters for later use ### 
ne=nRC+1                            #number of nodes in lumped ECN
nra=2*nlap+ne*(2*nlap-1)            #number of nodes in radial direction
ntotal=nx*ny*nra
b01=b0-delta_El/2-delta_Al/2; b02=b0+delta_El/2+delta_Al/2; b03=b0; b04=b0+delta_El+delta_Cu

#####################################
I_ext=1                       #discharge current
node_positive_0ind=0          #positive node 0index
node_negative_0ind=ntotal-1   #negative node oindex
###   getting Matrix1   ###
Matrix1=fun_matrix1()
node=Matrix1[0]; ax=Matrix1[1];  ra=Matrix1[2];  an=Matrix1[3];    lap=Matrix1[4];   theta=Matrix1[5];  mat=Matrix1[6]
xi=Matrix1[7];   yi=Matrix1[8];  zi=Matrix1[9];  jx1=Matrix1[10];  jx2=Matrix1[11];  jy1=Matrix1[12];   jy2=Matrix1[13]; jz1=Matrix1[14]; jz2=Matrix1[15]; V_ele=Matrix1[16]
######################################

###    visualization    ###
#fun_vln_node(yi, zi, theta, node)
######################################

### getting initial voltage potential, MatrixC and I ###
Uini=fun_Uini()
U_node1_plot=np.zeros([nt+1])
U_node1_plot[0]=Uini[0]
t_plot=dt*np.arange(0,nt+1)    #contain the t=0 point
#--------------------------------------------------------------------------
if CC=='No':      #if CC resistance is not considered, find the CC node 0index 
    indAlCu_temp=np.where(mat<=2)   #find current collector nodes (where mat==1 and mat==2)
    indAlCu_temp_pos=np.where(indAlCu_temp[0]==node_positive_0ind);  indAlCu_temp_neg=np.where(indAlCu_temp[0]==node_negative_0ind)
    indAlCu_del=np.append(indAlCu_temp_pos,indAlCu_temp_neg)    
    indAlCu=np.delete(indAlCu_temp,indAlCu_del)        #delete the positive and negative node from current collector nodes, indAlCu is 0index of the CC nodes
#--------------------------------------------------------------------------
for tn in np.arange(1,nt+1):   #tn: 1,2...nt   loop for time
    print('run tn=',tn)
    if tn==1:       #first time, initial voltage potential
        U1=Uini
    MatrixC=fun_matrixC()
    VectorI=fun_I()
                                                        
    if CC=='No':   #if CC resistance is not considered, modify the MatrixC and VectorI for the CC node 0index
        #---------------------------------------------------------------------
        for i0 in indAlCu:     #i0 is the CC 0index and also row for MatrixC
            MatrixC[i0]=0
            if mat[i0]==1:    #the CC node is positive
                MatrixC[i0,node_positive_0ind]=1; MatrixC[i0,i0]=-1
            if mat[i0]==2:    #the CC node is negative
                MatrixC[i0,i0]=-1
            VectorI[i0]=0
        MatrixC[node_positive_0ind]=0; MatrixC[node_positive_0ind,-nECN:]=1      #for positive node, i1+i2+...+inECN - I0 = 0
        VectorI[node_positive_0ind]=I_ext
        #---------------------------------------------------------------------
                    
    ######################################       
    U2=np.linalg.solve(MatrixC,VectorI)
    U_node1_plot[tn]=U2[0]
    U1=U2
    
plt.plot(t_plot,U_node1_plot,'ro')
plt.show()
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    