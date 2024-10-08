from mpi4py import MPI
import numpy as np
import time

start_time = time.time()


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


################################################################################
nt = 6
RVE_size = 200.0
total_volume = RVE_size**3
applied_strain = 1.0e-4
##########################Fiber-Properties######################################
E = 79.2e9;
mu = 0.1;
################################################################################


# Only the root process reads the strain data files
if rank == 0:
    readfile_start_time = time.time()
    for n in range(nt):
        ns = str(n+1)
        file_strain = f'strain_centroid_test{ns}.txt'
        with open(file_strain, 'r') as strain_data:
            straindata_o = [[float(num) for num in line.split(' ')] for line in strain_data]

        straindata = np.array(straindata_o)
        nelems = straindata.shape[0]

        if n == 0:
            file_volume = f'cell_volume_test{ns}.txt'
            with open(file_volume, 'r') as volume_data:
                Ve_o = [[float(num) for num in line.split(' ')] for line in volume_data]
                Ve = np.array(Ve_o)
            strain_xx = np.zeros((nelems, nt))
            strain_xy = np.zeros((nelems, nt))
            strain_xz = np.zeros((nelems, nt))
            strain_yy = np.zeros((nelems, nt))
            strain_yz = np.zeros((nelems, nt))
            strain_zz = np.zeros((nelems, nt))

      
        strain_xx[:, n] = straindata[:, 0]
        strain_xy[:, n] = straindata[:, 1]
        strain_xz[:, n] = straindata[:, 2]
        strain_yy[:, n] = straindata[:, 4]
        strain_yz[:, n] = straindata[:, 5]
        strain_zz[:, n] = straindata[:, 8]
    #np.savetxt(f'original_strain_xx_processor{rank}.txt', strain_xx)
    readfile_end_time = time.time()
    print(f"Readfile time: {readfile_end_time - readfile_start_time} seconds")
    
#comm.Barrier()

if rank == 0:
    
    #Split the matrix into a list of rows
    Ve_split = np.array_split(Ve, size, axis=0)
    strain_xx_split = np.array_split(strain_xx, size, axis=0)
    strain_xy_split = np.array_split(strain_xy, size, axis=0)
    strain_xz_split = np.array_split(strain_xz, size, axis=0)
    strain_yy_split = np.array_split(strain_yy, size, axis=0)
    strain_yz_split = np.array_split(strain_yz, size, axis=0)
    strain_zz_split = np.array_split(strain_zz, size, axis=0)
    
    m_strain_xx = np.zeros((1, 6));
    m_strain_yy = np.zeros((1, 6));
    m_strain_zz = np.zeros((1, 6));
    m_strain_yz = np.zeros((1, 6));
    m_strain_xz = np.zeros((1, 6));
    m_strain_xy = np.zeros((1, 6));

    m_strain_xx[0,0] = applied_strain
    m_strain_yy[0,1] = applied_strain
    m_strain_zz[0,2] = applied_strain
    m_strain_yz[0,3] = applied_strain
    m_strain_xz[0,4] = applied_strain
    m_strain_xy[0,5] = applied_strain
    
    m_strain_combined = [m_strain_xx, m_strain_yy, m_strain_zz, m_strain_yz, m_strain_xz, m_strain_xy]

else:
    Ve_split, strain_xx_split, strain_xy_split, strain_xz_split, strain_yy_split, strain_yz_split, strain_zz_split, m_strain_combined = (None, None, None, None, None, None, None, None)



# Scatter the data among all processes
Ve_sub = comm.scatter(Ve_split, root=0)
strain_xx_sub = comm.scatter(strain_xx_split, root=0)
strain_xy_sub = comm.scatter(strain_xy_split, root=0)
strain_xz_sub = comm.scatter(strain_xz_split, root=0)
strain_yy_sub = comm.scatter(strain_yy_split, root=0)
strain_yz_sub = comm.scatter(strain_yz_split, root=0)
strain_zz_sub = comm.scatter(strain_zz_split, root=0)
#np.savetxt(f'strain_xx_processor{rank}.txt', strain_xx_sub)
#np.savetxt(f'Ve_processor{rank}.txt', Ve_sub)

# Broadcast the combined m_strain to all processes
m_strain_combined = comm.bcast(m_strain_combined, root=0)
m_strain_xx, m_strain_yy, m_strain_zz, m_strain_yz, m_strain_xz, m_strain_xy = m_strain_combined[0], m_strain_combined[1], m_strain_combined[2], m_strain_combined[3], m_strain_combined[4], m_strain_combined[5]


#post-processing at each processor
nelems_sub = strain_xx_sub.shape[0]
m_M_sub = np.zeros((1, 36))
e_cal = np.zeros((36, 1))
e_ave = np.zeros((36, 36))


for i in range(nelems_sub):
    for j in range(nt):
        e_cal[j*6 , 0]    = strain_xx_sub[i, j]
        e_cal[j*6 + 1, 0] = strain_yy_sub[i, j]
        e_cal[j*6 + 2, 0] = strain_zz_sub[i, j]
        e_cal[j*6 + 3, 0] = strain_yz_sub[i, j] * 2
        e_cal[j*6 + 4, 0] = strain_xz_sub[i, j] * 2
        e_cal[j*6 + 5, 0] = strain_xy_sub[i, j] * 2
        #set up the averaged strain matrix
        for k in range(6):
            row_index = j * 6 + k
            e_ave[row_index, k * 6] = m_strain_xx[0,j]
            e_ave[row_index, k * 6 + 1] = m_strain_yy[0,j]
            e_ave[row_index, k * 6 + 2] = m_strain_zz[0,j]
            e_ave[row_index, k * 6 + 3] = m_strain_yz[0,j]
            e_ave[row_index, k * 6 + 4] = m_strain_xz[0,j]
            e_ave[row_index, k * 6 + 5] = m_strain_xy[0,j]

    #solve for the localization matrix
    e_ave_inv = np.linalg.inv(e_ave)
    Me = np.matmul(e_ave_inv, e_cal)
    m_M_sub +=  Me.T * Ve_sub[i, 0] / total_volume
    
 
# Gather all results of m_M_sub in the root process
m_M = comm.reduce(m_M_sub, op=MPI.SUM, root=0)


if rank == 0:
    M_ij = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            M_ij[i, j] = m_M[0, i * 6 + j]

    # Calculate the effective stiffness matrix
    Del = E / ((1 + mu) * (1 - 2 * mu))
    C11 = (1 - mu) * Del
    C22 = (1 - mu) * Del
    C33 = (1 - mu) * Del
    C44 = 1 / 2 * (1 - 2 * mu) * Del
    C55 = 1 / 2 * (1 - 2 * mu) * Del
    C66 = 1 / 2 * (1 - 2 * mu) * Del
    C12 = mu * Del
    C13 = mu * Del
    C23 = mu * Del

    C0 = np.array([[C11,  C12,  C13,   0.,     0.,      0.],
                  [C12,  C22,  C23,    0.,     0.,      0.],
                  [C13,  C23,  C33,    0.,     0.,      0.],
                  [0.,    0.,   0.,    C44,    0.,      0.],
                  [0.,    0.,   0.,    0.,     C55,     0.],
                  [0.,    0.,   0.,    0.,     0.,      C66]])

    Ce = np.matmul(C0, M_ij)
    V_fiber = np.sum(Ve, axis=0)
    Vf = V_fiber / total_volume
    ov = Ce.reshape(1, 36)

    with open("effective_stiffness_200um.txt", "a") as outfile:
        np.savetxt(outfile, ov, fmt='%1.4f')

    print('-------------------C0-------------------')
    print(C0)
    print('----------------------------------------')

    print('-------------------Ce-------------------')
    print(Ce)
    print('----------------------------------------')

    print('-------------------Vf-------------------')
    print(Vf)
    print('----------------------------------------')

    end_time = time.time()
    print(f"Computation time: {end_time - start_time} seconds")

