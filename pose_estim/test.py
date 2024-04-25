import numpy as np

def rotation_matrix_check(rotation_matrix):

    dot_product = np.dot(rotation_matrix, rotation_matrix.T)
    
  
    result_matrix = np.identity(3) - dot_product #closer the dot product mat is to zero the more orthogonal it is
    
   
    norm_result = np.linalg.norm(result_matrix)  #perfectly orthogonal if norm is zero
    
    return norm_result


def rotation_matrix_determinant_check(rotation_matrix):
    # Calculate the determinant of the rotation matrix
    determinant = np.linalg.det(rotation_matrix)
    
    # Check if the determinant is close to 1 (within a tolerance due to floating point errors)
    is_one = np.isclose(determinant, 1)
    
    return determinant, is_one


rot_mat=np.array( [[ 9.97933092e-01 , 1.27666055e-05, -6.42944603e-02],
 [ 1.21110879e-05, -1.00000496e+00 ,-1.02519772e-05],
 [-6.42943461e-02,  9.47834490e-06, -9.97932952e-01]])




determinant, is_zero = rotation_matrix_determinant_check(rot_mat)
norm_of_result = rotation_matrix_check(rot_mat)
print(norm_of_result)
print(determinant, is_zero)
