import numpy as np

def rotation_matrix_check(rotation_matrix):

    dot_product = np.dot(rotation_matrix, rotation_matrix.T)
    
  
    result_matrix = np.identity(3) - dot_product
    
   
    norm_result = np.linalg.norm(result_matrix)
    
    return norm_result


def rotation_matrix_determinant_check(rotation_matrix):
    # Calculate the determinant of the rotation matrix
    determinant = np.linalg.det(rotation_matrix)
    
    # Check if the determinant is equal to zero (within a tolerance due to floating point errors)
    is_zero = np.isclose(determinant, 0)
    
    return determinant, is_zero

# Call the function with the example rotation matrix





# rot_mat=np.array([[0.63304478,  0.70946388,  2.3855624],   
#                         [2.67578883  ,1.4228946  , 0.91158914],
#                         [1.2113317 , -0.44226942  ,8.65885199]])

rot_mat=np.array( [[-9.99998409e-01 , 8.66219176e-07, -8.39755961e-07],
 [ 1.35626010e-06,  9.99999516e-01 , 1.46209188e-06],
 [ 1.98137547e-08 , 1.17547681e-06 , 1.00000055e+00]])




determinant, is_zero = rotation_matrix_determinant_check(rot_mat)
norm_of_result = rotation_matrix_check(rot_mat)
print(norm_of_result)
print(determinant, is_zero)
