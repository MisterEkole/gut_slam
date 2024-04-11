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

rot_mat=np.array( [[-1.00032827e+00,  4.20239999e-05 , 1.92243865e-03],
 [ 6.69479845e-05,  1.00283498e+00, -7.39913499e-05],
 [ 3.09823499e-03,  4.59766712e-05 , 1.00268162e+00]])

determinant, is_zero = rotation_matrix_determinant_check(rot_mat)
norm_of_result = rotation_matrix_check(rot_mat)
print(norm_of_result)
print(determinant, is_zero)
