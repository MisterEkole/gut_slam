import numpy as np

class WarpField:
    def __init__(self, mesh_vertices, control_points):
        self.mesh_vertices = mesh_vertices
        self.control_points = control_points
        self.num_vertices = len(mesh_vertices)
        self.num_control_points = len(control_points)
        self.displacement_vectors = np.zeros((self.num_control_points, 3))

    def deform_mesh(self):
        deformed_mesh = np.zeros_like(self.mesh_vertices)
        for i in range(self.num_vertices):
            weighted_sum = np.zeros(3)
            total_weight = 0
            for j in range(self.num_control_points):
                distance = np.linalg.norm(self.mesh_vertices[i] - self.control_points[j])
                weight = self.weight_function(distance)
                weighted_sum += weight * self.displacement_vectors[j]
                total_weight += weight
            if total_weight > 0:
                deformed_mesh[i] = self.mesh_vertices[i] + weighted_sum / total_weight
            else:
                deformed_mesh[i] = self.mesh_vertices[i]
        return deformed_mesh

    def weight_function(self, distance):
        # Example: Linear falloff
        max_distance = 1.0
        if distance <= max_distance:
            return 1 - distance / max_distance
        else:
            return 0

if __name__ == "__main__":
    
    mesh_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    control_points = np.array([[0.2, 0.2, 0], [0.8, 0.2, 0]])
    
    warp_field = WarpField(mesh_vertices, control_points)
    
    # Assigning example displacement vectors
    warp_field.displacement_vectors[0] = np.array([0.1, 0.1, 0])
    warp_field.displacement_vectors[1] = np.array([-0.1, 0.1, 0])
    
    deformed_mesh = warp_field.deform_mesh()
    print("Deformed Mesh:")
    print(deformed_mesh)
