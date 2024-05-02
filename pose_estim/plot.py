import numpy as np
import matplotlib.pyplot as plt
def read_error_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    frames = []
    for i in range(0, len(lines), 13):  # Assuming each frame consists of 13 lines
        frame_info = {
            'reprojection_errors': [],
            'photometric_errors': []
        }
        for j in range(2, 12):  # Starting from line 2 to line 11 for error values
            line_index = i + j
            if line_index >= len(lines):
                print("IndexError: Line index out of range")
                print("Expected index:", line_index)
                break
            if not lines[line_index].strip():  # Skip empty lines
                continue
            if "Reprojection Error" not in lines[line_index] and "Photometric Error" not in lines[line_index]:
                print("Warning: Line does not contain expected format")
                print("Line:", lines[line_index])
                continue
            try:
                iteration_info = lines[line_index].strip().split(': ')[1].split(', ')
                frame_info['reprojection_errors'].append(float(iteration_info[0]))
                frame_info['photometric_errors'].append(float(iteration_info[1]))
            except IndexError:
                print("IndexError: Line structure does not match expected format")
                print("Line:", lines[line_index])
        frames.append(frame_info)
    return frames


def plot_errors(frames):
    num_frames = len(frames)
    for idx, frame in enumerate(frames):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, 11), frame['reprojection_errors'], marker='o')
        plt.title(f'Frame {idx}')
        plt.xlabel('Iteration')
        plt.ylabel('Reprojection Error')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, 11), frame['photometric_errors'], marker='o')
        plt.title(f'Frame {idx}')
        plt.xlabel('Iteration')
        plt.ylabel('Photometric Error')
        
        plt.tight_layout()
        plt.show()

def plot_total_errors(frames):
    total_reprojection_errors = []
    total_photometric_errors = []
    for frame in frames:
        total_reprojection_errors.append(np.mean(frame['reprojection_errors']))
        total_photometric_errors.append(np.mean(frame['photometric_errors']))
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(frames)+1), total_reprojection_errors, marker='o')
    plt.title('Total Mean Reprojection Error')
    plt.xlabel('Total Number of Frames')
    plt.ylabel('Mean Reprojection Error')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(frames)+1), total_photometric_errors, marker='o')
    plt.title('Total Mean Photometric Error')
    plt.xlabel('Total Number of Frames')
    plt.ylabel('Mean Photometric Error')
    
    plt.tight_layout()
    plt.show()

# Example usage:
file_path = '/Users/ekole/Dev/gut_slam/pose_estim/optimization_errors_all_frames.txt'  # Change this to your file path
frames = read_error_log(file_path)
plot_errors(frames)
plot_total_errors(frames)
