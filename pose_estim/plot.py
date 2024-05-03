'''
-------------------------------------------------------------
Utility script to parse and plot error data from a text file.

Author: Mitterand Ekole
Date: 043-05-2024
-------------------------------------------------------------
'''
import matplotlib.pyplot as plt

def parse_error_data(filepath):
    data = {}
    mean_errors = {'reprojection': [], 'photometric': []}
    with open(filepath, 'r') as file:
        frame_data = []
        current_frame = None
        for line in file:
            line = line.strip()  # Ensure no leading/trailing whitespace
            if 'Frame' in line:
                if current_frame is not None:
                    data[current_frame] = frame_data
                current_frame = int(line.split()[1])
                frame_data = []
            elif 'Reprojection Error:' in line and 'Photometric Error:' in line:
                parts = line.split('Photometric Error:')
                try:
                    repro_error = float(parts[0].split('Reprojection Error:')[1].strip().strip(','))
                    photo_error = float(parts[1].strip())
                    frame_data.append((repro_error, photo_error))
                except ValueError as e:
                    print(f"Skipping line due to value error: {line}. Error: {e}")
            elif 'Mean Reprojection Error:' in line:
                try:
                    mean_repro_error = float(line.split('Mean Reprojection Error:')[1].strip())
                    if current_frame is not None:
                        mean_errors['reprojection'].append((current_frame, mean_repro_error))
                except ValueError as e:
                    print(f"Skipping line due to value error: {line}. Error: {e}")
            elif 'Mean Photometric Error:' in line:
                try:
                    mean_photo_error = float(line.split('Mean Photometric Error:')[1].strip())
                    if current_frame is not None:
                        mean_errors['photometric'].append((current_frame, mean_photo_error))
                except ValueError as e:
                    print(f"Skipping line due to value error: {line}. Error: {e}")
                
        if current_frame is not None:
            data[current_frame] = frame_data

    return data, mean_errors

def plot_mean_errors(mean_errors, error_type="reprojection"):
    if mean_errors[error_type]:
        frames, errors = zip(*mean_errors[error_type])
        plt.figure()
        plt.plot(frames, errors, marker='o')
        plt.title(f"Mean {error_type.title()} Error Across Frames")
        plt.xlabel("Frame")
        plt.ylabel(f"Mean {error_type.title()} Error")
        plt.grid(True)
        plt.show()
    else:
        print(f"No data to plot for {error_type} mean errors.")

def plot_errors_for_each_frame(data, error_type="reprojection"):
    for frame, errors in data.items():
        iterations = range(1, len(errors) + 1)
        if error_type == "reprojection":
            error_values = [error[0] for error in errors]
            title = f"Reprojection Error for Frame {frame}"
            ylabel = "Reprojection Error"
        else:
            error_values = [error[1] for error in errors]
            title = f"Photometric Error for Frame {frame}"
            ylabel = "Photometric Error"
        
        plt.figure()
        plt.plot(iterations, error_values)
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()


filepath = './logs/Generated_Error_Data.txt'
data, mean_errors = parse_error_data(filepath)
plot_errors_for_each_frame(data, "reprojection")
plot_errors_for_each_frame(data, "photometric")
plot_mean_errors(mean_errors, "reprojection")
plot_mean_errors(mean_errors, "photometric")
