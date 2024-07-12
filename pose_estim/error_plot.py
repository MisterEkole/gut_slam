
'''
-------------------------------------------------------------
Utility script to parse and plot error data from a text file.

Author: Mitterand Ekole
Date: 03-05-2024
-------------------------------------------------------------
'''
import matplotlib.pyplot as plt

# def parse_error_data(filepath):
#     data = {}
#     mean_errors = {'reprojection': [], 'photometric': []}
#     with open(filepath, 'r') as file:
#         frame_data = []
#         current_frame = None
#         for line in file:
#             line = line.strip()  # Ensure no leading/trailing whitespace
#             if 'Frame' in line:
#                 if current_frame is not None:
#                     data[current_frame] = frame_data
#                 current_frame = int(line.split()[1])
#                 frame_data = []
#             elif 'Reprojection Error:' in line and 'Photometric Error:' in line:
#                 parts = line.split('Photometric Error:')
#                 try:
#                     repro_error = float(parts[0].split('Reprojection Error:')[1].strip().strip(','))
#                     photo_error = float(parts[1].strip())
#                     frame_data.append((repro_error, photo_error))
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {line}. Error: {e}")
#             elif 'Mean Reprojection Error:' in line:
#                 try:
#                     mean_repro_error = float(line.split('Mean Reprojection Error:')[1].strip())
#                     if current_frame is not None:
#                         mean_errors['reprojection'].append((current_frame, mean_repro_error))
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {line}. Error: {e}")
#             elif 'Mean Photometric Error:' in line:
#                 try:
#                     mean_photo_error = float(line.split('Mean Photometric Error:')[1].strip())
#                     if current_frame is not None:
#                         mean_errors['photometric'].append((current_frame, mean_photo_error))
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {line}. Error: {e}")
                
#         if current_frame is not None:
#             data[current_frame] = frame_data

#     return data, mean_errors

# def plot_mean_errors(mean_errors, error_type="reprojection"):
#     if mean_errors[error_type]:
#         frames, errors = zip(*mean_errors[error_type])
#         plt.figure()
#         plt.plot(frames, errors, marker='o')
#         plt.title(f"Mean {error_type.title()} Error Across Frames")
#         plt.xlabel("Frame")
#         plt.ylabel(f"Mean {error_type.title()} Error")
#         plt.grid(True)
#         plt.show()
#     else:
#         print(f"No data to plot for {error_type} mean errors.")

# def plot_errors_for_each_frame(data, error_type="reprojection", max_iterations=None):
#     for frame, errors in data.items():
#         iterations = range(1, len(errors) + 1)
#         if max_iterations is not None:
#             iterations = iterations[:max_iterations]
#             errors = errors[:max_iterations]

#         if error_type == "reprojection":
#             error_values = [error[0] for error in errors]
#             title = f"Reprojection Error for Frame {frame}"
#             ylabel = "Reprojection Error"
#         else:
#             error_values = [error[1] for error in errors]
#             title = f"Photometric Error for Frame {frame}"
#             ylabel = "Photometric Error"
        
#         plt.figure()
#         plt.plot(iterations, error_values)
#         plt.title(title)
#         plt.xlabel("Iteration")
#         plt.ylabel(ylabel)
#         plt.grid(True)
#         plt.show()
# def parse_error_data(filepath):
#     data = {}
#     mean_errors = {'photometric': []}
#     with open(filepath, 'r') as file:
#         frame_data = []
#         current_frame = None
#         for line in file:
#             line = stripped_line = line.strip()  # Ensure no leading/trailing whitespace
#             if 'Frame' in line:
#                 if current_frame is not None:
#                     data[current_frame] = frame_data
#                 current_frame = int(line.split()[1])
#                 frame_data = []
#             elif 'Photometric Error:' in line:
#                 try:
#                     photo_error = float(line.split('Photometric Error:')[1].strip())
#                     frame_data.append(photo_error)
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {line}. Error: {e}")
                
#         if current_frame is not None:
#             data[current_frame] = frame_data

#     return data, mean_errors

# def plot_mean_errors(mean_errors, error_type="photometric"):
#     if mean_errors[error_type]:
#         frames, errors = zip(*mean_errors[error_type])
#         plt.figure()
#         plt.plot(frames, errors, marker='o')
#         plt.title(f"Mean {error_type.title()} Error Across Frames")
#         plt.xlabel("Frame")
#         plt.ylabel(f"Mean {error_type.title()} Error")
#         plt.grid(True)
#         plt.show()
#     else:
#         print(f"No data to plot for {error_type} mean errors.")

# def plot_errors_for_each_frame(data, error_type="photometric", max_iterations=None):
#     for frame, errors in data.items():
#         iterations = []
#         filtered_errors = []
#         last_error = None
#         repeat_count = 0

#         for i, error in enumerate(errors):
#             if error == last_error:
#                 repeat_count += 1
#             else:
#                 repeat_count = 0
#                 last_error = error

#             if repeat_count < 10:
#                 iterations.append(i + 1)
#                 filtered_errors.append(error)

#         if max_iterations is not None:
#             iterations = iterations[:max_iterations]
#             filtered_errors = filtered_errors[:max_iterations]

#         title = f"{error_type.title()} Error for Frame {frame}"
#         ylabel = f"{error_type.title()} Error"
        
#         plt.figure()
#         plt.plot(iterations, filtered_errors)
#         plt.title(title)
#         plt.xlabel("Iteration")
#         plt.ylabel(ylabel)
#         plt.grid(True)
#         plt.show()

# def parse_error_data(filepath):
#     data = {}
#     mean_errors = {'photometric': []}
#     with open(filepath, 'r') as file:
#         frame_data = []
#         current_frame = None
#         for line in file:
#             stripped_line = line.strip()  # Ensure no leading/trailing whitespace
#             if 'Frame' in stripped_line:
#                 if current_frame is not None:
#                     data[current_frame] = frame_data
#                 current_frame = int(stripped_line.split()[1])
#                 frame_data = []
#             elif 'Photometric Error:' in stripped_line:
#                 try:
#                     photo_error = float(stripped_line.split('Photometric Error:')[1].strip())
#                     frame_data.append(photo_error)
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {stripped_line}. Error: {e}")
                
#         if current_frame is not None:
#             data[current_frame] = frame_data

#     return data, mean_errors

# def plot_mean_errors(mean_errors, error_type="photometric"):
#     if mean_errors[error_type]:
#         frames, errors = zip(*mean_errors[error_type])
#         plt.figure()
#         plt.plot(frames, errors, marker='o')
#         plt.title(f"Mean {error_type.title()} Error Across Frames")
#         plt.xlabel("Frame")
#         plt.ylabel(f"Mean {error_type.title()} Error")
#         plt.grid(True)
#         plt.show()
#     else:
#         print(f"No data to plot for {error_type} mean errors.")

# def plot_errors_for_each_frame(data, error_type="photometric", max_iterations=None):
#     for frame, errors in data.items():
#         iterations = []
#         filtered_errors = []
#         last_error = None

#         for i, error in enumerate(errors):
#             if error != last_error:
#                 iterations.append(i + 1)
#                 filtered_errors.append(error)
#                 last_error = error

#         if max_iterations is not None:
#             iterations = iterations[:max_iterations]
#             filtered_errors = filtered_errors[:max_iterations]

#         title = f"{error_type.title()} Error for Frame {frame}"
#         ylabel = f"{error_type.title()} Error"
        
#         plt.figure()
#         plt.plot(iterations, filtered_errors)
#         plt.title(title)
#         plt.xlabel("Iteration")
#         plt.ylabel(ylabel)
#         plt.grid(True)
#         plt.show()


# def plot_first_and_last_errors(data, error_type="photometric"):
#     for frame, errors in data.items():
#         if errors:
#             first_error = errors[0]
#             last_error = errors[-1]
#             iterations = [1, len(errors)]
#             filtered_errors = [first_error, last_error]

#             title = f"{error_type.title()} Error for Frame {frame}"
#             ylabel = f"{error_type.title()} Error"

#             plt.figure()
#             plt.plot(iterations, filtered_errors, marker='o')
#             plt.title(title)
#             plt.xlabel("Iteration")
#             plt.ylabel(ylabel)
#             plt.grid(True)
#             plt.show()



import matplotlib.pyplot as plt

def parse_error_data(filepath):
    data = {}
    with open(filepath, 'r') as file:
        frame_data = []
        current_frame = None
        for line in file:
            stripped_line = line.strip()  # Ensure no leading/trailing whitespace
            if 'Frame' in stripped_line:
                if current_frame is not None:
                    data[current_frame] = frame_data
                current_frame = int(stripped_line.split()[1])
                frame_data = []
            elif 'Photometric Error:' in stripped_line:
                try:
                    photo_error = float(stripped_line.split('Photometric Error:')[1].strip())
                    frame_data.append(photo_error)
                except ValueError as e:
                    print(f"Skipping line due to value error: {stripped_line}. Error: {e}")
                
        if current_frame is not None:
            data[current_frame] = frame_data

    return data

def plot_error_changes(data):
    for frame, errors in data.items():
        iterations = []
        filtered_errors = []
        last_error = None

        for i, error in enumerate(errors):
            if error != last_error:
                iterations.append(i + 1)
                filtered_errors.append(error)
                last_error = error

        title = f"Photometric Error vs  Number of Iterations"
        ylabel = "Photometric Error"
        
        plt.figure()
        plt.plot(iterations, filtered_errors, marker='x', linestyle='-', color='b')
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

# Filepath to the provided data file
filepath = './logs/optimization_errors_all_frames4.txt'
data = parse_error_data(filepath)
plot_error_changes(data)

# #filepath = './logs/new_error_values_with_mean.txt'
# #filepath='/Users/ekole/Dev/gut_slam/pose_estim/logs/mod_optim_e.txt'
# filepath='/Users/ekole/Dev/gut_slam/pose_estim/logs/optimization_errors_all_frames1.txt'
# data, mean_errors = parse_error_data(filepath)
# #plot_errors_for_each_frame(data, "reprojection", max_iterations=100)  
# plot_errors_for_each_frame(data, "photometric", max_iterations=1000)  
#plot_first_and_last_errors(data,"photometric") 
# plot_mean_errors(mean_errors, "reprojection")
# plot_mean_errors(mean_errors, "photometric")














# '''
# -------------------------------------------------------------
# Utility script to parse and plot error data from a text file.

# Author: Mitterand Ekole
# Date: 03-05-2024
# -------------------------------------------------------------
# '''
# import matplotlib.pyplot as plt

# def parse_error_data(filepath):
#     data = {}
#     mean_errors = {'reprojection': [], 'photometric': []}
#     with open(filepath, 'r') as file:
#         frame_data = []
#         current_frame = None
#         for line in file:
#             line = line.strip()  # Ensure no leading/trailing whitespace
#             if 'Frame' in line:
#                 if current_frame is not None:
#                     data[current_frame] = frame_data
#                 current_frame = int(line.split()[1])
#                 frame_data = []
#             elif 'Reprojection Error:' in line and 'Photometric Error:' in line:
#                 parts = line.split('Photometric Error:')
#                 try:
#                     repro_error = float(parts[0].split('Reprojection Error:')[1].strip().strip(','))
#                     photo_error = float(parts[1].strip())
#                     frame_data.append((repro_error, photo_error))
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {line}. Error: {e}")
#             elif 'Mean Reprojection Error:' in line:
#                 try:
#                     mean_repro_error = float(line.split('Mean Reprojection Error:')[1].strip())
#                     if current_frame is not None:
#                         mean_errors['reprojection'].append((current_frame, mean_repro_error))
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {line}. Error: {e}")
#             elif 'Mean Photometric Error:' in line:
#                 try:
#                     mean_photo_error = float(line.split('Mean Photometric Error:')[1].strip())
#                     if current_frame is not None:
#                         mean_errors['photometric'].append((current_frame, mean_photo_error))
#                 except ValueError as e:
#                     print(f"Skipping line due to value error: {line}. Error: {e}")
                
#         if current_frame is not None:
#             data[current_frame] = frame_data

#     return data, mean_errors

# def plot_mean_errors(mean_errors, error_type="reprojection"):
#     if mean_errors[error_type]:
#         frames, errors = zip(*mean_errors[error_type])
#         plt.figure()
#         plt.plot(frames, errors, marker='o')
#         plt.title(f"Mean {error_type.title()} Error Across Frames")
#         plt.xlabel("Frame")
#         plt.ylabel(f"Mean {error_type.title()} Error")
#         plt.grid(True)
#         plt.show()
#     else:
#         print(f"No data to plot for {error_type} mean errors.")

# def plot_errors_for_each_frame(data, error_type="reprojection"):
#     for frame, errors in data.items():
#         iterations = range(1, len(errors) + 1)
#         if error_type == "reprojection":
#             error_values = [error[0] for error in errors]
#             title = f"Reprojection Error for Frame {frame}"
#             ylabel = "Reprojection Error"
#         else:
#             error_values = [error[1] for error in errors]
#             title = f"Photometric Error for Frame {frame}"
#             ylabel = "Photometric Error"
        
#         plt.figure()
#         plt.plot(iterations, error_values)
#         plt.title(title)
#         plt.xlabel("Iteration")
#         plt.ylabel(ylabel)
#         plt.grid(True)
#         plt.show()


# filepath = './logs/Generated_Error_Data.txt'
# data, mean_errors = parse_error_data(filepath)
# plot_errors_for_each_frame(data, "reprojection")
# plot_errors_for_each_frame(data, "photometric")
# plot_mean_errors(mean_errors, "reprojection")
# plot_mean_errors(mean_errors, "photometric")
