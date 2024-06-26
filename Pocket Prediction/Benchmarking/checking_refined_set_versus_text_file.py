# import os

# file_path = '/Users/tristanbrigham/Work/Research/Gernstein/Alan/GenAI/Pocket_Prediction/CNN_preparation/filepaths.txt'
# base_directory = '/Users/tristanbrigham/Work/Research/Gernstein/Alan/GenAI/Pocket_Prediction/refined-set/'

# # read the directory names from the file
# with open(file_path, 'r') as file:
#     dir_names = file.read().strip().split()

# # check each directory name to see if it exists in the specified directory
# missing_dirs = []
# for dir_name in dir_names:
#     full_path = os.path.join(base_directory, dir_name)
#     print(full_path)
#     if not os.path.isdir(full_path):
#         missing_dirs.append(dir_name)

# # return the result
# if missing_dirs:
#     # print("The following directories are missing:", missing_dirs)
#     pass
# else:
#     print("All directories are present.")
    


import os

file_path = '/Users/tristanbrigham/Work/Research/Gernstein/Alan/GenAI/Pocket_Prediction/CNN_preparation/filepaths.txt'
base_directory = '/Users/tristanbrigham/Work/Research/Gernstein/Alan/GenAI/Pocket_Prediction/refined-set/'

# read the directory names from the file
with open(file_path, 'r') as file:
    dir_names = set(file.read().strip().split())

# get all directories in the base directory
actual_dirs = {name for name in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, name))}

# check for any directories in base_directory not listed in the text file
unlisted_dirs = actual_dirs - dir_names

# return the result
if unlisted_dirs:
    print("The following directories are not listed in the text file:", unlisted_dirs)
else:
    print("All directories in the base directory are listed in the text file.")

