import os

def remove_files_from_directory(directory, files_to_remove):
    files = os.listdir(directory)
    files.sort()
    print(f"Files in {directory}: {files}")
    for filename in files_to_remove:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
            except PermissionError:
                print(f"Permission error: {file_path}")
        else:
            print(f"Skipped non-file: {file_path}")

def process_subdirectories(base_directory, remove_files_indices):
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing directory: {subdir_path}")
            files = os.listdir(subdir_path)
            files.sort()
            files_to_remove = [files[i] for i in remove_files_indices]
            remove_files_from_directory(subdir_path, files_to_remove)

# Define paths
train_dir = './src/data/train/'
val_dir = './src/data/val/'

# Calculate indices for removal
def get_removal_indices(file_list, num_to_remove):
    total_files = len(file_list)
    return range(0, num_to_remove) if num_to_remove < total_files else []

# Number of images to remove
num_to_remove = 350

# Process validation images
for subdir in os.listdir(val_dir):
    subdir_path = os.path.join(val_dir, subdir)
    if os.path.isdir(subdir_path):
        files = os.listdir(subdir_path)
        files.sort()
        indices_to_remove = get_removal_indices(files, num_to_remove)
        remove_files_from_directory(subdir_path, [files[i+350] for i in indices_to_remove])

# Process training images
for subdir in os.listdir(train_dir):
    subdir_path = os.path.join(train_dir, subdir)
    if os.path.isdir(subdir_path):
        files = os.listdir(subdir_path)
        files.sort()
        indices_to_remove = get_removal_indices(files, num_to_remove)
        files_to_remove = [files[i] for i in indices_to_remove]
        remove_files_from_directory(subdir_path, files_to_remove)

