import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set file paths
inpath = './dataset/SMD/org/'  # input path
outpath = './dataset/SMD/processed/'  # output path

# Get all .npy files in the directory (excluding label files)
npy_files = [file for file in os.listdir(inpath) if file.endswith('.npy') and 'label' not in file]

# Sort file names alphabetically
npy_files.sort()

# Separate train and test files
train_files = [file for file in npy_files if 'train' in file]
test_files = [file for file in npy_files if 'test' in file]

print("Train file list:", train_files)
print("Test file list:", test_files)

# Load train data and compute mean_std_per_column
train_arrays = [np.load(os.path.join(inpath, file)) for file in train_files]
test_arrays = [np.load(os.path.join(inpath, file)) for file in test_files]

# Compute standard deviation for each file
stds_list = [np.std(data, axis=0) for data in train_arrays]

# Compute the mean of standard deviations across all train files
stds_array = np.vstack(stds_list)
mean_std_per_column = np.mean(stds_array, axis=0)

# Replace any zero values in mean_std_per_column with 1 (to prevent division by zero)
mean_std_per_column[mean_std_per_column == 0] = 1

print(f"Mean standard deviation per column (calculated from train data):\n{mean_std_per_column}")


plt.rc('grid', alpha=0.3)  # Set grid transparency
colors = plt.cm.tab20.colors  # Use Matplotlib's tab20 color palette

plt.figure(figsize=(10, 6))
for i, data in enumerate(train_arrays):  
    df = pd.DataFrame(data)
    if not df.empty:
        sns.kdeplot(df.iloc[:, 2], label=f"Dataset {i+1}", alpha=0.9, linewidth=2.5, color=colors[i % len(colors)])

# plt.xlim(0.42, 0.46)
plt.title("Before Normalization", fontsize=20, fontweight='bold')
plt.xlabel("Value", fontsize=18, fontweight='bold')
plt.ylabel("Density", fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.savefig("histogram_norm_train_before_col2.png")
plt.show()

plt.figure(figsize=(10, 6))
for i, data in enumerate(test_arrays):  
    df = pd.DataFrame(data)
    if not df.empty:
        sns.kdeplot(df.iloc[:, 2], label=f"Dataset {i+1}", alpha=0.9, linewidth=2.5, color=colors[i % len(colors)])

# plt.xlim(0.42, 0.46)
plt.title("Before Normalization", fontsize=20, fontweight='bold')
plt.xlabel("Value", fontsize=18, fontweight='bold')
plt.ylabel("Density", fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.savefig("histogram_norm_test_before_col2.png")
plt.show()



normalized_df_train_all = []
normalized_df_test_all = []

# Normalize train data and save
for file, data in zip(train_files, train_arrays):
    means = np.mean(data, axis=0)
    normalized_data = (data - means) / mean_std_per_column
    normalized_df_train_all.append(normalized_data)
    output_filename = os.path.splitext(file)[0] + '.npy'
    np.save(os.path.join(outpath, output_filename), normalized_data)

print("Train data normalization completed.")

# Normalize test data using mean_std_per_column from train data and save
for file in test_files:
    data = np.load(os.path.join(inpath, file))
    means = np.mean(data, axis=0)
    normalized_data = (data - means) / mean_std_per_column  # Use mean_std_per_column from train data
    normalized_df_test_all.append(normalized_data)
    output_filename = os.path.splitext(file)[0] + '.npy'
    np.save(os.path.join(outpath, output_filename), normalized_data)

print("Test data normalization completed.")


plt.figure(figsize=(10, 6))
for i, data in enumerate(normalized_df_train_all):  
    df = pd.DataFrame(data)
    if not df.empty:
        sns.kdeplot(df.iloc[:, 2], label=f"Dataset {i+1}", alpha=0.9,  linewidth=2.5, color=colors[i % len(colors)])

# plt.xlim(-5, 5)
plt.title("After Normalization", fontsize=20, fontweight='bold')
plt.xlabel("Value", fontsize=18, fontweight='bold')
plt.ylabel("Density", fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.savefig("histogram_norm_train_after_col2.png")
plt.show()

plt.figure(figsize=(10, 6))
for i, data in enumerate(normalized_df_test_all):  
    df = pd.DataFrame(data)
    if not df.empty:
        sns.kdeplot(df.iloc[:, 2], label=f"Dataset {i+1}", alpha=0.9,  linewidth=2.5, color=colors[i % len(colors)])

# plt.xlim(-5, 5)
plt.title("After Normalization", fontsize=20, fontweight='bold')
plt.xlabel("Value", fontsize=18, fontweight='bold')
plt.ylabel("Density", fontsize=18, fontweight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=12)
plt.savefig("histogram_norm_test_after_col2.png")
plt.show()