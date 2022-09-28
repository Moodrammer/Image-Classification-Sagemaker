# simple script to reduce datasize
import os
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'dogImages')
num_files = {
    'train' : 0, 'valid': 0, 'test': 0
}

for split in ['train', 'valid', 'test']:
    split_dir = os.path.join(data_dir, split)
    class_dirs = os.listdir(split_dir)
    for class_dir in class_dirs:
        files = os.listdir(os.path.join(split_dir, class_dir))
        for file in files[num_files[split]:]:
            file_path = os.path.join(split_dir, class_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)