import pickle
import os
base_dir = "./"
file_lists = os.listdir(base_dir)
pk_files = [item for item in file_lists if item[-3:]==".pk"]
for pk_file in pk_files:
    pk_file_path = os.path.join(base_dir, pk_file)
    with open(pk_file_path, "rb") as file:
        data = pickle.load(file)
        
        print(pk_file_path, " acc : ", data['acc'])
        
        print(pk_file_path, " f1 : ", data['f1'])

