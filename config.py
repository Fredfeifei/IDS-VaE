import os

batch_size =256
num_class_2017 = 13
num_class_2018 = 14

save_model_dir = os.getcwd()
freeze = True

train_file_dir_2017 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2017\\test_vector_2017.txt"
train_index_dir_2017 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2017\\test_index_2017.txt"
test_file_dir_2017 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2017\\train_vector_2017.txt"
test_index_dir_2017 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2017\\train_index_2017.txt"

train_file_dir_2018 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2018\\train_vector.txt"
train_index_dir_2018 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2018\\train_index.txt"
test_file_dir_2018 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2018\\valid_vector.txt"
test_index_dir_2018 = r"C:\\Users\\15516\Desktop\IDS_VaE\clean_data\IDS_2018\\valid_index.txt"