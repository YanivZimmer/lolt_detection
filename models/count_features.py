import pickle

file_path = "models/feature_names.pkl"
try:
    with open(file_path, "rb") as f:
        unpickled_list = pickle.load(f)
    print("List unpickled successfully:")
    print(unpickled_list)
    print(len(unpickled_list))
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred during unpickling: {e}")
