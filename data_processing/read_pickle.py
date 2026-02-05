import pickle

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    file_path = "/project/peilab/wzj/RoboTwin/policy/openpi_test/data_processing/tmp_data/episode_0_part0.pkl"
    data = read_pickle(file_path)
    print(len(data))
    #print(data)