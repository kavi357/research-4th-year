import pickle

with open("music_features_colab.pkl", "rb") as f:
    data = pickle.load(f)

print(len(data), "tracks loaded OK")
