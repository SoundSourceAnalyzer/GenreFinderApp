import os
import pickle
from neuralnet.parser import FeatureExtractor

root = os.path.join(os.getcwd(), 'neuralnet/data/genres')

genres = os.listdir(root)
mappings = dict(enumerate(genres))
mappings_rev = {v: k for k, v in mappings.items()}
dataset_xs = []
dataset_y = []

for folder in os.listdir(root):
    data_folder = os.path.join(root, folder)
    filenames = os.listdir(data_folder)
    for filename in filenames:
        path = os.path.join(data_folder, filename)
        genre = mappings_rev[folder]
        featureExtractor = FeatureExtractor(path)
        print (featureExtractor.fv.__len__())
        dataset_xs.append(featureExtractor.fv)
        dataset_y.append(genre)

print(dataset_xs.__len__())
print(dataset_xs[0].__len__())
print(dataset_y.__len__())

f = os.path.join(os.getcwd(), 'neuralnet/data/gztan_raw.pickle', 'wb')
save = {
    'dataset_xs': dataset_xs,
    'dataset_y': dataset_y,
    'mappings': mappings
}
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()