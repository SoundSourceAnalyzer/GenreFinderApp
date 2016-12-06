from parser import FeatureExtractor

featureExtractor = FeatureExtractor('rock_sample.au')
featureExtractor.extract_features()
print ("Extraction complete")
featureExtractor.export_features()
print(len(featureExtractor.fv))


#
# root = os.path.join(os.getcwd(), 'genres')
#
# genres = os.listdir(root)
# mappings = dict(enumerate(genres))
# mappings_rev = {v: k for k, v in mappings.items()}
# dataset_xs = []
# dataset_y = []
#
# for folder in os.listdir(root):
#     data_folder = os.path.join(root, folder)
#     filenames = os.listdir(data_folder)
#     for filename in filenames:
#         path = os.path.join(data_folder, filename)
#         genre = mappings_rev[folder]
#         featureExtractor = FeatureExtractor(path)
#         featureExtractor.extract_features()
#         featureExtractor.export_features()
#         dataset_xs.append(featureExtractor.fv)
#         dataset_y.append(genre)
