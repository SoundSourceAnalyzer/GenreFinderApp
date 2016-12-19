from neuralnet.model import NeuralNetModel
from neuralnet.parser import FeatureExtractor

featureExtractor = FeatureExtractor('neuralnet/data/blues.00000.au')
# print (featureExtractor.fv.__len__())
# print(featureExtractor.ase_per_band_avg.__len__())
# print(featureExtractor.ase_per_band_var.__len__())
# print(featureExtractor.sfm_per_band_avg.__len__())
# print(featureExtractor.sfm_per_band_var.__len__())
# print(featureExtractor.mfcc.tolist()[0].__len__())

model = NeuralNetModel()
model.predict_gztan(featureExtractor.fv)

model = NeuralNetModel()
model.predict_gztan(featureExtractor.fv)