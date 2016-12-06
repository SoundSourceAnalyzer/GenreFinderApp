# coding=utf-8
from __future__ import print_function
import os
import numpy as np
import sys
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from scipy import ndimage
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from six.moves import range


class FeatureExtractor:
    def __init__(self, audiofile):
        self.audiofile = audiofile
        self.temporal_centroid = None  # 1: Temporal Centroid
        self.spectral_centroid = None  # 2: Spectral Centroid average value
        self.ase_per_band_avg = []  # 4-37: Audio Spectrum Envelope (ASE) average values in 34 frequency bands
        self.ase_avg = None  # 38: ASE average value (averaged for all frequency bands)
        self.ase_per_band_var = []  # 39-72: ASE variance values in 34 frequency bands
        self.ase_var_avg = None  # 73: averaged ASE variance parameters
        self.centroid_avg = None  # 74: Audio Spectrum Centroid – average
        self.centroid_var = None  # 75: Audio Spectrum Centroid – variance
        self.spread_avg = None  # 76: Audio Spectrum Spread – average
        self.spread_var = None  # 77: Audio Spectrum Spread – variance
        self.sfm_per_band_avg = []  # 78-101: Spectral Flatness Measure (SFM) average values for 24 frequency bands
        self.sfm_avg = None  # 102: SFM average value (averaged for all frequency bands)
        self.sfm_per_band_var = []  # 103-126: Spectral Flatness Measure (SFM) variance values for 24 frequency bands
        self.sfm_var_avg = None  # 127: averaged SFM variance parameters
        self.mfcc = []  # 128-147: 20 first mel cepstral coefficients average values
        self.harmonic_spectral_centroid = None
        self.harmonic_spectral_deviation = None
        self.harmonic_spectral_spread = None
        self.harmonic_spectral_variation = None
        self.audio_fundamental_frequency_avg = None
        self.audio_fundamental_frequency_var = None
        self.harmonic_ratio_avg = None
        self.harmonic_ratio_var = None
        self.upper_limit_harmonicity_avg = None
        self.upper_limit_harmonicity_var = None
        self.audio_power_avg = None
        self.audio_power_var = None
        self.asp_per_band_avg = []
        self.asp_avg = None
        self.asp_per_band_var = []
        self.asp_var_avg = None
        self.log_attack_time = None
        self.sb_per_band_avg = []
        self.sb_avg = None
        self.sb_per_band_var = []
        self.sb_var_avg = None
        self.fv = []
        self.extract_features()
        self.export_features()

    def export_features(self):
        self.fv.append(self.temporal_centroid)
        self.fv.append(self.spectral_centroid)
        self.fv.extend(filter(lambda a: a != 0, self.ase_per_band_avg))
        self.fv.append(self.ase_avg)
        self.fv.extend(filter(lambda a: a != 0, self.ase_per_band_var))
        self.fv.append(self.ase_var_avg)
        self.fv.append(self.centroid_avg)
        self.fv.append(self.centroid_var)
        self.fv.append(self.spread_avg)
        self.fv.append(self.spread_var)
        self.fv.extend(self.sfm_per_band_avg)
        self.fv.append(self.sfm_avg)
        self.fv.extend(self.sfm_per_band_var)
        self.fv.append(self.sfm_var_avg)
        self.fv.extend(self.mfcc.tolist()[0])
        self.fv.append(self.harmonic_spectral_centroid)
        self.fv.append(self.harmonic_spectral_deviation)
        self.fv.append(self.harmonic_spectral_spread)
        self.fv.append(self.harmonic_spectral_variation)
        self.fv.append(self.audio_fundamental_frequency_avg)
        self.fv.append(self.audio_fundamental_frequency_var)
        self.fv.append(self.audio_power_avg)
        self.fv.append(self.audio_power_var)
        self.fv.extend(self.asp_per_band_avg)
        self.fv.append(self.asp_avg)
        self.fv.extend(self.asp_per_band_var)
        self.fv.append(self.asp_var_avg)
        self.fv.append(self.log_attack_time)
        self.fv.extend(self.sb_per_band_avg)
        self.fv.append(self.sb_avg)
        self.fv.extend(self.sb_per_band_var)
        self.fv.append(self.sb_var_avg)

    def extract_features(self):
        if os.path.exists(self.audiofile):
            print('Getting features from ' + self.audiofile)
        else:
            raise Exception('File ' + self.audiofile + ' not found')

        self.extract_mpeg7_features()
        self.extract_mfcc()

    def extract_mpeg7_features(self):
        ns = {'xmlns': 'urn:mpeg:mpeg7:schema:2001',
              'mpeg7': 'urn:mpeg:mpeg7:schema:2001',
              'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
              'xsi:schemaLocation': 'urn:mpeg:mpeg7:schema:2001 http://mpeg7audioenc.sourceforge.net/mpeg7audioenc.xsd'}

        print("waiting for result")
        result = subprocess.check_output(['java', '-jar', 'MPEG7AudioEnc.jar', self.audiofile, 'mpeg7config.xml'])
        print("result is here")
        root = ET.fromstring(result)

        self.extract_scalar_features(root, ns)
        self.extract_vector_features(root, ns)
        self.extract_matrix_features(root, ns)

    def extract_scalar_features(self, root, ns):

        self.temporal_centroid = self.parse_xml_scalar(root,
                                                       ".//mpeg7:AudioDescriptor[@xsi:type='TemporalCentroidType']", ns)
        self.spectral_centroid = self.parse_xml_scalar(root,
                                                       ".//mpeg7:AudioDescriptor[@xsi:type='SpectralCentroidType']", ns)
        self.harmonic_spectral_centroid = self.parse_xml_scalar(root,
                                                                ".//mpeg7:AudioDescriptor[@xsi:type='HarmonicSpectralCentroidType']",
                                                                ns)
        self.harmonic_spectral_deviation = self.parse_xml_scalar(root,
                                                                 ".//mpeg7:AudioDescriptor[@xsi:type='HarmonicSpectralDeviationType']",
                                                                 ns)
        self.harmonic_spectral_spread = self.parse_xml_scalar(root,
                                                              ".//mpeg7:AudioDescriptor[@xsi:type='HarmonicSpectralSpreadType']",
                                                              ns)
        self.harmonic_spectral_variation = self.parse_xml_scalar(root,
                                                                 ".//mpeg7:AudioDescriptor[@xsi:type='HarmonicSpectralVariationType']",
                                                                 ns)
        self.log_attack_time = self.parse_xml_scalar(root, ".//mpeg7:AudioDescriptor[@xsi:type='LogAttackTimeType']",
                                                     ns)

    def extract_vector_features(self, root, ns):
        audio_spectrum_centroid_values = self.parse_xml_vector(root,
                                                               ".//mpeg7:AudioDescriptor[@xsi:type='AudioSpectrumCentroidType']",
                                                               ns)
        self.centroid_avg = np.mean(audio_spectrum_centroid_values)
        self.centroid_var = np.var(audio_spectrum_centroid_values)

        audio_spectrum_spread_values = self.parse_xml_vector(root,
                                                             ".//mpeg7:AudioDescriptor[@xsi:type='AudioSpectrumSpreadType']",
                                                             ns)
        self.spread_avg = np.mean(audio_spectrum_spread_values)
        self.spread_var = np.var(audio_spectrum_spread_values)

        audio_fundamental_frequency_values = self.parse_xml_vector(root,
                                                                   ".//mpeg7:AudioDescriptor[@xsi:type='AudioFundamentalFrequencyType']",
                                                                   ns)
        self.audio_fundamental_frequency_avg = np.mean(audio_fundamental_frequency_values)
        self.audio_fundamental_frequency_var = np.var(audio_fundamental_frequency_values)

        harmonic_ratio_values = self.parse_xml_vector(root, ".//mpeg7:HarmonicRatio", ns)
        self.harmonic_ratio_avg = np.mean(harmonic_ratio_values)
        self.harmonic_ratio_var = np.var(harmonic_ratio_values)

        upper_limit_harmonicity_values = self.parse_xml_vector(root, ".//mpeg7:UpperLimitOfHarmonicity", ns)
        self.upper_limit_harmonicity_avg = np.mean(upper_limit_harmonicity_values)
        self.upper_limit_harmonicity_var = np.var(upper_limit_harmonicity_values)

        audio_power_values = self.parse_xml_vector(root, ".//mpeg7:AudioDescriptor[@xsi:type='AudioPowerType']", ns)
        self.audio_power_avg = np.mean(audio_power_values)
        self.audio_power_var = np.var(audio_power_values)

    def extract_matrix_features(self, root, ns):
        envelope_values = self.parse_2d_xml_vector(root,
                                                   ".//mpeg7:AudioDescriptor[@xsi:type='AudioSpectrumEnvelopeType']",
                                                   ns)
        self.ase_per_band_avg = [np.mean(band) for band in envelope_values]
        self.ase_avg = np.mean(self.ase_per_band_avg)
        self.ase_per_band_var = [np.var(band) for band in envelope_values]
        self.ase_var_avg = np.mean(self.ase_per_band_var)

        flatness_values = self.parse_2d_xml_vector(root,
                                                   ".//mpeg7:AudioDescriptor[@xsi:type='AudioSpectrumFlatnessType']",
                                                   ns)
        self.sfm_per_band_avg = [np.mean(band) for band in flatness_values]
        self.sfm_avg = np.mean(self.sfm_per_band_avg)
        self.sfm_per_band_var = [np.var(band) for band in flatness_values]
        self.sfm_var_avg = np.mean(self.sfm_per_band_var)

        asp_values = self.parse_2d_xml_vector(root, ".//mpeg7:AudioDescriptor[@xsi:type='AudioSpectrumProjectionType']",
                                              ns)
        self.asp_per_band_avg = [np.mean(band) for band in asp_values]
        self.asp_avg = np.mean(self.asp_per_band_avg)
        self.asp_per_band_var = [np.var(band) for band in asp_values]
        self.asp_var_avg = np.mean(self.asp_per_band_var)

        sb_values = self.parse_2d_xml_vector(root, ".//mpeg7:SpectrumBasis", ns)
        self.sb_per_band_avg = [np.mean(band) for band in sb_values]
        self.sb_avg = np.mean(self.sb_per_band_avg)
        self.sb_per_band_var = [np.var(band) for band in sb_values]
        self.sb_var_avg = np.mean(self.sb_per_band_var)

    def parse_xml_scalar(self, root, xml_selector, ns):
        element = root.find(xml_selector, ns)
        return float(element.find(".//mpeg7:Scalar", ns).text)

    def parse_xml_vector(self, root, xml_selector, ns):
        element = root.find(xml_selector, ns)
        values_splitted = element.find(".//mpeg7:Raw", ns).text.split()
        return map((lambda x: float(x)), values_splitted)

    def parse_2d_xml_vector(self, root, xml_selector, ns):
        element = root.find(xml_selector, ns)
        values_string = element.find(".//mpeg7:Raw", ns).text
        values_splitted = [s.strip().split() for s in values_string.splitlines()]
        values = [map((lambda x: float(x)), value) for value in values_splitted]  # cast to float
        transposed_values = list(map(list, zip(*values)))  # transpose matrix to have 1 long vector per 1 band
        return transposed_values

    def extract_mfcc(self):
        fp = FeaturePlan(sample_rate=22050, normalize=1)
        fp.addFeature('mfcc: MFCC CepsNbCoeffs=20')
        df = fp.getDataFlow()
        engine = Engine()
        engine.load(df)
        afp = AudioFileProcessor()
        afp.setOutputFormat('csv', 'features', {'Precision': '8', 'Metadata': 'False'})
        afp.processFile(engine, self.audiofile)
        engine.flush()
        feats = engine.readAllOutputs()
        self.mfcc = feats['mfcc']

    def get_features(self):
        return self.fv




