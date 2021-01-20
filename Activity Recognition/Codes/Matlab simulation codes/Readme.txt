		*Files Detail*

This directory contains only simulations codes used in MATLAB 2018a for testing purposes. It has two files beside the Readme.txt. The "FCFeatures.m" files is used to extract VGG_19 features from a whole dataset. "FeaturesEncoding.m" is used for encoding of the extracted features.

		*How to RUN?*

1. Setup the directories in "FCFeatures.m" and extract features of any dataset.
2. Run the "FeaturesEncoding.m" file to obtain the trained autoencoder over the features that are extracted in step-1
3. Use "Classification Learner" toolbox from MATLAB applications to try different variants of SVM classifiers.