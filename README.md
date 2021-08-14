# Emily's Summer Research

For an in-depth guide and log of my work, see [this Google doc](https://docs.google.com/document/d/188q6sSGN8BcHUqZJC1x4pu4Fj_XUL6MmuD9vFEqjNqA/edit?usp=sharing "Dummy's guide to my project")

## Purpose

A significant obstacle with training on real-world image data is the need for manually collecting and labeling image data. A promising alternative is to use 3D models of the objects to synthesize a dataset, which can easily be done in large volumes. The purpose of this project is to test the capabilities of synthetically-generated image data for training object detection and classification models for use on real-life data.

This concept is being used to create an assembly-state detector assistant for assembling building kits.

## Background (?)

Unity 3D has published a few articles on using synthetically-generated data for training purposes. The results show that a large volume of sythesized data is required for decent performance, and the synthetic data failed to outperform a smaller set of real-world data. However, combining a large amount of synthetic data with some real-world data was more effective than either alone. Our research aims to replicate their findings, but using more complex models. We also aim to optimize the effectiveness of the models by experimenting with various parameters.

Unity 3D created the Perception package and SynthDet project, which generate synthetic datasets from 3D models using various randomizers that create clutter and distractions. 

## What's available in this repository
Scripts I wrote or modified for this project (not including Unity scripts)

- Pipeline configurations for training and fine-tuning
- tftranslate: Converts Unity-generated datasets to TFRecords w/ sharding
- tftranslate_xml: Converts xml for PASCAL VOC dataset to TFRecords
- merge_tfrecord: Merges tfrecords into one with one class
- verify_tfrecord: Draws bounding boxes on images from TFRecords to verify their placement
- png_to_jpg: Creates jpg copies of png images in the same directory
- synthetic_classification_setup: Formats information from Unity-generated datasets for classification
- test_classification_setup: Formats information from 'CVAT for images 1.1' datasets for classification
- real_classification_setup: Formats information from TFRecords for classifications
- split_data: Splits classification training data into train, val, and test sets

The [Google doc](https://docs.google.com/document/d/188q6sSGN8BcHUqZJC1x4pu4Fj_XUL6MmuD9vFEqjNqA/edit?usp=sharing "Dummy's guide to my project") from earlier has instructions and references to necessary external files.

This [slideshow](https://docs.google.com/presentation/d/11UR2lp7ocq2jRAj6kw0i8Aq71NolocwWvvdS6JQQEAg/edit?usp=sharing) also summarizes the first part of my project, and the [Unity data generator project can be found here](https://github.com/exiaohuaz/SamplePerception).
