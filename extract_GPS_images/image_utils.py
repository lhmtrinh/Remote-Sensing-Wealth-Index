import tensorflow as tf
import numpy as np

def parse_tfrecord(example_proto, bands):
    """The parsing function.

    Read a serialized example into the structure defined by featuresDict.

    Args:
        example_proto: a serialized Example.

    Returns:
        A tuple of the predictors dictionary and the label, cast to an `int32`.
    """
    columns = [
        tf.io.FixedLenFeature(shape=(255,255), dtype=tf.float32) for _ in bands
    ]
    # Dictionary with names as keys, features as values.
    features_dict = dict(zip(bands, columns))
    parsed_features = tf.io.parse_example(example_proto, features_dict)
    return parsed_features

def normalize_and_clip(band_data, min_val = 0.0, max_val=0.3):
    return np.clip(band_data, min_val, max_val) / (max_val - min_val)