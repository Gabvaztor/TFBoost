# -*- coding: utf-8 -*-
"""

"""

import TFBoost.TFEncoder as tfencoder
import UsefulTools.UtilsFunctions as utils

import json

class Prediction():
    """

    """
    information = ""
    json = tfencoder.Dictionary.string_json_extension

# German signals images
class GermanSignal(Prediction):
    """

    """
    real_label = None
    prediction_label = None
    softmax_labels = None
    image_fullpath = None

    def __init__(self, information, image_fullpath, prediction_label, real_label=None, softmax_labels=None):
        self.real_label = real_label
        self.prediction_label = prediction_label
        self.image_fullpath = image_fullpath
        self.softmax_labels = softmax_labels
        self.information = information

    def _to_json(self):
        string_json = utils.object_to_json(object=self, attributes_to_delete=None)
        return string_json

    def save_json(self, save_fullpath):
        json = self._to_json()
        utils.write_string_to_pathfile(json, save_fullpath)

