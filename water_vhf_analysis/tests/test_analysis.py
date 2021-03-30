import pytest
import numpy as np
import scipy
from scipy.signal import argrelextrema
from water_vhf_analysis.tests.base_test import BaseTest
from water_vhf_analysis.utils import analysis


class TestAnalysis(BaseTest):
    def test_auc(self, ixs):
        auc = analysis.get_auc(ixs, 0)

    def test_check_auc_value(self, ixs):
        auc = analysis.get_auc(ixs, 0)
        assert np.isclose(auc, 0.45078)
