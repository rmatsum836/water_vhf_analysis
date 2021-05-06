import pytest
import numpy as np
import scipy
import os
from scipy.signal import argrelextrema
from water_vhf_analysis.tests.base_test import BaseTest
from water_vhf_analysis.utils import analysis
from water_vhf_analysis.analysis.plot_relaxation import (
    first_peak_auc,
    plot_first_peak_subplot,
)


class TestAnalysis(BaseTest):
    def test_auc(self, ixs):
        auc = analysis.get_auc(ixs, 0)

    def test_check_auc_value(self, ixs):
        auc = analysis.get_auc(ixs, 0)
        assert np.isclose(auc, 0.45078)

    @pytest.mark.parametrize("si", (True, False))
    def test_first_peak_auc(self, si, ixs):
        os.mkdir("./figures")
        os.mkdir("./tables")
        first_peak_auc([ixs], si=si)

    @pytest.mark.parametrize("si", (True, False))
    def test_first_peak_auc(self, si, ixs):
        os.mkdir("./figures")
        os.mkdir("./tables")
        plot_first_peak_subplot([ixs], si=si)
