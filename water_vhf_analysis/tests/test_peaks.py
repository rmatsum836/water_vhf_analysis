import pytest
import numpy as np
import pandas as pd
import scipy
import os
from scipy.signal import argrelextrema
from water_vhf_analysis.tests.base_test import BaseTest
from water_vhf_analysis.utils import analysis, utils
from scattering.utils.features import find_local_maxima, find_local_minima
from water_vhf_analysis.utils.utils import get_txt_file

class TestPeaks(BaseTest):
    """Test to ensure peak heights are accurate"""

    @pytest.mark.parametrize("model", ("spce", "tip3p_ew", "aimd", "bk3", "dftb"))
    def test_total_first_peak_max(self, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "t_random.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "vhf_random.txt")),
        }

        r = model_dict["r"]
        r_low = np.where(r > 0.15)[0][0]
        r_high = np.where(r < 0.32)[0][-1]
        r_range = r[r_low:r_high]

        maxs = list()
        # Only look at first 80 frames
        for i, frame in enumerate(model_dict["g"][:80]):
            g_range = frame[r_low:r_high]
            max_r, max_g = find_local_maxima(r_range, g_range, r_guess=0.28)
            maxs.append(max_r)

        maxs = np.array(maxs)

        assert np.all((maxs > 0.23) & (maxs < 0.32))

    @pytest.mark.parametrize("model", ("spce", "tip3p_ew", "aimd", "bk3", "dftb"))
    def test_total_second_peak_max(self, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "t_random.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "vhf_random.txt")),
        }

        maxs = list()
        # Only look at first 80 frames
        for i, frame in enumerate(model_dict["g"][:80]):
            max_r, max_g = find_local_maxima(model_dict["r"], frame, r_guess=0.44)
            maxs.append(max_r)

        maxs = np.array(maxs)

        assert np.all((maxs > 0.39) & (maxs < 0.5))

    @pytest.mark.parametrize("model", ("spce", "tip3p_ew", "aimd", "bk3", "dftb"))
    def test_hbond_peak_max(self, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/nvt_partial_data", "r_random_O_H.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/nvt_partial_data", "t_random_O_H.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/nvt_partial_data", "vhf_random_O_H.txt")),
        }

        r_low = np.where(model_dict["r"] > 0.16)[0][0]
        r_high = np.where(model_dict["r"] < 0.2)[0][-1]
        r_range = model_dict["r"][r_low:r_high]
        maxs = list()
        for frame in range(len(model_dict["t"][:10])):
            g_range = model_dict["g"][frame][r_low:r_high]
            max_r, max_g = find_local_maxima(r_range, g_range, r_guess=0.18)
            maxs.append(max_r)

        maxs = np.array(maxs)

        assert np.all((maxs > 0.17) & (maxs < 0.2))

    @pytest.mark.parametrize("model", ("spce", "tip3p_ew", "aimd", "bk3", "dftb"))
    def test_hh_peak_max(self, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/nvt_partial_data", "r_random_H_H.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/nvt_partial_data", "t_random_H_H.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/nvt_partial_data", "vhf_random_H_H.txt")),
        }

        r_low = np.where(model_dict["r"] > 0.16)[0][0]
        r_high = np.where(model_dict["r"] < 0.25)[0][-1]
        r_range = model_dict["r"][r_low:r_high]
        maxs = list()
        for frame in range(len(model_dict["t"])):
            g_range = model_dict["g"][frame][r_low:r_high]
            max_r, max_g = find_local_maxima(r_range, g_range, r_guess=0.23)
            maxs.append(max_r)

        maxs = np.array(maxs)

        assert np.all((maxs > 0.2) & (maxs < 0.3))
