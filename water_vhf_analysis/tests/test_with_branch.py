import pytest
import numpy as np
import pandas as pd
import scipy
import os
from scipy.signal import argrelextrema
from water_vhf_analysis.tests.base_test import BaseTest
from water_vhf_analysis.utils import analysis, utils
from water_vhf_analysis.analysis.plot_relaxation import (
    first_peak_auc,
    plot_first_peak_subplot,
)
from water_vhf_analysis.utils.utils import get_txt_file


class TestBranch(BaseTest):
    """Test to ensure data is similar to old branch data"""

    @pytest.mark.parametrize(
        "model", ("spce", "reaxff", "bk3", "aimd", "aimd/330k", "tip3p_ew")
    )
    def test_raw_data(self, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/overlap_nvt", "r_final.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/overlap_nvt", "t_final.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/overlap_nvt", "vhf_final.txt")),
        }

        branch_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/branch_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/branch_total_data", "t_random.txt")),
            "g": np.loadtxt(
                get_txt_file(f"{model}/branch_total_data", "vhf_random.txt")
            ),
        }

        assert np.allclose(model_dict["r"], branch_dict["r"], atol=1e-2)
        assert np.allclose(model_dict["t"], branch_dict["t"], atol=1e-2)
        # Corresponds to r = 0.227 nm to r = 0.35 nm
        assert np.allclose(model_dict["g"][:, 45:70], branch_dict["g"][:, 45:70], atol=0.15)
        # Corresponds r = 0.35 nm and greater
        assert np.allclose(model_dict["g"][:, 70:], branch_dict["g"][:, 70:], atol=0.1)

    @pytest.mark.parametrize(
        "model", ("spce", "reaxff", "bk3", "aimd", "aimd/330k", "tip3p_ew")
    )
    def test_auc(self, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/overlap_nvt", "r_final.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/overlap_nvt", "t_final.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/overlap_nvt", "vhf_final.txt")),
        }

        branch_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/branch_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/branch_total_data", "t_random.txt")),
            "g": np.loadtxt(
                get_txt_file(f"{model}/branch_total_data", "vhf_random.txt")
            ),
        }
        auc = analysis.get_auc(model_dict, 0)

        branch_auc = analysis.get_auc(branch_dict, 0)

        assert np.isclose(auc, branch_auc, atol=1e-2)
