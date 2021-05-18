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


class TestFF(BaseTest):
    """Test that form factors are accurate"""

    @pytest.mark.parametrize("model", ("spce", "reaxff", "bk3", "dftb"))
    def test_form_factors(self, model):
        total_data = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "t_random.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "vhf_random.txt")),
        }

        OO = {
            "name": model + "_OO",
            "r": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "r_random_O_O.txt")
            ),
            "t": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "t_random_O_O.txt")
            ),
            "g": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "vhf_random_O_O.txt")
            ),
        }

        OH = {
            "name": model + "_OH",
            "r": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "r_random_O_H.txt")
            ),
            "t": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "t_random_O_H.txt")
            ),
            "g": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "vhf_random_O_H.txt")
            ),
        }

        HH = {
            "name": model + "_HH",
            "r": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "r_random_H_H.txt")
            ),
            "t": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "t_random_H_H.txt")
            ),
            "g": np.loadtxt(
                get_txt_file(f"{model}/nvt_partial_data", "vhf_random_H_H.txt")
            ),
        }

        H_conc = float(2 / 3)
        O_conc = float(1 / 3)
        H_form = float(1 / 3)
        O_form = float(9 + 1 / 3)
        oo_coeff = O_conc ** 2 * O_form ** 2
        oh_coeff = O_conc * H_conc * O_form * H_form
        hh_coeff = H_conc * H_conc * H_form * H_form 
        norm = oo_coeff + oh_coeff + hh_coeff

        oo_part = oo_coeff * OO["g"]
        oh_part = oh_coeff * OH["g"]
        hh_part = hh_coeff * HH["g"]

        partial_total = (oo_part + oh_part + hh_part) / norm

        np.allclose(total_data["g"], partial_total, atol=1e-1)
