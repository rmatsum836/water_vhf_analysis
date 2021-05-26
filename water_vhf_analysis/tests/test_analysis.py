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


class TestAnalysis(BaseTest):
    def test_ixs_auc(self, ixs):
        auc = analysis.get_auc(ixs, 0)

    @pytest.mark.parametrize("model", ("spce", "reaxff", "bk3", "dftb"))
    def test_spce_auc(self, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "t_random.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "vhf_random.txt")),
        }
        auc = analysis.get_auc(model_dict, 0)

    def test_check_auc_value(self, ixs):
        auc = analysis.get_auc(ixs, 0)
        assert np.allclose(auc, 0.45078)

    @pytest.mark.parametrize("si", (True, False))
    def test_first_peak_ixs_auc(self, si, ixs):
        os.mkdir("./figures")
        os.mkdir("./tables")
        first_peak_auc([ixs], si=si)

    @pytest.mark.parametrize("si,model", [(True, "spce"), (False, "spce")])
    def test_first_peak_auc(self, si, model):
        model_dict = {
            "name": model,
            "r": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "t_random.txt")),
            "g": np.loadtxt(get_txt_file(f"{model}/nvt_total_data", "vhf_random.txt")),
        }
        os.mkdir("./figures")
        os.mkdir("./tables")
        first_peak_auc([model_dict], si=si)

    @pytest.mark.parametrize("si", (True, False))
    def test_ixs_auc(self, si, ixs, tmpdir):
        tf = tmpdir.mkdir("./figures")
        tt = tmpdir.mkdir("./tables")
        first_peak_auc([ixs], si=si)

        si_str = ""
        if si:
            si_str = "_si"

        table = pd.read_csv(tt.join("first_peak_fits{}.csv".format(si_str)))

        ref_table = pd.read_csv(
            utils.get_csv_file("first_peak_fits{}.csv".format(si_str))
        )

        assert (
            table[table["Model"] == "IXS"]["$A_{1}$"].values[0]
            == ref_table[ref_table["Model"] == "IXS"]["$A_{1}$"].values[0]
        )

        assert (
            table[table["Model"] == "IXS"]["$\gamma_{1}$"].values[0]
            == ref_table[ref_table["Model"] == "IXS"]["$\gamma_{1}$"].values[0]
        )

        assert (
            table[table["Model"] == "IXS"]["$A_{2}$"].values[0]
            == ref_table[ref_table["Model"] == "IXS"]["$A_{2}$"].values[0]
        )

        assert (
            table[table["Model"] == "IXS"]["$\gamma_{2}$"].values[0]
            == ref_table[ref_table["Model"] == "IXS"]["$\gamma_{2}$"].values[0]
        )

    @pytest.mark.parametrize("si", (True, False))
    def test_spce_auc(self, si, spce, tmpdir):
        tf = tmpdir.mkdir("./figures")
        tt = tmpdir.mkdir("./tables")
        first_peak_auc([spce], si=si)

        si_str = ""
        if si:
            si_str = "_si"

        table = pd.read_csv(tt.join("first_peak_fits{}.csv".format(si_str)))

        ref_table = pd.read_csv(
            utils.get_csv_file("first_peak_fits{}.csv".format(si_str))
        )

        assert (
            table[table["Model"] == "SPC/E (CMD)"]["$A_{1}$"].values[0]
            == ref_table[ref_table["Model"] == "SPC/E (CMD)"]["$A_{1}$"].values[0]
        )

        assert (
            table[table["Model"] == "SPC/E (CMD)"]["$\gamma_{1}$"].values[0]
            == ref_table[ref_table["Model"] == "SPC/E (CMD)"]["$\gamma_{1}$"].values[0]
        )

        assert (
            table[table["Model"] == "SPC/E (CMD)"]["$A_{2}$"].values[0]
            == ref_table[ref_table["Model"] == "SPC/E (CMD)"]["$A_{2}$"].values[0]
        )

        assert (
            table[table["Model"] == "SPC/E (CMD)"]["$\gamma_{2}$"].values[0]
            == ref_table[ref_table["Model"] == "SPC/E (CMD)"]["$\gamma_{2}$"].values[0]
        )
