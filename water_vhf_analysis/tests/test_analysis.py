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


class TestAnalysis(BaseTest):
    def test_ixs_auc(self, ixs):
        auc = analysis.get_auc(ixs, 0)

    def test_spce_auc(self, spce):
        auc = analysis.get_auc(spce, 0)

    def test_reaxff_auc(self, reaxff):
        auc = analysis.get_auc(reaxff, 0)

    def test_check_auc_value(self, ixs):
        auc = analysis.get_auc(ixs, 0)
        assert np.isclose(auc, 0.45078)

    @pytest.mark.parametrize("si", (True, False))
    def test_first_peak_auc(self, si, ixs):
        os.mkdir("./figures")
        os.mkdir("./tables")
        first_peak_auc([ixs], si=si)

    @pytest.mark.parametrize("si", (True, False))
    def test_plot_first_peak(self, si, ixs):
        os.mkdir("./figures")
        os.mkdir("./tables")
        plot_first_peak_subplot([ixs], si=si)

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
            table[table["Unnamed: 0"] == "IXS"].A_1.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "IXS"].A_1.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "IXS"].gamma_1.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "IXS"].gamma_1.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "IXS"].A_2.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "IXS"].A_2.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "IXS"].gamma_2.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "IXS"].gamma_2.values[0]
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
            table[table["Unnamed: 0"] == "SPC/E"].A_1.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "SPC/E"].A_1.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "SPC/E"].gamma_1.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "SPC/E"].gamma_1.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "SPC/E"].A_2.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "SPC/E"].A_2.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "SPC/E"].gamma_2.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "SPC/E"].gamma_2.values[0]
        )

    @pytest.mark.parametrize("si", (True, False))
    def test_reaxff_auc(self, si, reaxff, tmpdir):
        tf = tmpdir.mkdir("./figures")
        tt = tmpdir.mkdir("./tables")
        first_peak_auc([reaxff], si=si)

        si_str = ""
        if si:
            si_str = "_si"

        table = pd.read_csv(tt.join("first_peak_fits{}.csv".format(si_str)))

        ref_table = pd.read_csv(
            utils.get_csv_file("first_peak_fits{}.csv".format(si_str))
        )

        assert (
            table[table["Unnamed: 0"] == "CHON-2017_weak"].A_1.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "CHON-2017_weak"].A_1.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "CHON-2017_weak"].gamma_1.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "CHON-2017_weak"].gamma_1.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "CHON-2017_weak"].A_2.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "CHON-2017_weak"].A_2.values[0]
        )

        assert (
            table[table["Unnamed: 0"] == "CHON-2017_weak"].gamma_2.values[0]
            == ref_table[ref_table["Unnamed: 0"] == "CHON-2017_weak"].gamma_2.values[0]
        )
