import numpy as np
import pytest
from water_vhf_analysis.utils.utils import get_txt_file


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture
    def ixs(self):
        IXS = {
            "name": "IXS",
            "r": 0.1 * np.loadtxt(get_txt_file("expt", "R_1811pure.txt"))[0],
            "t": np.loadtxt(get_txt_file("expt", "t_1811pure.txt"))[:, 0],
            "g": 1 + np.loadtxt(get_txt_file("expt", "VHF_1811pure.txt")),
        }

        return IXS

    @pytest.fixture
    def spce(self):
        model_dict = {
            "name": "SPC/E",
            "r": np.loadtxt(get_txt_file("spce/nvt_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file("spce/nvt_total_data", "t_random.txt")),
            "g": np.loadtxt(get_txt_file("spce/nvt_total_data", "vhf_random.txt")),
        }

        return model_dict

    @pytest.fixture
    def reaxff(self):
        model_dict = {
            "name": "CHON-2017_weak",
            "r": np.loadtxt(get_txt_file("reaxff/nvt_total_data", "r_random.txt")),
            "t": np.loadtxt(get_txt_file("reaxff/nvt_total_data", "t_random.txt")),
            "g": np.loadtxt(get_txt_file("reaxff/nvt_total_data", "vhf_random.txt")),
        }

        return model_dict
