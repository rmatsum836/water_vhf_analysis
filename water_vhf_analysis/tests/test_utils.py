import pytest
import numpy as np
from water_vhf_analysis.utils import utils


class TestUtils(object):
    @pytest.mark.parametrize("model", ("spce", "bk3", "tip3p_ew"))
    def test_get_txt(self, model):
        file_path = model + "/nvt_total_data/"
        utils.get_txt_file(file_path, "t_random.txt")

    @pytest.mark.parametrize("model", ("spce", "bk3", "tip3p_ew"))
    def test_read_txt_file(self, model):
        file_path = model + "/nvt_total_data/"
        time = np.loadtxt(utils.get_txt_file(file_path, "t_random.txt"))

        assert time[0] == 0
