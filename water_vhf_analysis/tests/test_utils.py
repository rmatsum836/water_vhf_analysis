import pytest
import numpy as np
from water_vhf_analysis.utils import utils


class TestUtils(object):
    @pytest.mark.parametrize("model", ("spce", "bk3", "tip3p_ew", "reaxff"))
    def test_get_txt(self, model):
        file_path = model + "/nvt_total_data/"
        utils.get_txt_file(file_path, "t_random.txt")

    @pytest.mark.parametrize("model", ("spce", "bk3", "tip3p_ew", "reaxff"))
    def test_read_txt_file(self, model):
        file_path = model + "/nvt_total_data/"
        time = np.loadtxt(utils.get_txt_file(file_path, "t_random.txt"))

        if model == "reaxff":
            last = 1.98
        else:
            last = 1.99

        assert np.isclose(time[0], 0)
        assert np.isclose(time[-1], last, atol=1e-4)

    @pytest.mark.parametrize("model", ("spce", "bk3", "tip3p_ew"))
    def test_read_r_txt_file(self, model):
        file_path = model + "/nvt_total_data/"
        r = np.loadtxt(utils.get_txt_file(file_path, "r_random.txt"))

        assert np.isclose(r[0], 0.0025)
        assert np.isclose(r[-1], 0.9975)

    @pytest.mark.parametrize("model", ("spce", "bk3", "tip3p_ew"))
    def test_read_vhf_txt_file(self, model):
        file_path = model + "/nvt_total_data/"
        vhf = np.loadtxt(utils.get_txt_file(file_path, "vhf_random.txt"))

        assert np.isclose(vhf[0][-1], 1.0, rtol=1e-1)

    @pytest.mark.parametrize("temp", ("300k", "330k"))
    def test_read_aimd_txt_file(self, temp):
        if temp == "300k":
            file_path = "aimd/nvt_total_data/"
        else:
            file_path = "aimd/{}/nvt_total_data/".format(temp)
        time = np.loadtxt(utils.get_txt_file(file_path, "t_random.txt"))

        assert np.isclose(time[0], 0)
        assert np.isclose(time[-1], 1.99)

        r = np.loadtxt(utils.get_txt_file(file_path, "r_random.txt"))

        assert np.isclose(r[0], 0.0025)
        assert np.isclose(r[-1], 0.9975)

    def test_read_dftb_txt_file(self):
        file_path = "dftb/nvt_total_data"
        time = np.loadtxt(utils.get_txt_file(file_path, "t_random.txt"))

        assert np.isclose(time[0], 0)
        assert np.isclose(time[-1], 1.99)

        r = np.loadtxt(utils.get_txt_file(file_path, "r_random.txt"))

        assert np.isclose(r[0], 0.0025)
        assert np.isclose(r[-1], 0.9975)

        vhf = np.loadtxt(utils.get_txt_file(file_path, "vhf_random.txt"))

        assert np.isclose(vhf[0][-1], 1.0, rtol=1e-1)

    @pytest.mark.parametrize("model", ("spce", "bk3", "tip3p_ew", "reaxff", "dftb"))
    def test_read_partial_txt_file(self, model):
        file_path = model + "/nvt_partial_data/"
        if model == "reaxff":
            last = 1.98
        else:
            last = 1.99

        for pair in ["O_O", "O_H", "H_H"]:
            time = np.loadtxt(
                utils.get_txt_file(file_path, "t_random_{}.txt".format(pair))
            )

            assert np.isclose(time[0], 0)
            assert np.isclose(time[-1], last)

            r = np.loadtxt(
                utils.get_txt_file(file_path, "r_random_{}.txt".format(pair))
            )

            assert np.isclose(r[0], 0.002)
            assert np.isclose(r[-1], 0.798)

    @pytest.mark.parametrize("temp", ("300k", "330k"))
    def test_read_aimd_partial_txt_file(self, temp):
        if temp == "300k":
            file_path = "aimd/nvt_partial_data/"
        else:
            file_path = "aimd/{}/nvt_partial_data/".format(temp)
        for pair in ["O_O", "O_H", "H_H"]:
            time = np.loadtxt(
                utils.get_txt_file(file_path, "t_random_{}.txt".format(pair))
            )

            assert np.isclose(time[0], 0)
            assert np.isclose(time[-1], 1.99)

            r = np.loadtxt(
                utils.get_txt_file(file_path, "r_random_{}.txt".format(pair))
            )

            assert np.isclose(r[0], 0.002)
            assert np.isclose(r[-1], 0.798)
