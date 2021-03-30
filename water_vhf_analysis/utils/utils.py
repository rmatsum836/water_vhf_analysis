import pathlib
from pkg_resources import resource_filename

def get_root_path():
    import water_vhf_analysis
    return pathlib.Path(water_vhf_analysis.__file__).parent

def get_txt_file(model_path, name):
    """ Get path for txt file

    Parameters
    ----------
    model : str
       Simulation model directory path
    name : str
       Name of txt file to load

    Returns
    -------
    path : Path object
       Path of txt file
    """

    full_name = "/data/" + model_path + '/' + name
    path = pathlib.Path(str(get_root_path()) + full_name)

    return path
