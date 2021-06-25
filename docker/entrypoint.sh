#!/bin/sh

. /opt/conda/etc/profile.d/conda.sh
conda activate base
conda activate vanhove

if [ "$@" == "jupyter" ]; then
	jupyter notebook --no-browser --notebook-dir /water_vhf_analysis/notebooks --ip="0.0.0.0"
else
	$@
fi
