#!/bin/sh

. /opt/conda/etc/profile.d/conda.sh
conda activate base
conda activate vanhove

if [ "$@" == "none" ]; then
	bash
else
	$@
fi
