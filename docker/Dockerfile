FROM continuumio/miniconda3:4.8.2-alpine AS builder

EXPOSE 8888

ENV PATH /opt/conda/bin:$PATH

USER root

RUN apk --update-cache upgrade && \
    apk add --no-cache --virtual .build_deps git && \
    conda update conda -y && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name vanhove jupyter numpy scipy seaborn matplotlib mdtraj pytest progressbar2 \
    -c conda-forge -c anaconda -c omnia && \
    	echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/anaconda/.bashrc && \
	    echo "conda activate base" >> /home/anaconda/.bashrc && \
	    echo "conda activate vanhove" >> /home/anaconda/.bashrc && \
	    conda activate vanhove && \
    git clone https://github.com/mattwthompson/scattering && \
    cd scattering && \
    pip install . && cd ../ && \
    # Uncomment when water_vhf_analysis is public
    #git clone https://github.com/rmatsum836/water_vhf_analysis && \
    #cd water_vhf_analysis && \
    #python setup.py install && cd ../ && \
    mkdir /home/anaconda/data && \
    #chown -R anaconda:anaconda /water_vhf_analysis && \
    chown -R anaconda:anaconda /opt && \
    chown -R anaconda:anaconda /home/anaconda && \
    conda clean -afy && apk del .build_deps && \
    rm -rf /var/cache/apk/*

WORKDIR /home/anaconda

COPY entrypoint.sh /entrypoint.sh
RUN chmod a+x /entrypoint.sh

USER anaconda

ENTRYPOINT ["/entrypoint.sh"]
CMD ["jupyter"]
