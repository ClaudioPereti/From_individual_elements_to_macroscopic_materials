FROM jupyter/minimal-notebook:latest

ARG conda_env=python369
ARG py_ver=3.6.9

RUN mamba create --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" python=${py_ver} ipython ipykernel && \
    mamba clean --all -f -y

RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}"  && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" 
 

WORKDIR /AISC
COPY requirements.txt ./
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip" install --quiet --no-cache-dir -r requirements.txt && \
    mamba install -c conda-forge dvc && \
    mamba install -c conda-forge dvc-gdrive && \
    echo "conda activate ${conda_env}" >> "${HOME}/.bashrc" && \
    echo "pip install -e ." >>  "${HOME}/.bashrc"



ENTRYPOINT /bin/bash
    
