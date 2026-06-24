# Dockerfile for BixBench Inspect evaluation
#
# This image is built from scratch on Ubuntu 20.04 and installs only the
# dependencies required by BixBench. It intentionally omits the heavy ML/DL
# stack (tensorflow, torch, transformers, etc.) that previously shipped with
# the old mlebench-based base image, producing a significantly smaller image
# while keeping BixBench functionality intact.
#
# Build:
#   docker build -t jingedawang/bixbench-inspect-env:latest --platform linux/amd64 .
#
# If you need a proxy during build, pass it as build args:
#   docker build -t jingedawang/bixbench-inspect-env:latest \
#       --build-arg HTTP_PROXY=http://host:port \
#       --build-arg HTTPS_PROXY=http://host:port \
#       --platform linux/amd64 .

FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# -----------------------------------------------------------------------------
# 1. Base system packages
# -----------------------------------------------------------------------------
# These are the minimum packages needed to run conda, build any source-only
# Python/R extensions, and support headless matplotlib/opencv libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Install Miniforge (conda/mamba)
# -----------------------------------------------------------------------------
# Override with the official GitHub URL if you are outside mainland China.
ARG MINIFORGE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh"
RUN wget -q -O /tmp/miniforge.sh "${MINIFORGE_URL}" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean --all -f -y

ENV PATH="/opt/conda/bin:${PATH}"

# Use Tsinghua Anaconda mirrors for conda-forge / bioconda to speed up builds
# in mainland China. Override these if you are in a different region.
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/ \
    && conda config --set show_channel_urls yes

# -----------------------------------------------------------------------------
# 3. Create the bixbench conda environment
# -----------------------------------------------------------------------------
# bixbench.py hardcodes PATH to /opt/conda/envs/bixbench/bin, so keep the env
# name consistent with the code.
ARG CONDA_ENV_NAME=bixbench
ARG PYTHON_VERSION=3.11

RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y \
    && conda clean --all -f -y

# Make the bixbench environment the default for all subsequent conda run commands.
ENV PATH="/opt/conda/envs/${CONDA_ENV_NAME}/bin:/opt/conda/bin:${PATH}"

# Ensure strict priority for reproducible resolves.
RUN conda config --set channel_priority strict

# Proxy settings for builds behind restrictive firewalls.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ENV http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY}

# -----------------------------------------------------------------------------
# 4. Install Jupyter / notebook infrastructure
# -----------------------------------------------------------------------------
# Install Jupyter / notebook infrastructure directly into the bixbench
# environment because the runner relies on ipykernel.
RUN mamba install -n ${CONDA_ENV_NAME} -c conda-forge -y \
    jupyterlab \
    notebook \
    nbformat=5.10.4 \
    nbconvert=7.16.4 \
    ipykernel=6.29.5 \
    jupyter_client \
    && mamba clean --all -f -y

# -----------------------------------------------------------------------------
# 5. Install R base and core R packages
# -----------------------------------------------------------------------------
RUN mamba install -n ${CONDA_ENV_NAME} -c conda-forge -c bioconda -y \
    r-base=4.3.3 \
    r-irkernel=1.3.2 \
    r-tidyverse=2.0.0 \
    r-readxl=1.4.3 \
    r-ggrepel=0.9.6 \
    r-ggpubr=0.6.0 \
    r-pheatmap=1.0.12 \
    r-viridis=0.6.5 \
    && mamba clean --all -f -y

# -----------------------------------------------------------------------------
# 6. Install Bioconductor packages
# -----------------------------------------------------------------------------
RUN mamba install -n ${CONDA_ENV_NAME} -c conda-forge -c bioconda -y \
    bioconductor-deseq2=1.42.0 \
    bioconductor-clusterprofiler=4.10.0 \
    bioconductor-org.hs.eg.db=3.18.0 \
    bioconductor-enhancedvolcano=1.20.0 \
    bioconductor-summarizedexperiment=1.32.0 \
    bioconductor-apeglm=1.24.0 \
    bioconductor-genomicranges=1.54.1 \
    && mamba clean --all -f -y

# -----------------------------------------------------------------------------
# 7. Install Bioconductor annotation data packages
# -----------------------------------------------------------------------------
# These four tarballs are Bioconductor annotation data packages required by
# DESeq2/GenomeInfoDb and clusterProfiler/enrichGO. They contain large SQLite
# annotation databases (~130 MB total).
#
# Preferred: place the four files next to this Dockerfile before building. This
# avoids downloading them during the build and is much faster/reliable:
#
#   GenomeInfoDbData_1.2.15.tar.gz   (~13 MB)
#   GO.db_3.18.0.tar.gz              (~28 MB)
#   HDO.db_0.99.1.tar.gz             (~2 MB)
#   org.Hs.eg.db_3.18.0.tar.gz       (~94 MB)
#
# Fallback: if the files are not present, the build will automatically download
# them from Bioconductor using the URLs below. Override the base URLs if you
# have a closer mirror.
ARG BIOC_ANNOTATION_URL="https://bioconductor.org/packages/3.18/data/annotation/src/contrib"
ARG BIOC_RELEASE_URL="https://bioconductor.org/packages/release/data/annotation/src/contrib"

# This step uses BuildKit's bind mount to access the build context. BuildKit is
# required (the docker daemon used for the official image has it enabled; set
# DOCKER_BUILDKIT=1 if your local daemon does not).
RUN --mount=type=bind,source=.,target=/build-context,readonly \
    set -eux; \
    fetch_or_copy() { \
        local file="$1"; \
        local base_url="$2"; \
        if [ -f "/build-context/${file}" ]; then \
            echo "Using local ${file}"; \
            cp "/build-context/${file}" "/tmp/${file}"; \
        else \
            echo "Downloading ${file} from ${base_url} ..."; \
            wget -q --timeout=300 --tries=5 --retry-connrefused \
                 "${base_url}/${file}" -O "/tmp/${file}"; \
        fi; \
        test -s "/tmp/${file}"; \
    }; \
    fetch_or_copy "GenomeInfoDbData_1.2.15.tar.gz" "${BIOC_RELEASE_URL}"; \
    fetch_or_copy "GO.db_3.18.0.tar.gz" "${BIOC_ANNOTATION_URL}"; \
    fetch_or_copy "HDO.db_0.99.1.tar.gz" "${BIOC_ANNOTATION_URL}"; \
    fetch_or_copy "org.Hs.eg.db_3.18.0.tar.gz" "${BIOC_ANNOTATION_URL}"

RUN conda run -n ${CONDA_ENV_NAME} R CMD INSTALL --library=/opt/conda/envs/${CONDA_ENV_NAME}/lib/R/library \
        /tmp/GenomeInfoDbData_1.2.15.tar.gz \
    && conda run -n ${CONDA_ENV_NAME} R CMD INSTALL --library=/opt/conda/envs/${CONDA_ENV_NAME}/lib/R/library \
        /tmp/GO.db_3.18.0.tar.gz \
    && conda run -n ${CONDA_ENV_NAME} R CMD INSTALL --library=/opt/conda/envs/${CONDA_ENV_NAME}/lib/R/library \
        /tmp/HDO.db_0.99.1.tar.gz \
    && conda run -n ${CONDA_ENV_NAME} R CMD INSTALL --library=/opt/conda/envs/${CONDA_ENV_NAME}/lib/R/library \
        /tmp/org.Hs.eg.db_3.18.0.tar.gz \
    && rm -f /tmp/*.tar.gz

# -----------------------------------------------------------------------------
# 8. Install Python bioinformatics / data-science packages
# -----------------------------------------------------------------------------
# Installed via mamba first to keep binary dependencies in the conda ecosystem.
RUN mamba install -n ${CONDA_ENV_NAME} -c conda-forge -c bioconda -y \
    pydeseq2=0.4.12 \
    gseapy=1.1.4 \
    mygene=3.2.2 \
    biokit=0.5.0 \
    biopython=1.84 \
    scanpy=1.10.4 \
    anndata=0.11.1 \
    && mamba clean --all -f -y

# -----------------------------------------------------------------------------
# 9. Install remaining Python packages via pip
# -----------------------------------------------------------------------------
# scipy is pinned to 1.13.0 for compatibility with gseapy 1.1.4 (1.17+ causes
# ImportError). Use the Tsinghua PyPI mirror by default for better network
# stability in the target build environment; override with --build-arg.
ARG PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

RUN conda run -n ${CONDA_ENV_NAME} pip install --no-cache-dir -i ${PIP_INDEX_URL} \
    "scipy==1.13.0" \
    "statsmodels==0.14.4" \
    "matplotlib==3.10.0" \
    "seaborn==0.13.2" \
    "numpy==1.26.4" \
    "pandas==2.2.3" \
    "scikit-learn==1.6.0" \
    "plotly==5.24.1" \
    "openpyxl==3.1.5" \
    "cython==3.0.12" \
    "ete3==3.1.3" \
    "fcsparser==0.2.8" \
    "matplotlib-venn==1.1.1" \
    "optuna==4.1.0" \
    "umap-learn==0.5.7" \
    "goatools" \
    "mord" \
    "rpy2==3.5.11" \
    && conda run -n ${CONDA_ENV_NAME} pip cache purge

# -----------------------------------------------------------------------------
# 10. Install command-line bioinformatics tools
# -----------------------------------------------------------------------------
RUN mamba install -n ${CONDA_ENV_NAME} -c conda-forge -c bioconda -y \
    blast=2.16.0 \
    mafft=7.526 \
    iqtree=2.3.6 \
    clipkit=2.3.0 \
    fastqc=0.12.1 \
    phykit=2.0.1 \
    metaeuk=7.bba0d80 \
    spades=4.0.0 \
    && mamba clean --all -f -y

# Clear proxy env vars so they do not leak into the runtime image.
ENV http_proxy="" https_proxy=""

# -----------------------------------------------------------------------------
# 11. Register Jupyter kernels
# -----------------------------------------------------------------------------
RUN conda run -n ${CONDA_ENV_NAME} R -e 'IRkernel::installspec(name = "R", displayname = "R (4.3.3)")' \
    && conda run -n ${CONDA_ENV_NAME} python -m ipykernel install --user \
        --name python3 --display-name "Python 3 (ipykernel)"

# -----------------------------------------------------------------------------
# 12. Final setup
# -----------------------------------------------------------------------------
WORKDIR /workspace

# Keep the same default shell for compatibility.
SHELL ["/bin/bash", "-c"]
