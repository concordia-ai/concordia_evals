# BountyBench agent environment.
#
# Upstream runs agents in `cybench/bountyagent`, a Kali Linux image with the CTF
# tool set listed in `packages.list`, Python 3.9 as the default interpreter and
# the `bountytasks` Python requirements. The image is rebuilt from source here so
# the evaluation does not depend on a prebuilt image: only the upstream package
# list is used, taken from
# https://github.com/bountybench/bountybench (commit 8ece6aab).
FROM kalilinux/kali-last-release:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# The base image only ships the official mirrors.
COPY kali.sources /etc/apt/sources.list.d/kali.sources

COPY packages.list /tmp/packages.list
# ca-certificates first: HTTPS mirrors and pip need it, and the Kali base image
# does not ship it. `--no-install-recommends` keeps the upstream package list
# from pulling in large recommendations, so the tools the task scripts call
# directly are listed explicitly.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && xargs -a /tmp/packages.list apt-get install -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        git jq unzip wget docker-cli docker-compose \
        python3 python3-dev python3-venv nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# Upstream builds Python 3.9 for the agent: several invariant suites of the task
# data check that `python` is 3.9.
ARG PYTHON_VERSION=3.9.20
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libbz2-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev \
        libreadline-dev libsqlite3-dev libssl-dev tk-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && wget -q "https://mirrors.huaweicloud.com/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
        -O /tmp/python.tgz \
    && mkdir -p /tmp/python-src \
    && tar xzf /tmp/python.tgz -C /tmp/python-src --strip-components=1 \
    && cd /tmp/python-src \
    && ./configure --prefix=/opt/python3.9 --with-ensurepip=install > /dev/null \
    && make -j"$(nproc)" > /dev/null \
    && make altinstall \
    && cd / \
    && rm -rf /tmp/python-src /tmp/python.tgz

# Make Python 3.9 the default interpreter for the task scripts, as upstream does.
# /usr/bin/python3 keeps pointing at the distribution interpreter so that apt
# and dpkg keep working.
RUN ln -sf /opt/python3.9/bin/python3.9 /usr/local/bin/python3.9 \
    && ln -sf /opt/python3.9/bin/python3.9 /usr/local/bin/python3 \
    && ln -sf /opt/python3.9/bin/python3.9 /usr/local/bin/python \
    && ln -sf /opt/python3.9/bin/pip3.9 /usr/local/bin/pip3 \
    && ln -sf /opt/python3.9/bin/pip3.9 /usr/local/bin/pip \
    # A few task metadata files install their repository with `python3.11 -m pip`.
    && ln -sf /opt/python3.9/bin/python3.9 /usr/local/bin/python3.11

COPY agent_requirements.txt /tmp/agent_requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/agent_requirements.txt

# The target repository is provided as a read-only mount owned by the evaluator
# user, so git must not refuse it as a "dubious ownership" repository.
RUN git config --system --add safe.directory '*'

WORKDIR /app

CMD ["tail", "-f", "/dev/null"]
