# Genome Firewall — Streamlit app + AMRFinderPlus, self-contained.
#
# Bakes the AMRFinderPlus binary AND its AMR database into the image, so a
# freshly UPLOADED FASTA can be annotated with no host setup. Bundled example
# genomes work too (their AMRFinderPlus output is cached in the repo).
#
# Build (from the repo root):   docker build -t genome-firewall .
# Run:                          docker run --rm -p 8501:8501 genome-firewall
# Then open:                    http://localhost:8501
#
# Note: the build downloads the AMR database (~hundreds of MB) via `amrfinder -u`,
# so the FIRST build takes ~10-20 min and the image is ~2-3 GB. It is cached
# after that.

FROM mambaorg/micromamba:1.5.8

# AMRFinderPlus (+ its blast/hmmer deps) and a matching Python, from bioconda.
RUN micromamba install -y -n base -c conda-forge -c bioconda \
        python=3.11 \
        ncbi-amrfinderplus=4.2.7 \
    && micromamba clean --all --yes

# Activate the env for the remaining RUN steps so `amrfinder`/`pip` are found.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Put the conda env on PATH for EVERY process at runtime — not just steps that go
# through micromamba's activation entrypoint. Without this, Streamlit can start
# (via the entrypoint) yet its `amrfinder` subprocess inherits a PATH without
# /opt/conda/bin and fails with "amrfinder not found on PATH".
ENV PATH=/opt/conda/bin:$PATH

# Python deps for the app.
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download the AMR database INTO the image (so it isn't fetched on every start).
RUN amrfinder -u

WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

EXPOSE 8501
# CORS/XSRF are disabled so the app also works when embedded in an iframe or
# reverse-proxied (Hugging Face Spaces, a cloudflared tunnel). Harmless locally.
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
