FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /code

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . .

RUN apt update \
    && apt install -y git htop nano libpq-dev python3-dev build-essential python3-opencv python3.12-venv \
    && python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install hatchling \
    && uv pip install debugpy jupyter flake8 pytest parameterized \
    && uv pip install --no-build-isolation .

ENV PATH="/opt/venv/bin:$PATH"

CMD ["jupyter", "notebook", "--ip 0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
