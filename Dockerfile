FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /code

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml .

RUN apt update \
    && apt install -y git htop nano libpq-dev python3-dev build-essential python3-opencv \
    && uv pip install --system debugpy jupyter flake8 pytest parameterized \
    && uv pip install --system -e .

COPY . .
CMD ["jupyter", "notebook", "--ip 0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
