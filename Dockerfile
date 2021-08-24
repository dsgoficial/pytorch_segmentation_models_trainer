FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /code
COPY requirements.txt requirements.txt
RUN apt update \
    && apt install -y git htop nano libpq-dev python3-dev build-essential python3-opencv \
    && pip3 install -U debugpy jupyter flake8 pytest parameterized \
    && pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html \
    && pip3 install -r requirements.txt
COPY . .
RUN  pip3 install .
CMD ["jupyter", "notebook", "--ip 0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
