FROM pytorch/pytorch:latest
RUN apt update && apt install -y git htop nano && pip install debugpy && pip install -U jupyter git+https://github.com/phborba/pytorch_segmentation_models_trainer
CMD ["bash" "-c" "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/github_repos --ip 0.0.0.0 --no-browser --allow-root"]