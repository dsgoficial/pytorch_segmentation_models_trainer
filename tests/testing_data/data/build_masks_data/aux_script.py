# %%
import os
from pytorch_segmentation_models_trainer.utils.os_utils import create_folder
import shutil
import pandas as pd

# %%
base_dir = "/media/borba/Mestrado3/datasets/1cgeo/dataset_processado/"
output_dir = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(output_dir, "selected_tiles.csv"))
# %%
output_file = os.path.join(output_dir, df["image_path"][0])
os.path.dirname(output_file)

# %%
for file in df["image_path"]:
    filepath = os.path.join(base_dir, file)
    output_file = os.path.join(output_dir, file)
    create_folder(os.path.dirname(output_file))
    shutil.copyfile(str(filepath), os.path.join(output_dir, file))
# %%
