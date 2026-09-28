import pandas as pd
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image



def CME(file_name, image_tensor, df, one_hot_df):
    nb_categories = one_hot_df.shape[1]
    _, image_height, image_width = image_tensor.shape
    
    
    # récupération de la ligne
    idx = np.where(df.new_name_file == file_name.replace(".jpg","").split("/")[4])
    one_hot = torch.tensor(one_hot_df.iloc[idx].values, dtype=torch.float32)

    # ---- 2. Expansion spatiale ----
    metadata_channels = one_hot.view(nb_categories, 1, 1).expand(nb_categories, image_height, image_width)

    # ---- 3. Concatenation ----
    image_input = torch.cat([image_tensor, metadata_channels], dim=0)

    return image_input