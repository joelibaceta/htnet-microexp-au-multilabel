import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class AUOpticalFlowDataset(Dataset):
    def __init__(self, csv_path, images_folder, transform=None):
        self.data = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform

        # Detectar columnas AU_
        self.au_columns = [col for col in self.data.columns if col.startswith("AU_")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row["sub_filename"]
        image_path = os.path.join(self.images_folder, filename)
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Etiquetas AU
        labels = row[self.au_columns].values.astype("float32")
        return image, torch.tensor(labels)