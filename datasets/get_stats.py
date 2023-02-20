from tqdm import tqdm
import torch
import numpy as np

def get_stats(dataloader):
    count = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tqdm(dataloader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sumsqu = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (count * fst_moment + sum_) / (count + nb_pixels)
        snd_moment = (count * snd_moment + sumsqu) / (count + nb_pixels)
        count += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean, std
  

    

