from utils.rotation import *
import torch.nn as nn
import numpy as np

def embed(dataloader, model):
    model.eval()
    count = 0
    embed_list = []
    labels_list = []
    output_list = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        count += 1
        # inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            embed = model.get_embedding(inputs)
            outputs = model(inputs)
        embed_list.append(embed.detach().numpy())
        labels_list.append(labels.detach().numpy())
        output_list.append(outputs.detach().numpy())
        if count >= 4:
            break
    # for batch_idx, (inputs, labels) in enumerate(dataloader):
    #     with torch.no_grad():
    #         embed = model.get_embedding(inputs)
    #         outputs = model(inputs)
    #     break
    return embed_list, output_list, labels_list