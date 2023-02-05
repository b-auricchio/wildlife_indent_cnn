import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import make_grid
import torch
import time

def to_dataframe(history):
    output = pd.DataFrame(
    {'accuracy': history["acc"],
     'train_loss': history["train_loss"],
     'val_loss': history["val_loss"],
     'mean_lr': history["lr"]
    })

    return output

def show_batch(dl):
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        break
    plt.show()

def export(history, model, time_elapsed):
    history = to_dataframe(history)
    history["time_elapsed"] = [time_elapsed]

    filename = model.name + '_' + time.strftime("%Y%m%d_%H%M%S")

    history.to_csv((filename+'.csv'), encoding='utf-8', index=False)
    torch.save(model.state_dict(), './dict/'+ filename +'.pt')

def plot_history(history):
    acc = history['accuracy'].to_list()
    train_loss = history['train_loss'].to_list()
    val_loss = history['val_loss'].to_list()
    mean_lr = history['mean_lr'].to_list()

    fig = plt.figure(figsize=(13, 4))

    plt.subplot(1,3,1)
    plt.plot(acc,"-x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1,3,2)
    plt.plot(train_loss, "-bx")
    plt.plot(val_loss,"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss","val loss"])

    plt.subplot(1,3,3)
    plt.plot(mean_lr)
    plt.xlabel("Batch number")
    plt.ylabel("Learning rate")

    plt.show()

'''
def plot_result(model, dl, class_names):
    model.eval()
    for batch in dl:
        fig = plt.figure(figsize=(5,5))
        break
'''