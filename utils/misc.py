import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import make_grid
import torch
import matplotlib.ticker as mtick
import time

def to_dataframe(history):
    output = pd.DataFrame(
    {'val_acc': history["val_acc"],
     'train_acc': history["train_acc"],
     'train_loss': history["train_loss"],
     'val_loss': history["val_loss"],
     'mean_lr': history["lr"],
     'time_elapsed' : history["time_elapsed"]
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
    val_acc = history['val_acc'].to_list()
    train_acc = history['train_acc'].to_list()
    train_loss = history['train_loss'].to_list()
    val_loss = history['val_loss'].to_list()
    mean_lr = history['mean_lr'].to_list()

    fig = plt.figure(figsize=(14, 4))

    plt.subplot(1,3,1)
    plt.plot(train_acc,"-bx")
    plt.plot(val_acc,"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train acc","val acc"])

    plt.subplot(1,3,2)
    plt.plot(train_loss, "-bx")
    plt.plot(val_loss,"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss","val loss"])

    plt.subplot(1,3,3)
    plt.plot(mean_lr)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()
 

'''
def plot_result(model, dl, class_names):
    model.eval()
    for batch in dl:
        fig = plt.figure(figsize=(5,5))
        break
'''

