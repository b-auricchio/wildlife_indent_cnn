import matplotlib.pyplot as plt
import pandas as pd
import csv

def to_dataframe(history):
    output = pd.DataFrame(
    {'Accuracy': history["acc"],
     'Train Loss': history["train_loss"],
     'Val Loss': history["val_loss"],
     'Mean LR': history["lr"]
    })

    return output


def plot_acc(history):
    plt.plot(history["acc"],"-x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

def plot_loss(history):
    plt.plot(history["train_loss"], "-bx")
    plt.plot(history["val_loss"],"-rx")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train loss","val loss"])
    
def plot_lrs(history):
    plt.plot(history["lr"])
    plt.xlabel("Batch number")
    plt.ylabel("Learning rate")