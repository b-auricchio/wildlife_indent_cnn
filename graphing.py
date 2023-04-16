import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
import torchvision.transforms as transforms
import random


new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
matplotlib.rcParams.update(new_rc_params)

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

img_size = 224
dataset = 'cub'

dict_path = f'.\\output\\{dataset}'

acc_struct = pd.read_csv(os.path.join(dict_path, 'structured_pruning.csv'))
acc_struct = acc_struct.drop(acc_struct.columns[0], axis=1)
acc_struct = acc_struct.to_numpy()
acc_l1 = pd.read_csv(os.path.join(dict_path, 'unstructured_pruning.csv'))
acc_l1 = acc_l1.drop(acc_l1.columns[0], axis=1)
acc_l1 = acc_l1.to_numpy()

dataset = 'cub'

# dataset = eval(dataset)
# datasets = dataset.get_datasets(img_size, 0, download=False)

model_names = ['resnet10','resnet18', 'resnet18', 'resnet18', 'resnet18', 'resnet34', 'resnet50', 'resnet101']
ks = [1, 1.5, 2, 2.5, 1, 1, 1]
prune_percentage = [0, 0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]

results = pd.read_csv(f'./output/{dataset}/results.csv')
gen_gap = (- results['train_loss'] + results['val_loss']).to_list()
ece = results['ece'].to_list()
auroc = results['auroc'].to_list()
# n_auroc = results['new_auroc'].to_list()
train_acc = results['train_acc'].to_list()
val_acc = results['val_acc'].to_list()
params = results['#params'].to_list()
inf_time = results['inf_time'].to_list()
train_time = results['train_time'].to_list()
val_loss = results['val_loss'].to_list()
names = results['name'].to_list()

def plot_scaling(var1, var2, yoffset=[0]*len(train_acc), xoffset=[0]*len(train_acc), linestyle='--', light=1):
    plt.grid()
    plt.plot(var1[1:5], var2[1:5], marker='o', linestyle = linestyle, color=lighten_color('red', light))
    plt.plot(var1[0:2]+var1[5:9], var2[0:2]+var2[5:9], marker='o', linestyle = linestyle, color=lighten_color('blue', light))
    plt.legend(['Width scaling', 'Depth scaling'])

    # for i in range(len(var1)):
    #     plt.text(var1[i]+xoffset[i], var2[i]+0.01*(max(var2)-min(var2))+yoffset[i], names[i])
    # plt.title(f'{dataset} dataset')


# #Plot 2: Val accuracy vs AUROC
# plt.figure(figsize=(10, 5))
# plot_scaling(val_acc, auroc)
# plt.xlabel('Val Accuracy')
# plt.ylabel('AUROC')
# plt.show()

# plt.figure(figsize=(7, 5))
# plot_scaling(params, gen_gap, [0, 0, 0, 0, 0, 0.0005, -0.0002, 0])
# plt.ylabel('Generalization gap')
# plt.xlabel('No. parameters')
# plt.grid()

# plt.figure(figsize=(8, 4))
# plot_scaling(params, [i*1000 for i in inf_time], yoffset=[1, 1, -7, 1, 1, 1, 1, 0.2], xoffset=[0, 0, 0, 0, -7e6, 0, 0, 0])
# plt.ylabel('Inference Time (ms)')
# plt.xlabel('No. Parameters')
# plt.show()

# auroc vs params
# plt.figure(figsize=(10, 5))
# plot_scaling(params, auroc, [0, 0, 0, 0, 0, 0.0005, -0.0002, 0])
# plot_scaling(params, n_auroc, [0, 0, 0, 0, 0, 0.0005, -0.0002, 0], light=0.4, linestyle='--')
# plt.text(3.45e7, 0.63, 'Beach Litter')
# plt.text(3.39e7, 0.61, 'CSGRT')
# plt.xlabel('No. parameters')
# plt.ylabel('AUROC')
# plt.grid()
# plt.show()

# plt.figure(figsize=(10,5))
# # plt.subplot(1,2,1)
# # plt.title('a)', loc='left')
# plot_scaling(params, [1-i for i in val_acc], [0, 0, 0, 0, 0, 0.0005, -0.0002, 0])
# plt.xlabel('No. parameters')
# plt.ylabel('Test error')
# plt.grid()


# # Plot 4: INF time vs params
# plt.subplot(1,2,2)
# plt.title('b)', loc='left')
# plot_scaling(params, auroc)
# plt.xlabel('No. parameters')
# plt.ylabel('AUROC')
# plt.grid()

def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])




plt.legend(['Structured', 'L1'])
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.xlabel('Pruning Percentage')
plt.ylabel('Accuracy')
plt.title('a)', loc='left')
for i in range(4, len(model_names)):
    plt.plot(prune_percentage, acc_struct[i], marker ='o', linestyle='dashed', label=f'Unstructured {model_names[i]}', color=lighten_color('red', i/6))

for i in range(1,4):
    plt.plot(prune_percentage, acc_struct[i], marker ='o', linestyle='dashed', label=f'Unstructured {model_names[i]}', color=lighten_color('blue', i/4.5))

plt.plot(prune_percentage, acc_struct[0], marker ='o', linestyle='dashed', label=f'Unstructured {model_names[0]}', color='black')



plt.subplot(1, 2, 2)
plt.xlabel('Pruning Percentage')
plt.ylabel('Accuracy')
plt.title('b)', loc='left')
for i in range(4, len(model_names)):
    plt.plot(prune_percentage, acc_l1[i], marker ='o', linestyle='dashed', color=lighten_color('red', i/8))

for i in range(1,4):
    plt.plot(prune_percentage, acc_l1[i], marker ='o', linestyle='dashed', color=lighten_color('blue', i/4.5))

plt.plot(prune_percentage, acc_l1[0], marker ='o', linestyle='dashed', label=f'Unstructured {model_names[0]}', color='black')
plt.show()

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# root = './figure/grads/'
# # Load the images
# img1 = transforms.CenterCrop(280)(Image.open(root + 'albatross.jpg'))
# img2 = transforms.CenterCrop(224)(Image.open(root + 'albatross-18.jpg'))
# img3 = transforms.CenterCrop(224)(Image.open(root + 'albatross-101.jpg'))
# img4 = transforms.CenterCrop(224)(Image.open(root + 'albatross-18.25.jpg'))
# # img5 = transforms.CenterCrop(324)(Image.open(root + '2/2.png'))
# # img6 = transforms.CenterCrop(324)(Image.open(root + '2/3.png'))

# # Convert the images to numpy arrays
# arr1 = np.array(img1)
# arr2 = np.array(img2)
# arr3 = np.array(img3)
# arr4 = np.array(img4)
# # arr5 = np.array(img5)
# # arr6 = np.array(img6)

# # Create a figure and axis object
# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))
# #40, 21
# # Display the images in a grid
# ax[0].imshow(arr1)
# ax[0].set_title('Original', loc='left')
# ax[1].imshow(arr2)
# ax[1].set_title('ResNet18', loc='left')
# ax[2].imshow(arr3)
# ax[2].set_title('ResNet101', loc='left')
# ax[3].imshow(arr4)
# ax[3].set_title('ResNet18-2.5', loc='left')
# # ax[1,0].imshow(arr4)
# # ax[1,0].set_title('Class ID = 21', loc='left')
# # ax[1,1].imshow(arr5)
# # ax[1,2].imshow(arr6)

# # Remove the axis labels
# ax[0].axis('off')
# ax[1].axis('off')
# ax[2].axis('off')
# ax[3].axis('off')

# # ax[1,0].axis('off')
# # ax[1,1].axis('off')
# # ax[1,2].axis('off')

# # Show the plot


# plt.show()