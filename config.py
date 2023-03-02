DEBUG = True
debug_epochs = 10

###TRAINING
epochs = 90
print_freq = 100
eta = 1e-3
batch_size = 32

###DATASET
dataset = 'cub'
download = False
img_size = 128   #128, 224, 448

###OPTIMISER
optimiser = 'adam'
grad_clip = 0.1
weight_decay = 1e-4

###SCHEDULER
scheduler = 'onecycle'
#  cosine
num_restarts = 2

###LOSS
loss_fn = 'crossentropy'

###MODEL
model = 'wideresnet' #resnet 18, resnet 34, resnet 50 (resnet 101), wideresnet

depth_scaling = 4 #for wideresnet only (n)
width_scaling = 10 #for wideresnet only (k)

###TESTING
filename = 'resnet50_cub_64.4acc_90epochs_onecycle'
use_range = True
tau_num_steps = 5
tau_min = 0.999
tau_max = 0.99999

#  if range not used
tau = 0.85

#OUTPUT
dict_path = './output'
cloud_dict_path = '../drive/MyDrive/RP3'