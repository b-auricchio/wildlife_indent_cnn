DEBUG = False
debug_epochs = 1

###TRAINING
epochs = 10
print_freq = 100
eta = 1e-3
batch_size = 128

###DATASET
dataset = 'cub'
download = False
img_size = 224

###OPTIMISER
optimiser = 'adam'
grad_clip = 0.1
weight_decay = 1e-4

###SCHEDULER
scheduler = 'cosine'
#  cosine
num_restarts = 2

###LOSS
loss_fn = 'crossentropy'

###MODEL
model = 'resnet18'
width_scaling = 2

###TESTING
filename = 'resnet50_flowers_91.2acc_90epochs_onecycle'
use_range = True
tau_num_steps = 5
tau_min = 0.99
tau_max = 0.999

#  if range not used
tau = 0.85

#OUTPUT
dict_path = './output'
cloud_dict_path = '../drive/MyDrive/RP3'