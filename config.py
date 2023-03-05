DEBUG = False
debug_epochs = 10

#REQUIRED ARGS
img_size = None   #128, 224, 448
model = '' #resnet 18, resnet 34, resnet 50 (resnet 101), wideresnet
depth_scaling = None #for wideresnet only (n)
width_scaling = None #for wideresnet only (k)

###TRAINING
epochs = 100
print_freq = 50
lr = 1.5e-3
batch_size = 32
load_dict = False
train_filename = ''

###DATASET
dataset = 'cub'
download = False

###OPTIMISER
momentum = 0.9
grad_clip = 0.1
weight_decay = 1e-4

###SCHEDULER
scheduler = 'onecycle'
#  cosine
num_restarts = 2

###TESTING
test_filename = 'wrn_28_2_cub_size128_onecycle'
use_range = True
tau_num_steps = 5
tau_min = 0.9
tau_max = 0.99999

#OUTPUT
dict_path = './output'
cloud_dict_path = '../drive/MyDrive/RP3'