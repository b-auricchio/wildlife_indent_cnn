DEBUG = True
debug_epochs = 10

###TRAINING
epochs = 90
print_freq = 100
eta = 1e-3
batch_size = 32
load_dict = False
train_filename = ''

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
model = 'resnet34' #resnet 18, resnet 34, resnet 50 (resnet 101), wideresnet

depth_scaling = 4 #for wideresnet only (n)
width_scaling = 10 #for wideresnet only (k)

###TESTING
test_filename = 'wrn_28_10_cub_size128_onecycle'
use_range = True
tau_num_steps = 5
tau_min = 0.999
tau_max = 0.99999

#  if range not used
tau = 0.85

#OUTPUT
dict_path = './output'
cloud_dict_path = '../drive/MyDrive/RP3'