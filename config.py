###TRAINING
epochs = 90
batch_size = 64
print_freq = 100

###DATASET
dataset = 'flowers'
download = False

###OPTIMISER
optimiser = 'adam'
grad_clip = 0.1
weight_decay = 1e-4

###SCHEDULER
scheduler = 'onecycle'
#  Onecycle
eta_max = 0.01

###LOSS
loss_fn = 'crossentropy'
label_smoothing = 0

###MODEL
model = 'resnet34'
