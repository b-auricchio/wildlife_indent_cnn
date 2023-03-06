import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm

def show_batch(dl):
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        plt.show()
        break
        

def get_batch_lrs(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']

def train_accuracy(model, train_dl, batch_size, num_batches):
    correct = 0
    for i in range(num_batches):
        with torch.no_grad():
            X, y = next(iter(train_dl))
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().item()
    size = num_batches*batch_size
    return correct/size

def validate(model,dl,loss_fn):
    losses = []
    model.eval()
    size = dl.setlength()

    correct = 0
    for X,y in dl:
        with torch.no_grad():
            pred = model(X)
            losses.append(loss_fn(pred,y))
            correct += (pred.argmax(1) == y).sum().item()

    loss = torch.stack(losses).mean().item()
    acc = correct/size
    return acc, loss

def fit (batch_size,epochs,train_dl,test_dl,model,loss_fn,optimiser, cfg, scheduler=None,grad_clip=None, print_freq=100):
    torch.cuda.empty_cache()
    
    history = {'lr':[], 'val_loss':[], 'train_loss':[], 'val_acc':[], 'train_acc':[]}
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n ----------------------")

            model.train()

            train_losses = []

            lrs = []
            
            for i, batch in enumerate(train_dl):
                images, labels = batch
                predictions = model(images)
                loss = loss_fn(predictions, labels)
                
                train_losses.append(loss)
                
                optimiser.zero_grad()
                loss.backward()
                
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(),grad_clip)
                
                optimiser.step()

                if cfg.scheduler == 'cosine': 
                    scheduler.step(epoch+ i / len(train_dl))
                elif cfg.scheduler == 'onecycle':
                    scheduler.step()
                else:
                    print('no scheduler being used')

                if i % print_freq == 0:
                    print(f"Batch {i}:  [{batch_size*i:>5d}/{train_dl.setlength():>5d}]")
            
            val_acc, val_loss = validate(model,test_dl,loss_fn)
            train_acc = train_accuracy(model, train_dl, batch_size, 15)

            print(f"\nValidation Error: \n Accuracy: {(val_acc*100):>0.1f}%\n")
            print(f"Train Error: \n Accuracy: {(train_acc*100):>0.1f}%\n")

            train_loss = torch.stack(train_losses).mean().item()
            lr = optimiser.param_groups[0]['lr']

            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['lr'].append(lr)
    except KeyboardInterrupt:
        print('Interrupted training')
    return history

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

def get_scheduler(optim, cfg, epochs, steps_per_epoch=None):
    if cfg.scheduler == 'cosine':
        num_restarts = 2
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=int(epochs/(num_restarts+1)), eta_min=cfg.lr*0.001)

    if cfg.scheduler == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=cfg.lr, steps_per_epoch=steps_per_epoch, epochs=epochs, div_factor=10, final_div_factor=100)



class ToDeviceLoader:
    def __init__(self,dl,device):
        self.dataloader = dl
        self.device = device
        
    def __iter__(self):
        for batch in self.dataloader:
            yield to_device(batch,self.device)
            
    def __len__(self):
        return len(self.dataloader)

    def setlength(self):
        return len(self.dataloader.dataset)