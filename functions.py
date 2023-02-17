import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def show_batch(dl):
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        break

def get_lr(optimiser):
    for param_group in optimiser.param_groups:
        return param_group['lr']

def validate(model,loss_fn,dl):
    losses = []
    model.eval()
    size = dl.setlength()

    correct = 0
    for X,y in dl:
        with torch.no_grad():
            pred = model(X)
            losses.append(loss_fn(pred,y))
            correct += (pred.argmax(1) == y).sum().item()

    val_loss = torch.stack(losses).mean().item()
    acc = correct/size

    return val_loss, acc

def fit (batch_size,epochs,train_dl,test_dl,model,loss_fn,optimiser,accum_iter,scheduler=None,grad_clip=None):
    torch.cuda.empty_cache()
    
    history = {'lr':[0], 'val_loss':[0], 'train_loss':[0], 'acc':[0]}


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

            loss.backward()
            
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(),grad_clip)

            if ((i+1) % accum_iter == 0 or (i+1) == len(train_dl)):
                optimiser.zero_grad()
                optimiser.step()
                if scheduler is not None: 
                    scheduler.step()
            
            lrs.append(get_lr(optimiser))

            if i % 100 == 0:
                print(f"Batch {i}:  [{batch_size*i:>5d}/{train_dl.setlength():>5d}]")
                
        
        val_loss, accuracy = validate(model,loss_fn,test_dl)

        print(f"Test Error: \n Accuracy: {(accuracy*100):>0.1f}%\n")

        train_loss = torch.stack(train_losses).mean().item()
        lr = sum(lrs)/batch_size

        history['acc'].append(accuracy)
        history['val_loss'].append(val_loss)
        history['train_loss'].append(train_loss)
        history['lr'].append(lr)

    return history

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

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