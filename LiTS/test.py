import torch
from basic_setup import CFG, device
from construct_dataset import test_loader
from construct_model import LiTSModel
from plot_results import test_plot

def test_loop(dataloader, model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pred=[]
    target=[]

    for (X,y) in dataloader:
        for t in y:
            target.append(t[1].detach().tolist())

        X = X.to(device).float()
        y = y.to(device).float()

        output = model(X)

        for o in output:
            print(f"Output: {o}")
            pred.append(o[1].detach().cpu().tolist())

    return target, pred

model = LiTSModel().to(device)

target, pred = test_loop(test_loader, model, CFG["model_save_path"])

test_plot(target, pred)