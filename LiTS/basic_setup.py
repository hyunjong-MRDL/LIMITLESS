import os, torch, random
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = {"SEED": 42,
       "TEST_PORTION": 0.3,
       "EPOCHS": 30,
       "BATCH_SIZE": 16,
       "LR": 1e-4}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True