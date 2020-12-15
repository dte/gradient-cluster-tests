from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os


run_logdir = './logs'
writer = SummaryWriter(log_dir=run_logdir)
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

f = []
for (_, _, filenames) in os.walk(run_logdir):
    f.extend(filenames)
print(f)
