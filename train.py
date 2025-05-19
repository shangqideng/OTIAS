import torch.optim
from model.otias import *
import args_parser
from os.path import exists, join, basename
from torch.utils.data import DataLoader
from tools.dataset import DatasetFromHdf5
import os
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
import random
import torch.backends.cudnn as cudnn
from tools.evaluate import analysis_accu

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True

def get_training_set(root_dir):
    train_dir = join(root_dir, "train_cave(with_up)x4.h5")
    return DatasetFromHdf5(train_dir)

def get_val_set(root_dir):
    val_dir = join(root_dir, "validation_cave(with_up)x4.h5")
    return DatasetFromHdf5(val_dir)

opt = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(opt)

def save_checkpoint(model, epoch, t, data):
    model_out_path = "checkpoints/{}_{}_{}/model_epoch_{}.pth.tar".format(opt.arch, data,t,epoch)
    state = {"epoch": epoch, "model": model}

    if not os.path.exists("checkpoints/{}_{}_{}".format(opt.arch, data,t,epoch)):
        os.makedirs("checkpoints/{}_{}_{}".format(opt.arch, data, t,epoch))

    torch.save(state, model_out_path)

    print("Checkpoints saved to {}".format(model_out_path))

def main():
    # load data
    print('===> Loading datasets')
    train_set = get_training_set(opt.dataroot)
    val_set = get_val_set(opt.dataroot)

    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, batch_size=opt.testBatchSize, shuffle=False)

    if opt.dataset == 'cave_x4':
        opt.n_bands = 31
        opt.image_size = 64
        opt.n_bands_rgb = 3
        model = otias_c_x4(n_select_bands=3, n_bands=opt.n_bands, feat_dim=128,
                           guide_dim=128, sz=opt.image_size).cuda()

    elif opt.dataset == 'harvard_x4':
        opt.n_bands = 31
        opt.image_size = 64
        opt.n_bands_rgb = 3
        model = otias_h_x4(n_select_bands=3, n_bands=opt.n_bands, feat_dim=128,
                           guide_dim=128, sz=opt.image_size).cuda()


    # Loss and optimizer
    L1 = nn.L1Loss().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)  ## optimizer 1: AdamW

    # Load the trained model parameters

    if os.path.isfile(opt.model_path):
        print("=> loading checkpoint '{}'".format(opt.model_path))
        checkpoint = torch.load(opt.model_path)
        # print(checkpoint['state_dict'].keys())
        # opt.start_epochs = checkpoint["epoch"] + 1
        #
        # # model = torch.load(opt.model_path)
        # state_dict = checkpoint['state_dict']
        # dict = {}
        # for module in state_dict.items():
        #     k, v = module
        #     if 'model' in k:
        #         k = k.strip('model.')
        #     dict[k] = v
        # checkpoint['state_dict'] = dict
        # model.load_state_dict(checkpoint['state_dict'])
        opt.start_epochs = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.model_path))

    model.train()
    print ('Start Training: ')
    t = time.strftime("%Y%m%d%H%M")
    for epoch in range(opt.start_epochs, opt.n_epochs+1):
        # One epoch's training
        print ('Train_Epoch_{}: '.format(epoch))
        print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

        for iteration, batch in enumerate(training_data_loader, 1):
            input_rgb, ms, input_lr_u, ref = Variable(batch[0]).cuda(), Variable(batch[1]).cuda(), Variable( batch[2]).cuda(), Variable(batch[3], requires_grad=False).cuda()
            out = model(input_rgb, input_lr_u, ms)

            loss = L1(out, ref)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                    loss.item()))
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                total_metrics = {'SAM': 0.0, 'ERGAS': 0.0, 'PSNR': 0.0}
                count = 0
                for index, batch in enumerate(val_data_loader):
                    input_rgb, ms, input_lr_u, ref = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
                    out = model(input_rgb, input_lr_u, ms)

                    metrics = analysis_accu(ref[0].permute(1, 2, 0), out[0].permute(1, 2, 0), ratio=4)
                    for key in total_metrics:
                        total_metrics[key] += metrics[key].item() if isinstance(metrics[key], torch.Tensor) else metrics[key]
                    count += 1

            avg_metrics = {k: total_metrics[k] / count for k in total_metrics}
            print(avg_metrics)
        if epoch % 50 == 0:
            save_checkpoint(model, epoch, t, opt.dataset)


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DIVICES"] = "0"
    set_random_seed(10)
    main()
