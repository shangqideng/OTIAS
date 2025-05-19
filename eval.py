import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import h5py
from os.path import exists, join, basename
import torch.utils.data as data
from model.otias import *
from tools.split import model_fuse_patchwise
from tools.evaluate import analysis_accu
from tools.dataset import DatasetFromHdf5
import matplotlib
matplotlib.use('TkAgg')

def get_test_set(root_dir, name):
    train_dir = join(root_dir, name)
    return DatasetFromHdf5(train_dir)


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--dataset', type=str, default='cave_x4')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
opt = parser.parse_args()


def test():
    if opt.dataset == 'cave_x4':
        opt.n_bands = 31
        opt.image_size = 512
        opt.n_bands_rgb = 3
        opt.root = 'E:/HSI_data/cave_x4'
        opt.model_path = './pretrained model/cave_x4/1000.pth.tar'
        opt.testBatchSize = 1
        model = otias_c_x4(n_select_bands=3, n_bands=opt.n_bands, feat_dim=128,
                           guide_dim=128, sz=512).cuda()
        output = np.zeros((11, opt.image_size, opt.image_size, opt.n_bands))
        test_set = get_test_set(opt.root, "test_cave(with_up)x4.h5")

    elif opt.dataset == 'cave_x8':
        opt.n_bands = 31
        opt.image_size = 512
        opt.n_bands_rgb = 3
        opt.root = 'E:/HSI_data/cave_x8'
        opt.model_path = './pretrained model/cave_x8/1000.pth.tar'
        opt.testBatchSize = 1
        model = otias_c_x8(n_select_bands=3, n_bands=opt.n_bands, feat_dim=128,
                           guide_dim=128, sz=512).cuda()
        output = np.zeros((11, opt.image_size, opt.image_size, opt.n_bands))
        test_set = get_test_set(opt.root, "test_cave(with_up)x8.h5")

    elif opt.dataset == 'harvard_x4':
        opt.n_bands = 31
        opt.image_size = 1000
        opt.n_bands_rgb = 3
        opt.root = 'E:/HSI_data/harvard_x4'
        opt.model_path = './pretrained model/harvard_x4/1000.pth.tar'
        opt.testBatchSize = 1
        model = otias_h_x4(n_select_bands=3, n_bands=opt.n_bands, feat_dim=128,
                         guide_dim=128, sz=200).cuda()
        output = np.zeros((10, opt.image_size, opt.image_size, opt.n_bands))
        test_set = get_test_set(opt.root, "test_harvard(with_up)x4.h5")

    elif opt.dataset == 'harvard_x8':
        opt.n_bands = 31
        opt.image_size = 1000
        opt.n_bands_rgb = 3
        opt.root = 'E:/HSI_data/harvard_x8'
        opt.model_path = './pretrained model/harvard_x8/1000.pth.tar'
        opt.testBatchSize = 1
        model = otias_h_x8(n_select_bands=3, n_bands=opt.n_bands, feat_dim=128,
                         guide_dim=128, sz=200).cuda()
        output = np.zeros((10, opt.image_size, opt.image_size, opt.n_bands))
        test_set = get_test_set(opt.root, "test_harvard(with_up)x8.h5")


    elif opt.dataset == 'chikusei_x4':
        opt.n_bands = 128
        opt.image_size = 680
        opt.n_bands_rgb = 3
        opt.root = 'E:/HSI_data/chikusei_x4'
        opt.model_path = './pretrained model/chikusei_x4/1000.pth.tar'
        opt.testBatchSize = 1
        model = otias_ch_x4(n_select_bands=3, n_bands=opt.n_bands, feat_dim=192,
                         guide_dim=192, sz=340).cuda()
        output = np.zeros((6, opt.image_size, opt.image_size, opt.n_bands))
        test_set = get_test_set(opt.root, "test_Chikusei.h5")



    test_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)


    checkpoint = torch.load(opt.model_path)

    # model.load_state_dict(checkpoint["model"].state_dict())
    # model = torch.load(opt.model_path)

    # if you want to use the pretrained checkpoints, use the blow code instead.
    state_dict = checkpoint['state_dict']
    dict = {}
    for module in state_dict.items():
        k, v = module
        if 'model' in k:
            k = k.replace('model.','')
        dict[k] = v
    checkpoint['state_dict'] = dict
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    total_metrics = {'SAM': 0.0, 'ERGAS': 0.0, 'PSNR': 0.0}
    count = 0

    for index, batch in enumerate(test_data_loader):
        input_rgb, ms, input_lr_u, ref = Variable(batch[0]).cuda(), Variable(batch[1],).cuda(), Variable(batch[2]).cuda(), Variable(batch[3]).cuda()

        # ref = ref.cuda()

        if opt.dataset == 'cave_x4':
            out = model(input_rgb, input_lr_u, ms)
            metrics = analysis_accu(ref[0].permute(1, 2, 0), out[0].permute(1, 2, 0), ratio=4)

        elif opt.dataset == 'cave_x8':
            out = model(input_rgb, input_lr_u, ms)
            metrics = analysis_accu(ref[0].permute(1, 2, 0), out[0].permute(1, 2, 0), ratio=8)

        elif opt.dataset == 'harvard_x4':
            out = model_fuse_patchwise(model, input_rgb, input_lr_u, ms, patch_size=200, stride=200, dim=opt.n_bands, rate=4)
            metrics = analysis_accu(ref[0].permute(1, 2, 0), out[0].permute(1, 2, 0), ratio=4)

        elif opt.dataset == 'harvard_x8':
            out = model_fuse_patchwise(model, input_rgb, input_lr_u, ms, patch_size=200, stride=200, dim=opt.n_bands, rate=8)
            metrics = analysis_accu(ref[0].permute(1, 2, 0), out[0].permute(1, 2, 0), ratio=4)

        elif opt.dataset == 'chikusei_x4':
            out = model_fuse_patchwise(model, input_rgb, input_lr_u, ms, patch_size=340, stride=340, dim=opt.n_bands, rate=4)
            metrics = analysis_accu(ref[0].permute(1, 2, 0), out[0].permute(1, 2, 0), ratio=4)

        # # show pic
        # bands_to_show = [10, 20, 30]
        # img = out[0]
        # rgb = img[bands_to_show, :, :]
        # rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
        # rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(rgb_np)
        # plt.title("Selected Bands as RGB")
        # plt.axis('off')
        # plt.show()


        print(f"Sample {index}: {metrics}")

        for key in total_metrics:
            total_metrics[key] += metrics[key].item() if isinstance(metrics[key], torch.Tensor) else metrics[key]

        count += 1
        output[index, :, :, :] = out.permute(0, 2, 3, 1).cpu().detach().numpy()
    avg_metrics = {k: total_metrics[k] / count for k in total_metrics}
    print(f"\nAverage Results over {count} samples:")
    print(avg_metrics)

    # sio.savemat('cave11_x4-otias.mat', {'output': output})


test()











