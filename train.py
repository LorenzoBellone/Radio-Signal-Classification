import json
import os
import pandas as pd
import argparse
import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet, MyHisarDataSet
from utils import split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
from importlib import import_module


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device...")

    if os.path.exists(f"./weights/{args.model}_{args.L}_{args.C}/") is False:
        os.makedirs(f"./weights/{args.model}_{args.L}_{args.C}/")
    tb_writer = SummaryWriter(log_dir=f'./runs/{args.model}_{args.L}_{args.C}/' + args.model_name + '/')
    # 获得数据所在的文件路径
    data_dir = os.path.abspath(args.data_path)

    snrs_train = pd.read_csv(f'{data_dir}/Train/train_snr.csv', header=None)
    train_indexes = snrs_train[snrs_train[0]==args.snr].index.tolist()

    print("using train data size: {}".format(len(train_indexes)))

    # 实例化训练数据集
    train_dataset = MyHisarDataSet(hdf5_path=f'{data_dir}/Train/complex_train/complex_train_snr{args.snr}.h5',
                              labels_path=f'{data_dir}/Train/train_labels.csv',
                              indexes=train_indexes)

    snrs_test = pd.read_csv(f'{data_dir}/Test/test_snr.csv', header=None)
    test_indexes = snrs_test[snrs_test[0]==args.snr].index.tolist()

    print("using valid data size: {}".format(len(test_indexes)))

    # 实例化验证数据集
    val_dataset = MyHisarDataSet(hdf5_path=f'{data_dir}/Test/complex_test/complex_test_snr{args.snr}.h5',
                            labels_path=f'{data_dir}/Test/test_labels.csv',
                            indexes=test_indexes)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=False,
                                             num_workers=nw)
    model = import_module('models.' + args.model[:args.model.index('t')+1])
    if args.model == "convnet":
        net = model.net(num_classes=args.num_classes, L=args.L, C=args.C).to(device)
    elif args.model == "convnetBN":
        net = model.netBN(num_classes=args.num_classes, L=args.L, C=args.C).to(device)
    else:
        net = model.net(num_classes=args.num_classes).to(device)
    pg = get_params_groups(net, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=net,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=net,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < val_acc:
            torch.save(net.state_dict(), f"./weights/{args.model}_{args.L}_{args.C}/" + args.model_name + ".pth")
            best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnext', help='convnext, vision_transformer, or convnet')
    parser.add_argument('--L', type=int, default=1, help='Number of convolutional layers')
    parser.add_argument('--C', type=int, default=25, help='Number of filters per convolutional layer')
    parser.add_argument('--model_name', type=str, default='convnext', help='name of the model containing the weights')
    parser.add_argument('--snr', type=int, default=30)
    parser.add_argument('--num-classes', type=int, default=26)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # 数据集目录全称
    parser.add_argument('--data-path', type=str,
                        default="./data/2018/GOLD_XYZ_OSC.hdf5")  # hdf5.File()读取文件不支持相对路径

    opt = parser.parse_args()

    print("------------------------------------------------------------------")
    print(f"Training with model {opt.model}_{opt.L}_{opt.C} for snr {opt.snr}")
    print("------------------------------------------------------------------")

    main(opt)
