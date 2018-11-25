import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
from util import average_meter
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def random_data_loader(opt):
    transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=opt.dataroot,
        transform=transform
    )

    total_pics_size = len(train_dataset)
    indices = list(range(total_pics_size))
    split = int(np.floor(0.01 * total_pics_size))

    np.random.seed(123456)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    valid_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=valid_sampler,
        num_workers=int(opt.num_threads),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        num_workers=int(opt.num_threads),
    )
    train_batch_size = len(train_loader)
    print('#total images        = %d' % total_pics_size)
    print('#training batches    = %d' % train_batch_size)
    print('#valid batches       = %d' % len(valid_loader))

    return train_loader, valid_loader

if __name__ == '__main__':
    opt = TrainOptions().parse()
    # batch中的A，B乱序
    # random_train_loader, random_valid_loader = random_data_loader(opt)
    # batch中的A，B身份一致
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    print('#training images = %d' % len(data_loader))

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    top1 = average_meter.AverageMeter()
    top5 = average_meter.AverageMeter()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()

            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            # if opt.classify and total_steps % opt.validate_freq == 0:
            #     top1.reset()
            #     top5.reset()
            #     for j, (xv, yv) in enumerate(valid_loader):
            #         # measure accuracy
            #         model.set_input(xv, yv)
            #         output, target = model.forward_classify()
            #         prec1, prec5 = average_meter.accuracy(output.data, target.data, topk=(1, 5))
            #         top1.update(prec1[0], xv.size(0))
            #         top5.update(prec5[0], xv.size(0))
            #         log_str = 'Test: [{0}/{1}/{top1.count:}]\tepoch: {epoch:}\t' \
            #                   'top@1: {top1.val:.3f} ({top1.avg:.3f})\t' \
            #                   'top@5: {top5.val:.3f} ({top5.avg:.3f})\t' \
            #                   'prec@: {prec1[0]} <> {prec5[0]}\t'.format(
            #             j, len(valid_loader), epoch=epoch, top1=top1, top5=top5, prec1=prec1, prec5=prec5)
            #     print(log_str)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                if opt.classify:
                    losses['top1'] = top1.avg
                    losses['top5'] = top5.avg
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter), opt, losses)
                    # visualizer.plot_current_losses(epoch, float(epoch_iter) / train_batch_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
