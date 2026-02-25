from config import args as args_config
import time
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.autograd.set_detect_anomaly(True)

import utility
from model.EndoDDC import EndoDDC
from summary.gcsummary import EndoDDCSummary
from metric.dcmetric import DCMetric
from data import get as get_data
from loss.loss_all import Loss_All

# Multi-GPU and Mixed precision supports
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.cuda.amp as amp 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.start_epoch = checkpoint['epoch'] + 1
    return new_args

def train(gpu, args):
    # Initialize workers
    if args.multiprocessing:
        dist.init_process_group(backend='nccl', init_method='env://',
                               world_size=args.num_gpus, rank=gpu)
    else:
        dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.tcp_port}',
                               world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    try:
        data = get_data(args)
        data_train = data(args, 'train')
        data_val = data(args, 'val')

        sampler_train = DistributedSampler(
            data_train, num_replicas=args.num_gpus, rank=gpu)

        batch_size = args.batch_size

        loader_train = DataLoader(
            dataset=data_train, batch_size=batch_size, shuffle=False,
            num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
            drop_last=True, persistent_workers=True)
        loader_val = DataLoader(
            dataset=data_val, batch_size=1, shuffle=False,
            num_workers=4, drop_last=True, persistent_workers=True)

        if gpu == 0:
            print(f'Each GPU with training data {len(loader_train)}, validation data {len(loader_val)}!')

        if args.model == 'EndoDDC':
            net = EndoDDC(args)
        else:
            raise TypeError(args.model, ['EndoDDC', ])
        net.cuda(gpu)

        # Load pretrained model
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            net.load_state_dict(checkpoint['net'], strict=False)
            print('Load network parameters from : {}'.format(args.pretrain))

        # Loss
        if args.model == 'EndoDDC':
            loss_all = Loss_All(args)
            summ = EndoDDCSummary
        else:
            raise NotImplementedError

        loss_all.cuda(gpu)

        # Optimizer
        optimizer, scheduler = utility.make_optimizer_scheduler(args, net, len(loader_train))
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        
        # PyTorch AMP initialization
        scaler = amp.GradScaler(enabled=(args.opt_level != 'O0'))

        # Resume training
        if args.pretrain is not None and args.resume:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                scheduler.milestones = Counter(args.milestones)
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                print('Resume optimizer, scheduler and scaler from : {}'.format(args.pretrain))
            except KeyError:
                print('State dicts for resume are not saved. Use --save_full argument')

        net = DDP(net, device_ids=[gpu], find_unused_parameters=True)
        metric = DCMetric(args)
        best_val_rmse = 1e10

        if gpu == 0:
            utility.backup_source_code(args.save_dir + '/code')
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
            
            writer_train = summ(args.save_dir, 'train', args,
                                loss_all.loss_name, metric.metric_name)
            writer_val = summ(args.save_dir, 'val', args,
                              loss_all.loss_name, metric.metric_name)

            with open(args.save_dir + '/args.json', 'w') as args_json:
                json.dump(args.__dict__, args_json, indent=4)

        if args.warm_up:
            warm_up_cnt = 0.0
            warm_up_max_cnt = len(loader_train) + 1.0

        for epoch in range(args.start_epoch, args.epochs + 1):
            net.train()
            sampler_train.set_epoch(epoch)

            if gpu == 0:
                current_time = time.strftime('%y%m%d@%H:%M:%S')
                list_lr = [g['lr'] for g in optimizer.param_groups]
                print(f'=== Epoch {epoch:5d} / {args.epochs:5d} | Lr : {list_lr} | {current_time} ===')

            num_sample = len(loader_train) * loader_train.batch_size * args.num_gpus
            if gpu == 0:
                pbar = tqdm(total=num_sample)
                log_cnt = 0.0
                log_loss = 0.0

            init_seed(seed=int(time.time()))
            for batch, sample in enumerate(loader_train):
                sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}

                if torch.any(torch.isnan(sample['rgb'])) or torch.any(torch.isinf(sample['rgb'])):
                    print(f"NaN or Inf detected in input data at epoch {epoch}, batch {batch}")
                    break

                if epoch == 1 and args.warm_up:
                    warm_up_cnt += 1
                    for param_group in optimizer.param_groups:
                        lr_warm_up = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt
                        param_group['lr'] = lr_warm_up

                optimizer.zero_grad()

                with amp.autocast(enabled=(args.opt_level != 'O0')):
                    output = net(sample)
                    if torch.any(torch.isnan(output['pred'])):
                        print(f"NaN detected in model output at epoch {epoch}, batch {batch}")
                        break
                    loss_sum_all, loss_val_all = loss_all(sample, output)
                    loss_sum_all = loss_sum_all / loader_train.batch_size
                    loss_val_all = loss_val_all / loader_train.batch_size

                scaler.scale(loss_sum_all).backward()
                scaler.step(optimizer)
                scaler.update()

                if gpu == 0:
                    metric_val = metric.evaluate(sample, output, 'train')
                    writer_train.add(loss_val_all, metric_val)
                    log_cnt += 1
                    log_loss += loss_sum_all.item()

                    if batch % args.print_freq == 0:
                        current_time = time.strftime('%y%m%d@%H:%M:%S')
                        error_str = '{:<10s}| {} | Loss = {:.4f}'.format('Train', current_time, log_loss / log_cnt)
                        pbar.set_description(error_str)
                        pbar.update(loader_train.batch_size * args.num_gpus)

            scheduler.step()

            if gpu == 0:
                pbar.close()
                writer_train.update(epoch, sample, output)
                state = {
                    'epoch': epoch,
                    'net': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'args': args
                }
                torch.save(state, '{}/model_latest.pt'.format(args.save_dir))
                if epoch % 10 == 1 or epoch == args.epochs:
                    torch.save(state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))

            # Validation
            torch.set_grad_enabled(False)
            net.eval()
            if gpu == 0:
                pbar = tqdm(total=len(loader_val))
                log_cnt = 0.0
                log_loss = 0.0

            init_seed()
            for batch, sample in enumerate(loader_val):
                sample = {key: val.cuda(gpu) for key, val in sample.items() if val is not None}
                output = net(sample)
                loss_sum_all, loss_val_all = loss_all(sample, output)
                
                if gpu == 0:
                    metric_val = metric.evaluate(sample, output, 'val')
                    writer_val.add(loss_val_all / loader_val.batch_size, metric_val)
                    log_cnt += 1
                    log_loss += loss_sum_all.item()
                    if batch % args.print_freq == 0:
                        pbar.update(1)

            if gpu == 0:
                pbar.close()
                rmse = writer_val.update(epoch, sample, output)
                if rmse < best_val_rmse:
                    best_val_rmse = rmse
                    torch.save(state, '{}/model_best.pt'.format(args.save_dir))
                writer_val.add_scalar('Loss/epoch_val', log_loss / log_cnt, epoch)
                writer_val.add_scalar('RMSE/epoch_val', rmse, epoch)

            torch.set_grad_enabled(True)

        if gpu == 0:
            writer_train.close()
            writer_val.close()

    except KeyboardInterrupt:
        if gpu == 0:
            print(f"\n[GPU {gpu}] A manual interrupt (KeyboardInterrupt) has been detected. Preparing for safe shutdown...")
    except Exception as e:
        print(f"[GPU {gpu}] malfunction: {e}")
        raise e
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            if gpu == 0:
                print(f"[GPU {gpu}] NCCL process group has been securely destroyed, preventing zombie processes.")

def test(args):
    data = get_data(args)
    data_test = data(args, 'test')
    loader_test = DataLoader(dataset=data_test, batch_size=1, shuffle=False, num_workers=args.num_threads)

    net = EndoDDC(args)
    net.cuda()

    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        net.load_state_dict(checkpoint['net'], strict=False)
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)
    metric = DCMetric(args)
    os.makedirs(args.save_dir + '/test', exist_ok=True)
    writer_test = EndoDDCSummary(args.save_dir, 'test', args, None, metric.metric_name)

    net.eval()
    pbar = tqdm(total=len(loader_test))
    init_seed()
    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items() if val is not None}
        with torch.no_grad():
            output = net(sample)
        
        if args.test_augment:
            sample_fliplr = {key: torch.clone(val) for key, val in sample.items() if val is not None}
            sample_fliplr['rgb'] = torch.flip(sample_fliplr['rgb'], (3,))
            output_flip = net(sample_fliplr)
            output['pred'] = (torch.flip(output_flip['pred'], (3,)) + output['pred']) / 2.0

        metric_val = metric.evaluate(sample, output, 'test')
        writer_test.add(None, metric_val)
        writer_test.save(args.epochs, batch, sample, output)
        pbar.update(1)

    pbar.close()
    writer_test.update(args.epochs, sample, output)

def main(args):
    init_seed()
    if not args.test_only:
        if not args.multiprocessing:
            train(0, args)
        else:
            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,), join=False)
            while not spawn_context.join(): pass
    
    args.pretrain = '{}/model_best.pt'.format(args.save_dir)
    test(args)

if __name__ == '__main__':
    args_main = check_args(args_config)
    print('\n=== Arguments ===')
    for i, (key, val) in enumerate(sorted(vars(args_main).items())):
        print(f"{key}: {val}", end='  |  ')
        if (i + 1) % 4 == 0: print('')
    print('\n')
    main(args_main)