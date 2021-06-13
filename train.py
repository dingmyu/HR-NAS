"""Train and val."""
import logging
import os
import time

import torch
import subprocess

from utils.config import FLAGS, _ENV_EXPAND
from utils.common import get_params_by_name
from utils.common import set_random_seed
from utils.common import create_exp_dir
from utils.common import setup_logging
from utils.common import save_status
from utils.common import get_device
from utils.common import extract_item
from utils.common import get_data_queue_size
from utils.common import bn_calibration
from utils import dataflow
from utils import optim
from utils import distributed as udist
from utils import prune
from mmseg import seg_dataflow
from mmseg.loss import CrossEntropyLoss, JointsMSELoss, accuracy_keypoint

import models.mobilenet_base as mb
import common as mc
from mmseg.validation import SegVal, keypoint_val


def shrink_model(model_wrapper,
                 ema,
                 optimizer,
                 prune_info,
                 threshold=1e-3,
                 ema_only=False):
    r"""Dynamic network shrinkage to discard dead atomic blocks.

    Args:
        model_wrapper: model to be shrinked.
        ema: An instance of `ExponentialMovingAverage`, could be None.
        optimizer: Global optimizer.
        prune_info: An instance of `PruneInfo`, could be None.
        threshold: A small enough constant.
        ema_only: If `True`, regard an atomic block as dead only when
            `$$\hat{alpha} \le threshold$$`. Otherwise use both current value
            and momentum version.
    """
    model = mc.unwrap_model(model_wrapper)
    for block_name, block in model.get_named_block_list().items():  # inverted residual blocks
        assert isinstance(block, mb.InvertedResidualChannels)
        masks = [
            bn.weight.detach().abs() > threshold
            for bn in block.get_depthwise_bn()
        ]
        if ema is not None:
            masks_ema = [
                ema.average('{}.{}.weight'.format(
                    block_name, name)).detach().abs() > threshold
                for name in block.get_named_depthwise_bn().keys()
            ]
            if not ema_only:
                masks = [
                    mask0 | mask1 for mask0, mask1 in zip(masks, masks_ema)
                ]
            else:
                masks = masks_ema
        block.compress_by_mask(masks,
                               ema=ema,
                               optimizer=optimizer,
                               prune_info=prune_info,
                               prefix=block_name,
                               verbose=False)

    if optimizer is not None:
        assert set(optimizer.param_groups[0]['params']) == set(
            model.parameters())

    mc.model_profiling(model,
                       FLAGS.image_size,
                       FLAGS.image_size,
                       num_forwards=0,
                       verbose=False)
    if udist.is_master():
        logging.info('Model Shrink to FLOPS: {}'.format(model.n_macs))
        logging.info('Current model: {}'.format(mb.output_network(model)))


def get_prune_weights(model, use_transformer=False):
    """Get variables for pruning."""
    # ['features.2.ops.0.1.1.weight', 'features.2.ops.1.1.1.weight', 'features.2.ops.2.1.1.weight'...]
    if use_transformer:
        return get_params_by_name(mc.unwrap_model(model), FLAGS._bn_to_prune_transformer.weight)
    return get_params_by_name(mc.unwrap_model(model), FLAGS._bn_to_prune.weight)


@udist.master_only
def summary_bn(model, prefix):
    """Summary BN's weights."""
    weights = get_prune_weights(model)
    for name, param in zip(FLAGS._bn_to_prune.weight, weights):
        mc.summary_writer.add_histogram(
            '{}/{}/{}'.format(prefix, 'bn_scale', name), param.detach(),
            FLAGS._global_step)
    if len(FLAGS._bn_to_prune.weight) > 0:
        mc.summary_writer.add_histogram(
            '{}/bn_scale/all'.format(prefix),
            torch.cat([weight.detach() for weight in weights]),
            FLAGS._global_step)


@udist.master_only
def log_pruned_info(model, flops_pruned, infos, prune_threshold):
    """Log pruning-related information."""
    if udist.is_master():
        logging.info('Flops threshold: {}'.format(prune_threshold))
        for info in infos:
            if FLAGS.prune_params['logging_verbose']:
                logging.info(
                    'layer {}, total channel: {}, pruned channel: {}, flops'
                    ' total: {}, flops pruned: {}, pruned rate: {:.3f}'.format(
                        *info))
            mc.summary_writer.add_scalar(
                'prune_ratio/{}/{}'.format(prune_threshold, info[0]), info[-1],
                FLAGS._global_step)
        logging.info('Pruned model: {}'.format(
            prune.output_searched_network(model, infos, FLAGS.prune_params)))

    flops_remain = model.n_macs - flops_pruned
    if udist.is_master():
        logging.info(
            'Prune threshold: {}, flops pruned: {}, flops remain: {}'.format(
                prune_threshold, flops_pruned, flops_remain))
        mc.summary_writer.add_scalar('prune/flops/{}'.format(prune_threshold),
                                     flops_remain, FLAGS._global_step)


def run_one_epoch(epoch,
                  loader,
                  model,
                  criterion,
                  optimizer,
                  lr_scheduler,
                  ema,
                  rho_scheduler,
                  meters,
                  max_iter=None,
                  phase='train'):
    """Run one epoch."""
    assert phase in [
        'train', 'val', 'test', 'bn_calibration'
    ] or phase.startswith(
        'prune'), "phase not be in train/val/test/bn_calibration/prune."
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
    if phase == 'bn_calibration':
        model.apply(bn_calibration)

    if not FLAGS.use_hdfs:
        if FLAGS.use_distributed:
            loader.sampler.set_epoch(epoch)

    results = None
    data_iterator = iter(loader)
    if not FLAGS.use_hdfs:
        if FLAGS.use_distributed:
            if FLAGS.dataset == 'coco':
                data_fetcher = dataflow.DataPrefetcherKeypoint(data_iterator)
            else:
                data_fetcher = dataflow.DataPrefetcher(data_iterator)
        else:
            logging.warning('Not use prefetcher')
            data_fetcher = data_iterator

    for batch_idx, data in enumerate(data_fetcher):
        if FLAGS.dataset == 'coco':
            input, target, target_weight, meta = data
            # print(input.shape, target.shape, target_weight.shape, meta)
            # (4, 3, 384, 288), (4, 17, 96, 72), (4, 17, 1),
        else:
            input, target = data
        # if batch_idx > 400:
        #     break
        # used for bn calibration
        if max_iter is not None:
            assert phase == 'bn_calibration'
            if batch_idx >= max_iter:
                break

        target = target.cuda(non_blocking=True)
        if train:
            optimizer.zero_grad()
            rho = rho_scheduler(FLAGS._global_step)

            if FLAGS.dataset == 'coco':
                outputs = model(input)
                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += criterion(output, target, target_weight)
                else:
                    output = outputs
                    loss = criterion(output, target, target_weight)
                _, avg_acc, cnt, pred = accuracy_keypoint(output.detach().cpu().numpy(),
                                                          target.detach().cpu().numpy())  # cnt=17
                meters['acc'].cache(avg_acc)
                meters['loss'].cache(loss)
            else:
                loss = mc.forward_loss(model, criterion, input, target, meters, task=FLAGS.model_kwparams.task, distill=FLAGS.distill)
            if FLAGS.prune_params['method'] is not None:
                loss_l2 = optim.cal_l2_loss(model, FLAGS.weight_decay,
                                            FLAGS.weight_decay_method)  # manual weight decay
                loss_bn_l1 = prune.cal_bn_l1_loss(get_prune_weights(model),
                                                  FLAGS._bn_to_prune.penalty, rho)
                if FLAGS.prune_params.use_transformer:

                    transformer_weights = get_prune_weights(model, True)
                    loss_bn_l1 += prune.cal_bn_l1_loss(transformer_weights,
                                                       FLAGS._bn_to_prune_transformer.penalty, rho)

                    transformer_dict = []
                    for name, weight in zip(FLAGS._bn_to_prune_transformer.weight, transformer_weights):
                        transformer_dict.append(sum(weight > FLAGS.model_shrink_threshold).item())
                    FLAGS._bn_to_prune_transformer.add_info_list('channels', transformer_dict)
                    FLAGS._bn_to_prune_transformer.update_penalty()
                    if udist.is_master() and FLAGS._global_step % FLAGS.log_interval == 0:
                        logging.info(transformer_dict)
                        # logging.info(FLAGS._bn_to_prune_transformer.penalty)

                meters['loss_l2'].cache(loss_l2)
                meters['loss_bn_l1'].cache(loss_bn_l1)
                loss = loss + loss_l2 + loss_bn_l1
            loss.backward()
            if FLAGS.use_distributed:
                udist.allreduce_grads(model)

            if FLAGS._global_step % FLAGS.log_interval == 0:
                results = mc.reduce_and_flush_meters(meters)
                if udist.is_master():
                    logging.info('Epoch {}/{} Iter {}/{} Lr: {} {}: '.format(
                        epoch, FLAGS.num_epochs, batch_idx, len(loader), optimizer.param_groups[0]["lr"], phase)
                                 + ', '.join('{}: {:.4f}'.format(k, v)
                                             for k, v in results.items()))
                    for k, v in results.items():
                        mc.summary_writer.add_scalar('{}/{}'.format(phase, k),
                                                     v, FLAGS._global_step)

            if udist.is_master(
            ) and FLAGS._global_step % FLAGS.log_interval == 0:
                mc.summary_writer.add_scalar('train/learning_rate',
                                             optimizer.param_groups[0]['lr'],
                                             FLAGS._global_step)
                if FLAGS.prune_params['method'] is not None:
                    mc.summary_writer.add_scalar('train/l2_regularize_loss',
                                                 extract_item(loss_l2),
                                                 FLAGS._global_step)
                    mc.summary_writer.add_scalar('train/bn_l1_loss',
                                                 extract_item(loss_bn_l1),
                                                 FLAGS._global_step)
                mc.summary_writer.add_scalar('prune/rho', rho,
                                             FLAGS._global_step)
                mc.summary_writer.add_scalar(
                    'train/current_epoch',
                    FLAGS._global_step / FLAGS._steps_per_epoch,
                    FLAGS._global_step)
                if FLAGS.data_loader_workers > 0:
                    mc.summary_writer.add_scalar(
                        'data/train/prefetch_size',
                        get_data_queue_size(data_iterator), FLAGS._global_step)

            if udist.is_master(
            ) and FLAGS._global_step % FLAGS.log_interval_detail == 0:
                summary_bn(model, 'train')

            optimizer.step()
            if FLAGS.lr_scheduler == 'poly':
                optim.poly_learning_rate(optimizer,
                                         FLAGS.lr,
                                         epoch * FLAGS._steps_per_epoch + batch_idx + 1,
                                         FLAGS.num_epochs * FLAGS._steps_per_epoch)
            else:
                lr_scheduler.step()
            if FLAGS.use_distributed and FLAGS.allreduce_bn:
                udist.allreduce_bn(model)
            FLAGS._global_step += 1

            # NOTE: after steps count update
            if ema is not None:
                model_unwrap = mc.unwrap_model(model)
                ema_names = ema.average_names()
                params = get_params_by_name(model_unwrap, ema_names)
                for name, param in zip(ema_names, params):
                    ema(name, param, FLAGS._global_step)
        else:
            if FLAGS.dataset == 'coco':
                outputs = model(input)
                if isinstance(outputs, list):
                    loss = criterion(outputs[0], target, target_weight)
                    for output in outputs[1:]:
                        loss += criterion(output, target, target_weight)
                else:
                    output = outputs
                    loss = criterion(output, target, target_weight)
                _, avg_acc, cnt, pred = accuracy_keypoint(output.detach().cpu().numpy(),
                                                          target.detach().cpu().numpy())  # cnt=17
                meters['acc'].cache(avg_acc)
                meters['loss'].cache(loss)
            else:
                mc.forward_loss(model, criterion, input, target, meters, task=FLAGS.model_kwparams.task, distill=False)

    if not train:
        results = mc.reduce_and_flush_meters(meters)
        if udist.is_master():
            logging.info(
                'Epoch {}/{} {}: '.format(epoch, FLAGS.num_epochs, phase)
                + ', '.join(
                    '{}: {:.4f}'.format(k, v) for k, v in results.items()))
            for k, v in results.items():
                mc.summary_writer.add_scalar('{}/{}'.format(phase, k), v,
                                             FLAGS._global_step)
    return results


def train_val_test():
    """Train and val."""
    torch.backends.cudnn.benchmark = True  # For acceleration

    # model
    model, model_wrapper = mc.get_model()
    ema = mc.setup_ema(model)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()
    criterion_smooth = optim.CrossEntropyLabelSmooth(
        FLAGS.model_kwparams['num_classes'],
        FLAGS['label_smoothing'],
        reduction='mean').cuda()
    if model.task == 'segmentation':
        criterion = CrossEntropyLoss().cuda()
        criterion_smooth = CrossEntropyLoss().cuda()
    if FLAGS.dataset == 'coco':
        criterion = JointsMSELoss(use_target_weight=True).cuda()
        criterion_smooth = JointsMSELoss(use_target_weight=True).cuda()

    if FLAGS.get('log_graph_only', False):
        if udist.is_master():
            _input = torch.zeros(1, 3, FLAGS.image_size,
                                 FLAGS.image_size).cuda()
            _input = _input.requires_grad_(True)
            if isinstance(model_wrapper, (torch.nn.DataParallel, udist.AllReduceDistributedDataParallel)):
                mc.summary_writer.add_graph(model_wrapper.module, (_input,), verbose=True)
            else:
                mc.summary_writer.add_graph(model_wrapper, (_input,), verbose=True)
        return

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained,
                                map_location=lambda storage, loc: storage)
        if ema:
            ema.load_state_dict(checkpoint['ema'])
            ema.to(get_device(model))
        # update keys from external models
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if (hasattr(FLAGS, 'pretrained_model_remap_keys')
                and FLAGS.pretrained_model_remap_keys):
            new_checkpoint = {}
            new_keys = list(model_wrapper.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new, key_old in zip(new_keys, old_keys):
                new_checkpoint[key_new] = checkpoint[key_old]
                if udist.is_master():
                    logging.info('remap {} to {}'.format(key_new, key_old))
            checkpoint = new_checkpoint
        model_wrapper.load_state_dict(checkpoint)
        if udist.is_master():
            logging.info('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = optim.get_optimizer(model_wrapper, FLAGS)

    # check resume training
    if FLAGS.resume:
        checkpoint = torch.load(os.path.join(FLAGS.resume,
                                             'latest_checkpoint.pt'),
                                map_location=lambda storage, loc: storage)
        model_wrapper = checkpoint['model'].cuda()
        model = model_wrapper.module
        # model = checkpoint['model'].module
        optimizer = checkpoint['optimizer']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        # model_wrapper.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        if ema:
            # ema.load_state_dict(checkpoint['ema'])
            ema = checkpoint['ema'].cuda()
            ema.to(get_device(model))
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS, last_epoch=(last_epoch + 1) * FLAGS._steps_per_epoch)
        lr_scheduler.last_epoch = (last_epoch + 1) * FLAGS._steps_per_epoch
        best_val = extract_item(checkpoint['best_val'])
        train_meters, val_meters = checkpoint['meters']
        FLAGS._global_step = (last_epoch + 1) * FLAGS._steps_per_epoch
        if udist.is_master():
            logging.info('Loaded checkpoint {} at epoch {}.'.format(
                FLAGS.resume, last_epoch))
    else:
        lr_scheduler = optim.get_lr_scheduler(optimizer, FLAGS)
        # last_epoch = lr_scheduler.last_epoch
        last_epoch = -1
        best_val = 1.
        if not FLAGS.distill:
            train_meters = mc.get_meters('train', FLAGS.prune_params['method'])
            val_meters = mc.get_meters('val')
        else:
            train_meters = mc.get_distill_meters('train', FLAGS.prune_params['method'])
            val_meters = mc.get_distill_meters('val')
        if FLAGS.model_kwparams.task == 'segmentation':
            best_val = 0.
            if not FLAGS.distill:
                train_meters = mc.get_seg_meters('train', FLAGS.prune_params['method'])
                val_meters = mc.get_seg_meters('val')
            else:
                train_meters = mc.get_seg_distill_meters('train', FLAGS.prune_params['method'])
                val_meters = mc.get_seg_distill_meters('val')
        FLAGS._global_step = 0

    if not FLAGS.resume and udist.is_master():
        logging.info(model_wrapper)
    assert FLAGS.profiling, '`m.macs` is used for calculating penalty'
    # if udist.is_master():
    #     model.apply(lambda m: print(m))
    if FLAGS.profiling:
        if 'gpu' in FLAGS.profiling:
            mc.profiling(model, use_cuda=True)
        if 'cpu' in FLAGS.profiling:
            mc.profiling(model, use_cuda=False)

    if FLAGS.dataset == 'cityscapes':
        (train_set, val_set, test_set) = seg_dataflow.cityscapes_datasets(FLAGS)
        segval = SegVal(num_classes=19)
    elif FLAGS.dataset == 'ade20k':
        (train_set, val_set, test_set) = seg_dataflow.ade20k_datasets(FLAGS)
        segval = SegVal(num_classes=150)
    elif FLAGS.dataset == 'coco':
        (train_set, val_set, test_set) = seg_dataflow.coco_datasets(FLAGS)
        # print(len(train_set), len(val_set))  # 149813 104125
        segval = None
    else:
        # data
        (train_transforms, val_transforms,
         test_transforms) = dataflow.data_transforms(FLAGS)
        (train_set, val_set, test_set) = dataflow.dataset(train_transforms,
                                                          val_transforms,
                                                          test_transforms, FLAGS)
        segval = None
    (train_loader, calib_loader, val_loader,
     test_loader) = dataflow.data_loader(train_set, val_set, test_set, FLAGS)

    # get bn's weights
    if FLAGS.prune_params.use_transformer:
        FLAGS._bn_to_prune, FLAGS._bn_to_prune_transformer = prune.get_bn_to_prune(model, FLAGS.prune_params)
    else:
        FLAGS._bn_to_prune = prune.get_bn_to_prune(model, FLAGS.prune_params)
    rho_scheduler = prune.get_rho_scheduler(FLAGS.prune_params,
                                            FLAGS._steps_per_epoch)

    if FLAGS.test_only and (test_loader is not None):
        if udist.is_master():
            logging.info('Start testing.')
        test_meters = mc.get_meters('test')
        validate(last_epoch, calib_loader, test_loader, criterion, test_meters,
                 model_wrapper, ema, 'test')
        return

    # already broadcast by AllReduceDistributedDataParallel
    # optimizer load same checkpoint/same initialization

    if udist.is_master():
        logging.info('Start training.')

    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        # train
        results = run_one_epoch(epoch,
                                train_loader,
                                model_wrapper,
                                criterion_smooth,
                                optimizer,
                                lr_scheduler,
                                ema,
                                rho_scheduler,
                                train_meters,
                                phase='train')

        if (epoch + 1) % FLAGS.eval_interval == 0:
            # val
            results, model_eval_wrapper = validate(epoch, calib_loader, val_loader,
                                                   criterion, val_meters,
                                                   model_wrapper, ema, 'val', segval, val_set)

            if FLAGS.prune_params['method'] is not None and FLAGS.prune_params['bn_prune_filter'] is not None:
                prune_threshold = FLAGS.model_shrink_threshold  # 1e-3
                masks = prune.cal_mask_network_slimming_by_threshold(
                    get_prune_weights(model_eval_wrapper), prune_threshold)  # get mask for all bn weights (depth-wise)
                FLAGS._bn_to_prune.add_info_list('mask', masks)
                flops_pruned, infos = prune.cal_pruned_flops(FLAGS._bn_to_prune)
                log_pruned_info(mc.unwrap_model(model_eval_wrapper), flops_pruned,
                                infos, prune_threshold)
                if not FLAGS.distill:
                    if flops_pruned >= FLAGS.model_shrink_delta_flops \
                            or epoch == FLAGS.num_epochs - 1:
                        ema_only = (epoch == FLAGS.num_epochs - 1)
                        shrink_model(model_wrapper, ema, optimizer, FLAGS._bn_to_prune,
                                     prune_threshold, ema_only)
            model_kwparams = mb.output_network(mc.unwrap_model(model_wrapper))

            if udist.is_master():
                if FLAGS.model_kwparams.task == 'classification' and results['top1_error'] < best_val:
                    best_val = results['top1_error']
                    logging.info('New best validation top1 error: {:.4f}'.format(best_val))

                    save_status(model_wrapper, model_kwparams, optimizer, ema,
                                epoch, best_val, (train_meters, val_meters),
                                os.path.join(FLAGS.log_dir, 'best_model'))

                elif FLAGS.model_kwparams.task == 'segmentation' and FLAGS.dataset != 'coco' and results[
                    'mIoU'] > best_val:
                    best_val = results['mIoU']
                    logging.info('New seg mIoU: {:.4f}'.format(best_val))

                    save_status(model_wrapper, model_kwparams, optimizer, ema,
                                epoch, best_val, (train_meters, val_meters),
                                os.path.join(FLAGS.log_dir, 'best_model'))
                elif FLAGS.dataset == 'coco' and results > best_val:
                    best_val = results
                    logging.info('New Result: {:.4f}'.format(best_val))
                    save_status(model_wrapper, model_kwparams, optimizer, ema,
                                epoch, best_val, (train_meters, val_meters),
                                os.path.join(FLAGS.log_dir, 'best_model'))

                # save latest checkpoint
                save_status(model_wrapper, model_kwparams, optimizer, ema, epoch,
                            best_val, (train_meters, val_meters),
                            os.path.join(FLAGS.log_dir, 'latest_checkpoint'))

    return


def validate(epoch, calib_loader, val_loader, criterion, val_meters,
             model_wrapper, ema, phase, segval=None, val_set=None):
    """Calibrate and validate."""
    assert phase in ['test', 'val']
    model_eval_wrapper = mc.get_ema_model(ema, model_wrapper)

    # bn_calibration
    if FLAGS.prune_params['method'] is not None:
        if FLAGS.get('bn_calibration', False):
            if not FLAGS.use_distributed:
                logging.warning(
                    'Only GPU0 is used when calibration when use DataParallel')
            with torch.no_grad():
                _ = run_one_epoch(epoch,
                                  calib_loader,
                                  model_eval_wrapper,
                                  criterion,
                                  None,
                                  None,
                                  None,
                                  None,
                                  val_meters,
                                  max_iter=FLAGS.bn_calibration_steps,
                                  phase='bn_calibration')
            if FLAGS.use_distributed:
                udist.allreduce_bn(model_eval_wrapper)

    # val
    with torch.no_grad():
        if FLAGS.model_kwparams.task == 'segmentation':
            if FLAGS.dataset == 'coco':
                results = 0
                if udist.is_master():
                    results = keypoint_val(val_set, val_loader, model_eval_wrapper.module, criterion)
            else:
                assert segval is not None
                results = segval.run(epoch,
                                     val_loader,
                                     model_eval_wrapper.module if FLAGS.single_gpu_test else model_eval_wrapper,
                                     FLAGS)
        else:
            results = run_one_epoch(epoch,
                                    val_loader,
                                    model_eval_wrapper,
                                    criterion,
                                    None,
                                    None,
                                    None,
                                    None,
                                    val_meters,
                                    phase=phase)
    summary_bn(model_eval_wrapper, phase)
    return results, model_eval_wrapper


def main():
    """Entry."""
    NUM_IMAGENET_TRAIN = 1281167
    if FLAGS.dataset == 'cityscapes':
        NUM_IMAGENET_TRAIN = 2975
    elif FLAGS.dataset == 'ade20k':
        NUM_IMAGENET_TRAIN = 20210
    elif FLAGS.dataset == 'coco':
        NUM_IMAGENET_TRAIN = 149813
    mc.setup_distributed(NUM_IMAGENET_TRAIN)

    if FLAGS.net_params and FLAGS.model_kwparams.task == 'segmentation':
        tag, input_channels, block1, block2, block3, block4, last_channel = FLAGS.net_params.split('-')
        input_channels = [int(item) for item in input_channels.split('_')]
        block1 = [int(item) for item in block1.split('_')]
        block2 = [int(item) for item in block2.split('_')]
        block3 = [int(item) for item in block3.split('_')]
        block4 = [int(item) for item in block4.split('_')]
        last_channel = int(last_channel)

        inverted_residual_setting = []
        for item in [block1, block2, block3, block4]:
            for _ in range(item[0]):
                inverted_residual_setting.append(
                    [item[1], item[2:-int(len(item) / 2 - 1)], item[-int(len(item) / 2 - 1):]])

        FLAGS.model_kwparams.input_channel = input_channels
        FLAGS.model_kwparams.inverted_residual_setting = inverted_residual_setting
        FLAGS.model_kwparams.last_channel = last_channel

    if udist.is_master():
        FLAGS.log_dir = '{}/{}'.format(FLAGS.log_dir,
                                       time.strftime("%Y%m%d-%H%M%S"))
        # yapf: disable
        create_exp_dir(FLAGS.log_dir, FLAGS.config_path, blacklist_dirs=[
            'exp', '.git', 'pretrained', 'tmp', 'deprecated', 'bak', 'output'])
        # yapf: enable
        setup_logging(FLAGS.log_dir)
        for k, v in _ENV_EXPAND.items():
            logging.info('Env var expand: {} to {}'.format(k, v))
        logging.info(FLAGS)

    set_random_seed(FLAGS.get('random_seed', 0))
    with mc.SummaryWriterManager():
        train_val_test()


if __name__ == "__main__":
    main()
