import torch.nn.functional as F
import mmcv
from mmseg.utils import resize
from torch.distributed import get_world_size
import shutil
import tempfile
import os
import os.path as osp
import numpy as np
import torch
import torch.distributed as dist
from utils import distributed as udist
from utils.coco_dataset import flip_back
from mmseg.loss import accuracy, get_final_preds

def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = int(os.environ['RANK']), get_world_size()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


class SegVal:
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, num_classes=19):
        super(SegVal, self).__init__()

        self.align_corners = False
        self.stride = (513, 513)
        self.crop_size = (769, 769)
        self.num_classes = num_classes
        self.mode = 'whole'

    def run(self, epoch, loader, model, FLAGS):
        model.eval()
        dataset = loader.dataset
        data_iterator = iter(loader)

        results = []
        if udist.is_master():
            prog_bar = mmcv.ProgressBar(len(dataset))
        for batch_idx, input in enumerate(data_iterator):
            imgs = input['img']
            img_metas = input['img_metas'][0].data
            assert len(imgs) == len(img_metas)
            for img_meta in img_metas:
                ori_shapes = [_['ori_shape'] for _ in img_meta]
                assert all(shape == ori_shapes[0] for shape in ori_shapes)
                img_shapes = [_['img_shape'] for _ in img_meta]
                assert all(shape == img_shapes[0] for shape in img_shapes)
                pad_shapes = [_['pad_shape'] for _ in img_meta]
                assert all(shape == pad_shapes[0] for shape in pad_shapes)

            if len(imgs) == 1:
                result = self.simple_test(model,
                                          imgs[0].cuda() if FLAGS.single_gpu_test else imgs[0],
                                          img_metas[0])
            else:
                result = self.aug_test(model, imgs, img_metas)
            results.extend(result)
            if udist.is_master():
                batch_size = imgs[0].size(0)
                world_size = 1 if FLAGS.single_gpu_test else get_world_size()
                for _ in range(batch_size * world_size):
                    prog_bar.update()
        if not FLAGS.single_gpu_test:
            results = collect_results_cpu(results, len(dataset))
        performance = None
        if udist.is_master():
            performance = dataset.evaluate(results)
        dist.barrier()
        # dist.broadcast(performance, 0)
        return performance

    def encode_decode(self, model, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        out = model(img)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    # TODO refactor, support GPU
    def slide_inference(self, model, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.stride
        h_crop, w_crop = self.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_seg_logit = self.encode_decode(model, pad_img, img_meta)
                preds[:, :, y1:y2,
                      x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return preds

    def whole_inference(self, model, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(model, img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, model, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.mode == 'slide':
            seg_logit = self.slide_inference(model, img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(model, img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, model, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(model, img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, model, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(model, imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(model, imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


def keypoint_val(val_dataset, val_loader, model, criterion):
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, 17, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)

            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            FLIP_TEST = 1
            SHIFT_HEATMAP = 1
            if FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.cuda(non_blocking=True)

            # loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                  target.cpu().numpy())

            # measure elapsed time

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

        name_values, perf_indicator = val_dataset.evaluate(
            all_preds, 'output/', all_boxes, image_path, filenames, imgnums
        )

    return perf_indicator