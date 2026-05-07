import math
import os
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import torch
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class EMANetModel(BaseModel):
    """EMANet inference model."""

    def __init__(self, opt):
        super(EMANetModel, self).__init__(opt)

        self.net_g = build_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(
                self.net_g, load_path,
                self.opt['path'].get('strict_load_g', True),
                param_key=self.opt['path'].get('param_key', 'params'))


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.voxel = data['voxel'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if any('seq' in item for item in data):
            self.seq_name = data['seq']
            self.seq_name = self.seq_name[0]

        if any('origin_index' in item for item in data):
            self.origin_index = data['origin_index']
            self.origin_index = self.origin_index[0]

        self.times = data['times']
        if isinstance(self.times, torch.Tensor):
            self.times = self.times[0].numpy().astype(float)

        self.scale = data['scale']
        if isinstance(self.scale, torch.Tensor):
            self.scale = self.scale[0].numpy().astype(float)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            val_step = self.opt['val'].get('val_step', len(self.times))
            out = []
            for i in range(0, len(self.times), val_step):
                times = self.times[i:i + val_step]
                self.pre_process()
                if 'tile' in self.opt:
                    self.tile_process(times)
                else:
                    self.process(times)
                self.post_process()
                out.append(self.out)
            self.output = torch.cat(out, dim=1)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        get_root_logger()
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = self.opt.get('name')
        save_gt = self.opt['val'].get('save_gt', False)
        with_metrics = self.opt['val'].get('cal_metrics', True)

        all_avg_psnr = []
        all_avg_ssim = []
        all_center_psnr = []
        all_center_ssim = []

        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()

            del self.lq, self.voxel, self.output, self.out
            if hasattr(self, 'gt'):
                del self.gt
            del val_data
            torch.cuda.empty_cache()
            imgs_per_iter = visuals['result'].size(1)

            for frame_idx in range(visuals['result'].size(1) - 1):
                img_name = '{}_{:02d}'.format(self.origin_index, frame_idx)
                result = visuals['result'][0, frame_idx, :, :, :]
                sr_img = tensor2img([result])
                gt_img = None
                if 'gt' in visuals:
                    gt = visuals['gt'][0, frame_idx, :, :, :]
                    gt_img = tensor2img([gt])

                if save_img:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name, self.seq_name,
                        f'{img_name}.png')
                    imwrite(sr_img, save_img_path)
                    if save_gt and gt_img is not None:
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name, self.seq_name,
                            f'{img_name}_gt.png')
                        imwrite(gt_img, save_gt_img_path)

                if with_metrics and gt_img is not None:
                    psnr_values = calculate_psnr(sr_img, gt_img, 0, test_y_channel=True)
                    ssim_values = calculate_ssim(sr_img, gt_img, 0, test_y_channel=True)
                    all_avg_psnr.append(psnr_values)
                    all_avg_ssim.append(ssim_values)
                    if frame_idx == 0 or frame_idx == imgs_per_iter // 2:
                        all_center_psnr.append(psnr_values)
                        all_center_ssim.append(ssim_values)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
        if with_metrics and len(all_avg_psnr) > 0:
            self.avg_psnr = torch.mean(torch.tensor(all_avg_psnr))
            self.avg_ssim = torch.mean(torch.tensor(all_avg_ssim))
            self.center_psnr = torch.mean(torch.tensor(all_center_psnr))
            self.center_ssim = torch.mean(torch.tensor(all_center_ssim))
            self._log_validation_metric_values(current_iter, dataset_name)

    def _log_validation_metric_values(self, current_iter, dataset_name):
        log_str = f'Validation {dataset_name} [ST-VSR],\t'
        log_str += f'\t # avg_psnr: {self.avg_psnr:.4f}'
        log_str += f'\t # avg_ssim: {self.avg_ssim:.4f}'
        log_str += f'\t # center_psnr: {self.center_psnr:.4f}'
        log_str += f'\t # center_ssim: {self.center_ssim:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def pre_process(self):
        window_size = self.opt['val'].get('window_size', 4)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, _, h, w = self.lq.shape
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        pad_values = (0, self.mod_pad_w, 0, self.mod_pad_h, 0, 0)
        self.lq_pad = F.pad(self.lq, pad_values, 'replicate')
        self.voxel_pad = F.pad(self.voxel, pad_values, 'replicate')

    def process(self, times):
        self.net_g.eval()
        with torch.no_grad():
            self.out = self.net_g(image=self.lq_pad, event=self.voxel_pad,
                                  scale=self.scale, times=times)

    def tile_process(self, times):
        """Crop input to tiles, run inference per tile, then merge."""
        batch, _, channel, height, width = self.lq_pad.shape
        output_height = round(height * self.scale)
        output_width = round(width * self.scale)
        output_shape = (batch, len(times), channel, output_height, output_width)

        self.out = self.lq_pad.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                img_tile = self.lq_pad[:, :, :,
                                       input_start_y_pad:input_end_y_pad,
                                       input_start_x_pad:input_end_x_pad]
                voxel_tile = self.voxel_pad[:, :, :,
                                            input_start_y_pad:input_end_y_pad,
                                            input_start_x_pad:input_end_x_pad]

                try:
                    self.net_g.eval()
                    with torch.no_grad():
                        output_tile = self.net_g(
                            image=img_tile, event=voxel_tile,
                            scale=self.scale, times=times)
                except RuntimeError as error:
                    print('Error', error)

                output_start_x = round(input_start_x * self.scale)
                output_end_x = round(input_end_x * self.scale)
                output_start_y = round(input_start_y * self.scale)
                output_end_y = round(input_end_y * self.scale)

                output_start_x_tile = round((input_start_x - input_start_x_pad) * self.scale)
                output_end_x_tile = output_start_x_tile + round(input_tile_width * self.scale)
                output_start_y_tile = round((input_start_y - input_start_y_pad) * self.scale)
                output_end_y_tile = output_start_y_tile + round(input_tile_height * self.scale)

                self.out[:, :, :,
                         output_start_y:output_end_y,
                         output_start_x:output_end_x] = output_tile[
                             :, :, :,
                             output_start_y_tile:output_end_y_tile,
                             output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, _, h, w = self.out.shape
        output_height = h - round(self.mod_pad_h * self.scale)
        output_width = w - round(self.mod_pad_w * self.scale)
        self.out = self.out[:, :, :, 0:output_height, 0:output_width]
