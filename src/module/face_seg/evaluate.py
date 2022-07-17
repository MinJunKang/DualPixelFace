#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch

import os
import pdb
import sys
sys.path.append(os.getcwd())
from Module.External.FaceMasking.model import BiSeNet
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


def plot_matrix(map):

    map = map.cpu().numpy()
    map = map - np.min(map) / (np.max(map) - np.min(map))
    map = map * 255
    im = Image.fromarray(map.astype('uint8'))
    im.show()


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, img_path='vis_results/parsing_map_on_im.jpg'):

    respth = osp.join(osp.dirname(osp.realpath(__file__)), './res/test_res')
    if not os.path.exists(respth):
        os.makedirs(respth)
    save_path = osp.join(respth, img_path)

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


class FaceMaskEstimator(object):

    def __init__(self, option):
        self.option = option
        self.n_classes = option.eval_setting.masking.num_classes
        self.net = BiSeNet(n_classes=self.n_classes).cuda()
        self.cp = '79999_iter.pth'
        save_pth = osp.join(osp.dirname(osp.realpath(__file__)), osp.join('./res/cp', self.cp))
        self.net.load_state_dict(torch.load(save_pth))
        self.net.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def evaluate(self, input, vis=False, istensor=False):
        '''
        :param img: numpy array [H, W, 3]
        :param vis:
        :return:
        '''

        with torch.no_grad():
            if istensor:
                _, c, h, w = input.shape
                img = torch.nn.functional.interpolate(input, size=(512, 512), mode='bilinear')
            else:
                h, w, c = input.shape
                img = Image.fromarray(input)
                image = img.resize((512, 512), Image.BILINEAR)
                img = self.to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            if vis:
                vis_parsing_maps(image, parsing, stride=1, save_im=True, img_path='faceseg.JPG')

            mask_background = (parsing == 0)
            mask_foreground = ~mask_background
            mask_hair = (parsing == 17)
            mask_neck = (parsing == 14)
            mask_clothes = (parsing == 16)
            mask_face = ~mask_background ^ (mask_clothes + mask_neck + mask_hair)
            mask_face = cv2.resize(mask_face.astype('uint8'), dsize=(w, h),
                                   interpolation=cv2.INTER_NEAREST)
            mask_foreground = cv2.resize(mask_foreground.astype('uint8'), dsize=(w, h),
                                         interpolation=cv2.INTER_NEAREST)

        return mask_foreground > 0, mask_face > 0


def evaluate(imgs, cp='79999_iter.pth', vis=False):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes).cuda()
    net.cuda()
    save_pth = osp.join(osp.dirname(osp.realpath(__file__)), osp.join('./res/cp', cp))
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    outputs = []

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for idx, img in enumerate(imgs):
            img = Image.fromarray(img)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            if vis:
                vis_parsing_maps(image, parsing, stride=1, save_im=True, img_path='ex%04d.JPG' % idx)

            outputs.append(parsing)

    return outputs


if __name__ == "__main__":
    imgpath = osp.join(osp.dirname(osp.realpath(__file__)), './examples')
    names = os.listdir(imgpath)
    imgs = [np.asarray(Image.open(osp.join(imgpath, name)).rotate(180)) for name in names]
    evaluate(imgs=imgs, cp='79999_iter.pth', vis=True)


