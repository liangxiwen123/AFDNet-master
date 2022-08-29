import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import sys
from scipy.stats import wasserstein_distance
import cv2
import cv2 as cv
def filter_high_f(img,fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift


def filter_low_f(img,fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        #print('filter_img',filter_img.shape)
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift


def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg
def get_low_high_f(img, radius_ratio):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

    # 获取低频和高频部分
    hight_parts_fshift = filter_low_f(img,fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = filter_high_f(img,fshift.copy(), radius_ratio=radius_ratio)

    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)

    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high
def frequency_loss(im1, im2):
    loss=0
    radius_ratio = 0.5
    #print('im1.shape[0]',im1.shape[0])
    for i in range(3):
        #print(i)
        img=im1[i].cpu().detach().numpy()
        #print(img.shape)
        img1=im2[i].cpu().detach().numpy()
        img=np.transpose(img, (1, 2, 0))
        img1=np.transpose(img1, (1, 2, 0))
        low_freq_part_img1, high_freq_part_img1 = get_low_high_f(img, radius_ratio=radius_ratio)
        low_freq_part_img2, high_freq_part_img2 = get_low_high_f(img1, radius_ratio=radius_ratio)
        
        low_freq_part_img1=torch.from_numpy(low_freq_part_img1)
        low_freq_part_img2 = torch.from_numpy(low_freq_part_img2)
        high_freq_part_img1 = torch.from_numpy(high_freq_part_img1)
        high_freq_part_img2 = torch.from_numpy(high_freq_part_img2)

        real_loss = wasserstein_distance(low_freq_part_img1.reshape(im1[i].shape[0]*im1[i].shape[1]*im1[i].shape[2]).cuda().detach(),
                                         low_freq_part_img2.reshape(im1[i].shape[0]*im1[i].shape[1]*im1[i].shape[2]).cuda().detach())
        imag_loss = wasserstein_distance(high_freq_part_img1.reshape(im1[i].shape[0]*im1[i].shape[1]*im1[i].shape[2]).cuda().detach(),
                                         high_freq_part_img2.reshape(im1[i].shape[0]*im1[i].shape[1]*im1[i].shape[2]).cuda().detach())
        total_loss = real_loss + imag_loss
        loss += total_loss
    return torch.tensor(loss / (im1.shape[2] * im2.shape[3]))

class mutu(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(mutu, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x, org):
        batch_size = x.size()[0]
        h_x = x.size()[2]  # 256
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]  # 65280
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)  # 70
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)  # 100
        oh_tv = torch.pow((org[:, :, 1:, :] - org[:, :, :h_x - 1, :]), 2)  # 70
        ow_tv = torch.pow((org[:, :, :, 1:] - org[:, :, :, :w_x - 1]), 2)  # 100
        x_loss = ((h_tv + oh_tv) * torch.exp(-10 * (h_tv + oh_tv))).sum()
        y_loss = ((w_tv + ow_tv) * torch.exp(-10 * (w_tv + ow_tv))).sum()

        return self.TVLoss_weight * 2 * (x_loss / count_h + y_loss / count_w) / batch_size


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        #print('1')
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.a = mutu()
        self.tv = L_TV()
        #self.frequency_loss = frequency_loss()
        # self.e = L_exp(16, 0.8)
        self.L1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.scale1_color = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.TripletMarginLoss = nn.TripletMarginLoss()
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_C = self.Tensor(nb, opt.output_nc, size, size)
        
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn("./model")
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
        #                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, False)
            if self.opt.patchD:
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_patchD, opt.norm, use_sigmoid, self.gpu_ids, True)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            # self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
            # networks.print_network(self.netD_B)
        if opt.isTrain:
            self.netG_A.train()
            # self.netG_B.train()
        else:
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        #input_C = input['C' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        #self.input_C.resize_(input_C.size()).copy_(input_C)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)

    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        latent_real_A = util.tensor2im(self.latent_real_A.data)
        A_gray = util.atten2im(self.real_A_gray.data)
        # rec_A = util.tensor2im(self.rec_A.data)
        # if self.opt.skip == 1:
        #     latent_real_A = util.tensor2im(self.latent_real_A.data)
        #     latent_show = util.latent2im(self.latent_real_A.data)
        #     max_image = util.max2im(self.fake_B.data, self.latent_real_A.data)
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
        #                     ('latent_show', latent_show), ('max_image', max_image), ('A_gray', A_gray)])
        # else:
        #     return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
        # return OrderedDict([('fake_B', fake_B)])
        return OrderedDict([('', fake_B)])  # ,('latent_real_A', latent_real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake, use_ragan):
        # Real
        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())
        if self.opt.use_wgan:
            loss_D_real = pred_real.mean()
            loss_D_fake = pred_fake.mean()
            loss_D = loss_D_fake - loss_D_real + self.criterionGAN.calc_gradient_penalty(netD,
                                                                                         real.data, fake.data)
        elif self.opt.use_ragan and use_ragan:
           # loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
            #          self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_C, fake_B, True)
        self.loss_D_A.backward()

    def backward_D_P(self):
        if self.opt.hybrid_loss:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, False)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], False)
                self.loss_D_P = loss_D_P / float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        else:
            loss_D_P = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch, True)
            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    loss_D_P += self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i], True)
                self.loss_D_P = loss_D_P / float(self.opt.patchD_3 + 1)
            else:
                self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P * 2
        self.loss_D_P.backward()

    # def backward_D_B(self):
    #     fake_A = self.fake_A_pool.query(self.fake_A)
    #     self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_img, self.real_A_gray)
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)
        if self.opt.patchD:
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            self.fake_patch = self.fake_B[:, :, h_offset:h_offset + self.opt.patchSize,
                              w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:, :, h_offset:h_offset + self.opt.patchSize,
                              w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:, :, h_offset:h_offset + self.opt.patchSize,
                               w_offset:w_offset + self.opt.patchSize]
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                          w_offset_1:w_offset_1 + self.opt.patchSize])

            # w_offset_2 = random.randint(0, max(0, w - self.opt.patchSize - 1))
            # h_offset_2 = random.randint(0, max(0, h - self.opt.patchSize - 1))
            # self.fake_patch_2 = self.fake_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.real_patch_2 = self.real_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.input_patch_2 = self.real_A[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]

    def backward_G(self, epoch):
        pred_fake = self.netD_A.forward(self.fake_B)
        self.l1_loss = self.L1(self.fake_B, self.real_B)
        self.l2_loss = self.l2(self.fake_B, self.real_B)
        self.loss_mutu = 5 * self.a(self.fake_B, self.real_A)
        self.loss_tv = self.tv(self.fake_B)
        self.loss_frequency = 10*frequency_loss(self.fake_B,self.real_B).cuda()
        
        #print('self.loss_frequency',self.loss_frequency)
        #print(self.loss_frequency)
        # self.TripletMarginLoss_loss = self.TripletMarginLoss(self.fake_B,self.real_B,self.real_A)
        # self.loss_color=torch.mean(-1 * self.scale1_color(self.fake_B, self.real_B))
        # print('self.loss_color',self.loss_color)
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        elif self.opt.use_ragan:
            #pred_real = self.netD_A.forward(self.real_B) 这里可以减少时间
            #print('1',pred_fake.shape)
            #print('2',self.real_B.shape)
            #self.loss_G_A = (self.criterionGAN(pred_real - torch.mean(pred_fake), False) +
             #                self.criterionGAN(pred_fake - torch.mean(pred_real), True)) / 2
            self.loss_G_A = self.criterionGAN(pred_fake, True)               

        else:
            self.loss_G_A = self.criterionGAN(pred_fake, True)

        loss_G_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            if self.opt.hybrid_loss:
                loss_G_A += self.criterionGAN(pred_fake_patch, True)
            else:
                pred_real_patch = self.netD_P.forward(self.real_patch)

                loss_G_A += (self.criterionGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                             self.criterionGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
        if self.opt.patchD_3 > 0:
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                if self.opt.hybrid_loss:
                    loss_G_A += self.criterionGAN(pred_fake_patch_1, True)
                else:
                    pred_real_patch_1 = self.netD_P.forward(self.real_patch_1[i])

                    loss_G_A += (self.criterionGAN(pred_real_patch_1 - torch.mean(pred_fake_patch_1), False) +
                                 self.criterionGAN(pred_fake_patch_1 - torch.mean(pred_real_patch_1), True)) / 2

            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A / float(self.opt.patchD_3 + 1)
            else:
                self.loss_G_A += loss_G_A / float(self.opt.patchD_3 + 1) * 2
        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
            else:
                self.loss_G_A += loss_G_A * 2

        if epoch < 0:
            vgg_w = 0
        else:
            vgg_w = 1
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                             self.fake_B,
                                                             self.real_B) * self.opt.vgg if self.opt.vgg > 0 else 0
            if self.opt.patch_vgg:
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                                    self.fake_patch, self.real_patch) * self.opt.vgg
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg,
                                                                          self.fake_patch,
                                                                          self.input_patch) * self.opt.vgg
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        if not self.opt.IN_vgg:
                            loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg,
                                                                             self.fake_patch_1[i],
                                                                             self.real_patch_1[i]) * self.opt.vgg
                        else:
                            loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg,
                                                                                   self.fake_patch_1[i],
                                                                                   self.input_patch_1[i]) * self.opt.vgg
                    self.loss_vgg_b += loss_vgg_patch / float(self.opt.patchD_3 + 1)
                else:
                    self.loss_vgg_b += loss_vgg_patch
            self.loss_G = 0.5*self.loss_G_A + self.loss_vgg_b * vgg_w + self.l1_loss + self.l2_loss + 2*self.loss_tv+self.loss_frequency   + self.loss_mutu#+ self.TripletMarginLoss_loss
        elif self.opt.fcn > 0:
            self.loss_fcn_b = self.fcn_loss.compute_fcn_loss(self.fcn,
                                                             self.fake_B,
                                                             self.real_A) * self.opt.fcn if self.opt.fcn > 0 else 0
            if self.opt.patchD:
                loss_fcn_patch = self.fcn_loss.compute_vgg_loss(self.fcn,
                                                                self.fake_patch, self.input_patch) * self.opt.fcn
                if self.opt.patchD_3 > 0:
                    for i in range(self.opt.patchD_3):
                        loss_fcn_patch += self.fcn_loss.compute_vgg_loss(self.fcn,
                                                                         self.fake_patch_1[i],
                                                                         self.input_patch_1[i]) * self.opt.fcn
                    self.loss_fcn_b += loss_fcn_patch / float(self.opt.patchD_3 + 1)
                else:
                    self.loss_fcn_b += loss_fcn_patch
            self.loss_G = self.loss_G_A + self.loss_fcn_b * vgg_w
        # self.loss_G = self.L1_AB + self.L1_BA
        self.loss_G.backward()

    # def optimize_parameters(self, epoch):
    #     # forward
    #     self.forward()
    #     # G_A and G_B
    #     self.optimizer_G.zero_grad()
    #     self.backward_G(epoch)
    #     self.optimizer_G.step()
    #     # D_A
    #     self.optimizer_D_A.zero_grad()
    #     self.backward_D_A()
    #     self.optimizer_D_A.step()
    #     if self.opt.patchD:
    #         self.forward()
    #         self.optimizer_D_P.zero_grad()
    #         self.backward_D_P()
    #         self.optimizer_D_P.step()
    # D_B
    # self.optimizer_D_B.zero_grad()
    # self.backward_D_B()
    # self.optimizer_D_B.step()
    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if not self.opt.patchD:
            self.optimizer_D_A.step()
        else:
            # self.forward()
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            self.optimizer_D_A.step()
            self.optimizer_D_P.step()

    def get_current_errors(self, epoch):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()
        G_l1 = self.l1_loss.item()
        G_l2 = self.l2_loss.item()
        G_mu = self.loss_mutu.item()
        G_tv = self.loss_tv.item()
        G_fr = self.loss_frequency.item()
        # G_Tri=self.TripletMarginLoss_loss.item()
        if self.opt.vgg > 0:
            vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
            return OrderedDict(
                [('D_A', D_A), ('G_A', G_A), ("vgg", vgg), ("D_P", D_P), ("G_l1", G_l1), ("G_l2", G_l2), ("G_fr", G_fr),("G_mu", G_mu),
                 ("G_tv", G_tv)])
        elif self.opt.fcn > 0:
            fcn = self.loss_fcn_b.data[0] / self.opt.fcn if self.opt.fcn > 0 else 0
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ("fcn", fcn), ("D_P", D_P)])

    def get_current_visuals(self):
        #print('dsdssdsdsdsds',self.real_A.shape)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch), ('input_patch', input_patch),
                                            ('self_attention', self_attention)])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                        ('latent_show', latent_show), ('real_B', real_B)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                        ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                        ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)
        # self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        # self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):

        if self.opt.new_lr:
            lr = self.old_lr / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
