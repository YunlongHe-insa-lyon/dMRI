import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import scipy.io as scio
import math

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--sigma', type=float, default=0.5, help='weighted scale for manifold loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # 初步赋值self from opt via BaseModel
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_sim', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            # 将　opt.gan_mode　赋值给　ＧＡＮＬｏｓｓ　__init__
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            #self.weighted_average = networks.GANLoss(opt.gan_mode)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A = self.real_A.float() 
        self.real_B = self.real_B.float() 
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        # self.weight = scio.loadmat('/misc/raid/home/yhe/DeepLearning/dMRI_gan/models/weight.mat')

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        Asize = self.real_A
#        Asize.cpu().numpy().size()
#        print(Asize.shape)
#        Asize=Asize.float() 
        self.fake_B = self.netG(Asize)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        # detach 设置反向传播的截止点，阻止Ｇ的更新
        pred_fake = self.netD(fake_AB.detach())
        # 利用 __call__ in networks.GANLoss, 将class变为object
        # W=torch.from_numpy(self.weight['W_ang'])
        # n1=0
        # n2=48
        # wd_loss=networks.loss_corr(fake_AB, W, n1,n2,0.5,self.criterionMSE) 
        self.loss_D_fake = self.criterionGAN(pred_fake, False)  
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)  
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B         
        # W=torch.from_numpy(self.weight['W_ang'])
        # n1=0
        # n2=24  
        # if self.opt.MR_model == 'fy':    
        #     self.loss_G_inter=networks.loss_MR_fy(fake_AB, W, n1,n2,0.5,self.criterionMSE)
        # else:
        #     self.real_AB = torch.cat((self.real_A, self.real_B), 1)
        #     self.loss_G_inter=networks.loss_MR_fxy(self.real_AB,fake_AB, W, n1,n2,0.5,self.criterionMSE)  
        # f = open("/misc/raid/home/yhe/DeepLearning/dMRI_gan/out/s.txt", 'w+')
##        fm=torch.squeeze(fake_AB[:,13,:,:])
#        print(fm.shape, file=f)
#        print(wd, file=f)file:///D:/YunlongHE/projects/Diffusion_MRI_with_GAN/experiments/Main/4_directional_alignment&data_norm/cnn_input_12to24dirs_mat/codes_cluster/s.txt
 #       self.wg1=networks.test_mean(self.real_A)
        # real_AB = torch.cat((self.real_A, self.real_B), 1)
        # wgA=networks.test_mean(self.real_A)
        # wgB=networks.test_mean(self.real_B)
        # wgAB=networks.test_mean(real_AB)     
        # print(self.real_A.shape, '\n','\n', wgA,'\n','\n',wgB,'\n','\n',wgAB,file=f)  
        
        if self.opt.MR_loss == 'mse':     
            self.loss_G_sim = self.criterionMSE(self.fake_B, self.real_B)
        else:
            self.loss_G_sim = self.criterionL1(self.fake_B, self.real_B)
        # combine MR loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_sim*self.opt.lambda_G_sim 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
