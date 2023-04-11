import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import scipy.io as scio
import math
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, color_fa
from os.path import join as pjoin

class dmriTDmModel(BaseModel):
    """ This class implements the SR_dMRI_lossTDm model, for learning a mapping from input images to output images given paired data.

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
        self.loss_names = ['G_GAN', 'G_sim', 'G_const', 'D_real', 'D_fake']
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
        folder_name = "training_details/LossTDm_paras.mat"         
        constraints_name = pjoin(self.opt.dataroot, folder_name)
        f = open("/misc/raid/home/yhe/DeepLearning/dMRI_gan/out/s.txt", 'w+')
        print(constraints_name, file=f)
        data=scio.loadmat(constraints_name)
        bvecs_12a = np.squeeze(np.array(data['bvecs_A']))          
        bvecs_12b = np.squeeze(np.array(data['bvecs_B']))
        bvecs_24 = np.squeeze(np.array(data['bvecs_AB'])) 
        bvals_12a = np.squeeze(np.array(data['bvals_A']))
        bvals_12b = np.squeeze(np.array(data['bvals_B']))  
        bvals_24 = np.squeeze(np.array(data['bvals_AB']))        
        gtab_12a = gradient_table(bvals_12a, bvecs_12a) 
        gtab_12b = gradient_table(bvals_12b, bvecs_12b) 
        gtab_24 = gradient_table(bvals_24, bvecs_24) 
        self.tenmodel_12p = dti.TensorModel(gtab_12a)
        self.tenmodel_12o = dti.TensorModel(gtab_12b)
        self.tenmodel_24 = dti.TensorModel(gtab_24)
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
        
        #Second, min||G(A)-Y||
        if self.opt.sim_loss == 'mse':     
            self.loss_G_sim = self.criterionMSE(self.fake_B, self.real_B)
        else:
            self.loss_G_sim = self.criterionL1(self.fake_B, self.real_B)
        #Third, use constraint with diffusion tensors        
        data_S0 = self.real_A[:,0,43:85,43:85]
        data_S0 = data_S0.unsqueeze(1)
        data_GL = torch.cat((data_S0,self.fake_B[:,:,43:85,43:85]),1)
        data_H = torch.cat((data_S0,self.real_B[:,:,43:85,43:85]),1)   
        data_GL_p = torch.cat((data_S0,self.fake_B[:,0:12,43:85,43:85]),1)
        data_GL_o = torch.cat((data_S0,self.fake_B[:,12:25,43:85,43:85]),1)      
        data_H_p = torch.cat((data_S0,self.real_B[:,0:12,43:85,43:85]),1)
        data_H_o = torch.cat((data_S0,self.real_B[:,12:25,43:85,43:85]),1)   
        data_S0 = data_S0.cpu().detach().numpy()
        data_GL = data_GL.cpu().detach().numpy()
        data_H = data_H.cpu().detach().numpy()
        data_GL_p = data_GL_p.cpu().detach().numpy()
        data_GL_o = data_GL_o.cpu().detach().numpy() 
        data_H_p = data_H_p.cpu().detach().numpy()
        data_H_o = data_H_o.cpu().detach().numpy() 
#        np_data_A=self.real_A[:,:,43:85,43:85].cpu().detach().numpy()
#        np_data_B=self.fake_B[:,:,43:85,43:85].cpu().detach().numpy()
#        np_data_AB=torch.cat((self.real_A[:,:,43:85,43:85], self.fake_B[:,1:13,43:85,43:85]), 1)
#        np_data_AB=np_data_AB.cpu().detach().numpy()
#        np_data_A=np_data_A.transpose(2,3,0,1)
#        np_data_B=np_data_B.transpose(2,3,0,1)
#        np_data_AB=np_data_AB.transpose(2,3,0,1)
        data_S0 = data_S0.transpose(2,3,0,1)
        data_GL = data_GL.transpose(2,3,0,1)
        data_H = data_H.transpose(2,3,0,1)
        data_GL_p = data_GL_p.transpose(2,3,0,1)
        data_GL_o = data_GL_o.transpose(2,3,0,1)
        data_H_p = data_H_p.transpose(2,3,0,1)
        data_H_o = data_H_o.transpose(2,3,0,1)
#         f = open("/misc/raid/home/yhe/DeepLearning/dMRI_gan/out/s.txt", 'w+')
# ##        fm=torch.squeeze(fake_AB[:,13,:,:])
# #        print(fm.shape, file=f)
# #        print(wd, file=f)
#         print(np_data_A.shape, '\n',np_data_B.shape,'\n',file=f)
#         print(np_data_AB.shape, file=f)
        # loss_G_TDm_A_B=networks.loss_TDmeasures(np_data_A,np_data_B,self.tenmodel_A,self.tenmodel_B,self.criterionMSE)
        # loss_G_TDm_AB_A=networks.loss_TDmeasures(np_data_AB,np_data_A,self.tenmodel_AB,self.tenmodel_A,self.criterionMSE)
        # loss_G_TDm_AB_B=networks.loss_TDmeasures(np_data_AB,np_data_B,self.tenmodel_AB,self.tenmodel_B,self.criterionMSE)
#        loss_G_TDm_A_B=networks.loss_TDmeasures(np_data_A,np_data_B,self.tenmodel_A,self.tenmodel_B,self.criterionL1)
#        loss_G_TDm_AB_A=networks.loss_TDmeasures(np_data_AB,np_data_A,self.tenmodel_AB,self.tenmodel_A,self.criterionL1)
#        loss_G_TDm_AB_B=networks.loss_TDmeasures(np_data_AB,np_data_B,self.tenmodel_AB,self.tenmodel_B,self.criterionL1)
        loss_consistent_G_H = networks.loss_TDmeasures(data_GL,data_H,self.tenmodel_24,self.tenmodel_24,self.criterionMSE)
        consistent_sub_1_1 = networks.loss_TDmeasures(data_GL_p,data_H_p,self.tenmodel_12p,self.tenmodel_12p,self.criterionMSE)
        consistent_sub_1_2 = networks.loss_TDmeasures(data_GL_o,data_H_p,self.tenmodel_12o,self.tenmodel_12p,self.criterionMSE)        
        consistent_sub_2_1 = networks.loss_TDmeasures(data_GL_p,data_H_o,self.tenmodel_12p,self.tenmodel_12o,self.criterionMSE)
        consistent_sub_2_2 = networks.loss_TDmeasures(data_GL_o,data_H_o,self.tenmodel_12o,self.tenmodel_12o,self.criterionMSE)
#        Consistent_term_2 = networks.loss_TDmeasures(data_GL_p,data_H_p,self.tenmodel_12p,self.tenmodel_12p,self.criterionMSE)
#        Consistent_term_3 = networks.loss_TDmeasures(data_GL_p,data_H_p,self.tenmodel_12p,self.tenmodel_12p,self.criterionMSE)
#        Loss_Gx_Y_b = networks.loss_TDmeasures(data_GL_o,data_H_p,self.tenmodel_12o,self.tenmodel_12p,self.criterionMSE)
#        Loss_Gx_Y_ab = networks.loss_TDmeasures(data_Gx,data_Y,self.tenmodel_24,self.tenmodel_24,self.criterionMSE)
        loss_cycle_const = torch.mean(consistent_sub_1_1 + consistent_sub_1_2 + consistent_sub_2_1 + consistent_sub_2_2)
#        Loss_cycle_const = torch.mean(Loss_Gx_Y_a + Loss_Gx_Y_b + Loss_Gx_Y_ab)
#        Loss_Ga_Gb = networks.loss_TDmeasures(data_Gx_a,data_Gx_b,self.tenmodel_12a,self.tenmodel_12b,self.criterionMSE)
        self.loss_G_const = loss_consistent_G_H + loss_cycle_const
        # combine all the three loss to update parameters of G
        self.loss_G = self.loss_G_GAN*self.opt.lambda_G_GAN + self.loss_G_sim*self.opt.lambda_G_sim + self.loss_G_const*self.opt.lambda_G_const
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
