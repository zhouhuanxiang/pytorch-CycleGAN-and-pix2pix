import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import face_networks


class CycleFaceGANModel(BaseModel):
    def name(self):
        return 'CycleFaceGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        parser.set_defaults(dataset_mode="face")
        parser.set_defaults(checkpoints_dir="/home2/zhx/checkpoints")
        parser.set_defaults(lr=0.0001)
        parser.set_defaults(dataroot='/home2/zhx/data/CASIA-WebFace')
        parser.set_defaults(norm='batch')
        parser.set_defaults(save_epoch_freq=3)
        parser.set_defaults(display_ncols=8)

        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--class_n', type=int, default=10575, help='to do')
            parser.add_argument('--classify', action='store_true', help='to do')
            parser.add_argument('--validate_freq', type=int, default=10000, help='to do')
            parser.add_argument('--display_nrows', type=int, default=6, help='to do')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.total = 0
        self.correct = 0
        self.count = 0
        self.loss_precision = 0

        self.loss_D_A = 0
        self.loss_G_A = 0
        self.loss_cycle_A = 0
        self.loss_idt_A = 0
        self.loss_D_B = 0
        self.loss_G_B = 0
        self.loss_cycle_B = 0
        self.loss_idt_B = 0
        self.loss_classify = 0
        self.loss_identity = 0
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if not self.opt.classify:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'classify', 'identity']
        else:
            self.loss_names = ['precision', 'classify']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if not self.opt.classify:
            visual_names_A = ['real_AA', 'fake_BA', 'rec_AA']
            visual_names_B = ['real_BB', 'fake_AB', 'rec_BB']
        else:
            visual_names_A = ['real_AA']
            visual_names_B = ['real_BB']
        if not self.opt.classify and self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'D']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = face_networks.define_G(self.opt.class_n, opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = face_networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                            self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = face_networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss().to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionClassify = torch.nn.CrossEntropyLoss().to(self.device)
            self.criterionIndentity = torch.nn.L1Loss().to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_AA = input['A'].squeeze(0).to(self.device)
        self.real_BB = input['B'].squeeze(0).to(self.device)
        self.label_A = input['A_label'].squeeze(0).to(self.device)
        self.label_B = input['B_label'].squeeze(0).to(self.device)

        # print(self.label_A)
        # print(self.label_B)
        # print(self.real_AA.size())
        # print(self.real_BB.size())
        # print(self.label_A.size())
        # print(self.label_B.size())

    def forward(self):
        self.identity_AA, self.label_AA, self.fake_BA = self.netG(self.real_AA, self.real_BB)
        if not self.opt.classify:
            self.identity_BA, self.label_BA, self.rec_AA = self.netG(self.fake_BA, self.real_AA)

        self.identity_BB, self.label_BB, self.fake_AB = self.netG(self.real_BB, self.real_AA)
        if not self.opt.classify:
            self.identity_AB, self.label_AB, self.rec_BB = self.netG(self.fake_AB, self.real_BB)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_BA = self.fake_B_pool.query(self.fake_BA)
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_BB, fake_BA)

    def backward_D_B(self):
        fake_AB = self.fake_A_pool.query(self.fake_AB)
        self.loss_D_B = self.backward_D_basic(self.netD, self.real_AA, fake_AB)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if not self.opt.classify:
            # combined loss, two pathway step
            self.optimizer_G.zero_grad()
            # first, backward loss_others
            self.netG.module.set_requires_grad(iden_grad=False)
            # Identity loss
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed.
                _, _, self.idt_A = self.netG(self.real_BB, self.real_BB)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_BB) * lambda_B * lambda_idt
                # G_B should be identity if real_A is fed.
                _, _, self.idt_B = self.netG(self.real_AA, self.real_AA)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_AA) * lambda_A * lambda_idt
            else:
                self.loss_idt_A = 0
                self.loss_idt_B = 0

            # identity preserving
            self.loss_classify = self.criterionClassify(self.label_BA, self.label_A) \
                                 + self.criterionClassify(self.label_AB, self.label_B)
            # self.loss_classify = self.criterionClassify(self.label_AA, self.label_A) \
            #                      + self.criterionClassify(self.label_AB, self.label_B)
            self.loss_identity = self.criterionIndentity(self.identity_BA, self.identity_AA.detach()) \
                                 + self.criterionIndentity(self.identity_AB, self.identity_BB.detach())
            # self.loss_identity = torch.mean(torch.abs(self.identity_BA - self.identity_AA)) \
            #                      + torch.mean(torch.abs(self.identity_AB - self.identity_BB))

            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD(self.fake_BA), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD(self.fake_AB), True)
            # Forward cycle loss
            self.loss_cycle_A = self.criterionCycle(self.rec_AA, self.real_AA) * lambda_A
            # Backward cycle loss
            self.loss_cycle_B = self.criterionCycle(self.rec_BB, self.real_BB) * lambda_B
            self.loss_G = self.loss_G_A + self.loss_G_B \
                          + self.loss_cycle_A + self.loss_cycle_B \
                          + self.loss_idt_A + self.loss_idt_B \
                          + self.loss_classify + self.loss_identity
            self.loss_G.backward()
            # second, backward loss_classify
            self.netG.module.set_requires_grad(iden_grad=True)
            _, self.label_AA, _ = self.netG(self.real_AA, self.real_BB)
            _, self.label_BB, _ = self.netG(self.real_BB, self.real_AA)
            self.loss_classify = self.criterionClassify(self.label_AA, self.label_A) \
                                 + self.criterionClassify(self.label_BB, self.label_B)
            self.loss_classify.backward()
            # last, step grad
            self.optimizer_G.step()
        else:
            self.netG.module.set_requires_grad(iden_grad=True, attr_grad=True, gen_grad=True)
            self.optimizer_G.zero_grad()
            self.loss_classify = self.criterionClassify(self.label_AA, self.label_A) \
                                 + self.criterionClassify(self.label_BB, self.label_B)
            # test
            self.count = (self.count + 1) % 1000
            _, predicted = torch.max(self.label_AA, 1)
            self.total += self.label_A.size(0)
            self.correct += (predicted == self.label_A).sum().item()
            if self.count == 0:
                self.loss_precision = self.correct / self.total * 100.0
                self.total = 0
                self.correct = 0


            self.loss_G = self.loss_classify
            self.loss_G.backward()

            self.optimizer_G.step()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD], False)
        self.backward_G()
        # D_A and D_B
        if not self.opt.classify:
            self.set_requires_grad([self.netD], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()
