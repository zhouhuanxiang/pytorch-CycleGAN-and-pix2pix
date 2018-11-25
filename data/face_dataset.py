import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch


class FaceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(loadSize=72)
        parser.set_defaults(fineSize=64)
        parser.set_defaults(resize_or_crop="resize_and_crop")

        return parser

    def make_folder(self, dir):
        folders = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, dnames, _ in sorted(os.walk(dir)):
            for dname in dnames:
                path = os.path.join(root, dname)
                folders.append(path)

        return folders

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.A_paths = self.make_folder(self.root)

        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.transform = get_transform(opt)
        self.batch_size = opt.batch_size

        self.img_paths = []
        for i in range(0, self.A_size):
            self.img_paths.append(make_dataset(self.A_paths[i]))

        print('############\ntotal %d folders\n############\n' % self.A_size)

    def __getitem__(self, index_A):
        index_A = index_A % self.A_size
        if index_A % 2 == 0 and not self.opt.classify:
            A_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            index_B = index_A
            B_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]
        else:
            A_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            index_B = random.randint(0, self.A_size - 1)
            B_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        # index_B = index_A
        # if index_A % 2 == 1:
        #     index_B = random.randint(0, self.A_size - 1)
        #     if index_B == index_A:
        #         index_B = (index_B + 1) % self.A_size
        #
        #
        # A_img_paths = self.img_paths[index_A].copy()
        # B_img_paths = self.img_paths[index_B].copy()
        # A_img_paths = np.random.choice(A_img_paths, self.batch_size, replace=True)
        # B_img_paths = np.random.choice(B_img_paths, self.batch_size, replace=True)
        #
        # A = torch.tensor([]);
        # B = torch.tensor([]);
        # A_label = torch.tensor([index_A] * self.batch_size)
        # B_label = torch.tensor([index_B] * self.batch_size)
        # for i in range(0, self.batch_size):
        #     A_img = Image.open(A_img_paths[i]).convert('RGB')
        #     B_img = Image.open(B_img_paths[i]).convert('RGB')
        #     A = torch.cat([A, self.transform(A_img).unsqueeze(0)], 0)
        #     B = torch.cat([B, self.transform(B_img).unsqueeze(0)], 0)


        return {'A': A, 'B': B, 'A_label': index_A, 'B_label': index_B}

    def __len__(self):
        # return self.A_size
        return 494414

    def name(self):
        return 'FaceDataset'
