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
        if (random.randint(0, self.opt.same_person_ratio) or self.opt.same_person) and not self.opt.classify:
            A_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            # A1_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            # A2_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            index_B = index_A
            B_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]
            # B1_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]
            # B2_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]
        else:
            A_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            # A1_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            # A2_path = np.random.choice(self.img_paths[index_A], 1, replace=True)[0]
            index_B = random.randint(0, self.A_size - 1)
            B_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]
            # B1_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]
            # B2_path = np.random.choice(self.img_paths[index_B], 1, replace=True)[0]

        A_img = Image.open(A_path).convert('RGB')
        # A1_img = Image.open(A1_path).convert('RGB')
        # A2_img = Image.open(A2_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # B1_img = Image.open(B1_path).convert('RGB')
        # B2_img = Image.open(B2_path).convert('RGB')

        A = self.transform(A_img)
        # A1 = self.transform(A1_img)
        # A2 = self.transform(A2_img)
        B = self.transform(B_img)
        # B1 = self.transform(B1_img)
        # B2 = self.transform(B2_img)


        # return {'A': A, 'B': B, 'A_label': index_A, 'B_label': index_B,
        #         'A1': A1, 'A2': A2,
        #         'B1': B1, 'B2': B2}

        return {'A': A, 'B': B, 'A_label': index_A, 'B_label': index_B  }

    def __len__(self):
        # return self.A_size
        return 494414

    def name(self):
        return 'FaceDataset'
