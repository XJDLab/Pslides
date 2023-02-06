the code refer the following codes:

Reference:
1) https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms
2) https://iopscience.iop.org/article/10.3847/2041-8213/abc255
3) https://iopscience.iop.org/article/10.3847/2041-8213/ab9085 
4) https://github.com/JeongHyunJin/Pix2PixHD



import os
from os.path import split, splitext
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.ndimage import rotate
from random import randint
from PIL import Image
import argparse
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime   

#------------------------------------------------------------------------------
# [1] Preparing the Input and Target data sets

class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
                
        if opt.is_train:
            self.input_format = opt.data_format_input
            self.target_format = opt.data_format_target
            self.input_dir = opt.input_dir_train
            self.target_dir = opt.target_dir_train

            self.label_path_list = sorted(glob(os.path.join(self.input_dir, '*.' + self.input_format)))
            self.target_path_list = sorted(glob(os.path.join(self.target_dir, '*.' + self.target_format)))
            print(len(self.label_path_list), len(self.target_path_list))
        else:
            self.input_format = opt.data_format_input
            self.input_dir = opt.input_dir_test

            self.label_path_list = sorted(glob(os.path.join(self.input_dir, '*.' + self.input_format)))
            

    def __getitem__(self, index):
        list_transforms = []
        list_transforms += []

# [ Train data ] ==============================================================
        if self.opt.is_train:
            self.angle = randint(-self.opt.max_rotation_angle, self.opt.max_rotation_angle)

            self.offset_x = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            self.offset_y = randint(0, 2 * self.opt.padding_size - 1) if self.opt.padding_size > 0 else 0
            
# [ Train Input ] =============================================================
            if self.input_format in ["tif", "tiff", "png", "jpg", "jpeg"]:
                IMG_A0 = np.array(Image.open(self.label_path_list[index]), dtype=np.float32)
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            #elif self.input_format in ["fits", "fts", "fit"]:
                #IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data, dtype=np.float32)
            else:
                NotImplementedError("Please check data_format_input option. It has to be tif or npy or fits.")
          
            #--------------------------------------
            if len(IMG_A0.shape) == 3:
                IMG_A0 = IMG_A0.transpose(2, 0 ,1)

            #--------------------------------------
            UpIA = np.float(self.opt.saturation_upper_limit_input)
            LoIA = np.float(self.opt.saturation_lower_limit_input)
            
            if self.opt.saturation_clip_input == True:
                label_array = (np.clip(IMG_A0, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            else:
                label_array = (IMG_A0-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
                
            #--------------------------------------
            if self.opt.logscale_input == True:
                label_array[np.isnan(label_array)] = 0.1
                label_array[label_array == 0] = 0.1
                label_array = np.log10(label_array)
            else:
                label_array[np.isnan(label_array)] = 0
            
            #--------------------------------------
            label_array = self.__rotate(label_array)
            label_array = self.__pad(label_array, self.opt.padding_size)
            label_array = self.__random_crop(label_array)
            
            label_tensor = torch.tensor(label_array, dtype=torch.float32)
            
            #--------------------------------------
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
                
# [ Train Target ] ============================================================
            if self.input_format in ["tif", "tiff", "png", "jpg", "jpeg"]:
                IMG_B0 = np.array(Image.open(self.target_path_list[index]), dtype=np.float32)
            elif self.target_format in ["npy"]:
                IMG_B0 = np.load(self.target_path_list[index], allow_pickle=True)
            #elif self.target_format in ["fits", "fts", "fit"]:
                #IMG_B0 = np.array(fits.open(self.target_path_list[index])[0].data, dtype=np.float32)
            else:
                NotImplementedError("Please check data_format_target option. It has to be tif or npy or fits.")
            
            #--------------------------------------
            if len(IMG_B0.shape) == 3:
                IMG_B0 = IMG_B0.transpose(2, 0 ,1)
            
            #--------------------------------------
            IMG_B0[np.isnan(IMG_B0)] = 0
            UpIB = np.float(self.opt.saturation_upper_limit_target)
            LoIB = np.float(self.opt.saturation_lower_limit_target)
            
            if self.opt.saturation_clip_target == True:
                target_array = (np.clip(IMG_B0, LoIB, UpIB)-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            else:
                target_array = (IMG_B0-(UpIB+ LoIB)/2)/((UpIB - LoIB)/2)
            
            #--------------------------------------
            if self.opt.logscale_target == True:
                target_array[np.isnan(target_array)] = 0.1
                target_array[target_array == 0] = 0.1
                target_array = np.log10(target_array)
            else:
                target_array[np.isnan(target_array)] = 0
            
            #--------------------------------------
            target_array = self.__rotate(target_array)
            target_array = self.__pad(target_array, self.opt.padding_size)
            target_array = self.__random_crop(target_array)
            
            target_tensor = torch.tensor(target_array, dtype=torch.float32)
            
            #--------------------------------------
            if len(target_tensor.shape) == 2:
                target_tensor = target_tensor.unsqueeze(dim=0)  # Add channel dimension.


# [ Test data ] ===============================================================
        else:
# [ Test Input ] ==============================================================
            if self.input_format in ["tif", "tiff", "png", "jpg", "jpeg"]:
                IMG_A0 = np.array(Image.open(self.label_path_list[index]), dtype=np.float32)       
            elif self.input_format in ["npy"]:
                IMG_A0 = np.load(self.label_path_list[index], allow_pickle=True)
            #elif self.input_format in ["fits", "fts", "fit"]:                    
                #IMG_A0 = np.array(fits.open(self.label_path_list[index])[0].data, dtype=np.float32)
            else:
                NotImplementedError("Please check data_format_input option. It has to be tif or npy or fits.")
            
            #--------------------------------------
            if len(IMG_A0.shape) == 3:
                IMG_A0 = IMG_A0.transpose(2, 0 ,1)

            #--------------------------------------
            UpIA = np.float(self.opt.saturation_upper_limit_input)
            LoIA = np.float(self.opt.saturation_lower_limit_input)
            
            if self.opt.saturation_clip_input == True:
                label_array = (np.clip(IMG_A0, LoIA, UpIA)-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            else:
                label_array = (IMG_A0-(UpIA+LoIA)/2)/((UpIA - LoIA)/2)
            
            #--------------------------------------
            if self.opt.logscale_input == True:
                label_array[np.isnan(label_array)] = 0.1
                label_array[label_array == 0] = 0.1
                label_array = np.log10(label_array)
            else:
                label_array[np.isnan(label_array)] = 0
            
            label_tensor = torch.tensor(label_array, dtype=torch.float32)
            
            #--------------------------------------
            if len(label_tensor.shape) == 2:
                label_tensor = label_tensor.unsqueeze(dim=0)
            
            #--------------------------------------
            
            return label_tensor, splitext(split(self.label_path_list[index])[-1])[0]
        
        return label_tensor, target_tensor, splitext(split(self.label_path_list[index])[-1])[0], \
                   splitext(split(self.target_path_list[index])[-1])[0]

#------------------------------------------------------------------------------
# [2] Adjust or Measure the Input and Target data sets
                   
    def __random_crop(self, x):
        x = np.array(x)
        x = x[self.offset_x: self.offset_x + 1024, self.offset_y: self.offset_y + 1024]
        return x

    @staticmethod
    def __pad(x, padding_size):
        if type(padding_size) == int:
            if len(x.shape) == 3:
                padding_size= ((0, 0), (padding_size, padding_size), (padding_size, padding_size))
            else:
                padding_size = ((padding_size, padding_size), (padding_size, padding_size))
        return np.pad(x, pad_width=padding_size, mode="constant", constant_values=0)

    def __rotate(self, x):
        return rotate(x, self.angle, reshape=False)

    @staticmethod
    def __to_numpy(x):
        return np.array(x, dtype=np.float32)

    def __len__(self):
        return len(self.label_path_list)



##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################


class BaseOption(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--gpu_ids', type=int, default=0, help='gpu number. If -1, use cpu')
        self.parser.add_argument('--data_format_input', type=str, default='jpg',
                                 help="Input data extension. This will be used for loading and saving. [tif, png, jpg or npy or fits]")
        self.parser.add_argument('--data_format_target', type=str, default='jpg',
                                 help="Target data extension. This will be used for loading and saving. [tif, png, jpg or npy or fits]")
        
        #----------------------------------------------------------------------
        # data option
        self.parser.add_argument('--dataset_name', type=str, default='Pix2PixHD', help='dataset directory name')
        
        self.parser.add_argument('--input_ch', type=int, default=3, help="# of input channels for Generater")
        self.parser.add_argument('--target_ch', type=int, default=3, help="# of target channels for Generater")
        
        self.parser.add_argument('--data_size', type=int, default=1024, help='image size of the input and target data')
        
        self.parser.add_argument('--logscale_input', type=bool, default=False, help='use logarithmic scales to the input data sets')
        self.parser.add_argument('--logscale_target', type=bool, default=False, help="use logarithmic scales to the target data sets")
        
        self.parser.add_argument('--saturation_lower_limit_input', type=float, default=0, help="Saturation value (lower limit) of input")
        self.parser.add_argument('--saturation_upper_limit_input', type=float, default=255, help="Saturation value (upper limit) of input")
        self.parser.add_argument('--saturation_lower_limit_target', type=float, default=0, help="Saturation value (lower limit) of target")
        self.parser.add_argument('--saturation_upper_limit_target', type=float, default=255, help="Saturation value (upper limit) of target")
        
        self.parser.add_argument('--saturation_clip_input', type=bool, default=True, help="Saturation clip for input data")
        self.parser.add_argument('--saturation_clip_target', type=bool, default=True, help="Saturation clip for target data")

        #----------------------------------------------------------------------

        # data augmentation
        self.parser.add_argument('--batch_size', type=int, default=1, help='the number of batch_size')
        self.parser.add_argument('--data_type', type=int, default=32, help='float dtype')
        self.parser.add_argument('--image_mode', type=str, default='png', help='extension for saving image')
        
        # network option
        self.parser.add_argument('--n_gf', type=int, default=64, help='the number of channels in the first layer of G')
        self.parser.add_argument('--n_downsample', type=int, default=4, help='how many times you want to downsample input data in G')
        self.parser.add_argument('--n_residual', type=int, default=9, help='the number of residual blocks in G')
        self.parser.add_argument('--n_df', type=int, default=64, help='the number of channels in the first layer of D')
        self.parser.add_argument('--n_D', type=int, default=2, help='how many discriminators in differet scales you want to use')
        
        self.parser.add_argument('--n_workers', type=int, default=0, help='how many threads you want to use')
        self.parser.add_argument('--norm_type', type=str, default='InstanceNorm2d', help='[BatchNorm2d, InstanceNorm2d]')
        self.parser.add_argument('--padding_type', type=str, default='reflection', help='[reflection, replication, zero]')
        self.parser.add_argument('--padding_size', type=int, default=0, help='padding size')
        self.parser.add_argument('--max_rotation_angle', type=int, default=0, help='rotation angle in degrees')
        self.parser.add_argument('--val_during_train', action='store_true', default=False)
        
        self.parser.add_argument('--report_freq', type=int, default=100)
        self.parser.add_argument('--save_freq', type=int, default=2000)  #default=2000
        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--save_scale', type=float, default=1)
        self.parser.add_argument('--display_scale', type=float, default=1)

        
    def parse(self):
        opt = self.parser.parse_args(args=[])
        opt.format = 'jpg'  # extension for checking image 
        opt.flip = False
                    
        #--------------------------------
        if opt.data_type == 16:
            opt.eps = 1e-4
        elif opt.data_type == 32:
            opt.eps = 1e-8
        
        #--------------------------------
        dataset_name = opt.dataset_name

        os.makedirs(os.path.join(r"\checkpoints", dataset_name, 'Image', 'Train'), exist_ok=True)
        os.makedirs(os.path.join(r"\checkpoints", dataset_name, 'Image', 'Test'), exist_ok=True)
        os.makedirs(os.path.join(r"\checkpoints", dataset_name, 'Model'), exist_ok=True)
        
        if opt.is_train:
            opt.image_dir = os.path.join(r"\checkpoints", dataset_name, 'Image/Train')
        else:
            opt.image_dir = os.path.join(r"\checkpoints", dataset_name, 'Image/Test')

        opt.model_dir = os.path.join(r"\checkpoints1", dataset_name, 'Model')
        
        #--------------------------------
        return opt

#------------------------------------------------------------------------------

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()
        
        #----------------------------------------------------------------------
        # directory path for training
        self.parser.add_argument('--input_dir_train', type=str, default=r"datasets\IF_input_tiles", help='directory path of the input files for the model training')
        self.parser.add_argument('--target_dir_train', type=str, default=r"datasets\HE_target_tiles", help='directory path of the input files for the model training')
        #----------------------------------------------------------------------
        
        self.parser.add_argument('--is_train', type=bool, default=True, help='train flag')
        self.parser.add_argument('--n_epochs', type=int, default=0, help='how many epochs you want to train')
        self.parser.add_argument('--latest', type=int, default=0, help='Resume epoch')
        
        # hyperparameters
        self.parser.add_argument('--lambda_FM', type=int, default=10, help='weight for FM loss')
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--epoch_decay', type=int, default=100, help='when to start decay the lr')
        self.parser.add_argument('--lr', type=float, default=0.0002)

        self.parser.add_argument('--no_shuffle', action='store_true', default=False, help='if you want to shuffle the order')
        
#------------------------------------------------------------------------------

class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()
        
        #----------------------------------------------------------------------
        # directory path for test
        self.parser.add_argument('--input_dir_test', type=str, default=r"datasets\Test\Input", help='directory path of the input files for the model test')
        #----------------------------------------------------------------------
        
        self.parser.add_argument('--is_train', type=bool, default=False, help='test flag')
        self.parser.add_argument('--iteration', type=int, default=-1, help='if you want to generate from input for the specific iteration')
        self.parser.add_argument('--no_shuffle', type=bool, default=True, help='if you want to shuffle the order')


opt = TrainOption().parse()
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################


## Utils



#------------------------------------------------------------------------------
# [1] True or False grid

def get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid

#------------------------------------------------------------------------------
# [2] Set the Normalization method for the input layer

def get_norm_layer(type):
    if type == 'BatchNorm2d':
        layer = partial(nn.BatchNorm2d, affine=True)

    elif type == 'InstanceNorm2d':
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer

#------------------------------------------------------------------------------
# [3] Set the Padding method for the input layer

def get_pad_layer(type):
    if type == 'reflection':
        layer = nn.ReflectionPad2d

    elif type == 'replication':
        layer = nn.ReplicationPad2d

    elif type == 'zero':
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError("Padding type {} is not valid."
                                  " Please choose among ['reflection', 'replication', 'zero']".format(type))

    return layer

#------------------------------------------------------------------------------
# [4] Save or Report the model results 

class Manager(object):
    def __init__(self, opt):
        self.opt = opt
        self.dtype = opt.data_type

    @staticmethod
    def report_loss(package):
        print("Epoch: {} [{:.{prec}}%] Current_step: {} D_loss: {:.{prec}}  G_loss: {:.{prec}}".
              format(package['Epoch'], package['current_step']/package['total_step'] * 100, package['current_step'],
                     package['D_loss'], package['G_loss'], prec=4))

    def adjust_dynamic_range(self, data, drange_in, drange_out):
        if drange_in != drange_out:
            if self.dtype == 32:
                scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                            np.float32(drange_in[1]) - np.float32(drange_in[0]))
                bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            elif self.dtype == 16:
                scale = (np.float16(drange_out[1]) - np.float16(drange_out[0])) / (
                            np.float16(drange_in[1]) - np.float16(drange_in[0]))
                bias = (np.float16(drange_out[0]) - np.float16(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def tensor2image(self, image_tensor):
        np_image = image_tensor[0].squeeze().cpu().float().numpy()
        if len(np_image.shape) == 3:
            np_image = np.transpose(np_image, (1, 2, 0))  # HWC
        else:
            pass

        np_image = self.adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        return np_image

    def save_image(self, image_tensor, path):
        Image.fromarray(self.tensor2image(image_tensor)).save(path, self.opt.image_mode)

    def save(self, package, image=False, model=False):
        if image:
            path_real = os.path.join(self.opt.image_dir, str(package['current_step']) + '_' + 'real.png')
            path_fake = os.path.join(self.opt.image_dir, str(package['current_step']) + '_' + 'fake.png')
            self.save_image(package['target_tensor'], path_real)
            self.save_image(package['generated_tensor'], path_fake)

        elif model:
            path_D = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'D.pt')
            path_G = os.path.join(self.opt.model_dir, str(package['current_step']) + '_' + 'G.pt')
            torch.save(package['D_state_dict'], path_D)
            torch.save(package['G_state_dict'], path_G)

    def __call__(self, package):
        if package['current_step'] % self.opt.display_freq == 0:
            self.save(package, image=True)

        if package['current_step'] % self.opt.report_freq == 0:
            self.report_loss(package)

        if package['current_step'] % self.opt.save_freq == 0:
            self.save(package, model=True)

#------------------------------------------------------------------------------
# Update the learning rate 

def update_lr(old_lr, init_lr, n_epoch_decay, D_optim, G_optim):
    delta_lr = init_lr / n_epoch_decay
    new_lr = old_lr - delta_lr

    for param_group in D_optim.param_groups:
        param_group['lr'] = new_lr

    for param_group in G_optim.param_groups:
        param_group['lr'] = new_lr

    print("Learning rate has been updated from {} to {}.".format(old_lr, new_lr))

    return new_lr

#------------------------------------------------------------------------------
# Set the initial conditions of weights

def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################


## model definition

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        
        act = nn.ReLU(inplace=True)
        input_ch = opt.input_ch
        n_gf = opt.n_gf
        norm = get_norm_layer(opt.norm_type)
        output_ch = opt.target_ch
        pad = get_pad_layer(opt.padding_type)

        model = []
        model += [pad(3), nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0), norm(n_gf), act]

        for _ in range(opt.n_downsample):
            model += [nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1, stride=2), norm(2 * n_gf), act]
            n_gf *= 2

        for _ in range(opt.n_residual):
            model += [ResidualBlock(n_gf, pad, norm, act)]

        for _ in range(opt.n_downsample):
            model += [nn.ConvTranspose2d(n_gf, n_gf//2, kernel_size=3, padding=1, stride=2, output_padding=1),
                      norm(n_gf//2), act]
            n_gf //= 2

        model += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

        print(self)
        print("the number of G parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels), act]
        block += [pad(1), nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1), norm(n_channels)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)



#------------------------------------------------------------------------------
# [2] Discriminative Network

class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = opt.input_ch + opt.target_ch
        n_df = opt.n_df
        norm = nn.InstanceNorm2d

        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        
        for i in range(opt.n_D):
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator(opt))
        self.n_D = opt.n_D

        print(self)
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
        return result



#------------------------------------------------------------------------------
# [3] Objective (Loss) functions


class Loss(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:0' if opt.gpu_ids != -1 else 'cpu:0')
        self.dtype = torch.float16 if opt.data_type == 16 else torch.float32

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = opt.n_D


    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = get_grid(fake_features[i][-1], is_real=False).to(self.device, self.dtype)

            loss_D += (self.criterion(real_features[i][-1], real_grid) +
                       self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        fake_features = D(torch.cat((input, fake), dim=1))

        for i in range(self.n_D):
            real_grid = get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.criterion(fake_features[i][-1], real_grid)
            
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())
                
            loss_G += loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM

        return loss_D, loss_G, target, fake




##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

## model train


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
#------------------------------------------------------------------------------
  

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' )


if __name__ == '__main__':
    start_time = datetime.datetime.now()

#------------------------------------------------------------------------------
# [1] Initial Conditions Setup
    
    dtype = torch.float16 if opt.data_type == 16 else torch.float32

    if opt.val_during_train:
        from options import TestOption
        test_opt = TestOption().parse()
        save_freq = opt.save_freq

    init_lr = opt.lr
    lr = opt.lr

    dataset = CustomDataset(opt)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.n_workers,
                             shuffle=not opt.no_shuffle)

    G = Generator(opt).apply(weights_init).to(device=device, dtype=dtype)
    D = Discriminator(opt).apply(weights_init).to(device=device, dtype=dtype)

    criterion = Loss(opt)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)

    if opt.latest and os.path.isfile(opt.model_dir + '/' + str(opt.latest) + '_dict.pt'):
        pt_file = torch.load(opt.model_dir + '/' + str(opt.latest) + '_dict.pt')
        init_epoch = pt_file['Epoch']
        print("Resume at epoch: ", init_epoch)
        G.load_state_dict(pt_file['G_state_dict'])
        D.load_state_dict(pt_file['D_state_dict'])
        G_optim.load_state_dict(pt_file['G_optim_state_dict'])
        D_optim.load_state_dict(pt_file['D_optim_state_dict'])
        current_step = init_epoch * len(dataset)

        for param_group in G_optim.param_groups:
            lr = param_group['lr']

    else:
        init_epoch = 1
        current_step = 0

    manager = Manager(opt)
    
#------------------------------------------------------------------------------
# [2] Model training
    
    total_step = opt.n_epochs * len(data_loader)

    for epoch in range(init_epoch, opt.n_epochs + 1):
        for input, target, _, _ in tqdm(data_loader):
            G.train()
         
            current_step += 1
            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)

            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'D_loss': D_loss.detach().item(),
                       'G_loss': G_loss.detach().item(),
                       'D_state_dict': D.state_dict(),
                       'G_state_dict': G.state_dict(),
                       'D_optim_state_dict': D_optim.state_dict(),
                       'G_optim_state_dict': G_optim.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor.detach()}

            manager(package)

#------------------------------------------------------------------------------
# [2] Model Checking 

            if opt.val_during_train and (current_step % save_freq == 0):
                G.eval()
                test_image_dir = os.path.join(test_opt.image_dir, str(current_step))
                os.makedirs(test_image_dir, exist_ok=True)
                test_model_dir = test_opt.model_dir

                test_dataset = CustomDataset(test_opt)
                test_data_loader = DataLoader(dataset=test_dataset,
                                              batch_size=test_opt.batch_size,
                                              num_workers=test_opt.n_workers,
                                              shuffle=not test_opt.no_shuffle)

                for p in G.parameters():
                    p.requires_grad_(False)

                for input, target, _, name in tqdm(test_data_loader):
                    input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                    fake = G(input)
                    
                    np_fake = fake.cpu().numpy().squeeze()
                    np_real = target.cpu().numpy().squeeze()
                    
                    if opt.display_scale != 1:
                        np_fake = np.clip(np_fake*np.float(opt.display_scale), -1, 1)
                        np_real = np.clip(np_real*np.float(opt.display_scale), -1, 1)
                        
                    manager.save_image(np_fake, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_fake.png'))
                    manager.save_image(np_real, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_real.png'))
                    

                for p in G.parameters():
                    p.requires_grad_(True)

#------------------------------------------------------------------------------

        if epoch > opt.epoch_decay :
            lr = update_lr(lr, init_lr, opt.n_epochs - opt.epoch_decay, D_optim, G_optim)
    
    end_time = datetime.datetime.now()
    
    print("Total time taken: ", end_time - start_time)


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

## model test

opt = TestOption().parse()

if __name__ == '__main__':
    
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    
    STD = opt.dataset_name

    dataset = CustomDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    iters = opt.iteration
    step = opt.save_freq

#------------------------------------------------------------------------------
    
    G = Generator(opt).to(device)
    G.load_state_dict(torch.load(r"\checkpoints\pix2pixHD\Model\270000_G.pt"))
    dir_image_save =r"\dataset\Input"
    with torch.no_grad():
        G.eval()
        for input,  name in tqdm(test_data_loader):
            input = input.to(device)
            fake = G(input)
            np_fake = fake.cpu().numpy().squeeze()
            np_fake = np_fake.transpose(1, 2 ,0)
            np_fake = adjust_dynamic_range(np_fake, drange_in=[-1., 1.], drange_out=[0, 255])
            np_fake = np.clip(np_fake, 0, 255).astype(np.uint8)
            #np_fake=np_fake*255
            #np_fake = np.asarray(np_fake, np.uint8)
            pil_image = Image.fromarray(np_fake)
            pil_image.save(os.path.join(dir_image_save, name[0] + '.png'))