from functools import partial
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist

def get_loader(args, split, cuda):
    data_dir = args.root
    datalist_json = args.json_list

    if split == 'train':
        train_transform = transforms.Compose([transforms.LoadImaged(keys = ['image', 'label'],
                                                                    ensure_channel_first = True),
                                              transforms.ScaleIntensityRanged(keys = 'image',
                                                                              a_min = args.a_min,
                                                                              a_max = args.a_max,
                                                                              b_min = args.b_min,
                                                                              b_max = args.b_max,
                                                                              clip = True),
                                              transforms.CropForegroundd(keys = ['image', 'label'],
                                                                         source_key = 'image'),
                                              transforms.RandCropByPosNegLabeld(keys = ['image', 'label'],
                                                                                label_key = 'label',
                                                                                spatial_size = (args.roi_x, args.roi_y),
                                                                                pos = 1,
                                                                                neg = 1,
                                                                                num_samples = 4,
                                                                                image_key = 'image',
                                                                                image_threhsold = 0),
                                              transforms.RandFlipd(keys = ['image', 'label'],
                                                                   prob = args.randflip_prob,
                                                                   spatial_axis = 0),
                                              transforms.RandFlipd(keys = ['image', 'label'],
                                                                   prob = args.randflip_prob,
                                                                   spatial_axis = 1),
                                              transforms.RandRotate90d(keys = ['image', 'label'],
                                                                   prob = args.randrotate90_prob,
                                                                   max_k = 3),
                                              transforms.RandScaleIntensityd(keys = 'image',
                                                                             prob = args.randscaleintensity_prob,
                                                                             factors = 0.1),
                                              transforms.RandShiftIntensityd(keys = 'image',
                                                                             prob = args.randshiftintensity_prob,
                                                                             offsets = 0.1),
                                              transforms.Activationsd(keys = 'label',
                                                                      other = lambda x : partial(torch.bucketsize,
                                                                                                 boundaries = torch.tensor)),
                                              transforms.ToTensord(keys = ['image', 'label'])
                                            ])
        datalist = load_decathlon_datalist(datalist_json, True, 'training', base_dir = data_dir)
        train_ds = data.Dataset(data =  datalist, transform = train_transform)
        loader = data.DataLoader(train_ds,
                                 batch_size = args.batch_size,
                                 num_workers = args.num_workers,
                                 pin_memory = cuda)
    
    elif split == 'val':
        val_transform = transforms.Compose([transforms.LoadImaged(keys = ['image', 'label'],
                                                                  ensure_channel_first = True),
                                            transforms.ScaleIntensityRanged(keys = 'image',
                                                                            a_min = args.a_min,
                                                                            a_max = args.a_max,
                                                                            b_min = args.b_min,
                                                                            b_max = args.b_max,
                                                                            clip = True),
                                            transforms.CropForegroundd(keys = ['image', 'label'],
                                                                       source_key = 'image'),
                                            transforms.SpatialPadd(keys = ['image', 'label'],
                                                                   spatial_size = [512, 384]),
                                            transforms.Activationsd(keys = 'label',
                                                                    other = lambda x : partial(torch.bucketsize,
                                                                                               boundaries = torch.tensor)),
                                            transforms.ToTensord(keys = ['image', 'label'])
                                            ])
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir = data_dir)
        val_ds = data.Dataset(data = val_files, transform = val_transform)
        loader = data.DataLoader(val_ds,
                                 batch_size = 1,
                                 shuffle = False,
                                 num_workers = args.workers,
                                 pin_memory = cuda)
    
    elif split == 'test'
        test_transform = transforms.Compose([transforms.LoadImaged(keys = ['image', 'label'],
                                                                  ensure_channel_first = True),
                                            transforms.ScaleIntensityRanged(keys = 'image',
                                                                            a_min = args.a_min,
                                                                            a_max = args.a_max,
                                                                            b_min = args.b_min,
                                                                            b_max = args.b_max,
                                                                            clip = True),
                                            transforms.CropForegroundd(keys = ['image', 'label'],
                                                                       source_key = 'image'),
                                            transforms.SpatialPadd(keys = ['image', 'label'],
                                                                   spatial_size = [512, 384]),
                                            transforms.Activationsd(keys = 'label',
                                                                    other = lambda x : partial(torch.bucketsize,
                                                                                               boundaries = torch.tensor)),
                                            transforms.ToTensord(keys = ['image', 'label'])
                                            ])
        test_files = load_decathlon_datalist(datalist_json, True, "test", base_dir = data_dir)
        test_ds = data.Dataset(data = test_files, transform = test_transform)
        loader = data.DataLoader(val_ds,
                                 batch_size = 1,
                                 shuffle = False,
                                 num_workers = args.workers,
                                 pin_memory = cuda)
    
    return loader
        