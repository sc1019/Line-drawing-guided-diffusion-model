#from comet_ml import Experiment
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torchvision
import os
import errno
import shutil
import argparse

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', default=1000, type=int,
                        help="The number of steps the scheduler takes to go from clean image to an isotropic gaussian. This is also the number of steps of diffusion.")
    parser.add_argument('--train_steps', default=7000000, type=int,
                        help='The number of iterations for training.')
    parser.add_argument('--sample_steps', default=None, type=int)
    parser.add_argument('--save_folder', default='./results', type=str)
    parser.add_argument('--data_path', default='./root_celebA_128_test_new/', type=str)
    parser.add_argument('--load_path', default='./results/model.pt', type=str)
    parser.add_argument('--train_routine', default='Final', type=str)
    parser.add_argument('--sampling_routine', default='default', type=str,
                        help='The choice of sampling routine for reversing the diffusion process.')
    parser.add_argument('--remove_time_embed', action="store_true")
    parser.add_argument('--residual', action="store_true")
    parser.add_argument('--loss_type', default='l2', type=str)
    parser.add_argument('--test_type', default='test_data', type=str)
    parser.add_argument('--painting_path', default='./edge_img/', type=str)

    args = parser.parse_args()
    print(args)

    img_path=None
    if 'train' in args.test_type:
        img_path = args.data_path
    elif 'test' in args.test_type:
        img_path = args.data_path
    if args.test_type == 'test_painting':
        img_path = args.painting_path
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels=3,
        with_time_emb=not(args.remove_time_embed),
        residual=args.residual
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        channels = 3,
        timesteps = args.time_steps,   # number of steps
        loss_type = args.loss_type,    # L1 or L2
        train_routine = args.train_routine,
        sampling_routine = args.sampling_routine
    ).cuda()

    import torch
    diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))


    trainer = Trainer(
        diffusion,
        img_path,
        image_size = 128,
        train_batch_size = 1,
        train_lr = 2e-5,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        fp16 = False,                       # turn on mixed precision training with apex
        save_and_sample_every = 1000,
        results_folder = args.save_folder,
        load_path = args.load_path,
        dataset = 'train'
    )

    if args.test_type == 'train_data':
        trainer.test_from_data('train', s_times=args.sample_steps)

    elif args.test_type == 'test_data':
        trainer.test_from_data('test', s_times=args.sample_steps)

    elif args.test_type == 'test_painting':
        trainer.test_from_painting(True,input_path=img_path,img = '2.png')
    #### for FID and noise ablation ##
    elif args.test_type == 'test_sample_and_save_for_fid':
        trainer.sample_and_save_for_fid()

    ########## for paper ##########

    elif args.test_type == 'train_paper_showing_diffusion_images_cover_page':
        trainer.paper_showing_diffusion_images_cover_page()

if __name__ == '__main__':
    main()