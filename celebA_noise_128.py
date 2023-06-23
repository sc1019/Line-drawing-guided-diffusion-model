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
    parser.add_argument('--train_steps', default=700000, type=int,
                        help='The number of iterations for training.')
    parser.add_argument('--save_folder', default='./results', type=str)
    # parser.add_argument('--load_path', default='./results/model-latest.pt', type=str)
    parser.add_argument('--load_path', default=None, type=str)
    # parser.add_argument('--data_path', default='./root_celebA_128_train_new/', type=str)
    parser.add_argument('--data_path', default='root_celebA_128_train_new', type=str)
    parser.add_argument('--train_routine', default='Final', type=str)
    parser.add_argument('--sampling_routine', default='default', type=str,
                        help='The choice of sampling routine for reversing the diffusion process.')
    parser.add_argument('--remove_time_embed', action="store_true")
    parser.add_argument('--residual', action="store_true")
    parser.add_argument('--loss_type', default='l2', type=str)
  
    args = parser.parse_args()
    print(args)

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=4
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=args.time_steps,  # number of steps
        sampling_timesteps = 250
    ).cuda()
    import torch

    trainer = Trainer(
        diffusion,
        folder = args.data_path,
        train_batch_size = 16,
        train_lr = 8e-5,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 25,
        results_folder=args.save_folder,
        load_path=args.load_path,
        amp = False,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        save_model_every = 100000
    )
    trainer.train()


if __name__ == '__main__':
    main()
