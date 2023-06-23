#from comet_ml import Experiment
from Impasto_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import argparse
import torch


def main(mode):
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', default=50, type=int,
                        help="This is the number of steps in which a clean image looses information.")
    parser.add_argument('--sample_steps', default=None, type=int)
    parser.add_argument('--train_steps', default=100000, type=int,
                        help='The number of iterations for training.')
    parser.add_argument('--save_folder', default='./results', type=str)
    parser.add_argument('--data_path', default='./root_celebA_128_train_new/', type=str)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--train_routine', default='Final', type=str)
    parser.add_argument('--sampling_routine', default='x0_step_down', type=str,
                        help='The choice of sampling routine for reversing the diffusion process, when set as default it corresponds to Alg. 1 while when set as x0_step_down it stands for Alg. 2')
    parser.add_argument('--loss_type', default='l1', type=str)


    args = parser.parse_args()
    print(args)


    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels=3,
        with_time_emb=False,
        residual=True
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        device_of_kernel = 'cuda',
        channels = 3,
        timesteps = args.time_steps,   # number of steps
        loss_type = args.loss_type,    # L1 or L2
        train_routine = args.train_routine,
        sampling_routine = args.sampling_routine,
        discrete=False
    ).cuda()

    diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

    trainer = Trainer(
        diffusion,
        args.data_path,
        image_size = 128,
        train_batch_size = 24,
        train_lr = 2e-5,
        train_num_steps = args.train_steps,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        fp16 = False,                       # turn on mixed precision training with apex
        results_folder = args.save_folder,
        load_path = args.load_path,
        dataset = 'celebA'
    )

    if mode == '1':
        trainer.train()
    elif mode == '2':
        trainer.test_from_data('test', s_times=args.sample_steps)
    elif mode == '3':
        trainer.test_from_line('og-test.png')
    else:
        print("No such mode!")

if __name__ == '__main__':
    print("Pleae input the mode number, 1 for training, 2 for testing:")
    mode = input()
    main(mode)