import os
import torch
import torch.optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime


from model.ResNet_models import Generator, FCDiscriminator
from data import get_loader, SalObjDataset, label_perturbation, adversarial_to_uncertainty
from utils import AvgMeter, setup
from params import parser
from loss import uncertainty_aware_structure_loss, make_confidence_label, structure_loss
import visualisation


def train():
    # Load the arguments
    args = parser.parse_args()
    # print(args)
    setup(args.seed)

    torch.autograd.set_detect_anomaly(True)

    # Set up the distributed data parallel
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    n_gpu = torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()

    torch.distributed.barrier()

    # Load models
    COD_Net = Generator(channel=args.gen_reduced_channel).to(device)
    COD_Net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(COD_Net)
    COD_Net = DistributedDataParallel(COD_Net, device_ids=[args.local_rank], find_unused_parameters=True)
    COD_Net_params = COD_Net.parameters()
    COD_Net_optimiser = torch.optim.Adam(COD_Net_params, args.lr_gen, betas=[args.beta_gen, 0.999])

    OCE_Net = FCDiscriminator().to(device)
    OCE_Net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(OCE_Net)
    OCE_Net = DistributedDataParallel(OCE_Net, device_ids=[args.local_rank], find_unused_parameters=True)
    OCE_Net_params = OCE_Net.parameters()
    OCE_Net_optimiser = torch.optim.Adam(OCE_Net_params, args.lr_dis, betas=[args.beta_dis, 0.999])

    torch.distributed.barrier()

    # Set up the dataset and the dataloader
    train_dataset = SalObjDataset(args.train_image_root, args.train_gt_root, trainsize=args.trainsize)
    train_sampler = DistributedSampler(train_dataset, num_replicas=n_gpu, rank=args.local_rank)
    train_loader = get_loader(args.train_image_root, args.train_gt_root, batchsize=args.batchsize, trainsize=args.trainsize, sampler=train_sampler)
    train_step = len(train_loader)

    # Set up the learning rate scheduler
    COD_Net_scheduler = lr_scheduler.StepLR(OCE_Net_optimiser, step_size=20, gamma=0.1)
    # dis_scheduler = lr_scheduler.ExponentialLR(discriminator_optimiser, gamma=0.9)

    # Set up the loss function
    CE = torch.nn.BCELoss()

    # Set up the multi-scale training
    size_rates = [0.75, 1, 1.25]

    # commence training
    print("Ikou!")

    for epoch in range(1, (args.epoch + 1)):
        train_sampler.set_epoch(epoch)
        COD_Net_scheduler.step()
        COD_Net.train()
        OCE_Net.train()

        loss_record_COD = AvgMeter()
        loss_record_OCE = AvgMeter()

        print('Generator learning rate: {}\nDiscriminator learning rate: {}'.format\
              (COD_Net_optimiser.param_groups[0]['lr'], \
               OCE_Net_optimiser.param_groups[0]['lr']))

        for i, pack in enumerate(train_loader, start=1):
            torch.distributed.barrier()

            for rate in size_rates:
                COD_Net_optimiser.zero_grad()
                OCE_Net_optimiser.zero_grad()

                # Get the images and gts from the batch
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()

                # Format the size of images and gts
                trainsize = int(round(args.trainsize * rate / 32) * 32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                # Obtain initial and reference predictions from the generator
                init_pred, ref_pred = COD_Net(images)

                # Pass the generator predictions and ground truth through discriminator
                post_init = torch.sigmoid(init_pred.detach())
                post_ref = torch.sigmoid(ref_pred.detach())

                confi_init_label = make_confidence_label(gts=gts, pred=post_init)
                confi_ref_label = make_confidence_label(gts=gts, pred=post_ref)

                # Concatenate image with prediction as input for confidence estimation network
                post_init = torch.cat((post_init, images), dim=1)
                post_ref = torch.cat((post_ref, images), dim=1)

                # Predict the confidence map
                confi_init_pred = torch.sigmoid(OCE_Net.forward(post_init))
                confi_ref_pred = torch.sigmoid(OCE_Net.forward(post_ref))

                # Compute cross-entropy loss for the Online Confidence Estimation Network
                confi_loss_pred_init = CE(confi_init_pred, confi_init_label)
                confi_loss_pred_ref = CE(confi_ref_pred, confi_ref_label)
                confi_loss = 0.5 * (confi_loss_pred_init + confi_loss_pred_ref)

                # Backpropagate the loss through OCENet
                confi_loss.backward()
                OCE_Net_optimiser.step()

                torch.distributed.barrier()

                # Compute structure loss for the COD-Net
                struct_loss1 = structure_loss(pred=init_pred, mask=gts)
                struct_loss2 = structure_loss(pred=ref_pred, mask=gts)
                COD_loss = 0.5 * (struct_loss1 + struct_loss2)

                # Backpropagate loss through the COD-Net
                COD_loss.backward()
                COD_Net_optimiser.step()

                torch.distributed.barrier()

                # Update the loss record
                if rate == 1:
                    loss_record_COD.update(COD_loss.data, args.batchsize)
                    loss_record_OCE.update(confi_loss.data, args.batchsize)

                torch.distributed.barrier()

            if i % 10 == 0 or i == train_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}, Confi Loss: {:.4f}'.
                       format(datetime.now(), epoch, args.epoch, i, train_step, loss_record_COD.show(), loss_record_OCE.show()))

            torch.distributed.barrier()

        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)
        if epoch > 0 & epoch % 10 == 0:
            if torch.distributed.get_rank() == 0:
                torch.save(COD_Net.state_dict(), args.model_save_path + args.experiment_name + '/' + 'COD_Model' + '_%d' % epoch + '.pth')
                torch.save(OCE_Net.state_dict(), args.model_save_path + args.experiment_name + '/' + 'OCE_Model' + '_%d' % epoch + '.pth')
                print("Successfully save the trained models to {}".format(args.model_save_path))

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    train()
