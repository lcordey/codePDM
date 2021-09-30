import h5py
import math
import numpy as np
import torch
import pickle

from torch._C import dtype

from decoderSDF_rgb import DecoderSDF
from marching_cubes_rgb import *

###### parameter #####

TESTING = False

MODEL_PATH = "models_pth/decoderSDF.pth"
LATENT_VECS_PATH = "models_pth/latent_vecs.pth"

MODEL_PATH_TEST = "models_pth/decoderSDF_TEST.pth"
LATENT_VECS_PATH_TEST = "models_pth/latent_vecs_TEST.pth"

# input_file = "../../data_processing/sdf/sdf.h5"
input_dir = "../../data_processing/sdf/"

latent_size = 16
num_epoch = 100000
batch_size = 10000

eta_decoder = 1e-3
eta_latent_space = 1e-2
gammaLR = 0.99999


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('input', type=str, help='The input HDF5 file.')
    # parser.add_argument('output', type=str, help='Output directory for OFF files.')

    args = parser.parse_args()

    path_input = input_dir + args.input

    if not os.path.exists(path_input):
        print('Input file does not exist.')
        exit(1)

    # load file
    h5f = h5py.File(path_input, 'r')

    # sdf_data = torch.tensor(h5f["tensor"][()], dtype = torch.half)
    sdf_data = torch.tensor(h5f["tensor"][()], dtype = torch.float)

    resolution = sdf_data.shape[1]
    num_samples_per_scene = resolution * resolution * resolution
    num_scenes = len(sdf_data)

    assert(len(sdf_data.shape) == 5), "sdf data shoud have dimension: num_scenes x X_dim x Y_dim x Z_dim x 4 (sdf + r + g + b)"
    assert(sdf_data.shape[1] == sdf_data.shape[2] and sdf_data.shape[2] == sdf_data.shape[3]),"resolution should be the same in every direction"

    #fill tensors
    idx = torch.arange(num_scenes).type(torch.LongTensor).cuda()

    # xyz = torch.empty(num_samples_per_scene, 3,  dtype=torch.half).cuda()
    xyz = torch.empty(num_samples_per_scene, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])


    sdf_gt = np.reshape(sdf_data[:,:,:,:,0], [num_samples_per_scene * num_scenes])
    rgb_gt = np.reshape(sdf_data[:,:,:,:,1:], [num_samples_per_scene * num_scenes, 3])

    sdf_gt = sdf_gt.cuda()
    sdf_gt = sdf_gt /resolution

    rgb_gt = rgb_gt.cuda()

    threshold_precision = 1
    threshold_precision = threshold_precision/resolution


    # initialize random latent code for every shape

    # lat_vecs = torch.nn.Embedding(num_scenes, latent_size, dtype = torch.half).cuda()
    lat_vecs = torch.nn.Embedding(num_scenes, latent_size).cuda()
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        1.0 / math.sqrt(latent_size),
    )

    # decoder
    decoder = DecoderSDF(latent_size).cuda()


    loss = torch.nn.MSELoss

    #optimizer
    optimizer = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": eta_decoder,
                "eps": 1e-8,
            },
            {
                "params": lat_vecs.parameters(),
                "lr": eta_latent_space,
                "eps": 1e-8,
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gammaLR)



    ####################### Training loop ##########################
    decoder.train()
    # decoder.half()

    log_loss = []
    log_loss_sdf = []
    log_loss_rgb = []
    log_loss_reg = []

    for epoch in range(num_epoch):
        optimizer.zero_grad()

        #get random scene and samples
        batch_sample_idx = np.random.randint(num_samples_per_scene, size = batch_size)
        batch_scenes_idx = np.random.randint(num_scenes, size = batch_size)

        sdf_pred = decoder(lat_vecs(idx[batch_scenes_idx]), xyz[batch_sample_idx])

        # assign weight of 0 for easy samples that are well trained
        weight_sdf = ~((sdf_pred[:,0] > threshold_precision).squeeze() * (sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx] > threshold_precision).squeeze()) \
            * ~((sdf_pred[:,0] < -threshold_precision).squeeze() * (sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx] < -threshold_precision).squeeze())

        
        #L1 loss, only for hard samples
        loss_sdf = loss(reduction='none')(sdf_pred[:,0].squeeze(), sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx])
        loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()

        # loss rgb
        lambda_rgb = 1/100
        
        rgb_gt_normalized = rgb_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx,:]/255
        loss_rgb = loss(reduction='none')(sdf_pred[:,1:], rgb_gt_normalized)
        loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean() * weight_sdf.numel()/weight_sdf.count_nonzero() * lambda_rgb
        

        #regularization loss
        lambda_reg_std = 1/100
        lambda_reg_mean = 1/100
        loss_reg_std = lambda_reg_std * abs(1.0 / math.sqrt(latent_size) - (lat_vecs.weight).std())
        loss_reg_mean = lambda_reg_mean * abs((lat_vecs.weight).mean())
        loss_reg = loss_reg_mean + loss_reg_std


        loss_pred = loss_sdf + loss_rgb + loss_reg_std + loss_reg_mean

        #log
        log_loss.append(loss_pred.detach().cpu())
        log_loss_sdf.append(loss_sdf.detach().cpu())
        log_loss_rgb.append(loss_rgb.detach().cpu())
        log_loss_reg.append(loss_reg.detach().cpu())

        #update weights
        loss_pred.backward()
        optimizer.step()
        scheduler.step()

        print("After {} epoch,  loss sdf: {:.5f}, loss rgb: {:.5f}, loss reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, lr: {:f}, lat_vec std/mean: {:.2f}/{:.2f}".format(\
            epoch, torch.Tensor(log_loss_sdf[-10:]).mean(), torch.Tensor(log_loss_rgb[-10:]).mean(), torch.Tensor(log_loss_reg[-10:]).mean(), sdf_pred[:,0].min() * resolution, \
            sdf_pred[:,0].max() * resolution, sdf_pred[:,1:].min() * 255, sdf_pred[:,1:].max() * 255, optimizer.param_groups[0]['lr'], (lat_vecs.weight).std(), (lat_vecs.weight).mean()))

    #save model
    if (TESTING == True):
        torch.save(decoder, MODEL_PATH_TEST)
        torch.save(lat_vecs, LATENT_VECS_PATH_TEST)
    else:
        torch.save(decoder, MODEL_PATH)
        torch.save(lat_vecs, LATENT_VECS_PATH)


    print("final loss sdf: {:f}".format(torch.Tensor(log_loss_sdf[-100:]).mean()))
    print("final loss rgb: {:f}".format(torch.Tensor(log_loss_rgb[-100:]).mean()))


    #save sdf results
    sdf_output = np.empty([num_scenes , resolution, resolution, resolution, 4], dtype = np.float16)

    decoder.eval()
    for i in range(num_scenes):
        
        # free variable for memory space
        try:
            del sdf_pred
        except:
            print("sdf_pred wasn't defined")

        sdf_result = np.empty([resolution, resolution, resolution, 4], dtype = np.float16)
        for x in range(resolution):
            sdf_pred = decoder(lat_vecs(idx[i].repeat(resolution * resolution)),xyz[x * resolution * resolution: (x+1) * resolution * resolution])

            sdf_pred[:,0] = sdf_pred[:,0] * resolution
            sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
            sdf_pred[:,1:] = sdf_pred[:,1:] * 255
            
            sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].detach().cpu(), [resolution, resolution, 4])

        # sdf_pred = decoder(lat_vecs(idx[i].repeat(num_samples_per_scene)),xyz)

        # sdf_pred[:,0] = sdf_pred[:,0] * resolution
        # sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
        # sdf_pred[:,1:] = sdf_pred[:,1:] * 255


        # sdf_result = np.empty([resolution, resolution, resolution, 4], dtype = np.float16)
        # sdf_result[:,:,:,:] = np.reshape(sdf_pred[:,:].type(torch.float16).detach().cpu(), [resolution, resolution, resolution, 4])

        sdf_output[i] = sdf_result

        print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
        if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
            vertices, faces = marching_cubes(sdf_result[:,:,:,0])
            colors_v = exctract_colors_v(vertices, sdf_result)
            colors_f = exctract_colors_f(colors_v, faces)
            off_file = '../../data_processing/output_prediction/%d.off' % i
            write_off(off_file, vertices, faces, colors_f)
            print('Wrote %s.' % off_file)
        else:
            print("surface level: 0, should be comprise in between the minimum and maximum value")


    #save sdf
    with h5py.File('../../data_processing/sdf/sdf_output_half.h5', 'w') as f:
        dset = f.create_dataset("tensor", data = sdf_output)


    #save logs plot
    avrg_loss = []
    avrg_loss_sdf = []
    avrg_loss_rgb = []
    for i in range(0,len(log_loss)):
        avrg_loss.append(torch.Tensor(log_loss[i-20:i]).mean())
        avrg_loss_sdf.append(torch.Tensor(log_loss_sdf[i-20:i]).mean())
        avrg_loss_rgb.append(torch.Tensor(log_loss_rgb[i-20:i]).mean())
        

    from matplotlib import pyplot as plt
    plt.figure()
    plt.title("Total loss")
    plt.semilogy(avrg_loss[:])
    plt.savefig("../../data_processing/logs/log_total")
    plt.figure()
    plt.title("SDF loss")
    plt.semilogy(avrg_loss_sdf[:])
    plt.savefig("../../data_processing/logs/log_sdf")
    plt.figure()
    plt.title("RGB loss")
    plt.semilogy(avrg_loss_rgb[:])
    plt.savefig("../../data_processing/logs/log_rgb")

    with open("../../data_processing/logs/log.txt", "wb") as fp:
        pickle.dump(avrg_loss, fp)
