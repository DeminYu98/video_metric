import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
from calculate_mse import calculate_mse

# ps: pixel value should be in [0, 1]!

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 10
CHANNEL = 3
SIZE = 64
CALCULATE_PER_FRAME = 8
CALCULATE_FINAL = True
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')



def videos_metric(real, pred, cond=None):
    # B, T, C, H, W
    # pixel value should be in [0, 1]!
    
    mse = calculate_mse(real, pred)['mse']['final']
    psnr = calculate_psnr(real, pred)['psnr']['final']
    ssim = calculate_ssim(real, pred)['ssim']['final']
    lpips = calculate_lpips(real, pred, device)['lpips']['final']
    if cond is not None:
        pred = torch.cat((cond,pred), dim=1)
        real = torch.cat((cond,real), dim=1)
    fvd = calculate_fvd(real, pred, device)['fvd']['final']

    return mse, psnr, ssim, lpips, fvd

if __name__=='__main__':
    # We can use videos with ones and zeros to test metric code
    # MSE should to be 1 * (H * W) for demo videos (ones and zeros)
    print(videos_metric(videos1,videos2))



