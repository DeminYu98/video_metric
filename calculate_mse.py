import numpy as np
import torch
from tqdm import tqdm
import math

def img_mse(img1, img2):
    # [0,1]
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # # compute mse
    # if mse < 1e-10:
    #     return 100
    # mse = 20 * math.log10(1 / math.sqrt(mse))
    return mse

def trans(x):
    return x

def calculate_mse(videos1, videos2, calculate_final=True, calculate_per_frame=0):
    # print("calculate_mse...")

    # videos [batch_size, timestamps, channel, h, w]
    b,t,c,h,w = videos1.shape
    
    assert videos1.shape == videos2.shape

    videos1 = trans(videos1)
    videos2 = trans(videos2)
    
    mse_results = []
    
    for video_num in range(videos1.shape[0]):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]

        mse_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].numpy()
            img2 = video2[clip_timestamp].numpy()
            
            # calculate mse of a video
            mse_results_of_a_video.append(img_mse(img1, img2))

        mse_results.append(mse_results_of_a_video)
    
    mse_results = np.array(mse_results)
    
    mse = {}
    mse_std = {}

    if calculate_per_frame > 0:
        for clip_timestamp in range(calculate_per_frame, len(video1)+1, calculate_per_frame):
            mse[f'avg[:{clip_timestamp}]'] = np.mean(mse_results[:,:clip_timestamp]) * (h*w)
            mse_std[f'std[:{clip_timestamp}]'] = np.std(mse_results[:,:clip_timestamp])

    # multi (h*w) to calaulate image mse
    if calculate_final:
        mse['final'] = np.mean(mse_results)*(h*w)
        mse_std['final'] = np.std(mse_results)
    
    result = {
        "mse": mse,
        "mse_std": mse_std,
        "mse_per_frame": calculate_per_frame,
        "mse_video_setting": video1.shape,
        "mse_video_setting_name": "time, channel, heigth, width",
    }

    return result

# test code / using example

def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    CALCULATE_PER_FRAME = 5
    CALCULATE_FINAL = True
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    videos2 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
    device = torch.device("cuda")

    import json
    result = calculate_mse(videos1, videos2)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()