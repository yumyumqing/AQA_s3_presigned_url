# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import os
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_C3DAVG import VideoDataset
import random
import scipy.stats as stats
import torch.optim as optim
import torch.nn as nn
from models.C3DAVG.C3D_altered import C3D_altered
from models.C3DAVG.my_fc6 import my_fc6
from models.C3DAVG.score_regressor import score_regressor
from models.C3DAVG.dive_classifier import dive_classifier
from models.C3DAVG.S2VTModel import S2VTModel
from models.C3D_model import C3D
from opts import *
from utils import utils_1
import numpy as np
import streamlit as st
import subprocess
import os
import cv2 as cv
import tempfile
from torchvision import transforms
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image

torch.manual_seed(randomseed);
torch.cuda.manual_seed_all(randomseed);
random.seed(randomseed);
np.random.seed(randomseed)
torch.backends.cudnn.deterministic = True

def center_crop(img, dim):
      """Returns center cropped image

      Args:Image Scaling
      img: image to be center cropped
      dim: dimensions (width, height) to be cropped from center
      """
      width, height = img.shape[1], img.shape[0]
      #process crop width and height for max available dimension
      crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
      crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
      mid_x, mid_y = int(width/2), int(height/2)
      cw2, ch2 = int(crop_width/2), int(crop_height/2)
      crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
      return crop_img


def action_classifier(frames):
    """
    Main function.
    """
    with torch.no_grad():
        X = torch.zeros((1, 3, 16, 112, 112))
        frames2keep = np.linspace(0, frames.shape[2] - 1, 16, dtype=int)
        ctr = 0
        for i in frames2keep:
            X[:, :, ctr, :, :] = frames[:, :, i, :, :]
            ctr += 1
        # X = np.float32(X)
        print('X shape: ', X.shape)

        # modifying

        # get network pretrained model
        net = C3D()
        net.load_state_dict(torch.load(m5_path))
        # net.cuda()
        net.eval()

        # perform prediction
        X = X*255
        X = torch.flip(X, [1])
        prediction = net(X)
        prediction = prediction.data.cpu().numpy()
        # print('prediction: ', prediction)

        # read labels
        # labels = read_labels_from_file('labels.txt')

        # print top predictions
        top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
        print('\nTop 5:')
        print('Top inds: ', top_inds)
        # for i in top_inds:
        #     print('{:.5f} {}'.format(prediction[0][i], labels[i]))
    return top_inds[0]


def action_scoring(video):
    # loading the altered C3D backbone (ie C3D upto before fc-6)
    model_CNN = C3D_altered()  # .cuda()
    model_CNN.load_state_dict(torch.load(m1_path, map_location={'cuda:0': 'cpu'}))

    # loading our fc6 layer
    model_my_fc6 = my_fc6()  # .cuda()
    model_my_fc6.load_state_dict(torch.load(m2_path, map_location={'cuda:0': 'cpu'}))

    # loading our score regressor
    model_score_regressor = score_regressor()  # .cuda()
    model_score_regressor.load_state_dict(torch.load(m3_path, map_location={'cuda:0': 'cpu'}))
    # print('Using Final Score Loss')

    with torch.no_grad():
        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()

        # for video in frames:

        clip_feats = torch.Tensor([])
        print('len of video: ', len(video))
        print('video shape: ', video.shape)
        for i in np.arange(0, video.shape[2], 16):
            print('i: ', i)
            clip = video[:, :, i:i + 16, :, :]
            # print(f"clip shape: {clip.shape}") # clip shape: torch.Size([1, 3, 16, 112, 112])
            # print(f"clip type: {clip.type()}") # clip type: torch.DoubleTensor
            model_CNN = model_CNN.double()
            clip_feats_temp = model_CNN(clip)

            # print(f"clip_feats_temp shape: {clip_feats_temp.shape}")
            # clip_feats_temp shape: torch.Size([9, 8192])

            clip_feats_temp.unsqueeze_(0)

            # print(f"clip_feats_temp unsqueeze shape: {clip_feats_temp.shape}")
            # clip_feats_temp unsqueeze shape: torch.Size([1, 9, 8192])

            clip_feats_temp.transpose_(0, 1)

            # print(f"clip_feats_temp transposes shape: {clip_feats_temp.shape}")
            # clip_feats_temp transposes shape: torch.Size([9, 1, 8192])

            clip_feats = torch.cat((clip_feats, clip_feats_temp), 1)

            # print(f"clip_feats shape: {clip_feats.shape}")
            # clip_feats shape: torch.Size([9, 1, 8192])

        clip_feats_avg = clip_feats.mean(1)

        # print(f"clip_feats_avg shape: {clip_feats_avg.shape}") # clip_feats_avg shape: torch.Size([9, 8192])
        # clip_feats_avg shape: torch.Size([9, 8192])

        model_my_fc6 = model_my_fc6.double()
        sample_feats_fc6 = model_my_fc6(clip_feats_avg)
        model_score_regressor = model_score_regressor.double()
        temp_final_score = model_score_regressor(sample_feats_fc6) * 17
        # pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])
        pred_scores = [element[0] for element in temp_final_score.data.cpu().numpy()]
        # print('pred shape: ', len(pred_scores))

    return pred_scores


def dataloading(vf):
    transform = transforms.Compose([  # transforms.CenterCrop(H),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # https: // discuss.streamlit.io / t / how - to - access - uploaded - video - in -streamlit - by - open - cv / 5831 / 8
    # frames = None
    frames = None
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # print('shape of frame: ', frame.shape)
        frame = cv.resize(frame, input_resize, interpolation=cv.INTER_LINEAR)  # frame resized: (128, 171, 3)
        # frame = Image.fromarray(frame)
        frame = center_crop(frame, (H, H))
        frame = transform(frame).unsqueeze(0)
        if frames is not None:
            frames = np.vstack((frames, frame))
            # frames = frame
        else:
            frames = frame
    print('frames shape: ', frames.shape)

    vf.release()
    cv.destroyAllWindows()
    rem = len(frames) % 16
    # rem = 16 - rem

    if rem != 0:
        # padding routine
        # padding = np.zeros((rem, C, H, H))
        # # print(padding.shape)
        # frames = np.vstack((frames, padding))
        # removing routine
        frames = frames[:-rem, :, :, :]

    # just considering 96 frames
    # frames = frames[-96:]

    # frames = np.expand_dims(frames, axis=0)
    # frames = frames.unsqueeze(0)
    print(f"frames shape: {frames.shape}")
    # frames shape: (137, 3, 112, 112)

    # frames = DataLoader(frames, batch_size=test_batch_size, shuffle=False)
    video = torch.from_numpy(frames).unsqueeze(0)

    # print(f"video shape: {video.shape}") # video shape: torch.Size([1, 144, 3, 112, 112])
    video = video.transpose_(1, 2)
    video = video.double()
    return video


def main():
    st.title("Olympics diving")
    video_file = st.file_uploader("Upload a video")
    print('video file: ', video_file)

    pred_scores = []; true_scores = []

    mat = scipy.io.loadmat('E:\PyCharmProjects_E\MTL-AQA\AQA-7\AQA-7\Split_4\split_4_train_list.mat')
    data = mat['consolidated_train_list']

    # if 1 == 1: #video_file is not None:
    for sample in range(3):
        print('processing sample #: ', sample)
        if data[sample][0] == 1:
            true_scores.append(data[sample][2])
            # tfile = tempfile.NamedTemporaryFile(delete=False)
            # tfile.write(video_file.read())

            # vf = cv.VideoCapture(tfile.name)
            # vf = cv.VideoCapture('E:/Downloads/david_b_oly_2016.mp4')
            vf = cv.VideoCapture(
                'E:/PyCharmProjects_E/MTL-AQA/AQA-7/AQA-7/Actions/diving/' + format(int(data[sample][1]),
                                                                                    '03d') + '.avi')

            video = dataloading(vf)
            action_class = action_classifier(video)
            if action_class == 463:
                pred_scores.extend(action_scoring(video))
            else:
                print('Non-diving action class detected!')
                pred_scores = -1

    print('Predicted scores: ', pred_scores)
    print('True scores: ', true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    print('Correlation: ', rho)
    plt.plot(pred_scores)
    plt.plot(true_scores)
    plt.show()


if __name__ == '__main__':
    main()