from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from collections import defaultdict

import numpy as np
import scipy.stats as st
import scipy
import scipy.io
import matplotlib.pyplot as plt


import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import torchvision
from torchvision import transforms, utils, datasets, models

torch.set_printoptions(linewidth=120)

labels_name = ['n02085620-Chihuahua/n02085620_10074',
 'n02085782-Japanese_spaniel/n02085782_1039',
 'n02085936-Maltese_dog/n02085936_10073',
 'n02086079-Pekinese/n02086079_10059',
 'n02086240-Shih-Tzu/n02086240_1011',
 'n02086646-Blenheim_spaniel/n02086646_1002',
 'n02086910-papillon/n02086910_10147',
 'n02087046-toy_terrier/n02087046_1004',
 'n02087394-Rhodesian_ridgeback/n02087394_10014',
 'n02088094-Afghan_hound/n02088094_1003',
 'n02088238-basset/n02088238_10005',
 'n02088364-beagle/n02088364_10108',
 'n02088466-bloodhound/n02088466_10083',
 'n02088632-bluetick/n02088632_101',
 'n02089078-black-and-tan_coonhound/n02089078_1021',
 'n02089867-Walker_hound/n02089867_1029',
 'n02089973-English_foxhound/n02089973_1',
 'n02090379-redbone/n02090379_1006',
 'n02090622-borzoi/n02090622_10281',
 'n02090721-Irish_wolfhound/n02090721_1002',
 'n02091032-Italian_greyhound/n02091032_10079',
 'n02091134-whippet/n02091134_10107',
 'n02091244-Ibizan_hound/n02091244_100',
 'n02091467-Norwegian_elkhound/n02091467_1110',
 'n02091635-otterhound/n02091635_1043',
 'n02091831-Saluki/n02091831_10215',
 'n02092002-Scottish_deerhound/n02092002_10060',
 'n02092339-Weimaraner/n02092339_1013',
 'n02093256-Staffordshire_bullterrier/n02093256_10078',
 'n02093428-American_Staffordshire_terrier/n02093428_10164',
 'n02093647-Bedlington_terrier/n02093647_1022',
 'n02093754-Border_terrier/n02093754_1062',
 'n02093859-Kerry_blue_terrier/n02093859_10',
 'n02093991-Irish_terrier/n02093991_1026',
 'n02094114-Norfolk_terrier/n02094114_1020',
 'n02094258-Norwich_terrier/n02094258_1003',
 'n02094433-Yorkshire_terrier/n02094433_10123',
 'n02095314-wire-haired_fox_terrier/n02095314_1033',
 'n02095570-Lakeland_terrier/n02095570_1031',
 'n02095889-Sealyham_terrier/n02095889_10',
 'n02096051-Airedale/n02096051_1017',
 'n02096177-cairn/n02096177_1000',
 'n02096294-Australian_terrier/n02096294_1111',
 'n02096437-Dandie_Dinmont/n02096437_1006',
 'n02096585-Boston_bull/n02096585_10380',
 'n02097047-miniature_schnauzer/n02097047_1028',
 'n02097130-giant_schnauzer/n02097130_1032',
 'n02097209-standard_schnauzer/n02097209_1',
 'n02097298-Scotch_terrier/n02097298_1007',
 'n02097474-Tibetan_terrier/n02097474_1023',
 'n02097658-silky_terrier/n02097658_10020',
 'n02098105-soft-coated_wheaten_terrier/n02098105_100',
 'n02098286-West_Highland_white_terrier/n02098286_1009',
 'n02098413-Lhasa/n02098413_10144',
 'n02099267-flat-coated_retriever/n02099267_1018',
 'n02099429-curly-coated_retriever/n02099429_1039',
 'n02099601-golden_retriever/n02099601_10',
 'n02099712-Labrador_retriever/n02099712_1150',
 'n02099849-Chesapeake_Bay_retriever/n02099849_1024',
 'n02100236-German_short-haired_pointer/n02100236_1054',
 'n02100583-vizsla/n02100583_10249',
 'n02100735-English_setter/n02100735_10030',
 'n02100877-Irish_setter/n02100877_102',
 'n02101006-Gordon_setter/n02101006_1016',
 'n02101388-Brittany_spaniel/n02101388_10017',
 'n02101556-clumber/n02101556_1018',
 'n02102040-English_springer/n02102040_1055',
 'n02102177-Welsh_springer_spaniel/n02102177_1022',
 'n02102318-cocker_spaniel/n02102318_10000',
 'n02102480-Sussex_spaniel/n02102480_101',
 'n02102973-Irish_water_spaniel/n02102973_1037',
 'n02104029-kuvasz/n02104029_1075',
 'n02104365-schipperke/n02104365_10071',
 'n02105056-groenendael/n02105056_1018',
 'n02105162-malinois/n02105162_10076',
 'n02105251-briard/n02105251_12',
 'n02105412-kelpie/n02105412_1031',
 'n02105505-komondor/n02105505_1018',
 'n02105641-Old_English_sheepdog/n02105641_10048',
 'n02105855-Shetland_sheepdog/n02105855_10095',
 'n02106030-collie/n02106030_10021',
 'n02106166-Border_collie/n02106166_1031',
 'n02106382-Bouvier_des_Flandres/n02106382_1000',
 'n02106550-Rottweiler/n02106550_10048',
 'n02106662-German_shepherd/n02106662_10122',
 'n02107142-Doberman/n02107142_10009',
 'n02107312-miniature_pinscher/n02107312_105',
 'n02107574-Greater_Swiss_Mountain_dog/n02107574_1007',
 'n02107683-Bernese_mountain_dog/n02107683_1003',
 'n02107908-Appenzeller/n02107908_1030',
 'n02108000-EntleBucher/n02108000_1011',
 'n02108089-boxer/n02108089_1',
 'n02108422-bull_mastiff/n02108422_1013',
 'n02108551-Tibetan_mastiff/n02108551_10182',
 'n02108915-French_bulldog/n02108915_10204',
 'n02109047-Great_Dane/n02109047_1005',
 'n02109525-Saint_Bernard/n02109525_10032',
 'n02109961-Eskimo_dog/n02109961_10021',
 'n02110063-malamute/n02110063_10025',
 'n02110185-Siberian_husky/n02110185_10047',
 'n02110627-affenpinscher/n02110627_10147',
 'n02110806-basenji/n02110806_1013',
 'n02110958-pug/n02110958_10',
 'n02111129-Leonberg/n02111129_1',
 'n02111277-Newfoundland/n02111277_1008',
 'n02111500-Great_Pyrenees/n02111500_1031',
 'n02111889-Samoyed/n02111889_1',
 'n02112018-Pomeranian/n02112018_10129',
 'n02112137-chow/n02112137_1005',
 'n02112350-keeshond/n02112350_10023',
 'n02112706-Brabancon_griffon/n02112706_1041',
 'n02113023-Pembroke/n02113023_10636',
 'n02113186-Cardigan/n02113186_10077',
 'n02113624-toy_poodle/n02113624_1008',
 'n02113712-miniature_poodle/n02113712_1036',
 'n02113799-standard_poodle/n02113799_1057',
 'n02113978-Mexican_hairless/n02113978_1006',
 'n02115641-dingo/n02115641_10021',
 'n02115913-dhole/n02115913_1010']
#model = torch.load('./torch_model.pt')
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        #print("rescale sample is : ", sample)


        image = sample
        #print(image.size)

        h, w = image.size
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        new_h, new_w = int(self.output_size), int(self.output_size)
        img = image.resize((new_h, new_w))

        #img = transform.resize(image, (new_h, new_w))

        return img

class RealRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        #print("rescale sample is : ", sample)


        image = sample
        #print(image.size)

        h, w = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_h, new_w))

        #img = transform.resize(image, (new_h, new_w))



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.conv3 = nn.Conv2d(8, 16, 5)

        self.fc1 = nn.Linear(in_features =  16*576, out_features = 2000)
        self.fc2 = nn.Linear(in_features = 2000, out_features = 200)
        self.out = nn.Linear(in_features = 200, out_features = 120)


    def forward(self, t):

        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #print(t.shape)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #print(t.shape)

        # (3.5) hidden conv layer
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #print(t.shape)

        # (4) hidden linear layer
        #[64, 8, 53, 53]
        #print("before : ", t.shape)
        t = t.view(-1, 16*576)
        #print("after : ", t.shape)
        t = self.fc1(t)
        t = F.relu(t)
        #print(t.shape)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        #print(t.shape)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        #print(t.shape)

        return t
        #x = F.relu(self.conv1(x))
        #return F.relu(self.conv2(x))
        #return x

img_link = "./test_img.jpg"
pic = Image.open(img_link)

transform_img = torchvision.transforms.Compose([
    Rescale(224),
    transforms.ToTensor()
])
network = torch.load('Final_Model.pt')

def exec_P6(image):
    image_tens = transform_img(image).unsqueeze(0)
    print(image_tens.shape, '\n')
    pred = network(image_tens)
    _, out = torch.max(pred, 1)

    return torch.add(out, 1)

result_tens = exec_P6(pic)
print(result_tens, labels_name[result_tens.item()])
print()
