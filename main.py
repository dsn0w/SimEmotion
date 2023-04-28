import os
# import CLIP.clip as clip  # choose by your situation
import clip
import torch
import random
from PIL import ImageFile, Image
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
import copy
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InputException(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        print('Got a wrong input for ' + self.name + '!')


def datasetInit(datasetname):
    if datasetname == 'FI_8':
        _classtype = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        _data_dir = '/data/dataset/FI/'
        _catepath = '/data/dataset/cate info/FI/'
    elif datasetname == 'EmotionROI_6':
        _classtype = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        _data_dir = '/data/dataset/Emotion6/'
        _catepath = '/data/dataset/cate info/EmotionROI/'
    elif datasetname == 'FI_2':
        _classtype = ['negative', 'positive']
        _data_dir = '/data/dataset/FI2/'
        _catepath = '/data/dataset/cate info/FI/'
    elif datasetname == 'EmotionROI_2':
        _classtype = ['negative', 'positive']
        _data_dir = '/data/dataset/Emotion6_2/'
        _catepath = '/data/dataset/cate info/EmotionROI/'
    elif datasetname == 'TwitterI':
        _classtype = ['negative', 'positive']
        # _data_dir = '/data/dataset/TwitterI/total/' + str(ind + 1) + '/'  # it can be loaded by a loop
        _data_dir = '/data/dataset/TwitterI/total/4/'  # take split No."4" as an example
        _catepath = '/data/dataset/cate info/TwitterI/'
    elif datasetname == 'TwitterII':
        _classtype = ['negative', 'positive']
        _data_dir = '/data/dataset/TwitterII/'
        _catepath = '/data/dataset/cate info/TwitterII/'
    else:
        raise InputException

    _catedir = {}
    _catedirnlp = {}
    for file in os.listdir(_catepath):
        with open(_catepath + file, 'r') as catefile:
            cateinfo = catefile.readline().strip(' ')
            idx = '_' + file.strip('.txt') + '.jpg'
            _catedir[file] = cateinfo
            wordlist = cateinfo.split(' ')
            wordinfo = ''
            for order in range(len(wordlist)):
                if order == len(wordlist) - 1:
                    wordinfo = wordinfo + 'and ' + wordlist[order]
                else:
                    wordinfo = wordinfo + wordlist[order] + ', '
            _catedirnlp[file] = wordinfo
    return _classtype, _data_dir, _catedir, _catedirnlp


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    os.environ['PYTHONHASHSEED'] = str(seed)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class Loaddata(torch.utils.data.Dataset):
    def __init__(self, setname):
        self.images = []
        path = data_dir
        settype = []
        settype.append(setname)
        if args.datasetname == 'FI_8':
            self.classindex = {'amusement': 0, 'anger': 1, 'awe': 2, 'contentment': 3, 'disgust': 4, 'excitement': 5,
                               'fear': 6, 'sadness': 7}
            self.subclassindex = ['ecstasy, joy, or serenity',
                                  'rage, anger, or annoyance',
                                  'amazement, surprise, or distraction',
                                  'admiration, trust, or acceptance',
                                  'loathing, disgust, or boredom',
                                  'vigilance, anticipation, or interest',
                                  'terror, fear, or apprehension',
                                  'grief, sadness, or pensiveness']
        elif args.datasetname == 'EmotionROI_6':
            self.classindex = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5}
            self.subclassindex = ['rage, anger, or annoyance',
                                  'loathing, disgust, or boredom',
                                  'terror, fear, or apprehension',
                                  'ecstasy, joy, or serenity',
                                  'grief, sadness, or pensiveness',
                                  'amazement, surprise, or distraction']
        else:
            self.classindex = {'negative': 0, 'positive': 1}
            self.subclassindex = ['negative, passive, or feminine',
                                  'positive, active, or masculine']

        for s in settype:
            for ct in classtype:
                filepath = path + s + '/' + ct + '/'
                file_list = os.listdir(filepath)
                for filename in file_list:
                    img_path = filepath + filename
                    img_set = s
                    img_label = ct
                    catepath = filename.strip(ct + '_').strip('jpg') + 'txt'
                    if catepath in catedir:
                        img_cate = 'a photo contains ' + catedirnlp[
                            catepath] + ', and it seems to express some feelings like ' + self.subclassindex[
                                       self.classindex[img_label]]
                    else:
                        img_cate = 'a photo seems to express some feelings like ' + self.subclassindex[
                            self.classindex[img_label]]
                    self.images.append((img_path, img_set, img_label, img_cate))

    def __getitem__(self, item):
        image, imgset, label, cateinfo = self.images[item]
        text = clip.tokenize(cateinfo)[0]
        img = preprocess(Image.open(image)).to(device)
        labelindex = self.classindex[label]
        return img, text, labelindex

    def __len__(self):
        return len(self.images)


def get_features(traindataset, valdataset):
    topacc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    best_fc = copy.deepcopy(fc.state_dict())
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        running_lossCE = 0.0
        running_lossTI = 0.0
        running_corrects = 0

        for images, text, labels in tqdm(DataLoader(traindataset, batch_size=args.batch_size, shuffle=True)):
            images = images.to(device)
            text = text.to(device)
            labels = labels.to(device)
            if epoch >= args.warmepoch:
                optimizer.zero_grad()
            optimizer_fc.zero_grad()
            with torch.set_grad_enabled(True):
                features = model.encode_image(images).float()
                textout = model.encode_text(text).float()
                outputs = fc(features)

                features = features / features.norm(dim=-1, keepdim=True)
                textout = textout / textout.norm(dim=-1, keepdim=True)

                losstextimg = torch.mean(1 - torch.cosine_similarity(features, textout, dim=-1))
                lossCE = criterion(outputs, labels)
                loss = (1 - args.alpha) * lossCE + args.alpha * losstextimg
                confidences, preds = torch.max(outputs, 1)
                loss.backward()
                if epoch >= args.warmepoch:
                    convert_models_to_fp32(model.visual)
                    optimizer.step()
                    clip.model.convert_weights(model.visual)
                optimizer_fc.step()
            running_loss += loss.item() * features.size(0)
            running_lossCE += lossCE.item() * features.size(0)
            running_lossTI += losstextimg.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_losstrain = running_loss / len(traindataset)
        epoch_losstrainCE = running_lossCE / len(traindataset)
        epoch_losstrainTI = running_lossTI / len(traindataset)
        epoch_acctrain = running_corrects.double() / len(traindataset)
        loginfo = 'Train Epoch:{} Loss: {:.4f} LossCE: {:.4f} lossTI: {:.4f} Acc: {:.4f}'.format(epoch, epoch_losstrain,
                                                                                                 epoch_losstrainCE,
                                                                                                 epoch_losstrainTI,
                                                                                                 epoch_acctrain)
        log.append(loginfo + '\n')
        print(loginfo)

        if epoch >= args.warmepoch:
            scheduler.step()
        scheduler_fc.step()
        running_lossval = 0.0
        running_lossCEval = 0.0
        running_lossTIval = 0.0
        running_correctsval = 0

        for images, text, labels in tqdm(DataLoader(valdataset, batch_size=args.batch_size, shuffle=False)):
            images = images.to(device)
            text = text.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                features = model.encode_image(images).float()
                textout = model.encode_text(text).float()
                outputs = fc(features)
                features = features / features.norm(dim=-1, keepdim=True)
                textout = textout / textout.norm(dim=-1, keepdim=True)
                losstextimg = torch.mean(1 - torch.cosine_similarity(features, textout, dim=-1))
                lossCE = criterion(outputs, labels)
                loss = (1 - args.alpha) * lossCE + args.alpha * losstextimg
                confidences, preds = torch.max(outputs, 1)
            running_lossval += loss.item() * features.size(0)
            running_lossCEval += lossCE.item() * features.size(0)
            running_lossTIval += losstextimg.item() * features.size(0)
            running_correctsval += torch.sum(preds == labels.data)
        epoch_lossval = running_lossval / len(valdataset)
        epoch_lossvalCE = running_lossCEval / len(valdataset)
        epoch_lossvalTI = running_lossTIval / len(valdataset)
        epoch_accval = running_correctsval.double() / len(valdataset)
        if epoch_accval > topacc:
            topacc = epoch_accval
            best_model = copy.deepcopy(model.state_dict())
            best_fc = copy.deepcopy(fc.state_dict())

        loginfo = 'Val Epoch:{} Loss: {:.4f} LossCE: {:.4f} lossTI: {:.4f} Acc: {:.4f}'.format(epoch, epoch_lossval,
                                                                                               epoch_lossvalCE,
                                                                                               epoch_lossvalTI,
                                                                                               epoch_accval)
        log.append(loginfo + '\n')
        topaccinfo = 'topacc: ' + str(topacc) + '\n'
        log.append(topaccinfo)
        print(loginfo)
        print(topaccinfo)
        # get_features_test(testset)
    model.load_state_dict(best_model)
    fc.load_state_dict(best_fc)
    torch.save(model.state_dict(), './savemodel/model' + args.modelname + '_' + args.datasetname + '_' + str(starttime) + '.pth')
    torch.save(fc.state_dict(), './savemodel/fc' + args.modelname + '_' + args.datasetname + '_' + str(starttime) + '.pth')
    get_features_test(testset)


def get_features_test(dataset):
    running_loss = 0.0
    running_corrects = 0
    testResult = [[0 for col in range(len(classtype))] for row in range(len(classtype))]  #
    with torch.no_grad():
        for images, text, labels in tqdm(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
            images = images.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                features = model.encode_image(images).float()
                outputs = fc(features)
                loss = criterion(outputs, labels)
                confidences, preds = torch.max(outputs, 1)
            predTemp = preds.cpu().numpy().tolist()  #
            testTemp = labels.cpu().numpy().tolist()  #
            for index in range(len(testTemp)):  #
                testResult[testTemp[index]][predTemp[index]] += 1  #
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)
        print('Answers in testlist:', testResult)  #
        loginfo = 'Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc)
        log.append(loginfo + '\n')
        log.append(str(testResult))  #
        print(loginfo)

        return round(float(epoch_acc.cpu().detach().numpy()), 6)


class FCN(nn.Module):

    def __init__(self, dim=512):
        super(FCN, self).__init__()
        # self.classifier = nn.Linear(dim, len(classtype))
        self.classifier = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, len(classtype)),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    # load settings
    parser = argparse.ArgumentParser(description='Image Emotion Classification by SimEmotion')
    parser.add_argument('--datasetname', type=str, default='FI_8')
    parser.add_argument('--modelname', type=str, default='ViT-L/14', help='differnet backbone can be chosen: RN50, RN101, ViT-B/16, ViT-B/32, ViT-L/14, ViT-L/14@336px')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--lr_BB', type=float, default=1e-6)
    parser.add_argument('--wd_BB', type=float, default=1e-5)
    parser.add_argument('--stepsize_BB', type=int, default=6)
    parser.add_argument('--gamma_BB', type=float, default=0.5)
    parser.add_argument('--lr_FC', type=float, default=1e-3)
    parser.add_argument('--stepsize_FC', type=int, default=6)
    parser.add_argument('--gamma_FC', type=float, default=0.5)
    parser.add_argument('--warmepoch', type=int, default=3)
    args = parser.parse_args()
    classtype, data_dir, catedir, catedirnlp = datasetInit(args.datasetname)
    args.modelname = args.modelname.replace('/', '-')
    set_seed(args.seed)
    log = []

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.modelname, device)
    clip.model.convert_weights(model)
    if args.modelname == 'RN50':
        fcdim = 1024
    elif args.modelname == 'ViT-L/14' or args.modelname == 'ViT-L/14@336px':
        fcdim = 768
        args.batch_size = 32
    else:
        fcdim = 512
    fc = FCN(dim=fcdim).to(device)
    logit = model.logit_scale

    # load the dataset
    trainset = Loaddata('train')
    warmpos = 0.1 * len(trainset) / args.batch_size
    if args.datasetname == 'FI_8' or args.datasetname == 'FI_2':
        valset = Loaddata('val')
    else:
        valset = Loaddata('test')
    testset = Loaddata('test')

    # load the training setting
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.visual.parameters(), lr=args.lr_BB, weight_decay=args.wd_BB)
    optimizer_fc = optim.Adam(fc.parameters(), lr=args.lr_FC)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize_BB, gamma=args.gamma_BB)
    scheduler_fc = lr_scheduler.StepLR(optimizer_fc, step_size=args.stepsize_FC, gamma=args.gamma_FC)

    # main phase
    print("New experiment start.")
    starttime = time.strftime('%Y%m%d%H%M%S')
    print(starttime)
    startpoint = time.time()
    get_features(trainset, valset)
    acc = 100 * get_features_test(testset)
    endpoint = time.time()
    timecost = endpoint - startpoint
    print('cost: ' + str(timecost) + 's.')

    # log saving
    paraminfo = args.modelname + '_' +args.datasetname + ' lr_FC:' + str(args.lr_FC) + ' stepsize_FC:' \
                + str(args.stepsize_FC) + ' gamma_FC' + str(args.gamma_FC) + ' lr_BB:' + str(args.lr_BB) \
                + ' stepsize_BB:' + str(args.stepsize_BB) + ' gamma_BB' + str(args.gamma_BB)
    print(paraminfo)
    log.append(paraminfo + '\n')
    with open('./log/' + args.modelname + '_' + args.datasetname + str(acc) + '_' + str(starttime) + '.txt', 'a+') as flog:
        for line in log:
            flog.write(str(line))
        print('Log saved successfully.')

    # os.system("shutdown")
