'''
Author: HelinXu xuhelin1911@gmail.com
Date: 2022-06-15 21:59:33
LastEditTime: 2022-06-23 03:37:47
Description: Inference, save to predicted.json, merge to final_json.json
'''

import torch
import torch.nn as nn
from glob import glob
from os.path import join as pjoin
import json
from PIL import Image

from dataset import TaobaoDataset, tag2label
import torchvision.transforms as transforms
from misc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(num_classes=len(LABEL_1)+1, val=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet161', pretrained=False)
    model.classifier = nn.Linear(2208, num_classes)

    # load model from last model
    with open('./model/last_model.txt', 'r') as f:
        last_model = f.read()
    model.load_state_dict(torch.load('./model/{}'.format(last_model)))
    print('Loaded model from {}'.format(last_model))

    # Send the model to GPU
    model = model.to(device)

    if val:
        val_loader = torch.utils.data.DataLoader(
            dataset=TaobaoDataset(train=False, data_root=DATA_ROOT),
            batch_size=128,
            shuffle=False,
            num_workers=16,
            drop_last=True
        )

        # inference
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0.1
            total = 0.1
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
            print('Accuracy: {:.4f}'.format(correct / total))
    
    # annotate the test set
    transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for file in glob((pjoin(DATA_ROOT, "test/*/profile.json"))):
        with open(file, 'r') as f:
            jf = json.load(f)
            for i, img in enumerate(jf['imgs_tags']):
                for k, _ in img.items():
                    img_path = pjoin(DATA_ROOT, "test", k.split('_')[0], k)
                    img = Image.open(img_path)
                    img = transform(img)
                    output = model(img.unsqueeze(0).to(device))
                    pred = output.data.cpu()
                    
                    # pred: [1, num_classes]
                    best_match_tag = jf['optional_tags'][0]
                    for tag in jf['optional_tags']:
                        label = torch.tensor(tag2label(tag)).unsqueeze(0)
                        if criterion(pred, label) < criterion(pred, torch.tensor(tag2label(best_match_tag)).unsqueeze(0)):
                            best_match_tag = tag

                    jf['imgs_tags'][i][k] = best_match_tag

            # save the annotated json file
            predicted_json = file.replace('profile', 'predicted')
            with open(predicted_json, 'w', encoding='utf8') as f:
                json.dump(jf, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()

    # merge
    final_json = {}

    for file in glob((pjoin(DATA_ROOT, "test/*/predicted.json"))):
        with open(file, 'r') as f:
            jf = json.load(f)
            img_name = file.split('/')[-2]
            final_json[img_name] = jf

    # save final json
    with open(pjoin(DATA_ROOT, "final_json.json"), 'w', encoding='utf8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)