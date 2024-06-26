"""
In this file we finetune the ImageBind model on the UTD-MHAD dataset using the linear probing method 
(training a linear layer ontop of the embeddings).
"""

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from linked_dataset import RGB_IMU_Dataset
import os

import wandb
import torchvision.transforms as transforms
import torch.nn as nn
import argparse
args = argparse.ArgumentParser()

actions_dict = {
    1: 'A person swipes their hand to the left.',
    2: 'A person swipes their hand to the right.',
    3: 'A person waves.',
    4: 'A person claps.',
    5: 'A person throws an object.',
    6: 'A person crosses their arms.',
    7: 'A person shoots a basketball.',
    8: 'A person draws an X.',
    9: 'A person draws a circle clockwise.',
    10: 'A person draws a circle counter clockwise.',
    11: 'A person draws a triangle.',
    12: 'A person is bowling.',
    13: 'A person is boxing.',
    14: 'A person swings a baseball bat.',
    15: 'A person swings a tennis racket.',
    16: 'A person does an arm curl.',
    17: 'A person serves in tennis.',
    18: 'A person pushes an object.',
    19: 'A person knocks on a door.',
    20: 'A person catches an object.',
    21: 'A person picks up and throws an object.',
    22: 'A person is jogging.',
    23: 'A person is walking.',
    24: 'A person moves from sitting to standing.',
    25: 'A person moves from standing to sitting.',
    26: 'A person does a lunge.',
    27: 'A person does a squat.'
}

def load_and_transform_imu_data(imu_data, device):
    # currently imu is [batch_size,180,6] but accroding to https://github.com/facebookresearch/ImageBind/issues/66 
        # it should be [batch_size, 6, 2000]
    imu_data = torch.tensor(imu_data).to(device)
    imu_data = imu_data.permute(0,2,1)

    target_imu = torch.zeros(imu_data.shape[0],6,2000)
    target_imu[:,:,:imu_data.shape[2]] = imu_data
    target_imu = target_imu.to(device)
    
    return target_imu


def zero_shot_imagebind(val_loader, model, sensors, device):
    model.eval()

    text_list=list(actions_dict.values())
    predictions = []
    correct = 0
    # for rgb_path, imu_path, class_idx, pid_idx in val_dataset.videos:
    for frames, accel_data, class_idx, pid_idx, rgb_path, imu_path  in val_loader:

        # print("RGB Path: ", rgb_path, class_idx)
        # Load and transform video data
        text = data.load_and_transform_text(text_list, device)
        video = data.load_and_transform_video_data(rgb_path, device)
        imu = load_and_transform_imu_data(accel_data, device)
        inputs = {
            ModalityType.TEXT: text,
            ModalityType.VISION: video,
            ModalityType.IMU: imu
        }

        # print("Computing predictions... for ", v)
        with torch.no_grad():
            emb = model(inputs)
            # sftmx = torch.softmax(emb[ModalityType.VISION] @ emb[ModalityType.TEXT].T, dim=-1)
            sftmx = torch.softmax(emb[ModalityType.IMU] @ emb[ModalityType.TEXT].T, dim=-1)
        # print(sftmx)
        pred = torch.argmax(sftmx, dim=-1)
        predictions.append(pred.item())
        # print(predictions)
        if pred.item() != class_idx:
            # print(f"Prediction: {pred.item()} | Label: {class_idx}")
            pass
        else:
            correct += 1


    print("Computing accuracy...")
    # print("Vision x Text: ", sftmx)

    accuracy = correct / len(val_dataset) * 100
    print("Accuracy: ", accuracy)

def train_linear(train_loader, val_loader, model, model_linear, sensors, device):
    # Train the model
    model.eval()
    model_linear.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_linear.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Add scheduler
    num_epochs = 100
    epoch=0

    for epoch in range(num_epochs):

        for i,(frames, accel_data, labels, pid_idx, rgb_path, imu_path)  in enumerate(train_loader):

            # Load and transform video data
            with torch.no_grad():
                if sensors =="both":
                    inputs = {
                        ModalityType.VISION: data.load_and_transform_video_data(rgb_path, device),
                        ModalityType.IMU: load_and_transform_imu_data(accel_data, device)
                    }
                    emb = model(inputs)
                    embedding_vector = (emb[ModalityType.VISION]+emb[ModalityType.IMU])/2
                elif sensors == "vision":
                    inputs = {
                        ModalityType.VISION: data.load_and_transform_video_data(rgb_path, device),
                    }
                    emb = model(inputs)
                    embedding_vector = emb[ModalityType.VISION]

                elif sensors == "imu":
                    inputs = {
                        ModalityType.IMU: load_and_transform_imu_data(accel_data, device)
                    }
                    emb = model(inputs)
                    embedding_vector = emb[ModalityType.IMU]

            labels = labels.to(device)
            outputs = model_linear(embedding_vector)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                wandb.log({"Train Loss": loss.item()})
        
        scheduler.step()  # Step the scheduler

        if (epoch+1) % 2 == 0:
            acc = evaluate(model, model_linear, sensors, val_loader, device)
            print(f'Epoch [{epoch+1}/{num_epochs}],  Val Acc: {acc:.4f}')
            wandb.log({"Val Accuracy": acc})

            #save the model 
            torch.save(model_linear.state_dict(), f'model_linear_{sensors}.ckpt')

    print('Finished Training')

def evaluate(model, model_linear, sensors, val_loader, device):
    model.eval()
    model_linear.eval()
    correct = 0
    total = 0
    for i,(frames, accel_data, labels, pid_idx, rgb_path, imu_path)  in enumerate(val_loader):
        with torch.no_grad():
            if sensors =="both":
                inputs = {
                    ModalityType.VISION: data.load_and_transform_video_data(rgb_path, device),
                    ModalityType.IMU: load_and_transform_imu_data(accel_data, device)
                }
                emb = model(inputs)
                embedding_vector = (emb[ModalityType.VISION]+emb[ModalityType.IMU])/2
            elif sensors == "vision":
                inputs = {
                    ModalityType.VISION: data.load_and_transform_video_data(rgb_path, device),
                }
                emb = model(inputs)
                embedding_vector = emb[ModalityType.VISION]
            elif sensors == "imu":
                inputs = {
                    ModalityType.IMU: load_and_transform_imu_data(accel_data, device)
                }
                emb = model(inputs)
                embedding_vector = emb[ModalityType.IMU]

            out = model_linear(embedding_vector)
            pred = out.cpu().argmax(dim=-1).type(torch.int)
            labels = labels.cpu()
            total += labels.size(0)
            correct += (pred == labels).sum()
     
    return 100.* correct/total


if __name__ == "__main__":    
    # video_paths=["/media/abhi/Seagate-FireCUDA/utd-mhad/RGB/a27_s4_t3_color.avi"]

    args.add_argument("--sensors", type=str, default="vision", help="Choose between vision, imu or both")
    args.add_argument("--zero_shot", type=bool, default=False, help="Choose between vision, imu or both")
    args.add_argument("--batch_size", type=int, default=4, help="Choose between vision, imu or both")
    args = args.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.to(device)
    model.eval()

    print("Loading test data...")
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize frames
        transforms.ToTensor(),           # Convert frames to tensors
    ])
    rgb_video_length = 30
    datapath = "Both_splits/both_45_45_10_#1"
    base_path = "/home/akamboj2/data/utd-mhad/"
    train_dir = os.path.join("/home/akamboj2/data/utd-mhad/",datapath,"train.txt")
    val_dir = os.path.join("/home/akamboj2/data/utd-mhad/",datapath,"val.txt")
    train_dataset = RGB_IMU_Dataset(train_dir, video_length=rgb_video_length, transform=transforms, base_path=base_path, return_path=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = RGB_IMU_Dataset(val_dir, video_length=rgb_video_length, transform=transforms, base_path=base_path, return_path=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    if args.zero_shot:
        print("running zero shot ")
        zero_shot_imagebind(val_loader, model, args.sensors, device)
    else:
        print("finetuning a linear layer:")
        wandb.init(project="imagebind-finetune")

        # Pytorch 2 layer MLP
        model_linear = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 27)
        ) # don't need another relu, bc softmax (sigmoid activation) is applied in the loss function
        model_linear.to(device)

        train_linear(train_loader, val_loader, model, model_linear, args.sensors, device)


   