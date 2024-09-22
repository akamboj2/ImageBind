"""
In this file we finetune the ImageBind model on the UTD-MHAD dataset using the linear probing method 
(training a linear layer ontop of the embeddings).
"""

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from datasets.dataset import RGB_IMU_Dataset, load_dataloaders
# from dataset_mmact import MMACT
# from dataset_mhad import CZUMHADDataset

import os
import sys
import wandb
import torchvision.transforms as transforms
import torch.nn as nn
import argparse

import torch.distributed as dist

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

    target_imu = torch.zeros(imu_data.shape[0],6,2000) # could also try upsampling instead of concat with zeros.
    target_imu[:,:,:imu_data.shape[2]] = imu_data[:,:6,:] #if the imu data has more than 6 channels just use the first 6
    # it's not my fault imagebind is rigid and not adaptable to more imu channels -> i shouldn't have to spend time trying to extend it's funcitonality
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

    accuracy = correct / len(val_loader) * 100
    print("Accuracy: ", accuracy)


class ImageBind_baseline(nn.Module):
    def __init__(self, output_size):
        super(ImageBind_baseline, self).__init__()
        # Instantiate model
        self.model_ib = imagebind_model.imagebind_huge(pretrained=True)
        self.model_linear = nn.Sequential(# Pytorch 2 layer MLP
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        ) # don't need another relu, bc softmax (sigmoid activation) is applied in the loss function

    def forward(self, inputs, sensors, device, fine_tune=False, clip_align=False):
        # if fine_tune: # moving this to train function
        #     self.model_ib.eval() # Might need to manually freeze the weights, this might not be enough
        #     self.model_linear.train()
        # Load and transform video data
        frames, accel_data, labels, pid_idx, rgb_path, imu_path  = inputs
        with torch.no_grad():
            if sensors =="both":
                inputs = {
                    ModalityType.VISION: data.load_and_transform_video_data(rgb_path, device),
                    ModalityType.IMU: load_and_transform_imu_data(accel_data, device)
                }
                emb = self.model_ib(inputs)
                embedding_vector = (emb[ModalityType.VISION]+emb[ModalityType.IMU])/2
            elif sensors == "vision": #might be an easier way to write this
                inputs = {
                    ModalityType.VISION: data.load_and_transform_video_data(rgb_path, device),
                }
                emb = self.model_ib(inputs)
                embedding_vector = emb[ModalityType.VISION]

            elif sensors == "imu":
                inputs = {
                    ModalityType.IMU: load_and_transform_imu_data(accel_data, device)
                }
                emb = self.model_ib(inputs)
                embedding_vector = emb[ModalityType.IMU]
            elif sensors == "depth":
                inputs = {
                    ModalityType.DEPTH: frames.to(device),
                }
                emb = self.model_ib(inputs)
                embedding_vector = emb[ModalityType.DEPTH]
            elif sensors == "both_depth":
                inputs = {
                    ModalityType.IMU: load_and_transform_imu_data(accel_data, device),
                    ModalityType.DEPTH: frames.to(device)
                }
                emb = self.model_ib(inputs)
                embedding_vector = (emb[ModalityType.IMU]+emb[ModalityType.DEPTH])/2
            else:
                raise ValueError("Invalid sensor type: ", sensors)

        outputs = self.model_linear(embedding_vector)
        return outputs
        # emb = self.model_ib(inputs)
        # return self.model_linear(emb)
    
#NOTE: UNFINISHED! Need to implement the contrastive train
# def train_contrastive(train_loader, val_loader, model, sensors, device, args):
#     # Train the model
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Add scheduler
#     num_epochs = args.num_epochs
    
#     for epoch in range(num_epochs):

#         for i,inputs  in enumerate(train_loader):
#             # break # used to quick test eval function
#             frames, accel_data, labels, pid_idx, rgb_path, imu_path  = inputs

#             # outputs = model(inputs, sensors, fine_tune=True)
#             # labels = labels.to(device)
#             # loss = criterion(outputs, labels)
#             # # Backward and optimize
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()

#             if (i+1) % 1 == 0:
#                 if args.rank == 0 or args.single_gpu:
#                     print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#                     if not args.no_wandb: wandb.log({"Train Loss": loss.item()})
        
#         scheduler.step()  # Step the scheduler

#         if (epoch+1) % args.eval_every == 0:
#             acc = evaluate_ib(model, sensors, val_loader, device)
#             if args.rank == 0 or args.single_gpu: 
#                 print(f'Epoch [{epoch+1}/{num_epochs}],  Val Acc: {acc:.4f}')
#                 if not args.no_wandb: wandb.log({"Val Accuracy": acc})

#                 #save the model 
#                 torch.save(model.state_dict(), f'model_linear_{sensors}.ckpt')

#     print('Finished CLIPTraining')

def train_linear(train_loader, val_loader, model, sensors, args):
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # Add scheduler
    num_epochs = args.num_epochs
    device = args.device
    if args.dataset == "czu-mhad":
        sensors = "depth"
    
    if args.single_gpu:
        model.model_linear.train()
        # freeze model.model_ib weights
        for param in model.model_ib.parameters():
            param.requires_grad = False
    else:
        model.module.model_linear.train()
        # DDP
        for param in model.module.model_ib.parameters():
            param.requires_grad = False

    
    for epoch in range(num_epochs):
        for i,inputs  in enumerate(train_loader):
            # break # used to quick test eval function
            frames, accel_data, labels, pid_idx, rgb_path, imu_path  = inputs

            # outputs = model.model_linear(embedding_vector)
            outputs = model(inputs, sensors, device, fine_tune=True)
            labels = labels.to(device)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1 == 0:
                if args.rank == 0 or args.single_gpu:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    if not args.no_wandb: wandb.log({"Train Loss": loss.item()})
        
        scheduler.step()  # Step the scheduler

        if (epoch+1) % args.eval_every == 0:
            # Evaluation without wandb log
            # acc = evaluate_ib(model, sensors, val_loader, args)
            # Evaluate log with all combinations of sensors
            acc_rgb, acc_imu, acc_both = eval_log_ib(model, val_loader, args)
            # Eval log with only vision
            # acc_rgb = eval_log_ib(model, val_loader, args, eval_sensors=["vision"])[0]
            acc = acc_rgb
            if args.rank == 0 or args.single_gpu: 
                # print(f'Epoch [{epoch+1}/{num_epochs}],  Val Acc RGB: {acc:.4f}')
                # if not args.no_wandb: wandb.log({"Val Accuracy": acc})

                #save the model 
                torch.save(model.state_dict(), f'./models/ib_linear_{sensors}.ckpt')

    print('Finished Training')

def evaluate_ib(model, sensors, val_loader, args):
    model.eval()
    correct = 0
    total = 0
    device = args.device
    if args.dataset == "czu-mhad":
        if sensors == "vision":
            sensors = "depth"
        elif sensors=="both":
            sensors = "both_depth"
    for i,inputs  in enumerate(val_loader):
        frames, accel_data, labels, pid_idx, rgb_path, imu_path  = inputs
        out = model(inputs, sensors, device)
        pred = out.cpu().argmax(dim=-1).type(torch.int)
        labels = labels.cpu()
        total += labels.size(0)
        correct += (pred == labels).sum()

        # Convert to tensor and aggregate across all processes
        correct_tensor = torch.tensor(correct, dtype=torch.float32, device=device)
        total_tensor = torch.tensor(total, dtype=torch.float32, device=device)

        if not args.single_gpu:
            # All reduce
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
     
    return 100.* correct/total


def eval_log_ib(model, val_loader, args, eval_sensors=["vision", "imu", "both"]):
    device = args.device

    return_vals = []
    #Finally evaluate on camera->HAR, imu->HAR, camera+imu->HAR
    if "vision" in eval_sensors:
        if args.rank==0 or args.single_gpu: print("Evaluating on RGB only")
        acc_rgb = evaluate_ib(model, "vision", val_loader, args)
        if args.rank==0: 
            if not args.no_wandb: wandb.log({'val_acc_RGB': acc_rgb})
            print('Test accuracy RGB: {:.4f} %'.format(acc_rgb))
        return_vals.append(acc_rgb.item())

    if "imu" in eval_sensors:
        if args.rank==0 or args.single_gpu: print("Evaluating on IMU only")
        acc_imu = evaluate_ib(model, "imu", val_loader, args)
        if args.rank==0 or args.single_gpu: 
            if not args.no_wandb: wandb.log({'val_acc_IMU': acc_imu})
            print('Test accuracy IMU: {:.4f} %'.format(acc_imu))
        return_vals.append(acc_imu.item())

    if "both" in eval_sensors:
        if args.rank==0 or args.single_gpu: print("Evaluating on RGB and IMU")
        acc_both = evaluate_ib(model, "both", val_loader, args)
        if args.rank==0 or args.single_gpu: 
            if not args.no_wandb: wandb.log({'val_acc_both': acc_both})
            print('Test accuracy: {:.4f} %'.format(acc_both))
        return_vals.append(acc_both.item())

    return return_vals
    
if __name__ == "__main__":    
    # video_paths=["/media/abhi/Seagate-FireCUDA/utd-mhad/RGB/a27_s4_t3_color.avi"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = argparse.ArgumentParser() # can't keep this global otherwise will conflict with FACT/train.py, when that file imports this file.
    args.add_argument("--sensors", type=str, default="vision", help="Choose between vision, imu or both")
    args.add_argument("--zero_shot", type=bool, default=False, help="Choose between vision, imu or both")
    args.add_argument("--batch_size", type=int, default=16, help="Choose between vision, imu or both")
    args.add_argument("--device", type=str, default=device, help="cuda device")
    args.add_argument("--dataset", type=str, default="utd-mhad", help="Choose between utd-mhad, mmact, mmea, czu-mhad")
    args.add_argument('--no_wandb', action='store_true',default=False)
    args.add_argument('--single_gpu', action='store_true',default=True) # default use DDP multi GPU.
    args.add_argument('--eval_every', type=int, default=10) # how often to perform eval (every eval_every epochs)
    args.add_argument('--num_epochs', type=int, default=10) 

    args = args.parse_args()

    device = args.device

    #dummies to be compatible with fact train
    args.rank = 0
    args.signal_gpu = True



    # print(model.device)
    print("Loading test data...")
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize frames
        transforms.ToTensor(),           # Convert frames to tensors
    ])
    rgb_video_length = 30

    model_info = {
        'sensors' : ['RGB', 'IMU'], #['RGB', 'IMU'] #NOTE: Keep the order here consistent for naming purposes
        'tasks' : ['HAR'], #['HAR', 'PID'],
        'fusion_type' : 'imagebind', #'middle', #'cross_modal', # 'early', 'middle', 'late', 'cross_modal', 'student_teacher
        'num_classes' : -1, # Will be replaced in load_dataloaders depending on the dataset
        'project_name' : ""
    }

    HOME_DIR = os.environ['HOME']
    dataset = args.dataset
    rgb_video_length = 30
    imu_length = 180
    world_size = 1

    # Load the dataloaders
    train_loader, train_2_loader, val_loader, test_loader, model_info = load_dataloaders(dataset, model_info, rgb_video_length, imu_length, world_size, args, return_path=True)

    model = ImageBind_baseline(output_size=model_info['num_classes'])
    model.to(device)

    print(args)
    print(model_info)
    if args.zero_shot:
        print("running zero shot ")
        zero_shot_imagebind(val_loader, model.model_ib, args.sensors, device)
    else:
        print("finetuning a linear layer:")
        if not args.no_wandb: wandb.init(project="imagebind-finetune")


        # model_linear.to(device)
        # TODO: Maybe should first align with both imu and vision modalities
        
        # Train the task head with just vision modality
        args.sensors = "vision"
        train_linear(train_2_loader, test_loader, model, args.sensors, device, args)

        # args.sensors = "imu"
        # train_linear(train_loader, test_loader, model, model_linear, args.sensors, device)

        # evaluate_ib(model, model_linear, args.sensors, test_loader, device)