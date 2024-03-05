from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


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

text_list=list(actions_dict.values())
# video_paths=["/media/abhi/Seagate-FireCUDA/utd-mhad/RGB/a27_s4_t3_color.avi"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


print("Loading test data...")
test_file="/home/abhi/data/utd-mhad/RGB_splits/Action_80_20_#1/val.txt"
# Load video paths and labels from a file
with open(test_file, "r") as file:
    lines = [line.strip().split() for line in file]
    video_paths = [line[0] for line in lines]
    labels = [int(line[1]) for line in lines]

predictions = []
for v in video_paths:
    # Load and transform video data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_video_data([v], device),
    }

    print("Computing predictions... for ", v)
    with torch.no_grad():
        emb = model(inputs)
        sftmx = torch.softmax(emb[ModalityType.VISION] @ emb[ModalityType.TEXT].T, dim=-1)
    print(sftmx)
    pred = torch.argmax(sftmx, dim=-1)
    predictions.append(pred.item())
    print(predictions)


print("Computing accuracy...")
# print("Vision x Text: ", sftmx)


correct = 0
total = len(labels)
print("Lengths", len(predictions), len(labels))

for p, label in zip(predictions, labels):
    if p == label:
        correct += 1

accuracy = correct / total
print("Accuracy: ", accuracy)
