from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])


""" This code below can run successfully on one video.... i think"""

# from imagebind import data
# import torch
# from imagebind.models import imagebind_model
# from imagebind.models.imagebind_model import ModalityType


# actions_dict = {
#     1: 'A person swipes their hand to the left.',
#     2: 'A person swipes their hand to the right.',
#     3: 'A person waves.',
#     4: 'A person claps.',
#     5: 'A person throws an object.',
#     6: 'A person crosses their arms.',
#     7: 'A person shoots a basketball.',
#     8: 'A person draws an X.',
#     9: 'A person draws a circle clockwise.',
#     10: 'A person draws a circle counter clockwise.',
#     11: 'A person draws a triangle.',
#     12: 'A person is bowling.',
#     13: 'A person is boxing.',
#     14: 'A person swings a baseball bat.',
#     15: 'A person swings a tennis racket.',
#     16: 'A person does an arm curl.',
#     17: 'A person serves in tennis.',
#     18: 'A person pushes an object.',
#     19: 'A person knocks on a door.',
#     20: 'A person catches an object.',
#     21: 'A person picks up and throws an object.',
#     22: 'A person is jogging.',
#     23: 'A person is walking.',
#     24: 'A person moves from sitting to standing.',
#     25: 'A person moves from standing to sitting.',
#     26: 'A person does a lunge.',
#     27: 'A person does a squat.'
# }

# text_list=list(actions_dict.values())
# # video_paths=["/media/abhi/Seagate-FireCUDA/utd-mhad/RGB/a27_s4_t3_color.avi"]

# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # Instantiate model
# model = imagebind_model.imagebind_huge(pretrained=True)
# model.eval()
# model.to(device)

# # Load data
# inputs = {
#     ModalityType.TEXT: data.load_and_transform_text(text_list, device),
#     ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
#     # ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
# }

# with torch.no_grad():
#     embeddings = model(inputs)

# sftmx = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
# print("Vision x Text: ", sftmx)

# pred = torch.argmax(sftmx, dim=-1)
# print("Predicted Action: ", actions_dict[pred.item()])
