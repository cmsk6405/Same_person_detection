from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np

def resnet_embedding(folder_path, save_folder):

    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    cant = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            img = Image.open(image_path).convert('RGB')
            img = transform(img).unsqueeze(0)

            if img is not None :
                embedding = resnet(img).detach()
                save_embeding = embedding.numpy()
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = os.path.join(save_folder, file_name.replace('.png', '.npy'))
                np.save(save_path, save_embeding)
            else:
                cant.append(image_path)
    
    if len(cant) > 0:
        if not os.path.exists("./cantlist"):
            os.makedirs("./cantlist")
        np.savetxt('./cantlist/cant_emb_list.txt', cant, fmt='%s')
        

