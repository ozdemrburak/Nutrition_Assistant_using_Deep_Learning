import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from transformers import AutoModel


class Siglip2Regressor(nn.Module):

  def __init__(self, model_name = "google/siglip2-base-patch16-224", output_dim = 5):
    super().__init__()
    full_model = AutoModel.from_pretrained(model_name)
    self.vision_encoder = full_model.vision_model # text kısmı gerekli olmadığından yalnızca vision_encoder aldım
    # freeze model
    for param in self.vision_encoder.parameters():
      param.requires_grad = False
    for param in self.vision_encoder.head.mlp.parameters():
      param.requires_grad = True

    hidden_size = self.vision_encoder.head.mlp.fc2.out_features
    self.reg_head = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_size, output_dim)
    )

  def forward(self, x):
    x = self.vision_encoder(x)
    x = x.pooler_output #siglip finetune ederken gerekli, kullanmazsa hata veriyor
    x = self.reg_head(x)
    return x


def image_transform(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
    return transform(image).unsqueeze(0)


def predict_image(image_path):
    """
    - Eğittiğim modeli huggingface'e yükledim ve oradan çekiyorum.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image_transform(image_path).to(device)
    with torch.no_grad():
        scaled_prediction = model(image)

    return unscale_prediction(scaled_prediction)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Siglip2Regressor()
model_path = hf_hub_download(repo_id="theycallmeburki/siglip2_regressor",
                             filename="siglip2_regressor_state_dict.pth")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


def unscale_prediction(output):
    """
    - Model eğitirken StandardScaler kullandık. Çıktılar bu sebepten ölçeklenmiş halde dönüyor.
    - Bu fonksiyon gerçek çıktıları döndürür.
    """
    y_mean = torch.tensor([182.817254, 217.43218233, 10.88178516, 16.80198885, 15.04939805], device=output.device)
    y_std = torch.tensor([143.11745907, 196.06303582, 12.62122967, 15.10990037, 18.22648705], device=output.device)
    y_pred_scaled = output.cpu().numpy()
    y_pred_original = y_pred_scaled * y_std + y_mean
    y_pred_original = torch.clamp(y_pred_original, min=0)

    return y_pred_original

