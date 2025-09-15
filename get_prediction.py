import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download


class ResNetRegressor(nn.Module):
  def __init__(self, num_outputs=5):
    super(ResNetRegressor, self).__init__()
    self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    self.reg_head = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features=1000, out_features=num_outputs)
    )

  def forward(self, x):
    x = self.backbone(x)
    x = self.reg_head(x)
    return x


def image_transform(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    return transform(image).unsqueeze(0)


def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image_transform(image_path)
    model = ResNetRegressor()
    model_path = hf_hub_download(repo_id="theycallmeburki/resnet_regressor_nutrition5k", filename="resnet_regressor_state_dict.pth")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    image = image.to(device)
    model = model.to(device)
    with torch.no_grad():
        scaled_prediction = model(image)

    return unscale_prediction(scaled_prediction)


def unscale_prediction(output):
    y_mean = [211.20069936, 253.61141603, 12.79815003, 18.99648289, 17.76016044]
    y_std = [151.06439266, 206.40657935, 13.34529479, 16.02617264, 19.58984867]
    y_pred_scaled = output.cpu().numpy()
    y_pred_original = y_pred_scaled * y_std + y_mean

    return y_pred_original


def main():
    print(predict_image(r"C:\Users\ozdem\OneDrive\Masaüstü\Nutristant\tavuk.jpg"))


if __name__ == "__main__":
    main()
