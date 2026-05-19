import torch
from torchvision import transforms
from PIL import Image

from app.training.model import CIFAR10CNN


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


model = CIFAR10CNN()

model.load_state_dict(
    torch.load(
        "models/cifar10_cnn.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

    return CLASSES[predicted.item()]

