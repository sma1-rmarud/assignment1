import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import random
from transformers import ViTForImageClassification
import logging
import os

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로그 및 파일 핸들러 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
loggers = {
    "fgsm_targeted": logging.getLogger("FGSM_Targeted"),
    "fgsm_untargeted": logging.getLogger("FGSM_Untargeted"),
    "pgd_targeted": logging.getLogger("PGD_Targeted"),
    "pgd_untargeted": logging.getLogger("PGD_Untargeted"),
}

for attack_type in loggers:
    os.makedirs('logs', exist_ok=True)
    handler = logging.FileHandler(f"./logs/log_{attack_type}.txt", mode="w")
    loggers[attack_type].addHandler(handler)

# 모델 로드
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=10, ignore_mismatched_sizes=True)
model.to(device)
model.eval()

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# CIFAR-10 데이터셋 로드
os.makedirs('datasets', exist_ok=True)
testset = datasets.CIFAR10(root="./datasets", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)


# Targeted FGSM attack 시행
def fgsm_targeted(model, x, target, eps):
    x = x.clone().detach().requires_grad_(True).to(device)
    target = target.to(device)
    
    output = model(x).logits

    loss = nn.functional.cross_entropy(output, target)

    model.zero_grad()
    loss.backward()

    if x.grad is None:
        raise ValueError("x.grad is None! Backpropagation might have failed.")

    adv_x = x - eps * x.grad.sign()
    return torch.clamp(adv_x, 0, 1).detach()


# Untargeted FGSM attack 시행
def fgsm_untargeted(model, x, label, eps):
    x = x.clone().detach().requires_grad_(True).to(device)
    label = label.to(device)

    output = model(x).logits
    loss = nn.functional.cross_entropy(output, label)

    model.zero_grad()
    loss.backward()

    if x.grad is None:
        raise ValueError("x.grad is None!")

    adv_x = x + eps * x.grad.sign()
    return torch.clamp(adv_x, 0, 1).detach()

# Targeted PGD attack 시행
def pgd_targeted(model, x, target, eps, alpha=0.01, iters=10):
    x_orig = x.clone().detach().to(device)
    x_adv = x.clone().detach().to(device)
    target = target.to(device)

    for _ in range(iters):
        x_adv.requires_grad_()
        output = model(x_adv).logits
        loss = nn.functional.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()

        if x_adv.grad is None:
            raise RuntimeError("Gradient is None!")

        x_adv = x_adv - alpha * x_adv.grad.sign()
        eta = torch.clamp(x_adv - x_orig, -eps, eps)  # Projection
        x_adv = torch.clamp(x_orig + eta, 0, 1).detach()

    return x_adv


# Untargeted PGD attack 시행
def pgd_untargeted(model, x, label, eps, alpha=0.01, iters=10):
    x_orig = x.clone().detach().to(device)
    x_adv = x.clone().detach().to(device)
    label = label.to(device)

    for _ in range(iters):
        x_adv.requires_grad_()
        output = model(x_adv).logits
        loss = nn.functional.cross_entropy(output, label)

        model.zero_grad()
        loss.backward()

        if x_adv.grad is None:
            raise RuntimeError("Gradient is None!")

        x_adv = x_adv + alpha * x_adv.grad.sign()
        eta = torch.clamp(x_adv - x_orig, -eps, eps)   # Projection
        x_adv = torch.clamp(x_orig + eta, 0, 1).detach()

    return x_adv

# 평가 함수
def evaluate_attack(model, attack_fn, eps, targeted):
    correct, total = 0, 0
    success_examples = []

    for x, label in testloader:
        x, label = x.to(device), label.to(device)

        if targeted:
            target_labels = (label + 1) % 10  # 타겟 레이블 설정
            adv_x = attack_fn(model, x, target_labels, eps)
        else:
            adv_x = attack_fn(model, x, label, eps)

        # ViT에서는 logits로 접근해야 함
        adv_output = model(adv_x).logits
        adv_pred = adv_output.argmax(dim=1)

        total += label.size(0)

        if targeted:
            correct += (adv_pred == target_labels).sum().item()
        else:
            correct += (adv_pred != label).sum().item()

        for j in range(len(label)):
            if (targeted and adv_pred[j] == target_labels[j]) or (not targeted and adv_pred[j] != label[j]):
                success_examples.append((x[j].cpu(), adv_x[j].cpu(), label[j].item(), adv_pred[j].item()))

        if len(success_examples) >= 10:
            break

    success_rate = correct / total
    return success_rate, success_examples

# 예시 저장
def plot_examples(eps, examples, title, attack_type):
    random.shuffle(examples)
    examples = examples[:5]
    
    fig, axes = plt.subplots(2, len(examples), figsize=(15, 6))
    for i, (orig, adv, true_label, adv_label) in enumerate(examples):
        # 정규화 해제: * 0.5 + 0.5 / 채널 순서 변경
        axes[0, i].imshow(orig.permute(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].set_title(f"Original: {true_label}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(adv.permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].set_title(f"Adversarial: {adv_label}", fontsize=10)
        axes[1, i].axis("off")

    plt.suptitle(title, fontsize=14)
    
    examples_folder = f"./examples/{attack_type}_examples"
    os.makedirs(examples_folder, exist_ok=True)
    
    filename = f"eps_{eps:.2f}.png"
    filepath = f"{examples_folder}/{filename}"
    try:
        plt.savefig(filepath)
        print(f"Saved {filepath}")
    except Exception as e:
        print(f"Error saving image: {e}")
    finally:
        plt.close()


def attack_logging(logger, eps, success_rate, attack_type):
    logger.info(f"Epsilon: {eps}, {attack_type} Success Rate: {success_rate * 100:.2f}%")


# 다양한 엡실론 값 설정
eps_values = [0.05, 0.1, 0.15, 0.2, 0.3]

for eps in eps_values:
    for attack_fn, attack_name, logger, is_targeted in [
        (fgsm_targeted, "FGSM_Targeted", loggers["fgsm_targeted"], True),
        (fgsm_untargeted, "FGSM_Untargeted", loggers["fgsm_untargeted"], False),
        (pgd_targeted, "PGD_Targeted", loggers["pgd_targeted"], True),
        (pgd_untargeted, "PGD_Untargeted", loggers["pgd_untargeted"], False),
    ]:
        success_rate, examples = evaluate_attack(model, attack_fn, eps, targeted=is_targeted)
        attack_logging(logger, eps, success_rate, attack_name)
        
        if examples:
            plot_examples(eps, examples, f"{attack_name} Attack", attack_name.lower())
