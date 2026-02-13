import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torchvision import models, transforms
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv

# Config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
config = {"batch": 16, "lr": 1e-4, "epochs": 5, "path": "model.pth", "device": "cuda" if torch.cuda.is_available() else "cpu"}
client = OpenAI(api_key=OPENAI_API_KEY)

# Data Processing
def preprocess(img):
    if img.mode != 'RGB': img = img.convert('RGB')
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(img)

def collate_fn(batch):
    imgs, labels = [], []
    for item in batch:
        k_img = 'Image' if 'Image' in item else 'image'
        imgs.append(preprocess(item[k_img]))
        
        # Label Logic: 1 if Disease Label indicates DME/DR or dr_level > 0
        l = 0
        if 'Disease Label' in item:
            val = item['Disease Label']
            l = 1 if (isinstance(val, str) and ("DME" in val or "DR" in val)) or (isinstance(val, (int, float)) and val > 0) else 0
        elif 'dr_level' in item:
            l = 1 if item['dr_level'] > 0 else 0
        elif 'label' in item:
            l = item['label']
        labels.append(l)
    return torch.stack(imgs), torch.tensor(labels)

# Main
if __name__ == "__main__":
    print(f"Loading OLIVES Dataset... Device: {config['device']}")
    ds = load_dataset("gOLIVES/OLIVES_Dataset", "disease_classification", split="train", streaming=True)
    
    # Create simple lists for demo (streaming subset)
    data = []
    for i, item in enumerate(ds):
        if i >= 120: break
        data.append(item)
    
    train_dl = DataLoader(data[:100], batch_size=config['batch'], shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(data[100:], batch_size=config['batch'], shuffle=False, collate_fn=collate_fn)

    # Model
    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(config['device'])
    
    # Train
    crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=config['lr'])
    print("Training...")
    for e in range(config['epochs']):
        model.train()
        losses = []
        for X, y in train_dl:
            opt.zero_grad()
            loss = crit(model(X.to(config['device'])), y.to(config['device']))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {e+1}: Loss {sum(losses)/len(losses):.4f}")
    
    torch.save(model.state_dict(), config['path'])
    print(f"Saved to {config['path']}")

    # Eval
    model.eval()
    X, y = next(iter(test_dl))
    with torch.no_grad():
        out = model(X[0].unsqueeze(0).to(config['device']))
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
    
    lbls = {0: "Healthy", 1: "At Risk (DR/DME)"}
    print(f"\nPred: {lbls[pred.item()]} ({conf.item():.1%}) | Actual: {lbls[y[0].item()]}")

    # GPT Explanation
    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Explain diagnosis {lbls[pred.item()]} with {conf.item():.1%} confidence for a retinal scan briefly to a doctor."}]
        )
        print(f"\nAI Note: {res.choices[0].message.content}")
    except Exception as e: print(e)
