python'''
# --- 2. IMPORTS ---
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
import tifffile as tiff
import segmentation_models_pytorch as smp

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 3. CONFIGURATION ---
CONFIG = {
    "SEED": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPOCHS": 15,
    "BATCH_SIZE": 16,
    "LR": 3e-4,
    "IMG_SIZE": 256,
    "DATA_DIR": "/kaggle/input/med-informatics-liver-segmentation",
    "MODEL_NAME": "Unet",
    "BACKBONE": "resnet34"
}

# --- 4. DATA PREPARATION ---
def get_dataframes(root_dir):
    train_img_paths = []
    train_mask_paths = []
    test_img_paths = []
    test_ids = []

    # 1. Original Images
    orig_img_dir = os.path.join(root_dir, 'images')
    orig_mask_dir = os.path.join(root_dir, 'masks')
    if os.path.exists(orig_img_dir):
        for f in os.listdir(orig_img_dir):
            if f.endswith('.tiff'):
                mask_name = f.replace('orig_image', 'orig_mask')
                if os.path.exists(os.path.join(orig_mask_dir, mask_name)):
                    train_img_paths.append(os.path.join(orig_img_dir, f))
                    train_mask_paths.append(os.path.join(orig_mask_dir, mask_name))

    # 2. Augmented Images
    aug_img_dir = os.path.join(root_dir, 'aug_img')
    aug_mask_dir = os.path.join(root_dir, 'aug_mask')
    if os.path.exists(aug_img_dir):
        for f in os.listdir(aug_img_dir):
            if f.endswith('.tiff'):
                mask_name = f.replace('augmented_image', 'augmented_mask')
                if os.path.exists(os.path.join(aug_mask_dir, mask_name)):
                    train_img_paths.append(os.path.join(aug_img_dir, f))
                    train_mask_paths.append(os.path.join(aug_mask_dir, mask_name))

    # 3. Test Images
    test_dir = os.path.join(root_dir, 'test_image')
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.endswith('.tiff'):
                test_img_paths.append(os.path.join(test_dir, f))
                test_ids.append(f)

    train_df = pd.DataFrame({'image': train_img_paths, 'mask': train_mask_paths})
    test_df = pd.DataFrame({'image': test_img_paths, 'id': test_ids})
    
    return train_df, test_df

# --- 5. DATASET & LOADING ---
class LiverDataset(Dataset):
    def __init__(self, df, transform=None, mode='train'):
        self.df = df
        self.transform = transform
        self.mode = mode

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image']
        
        # Load Image (tifffile handles 64-bit float)
        try:
            image = tiff.imread(img_path).astype('float32')
            # Normalize to 0-255 uint8
            image = image - image.min()
            image = image / (image.max() + 1e-6)
            image = (image * 255).astype('uint8')
            if len(image.shape) == 2:
                image = np.stack([image]*3, axis=-1)
        except:
            image = np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'], 3), dtype='uint8')

        if self.mode != 'test':
            mask_path = self.df.iloc[idx]['mask']
            try:
                mask = tiff.imread(mask_path).astype('float32')
                mask = np.where(mask > 0, 1.0, 0.0)
            except:
                mask = np.zeros((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']), dtype='float32')

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            return image, mask.unsqueeze(0)
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, self.df.iloc[idx]['id']

    def


__len__(self):
        return len(self.df)

# --- 6. UTILS ---
def get_transforms(split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.Normalize(),
            ToTensorV2()
        ])

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# --- 7. EXECUTION ---
train_df, test_df = get_dataframes(CONFIG['DATA_DIR'])
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Train
train_dataset = LiverDataset(train_df, get_transforms('train'), mode='train')
train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
test_dataset = LiverDataset(test_df, get_transforms('test'), mode='test')
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)

model = smp.Unet(encoder_name=CONFIG['BACKBONE'], encoder_weights="imagenet", in_channels=3, classes=1).to(CONFIG['DEVICE'])
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
loss_fn = smp.losses.DiceLoss(mode="binary")

print("Training...")
for epoch in range(CONFIG['EPOCHS']):
    model.train()
    for images, masks in tqdm(train_loader):
        images = images.to(CONFIG['DEVICE'])
        masks = masks.to(CONFIG['DEVICE'])
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

# --- 8. SUBMISSION (FIXED HEADERS) ---
print("Generating Submission...")
model.eval()
submission = []

with torch.no_grad():
    for images, ids in tqdm(test_loader):
        images = images.to(CONFIG['DEVICE'])
        logits = model(images)
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float().cpu().numpy()
        
        for i, pred_mask in enumerate(preds):
            encoded = rle_encode(pred_mask[0])
            # FIXED: Column names are now 'id' and 'target'
            submission.append({'id': ids[i], 'target': encoded})

sub_df = pd.DataFrame(submission)
sub_df.to_csv('submission.csv', index=False)
print("Saved submission.csv with columns: id, target")
'''
