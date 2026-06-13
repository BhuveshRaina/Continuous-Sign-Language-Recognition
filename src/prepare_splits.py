import os
import json
import random
import config

def create_splits_and_labels():
    if not os.path.exists(config.METADATA_DIR):
        os.makedirs(config.METADATA_DIR)

    # 1. Get all sentence folders
    sentences = sorted([d for d in os.listdir(config.KEYPOINTS_DIR) 
                        if os.path.isdir(os.path.join(config.KEYPOINTS_DIR, d))])
    
    # 2. Create and save Label Map
    label_map = {sentence: idx for idx, sentence in enumerate(sentences)}
    with open(config.LABEL_MAP_FILE, 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"✅ Saved label_map.json with {len(label_map)} classes.")

    # 3. Split data (70% Train, 15% Val, 15% Test)
    splits = {'train': [], 'val': [], 'test': []}
    
    for sentence in sentences:
        folder_path = os.path.join(config.KEYPOINTS_DIR, sentence)
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        # Shuffle to ensure random distribution
        random.seed(42)
        random.shuffle(files)
        
        n_files = len(files)
        n_train = int(n_files * 0.7)
        n_val = int(n_files * 0.15)
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        
        # Store relative paths: "sentence_folder/video.npy"
        for f in train_files: splits['train'].append(os.path.join(sentence, f))
        for f in val_files: splits['val'].append(os.path.join(sentence, f))
        for f in test_files: splits['test'].append(os.path.join(sentence, f))

    # 4. Save Splits
    with open(config.SPLITS_FILE, 'w') as f:
        json.dump(splits, f, indent=4)
    
    print(f"✅ Saved data_splits.json")
    print(f"   Train samples: {len(splits['train'])}")
    print(f"   Val samples: {len(splits['val'])}")
    print(f"   Test samples: {len(splits['test'])}")

if __name__ == "__main__":
    create_splits_and_labels()