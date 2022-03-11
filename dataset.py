from training_params import TOKENIZER, MAX_LEN
from torch.utils.data import Dataset, DataLoader
import torch
from utils import process_csv
from training_params import TRAIN_DATA

class SentimentDataset(Dataset):

    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = TOKENIZER.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            # 'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

if __name__ == "__main__":
    texts, targets = process_csv(TRAIN_DATA)
    # first row
    data = SentimentDataset(texts, targets).__getitem__(0)
    print(type(data))
    print(data)
    print(TOKENIZER.convert_ids_to_tokens(data['input_ids']))

