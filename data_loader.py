import os, math
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer


def collate_fn(batch):
    """Custom collate function to handle batching of dictionary items"""
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    token_type_ids = torch.stack([torch.tensor(item['token_type_ids']) for item in batch])

    # Handle labeled vs unlabeled data
    if 'labels' in batch[0]:
        labels = torch.stack([torch.tensor(item['labels']) for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }
    else:
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }


class MNLIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, domains, sentiments):
        super().__init__()
        self.data_dir = data_dir
        self.domains = domains
        self.sentiments = sentiments
        self.domain_dict = dict()
        self.sentiment_dict = dict()
        for idx, domain in enumerate(domains):
            self.domain_dict[domain] = idx
        for idx, sent in enumerate(sentiments):
            self.sentiment_dict[sent] = idx

        self.batch_size = 1
        self.stage = None
        self.task = None


    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_predict_suffix(self, suffix):
        """Set the suffix for prediction data (e.g., '.test_matched')"""
        self.predict_suffix = suffix

    def prepare_data(self):
        mnli_preprocess(self.data_dir, self.domains)

    def setup(self, stage, task='seq_pair_classif', domain=None, ratio=None):
        self.stage = stage
        self.task = task
        target_domains = self.domains if domain == "all" else [domain]
        print(f'Setup datasets, target domains: {target_domains}')
        if task == 'seq_pair_classif':
            if stage == 'fit':  # Training + Validation
                # Entire training set
                self.train = MNLIDataset(
                    self.data_dir, target_domains, self.domain_dict,
                    self.sentiment_dict, 'train', task
                )
                # Validation set (dev_matched)
                self.valid = ConcatDataset(
                    [MNLIDataset(
                    self.data_dir, target_domains, self.domain_dict,
                    self.sentiment_dict, 'dev_matched', task
                    ),
                    MNLIDataset(
                        self.data_dir, target_domains, self.domain_dict,
                        self.sentiment_dict, 'dev_mismatched', task
                    )
                    ]
                )
                print(f'Setup stage {stage} | train size {len(self.train)} | validation size {len(self.valid)}')

            elif stage == 'predict':  # Unlabeled data
                self.predict = ConcatDataset(
                    [MNLIDataset(
                        self.data_dir, target_domains, self.domain_dict,
                        self.sentiment_dict, 'test_matched', task, has_labels=False
                    ),
                    MNLIDataset(
                        self.data_dir, target_domains, self.domain_dict,
                        self.sentiment_dict, 'test_mismatched', task, has_labels=False
                    )
                    ]
                )
                print(f'Setup stage {stage} | predict size {len(self.predict)}')


        elif task == 'domain_classif':
            # Combine data from all domains
            if stage == 'fit':  # Training + Validation
                # Training set combines all domains
                self.train = MNLIDataset(
                    self.data_dir, target_domains, self.domain_dict,
                    self.sentiment_dict, 'train', task
                )
                # Validation set combines dev splits
                self.valid = ConcatDataset([
                    MNLIDataset(
                        self.data_dir, target_domains, self.domain_dict,
                        self.sentiment_dict, 'dev_matched', task
                    ),
                    MNLIDataset(
                        self.data_dir, target_domains, self.domain_dict,
                        self.sentiment_dict, 'dev_mismatched', task
                    )
                ])
                print(f'Domain Classif: stage {stage} | train size {len(self.train)} | valid size {len(self.valid)}')

            elif stage == 'predict':  # Test sets
                self.predict = ConcatDataset([
                    MNLIDataset(
                        self.data_dir, target_domains, self.domain_dict,
                        self.sentiment_dict, 'test_matched', task, has_labels=False
                    ),
                    MNLIDataset(
                        self.data_dir, target_domains, self.domain_dict,
                        self.sentiment_dict, 'test_mismatched', task, has_labels=False
                    )
                ])
                print(f'Domain Classif: stage {stage} | predict size {len(self.predict)}')


    def train_dataloader(self):
        assert self.stage == 'fit'
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )

    def val_dataloader(self):
        assert self.stage == 'fit'
        return DataLoader(
            self.valid,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )

    def predict_dataloader(self):
        assert self.stage == 'predict'
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=collate_fn
        )


class MNLIDataset(Dataset):
    def __init__(self, file_dir, domains, domain_dict, sentiment_dict,
                 suffix, task, ratio=None, has_labels=True):
        self.file_dir = file_dir
        self.domains = domains if isinstance(domains, list) else [domains]
        self.tokenizer = AutoTokenizer.from_pretrained(
            r'C:\Users\tangc\PycharmProjects\domain-adaptation\bert-base-uncased-cache'
        )
        self.suffix = suffix
        self.task = task
        self.domain_dict = domain_dict
        self.sentiment_dict = sentiment_dict
        self.has_labels = has_labels
        self.examples = []
        self.length = 0

        for domain in self.domains:
            filename = f"{domain}.{suffix}.tmp"
            file_path = os.path.join(self.file_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    json_dict = json.load(f)
                domain_examples = json_dict["data"]
                if ratio is not None:
                    length = math.floor(len(domain_examples) * ratio)
                    domain_examples = domain_examples[:length]

                if self.task == "domain_classif":
                    # modify label to domain label here
                    for example in domain_examples:
                        example[-1] = domain

                self.examples.extend(domain_examples)

        self.length = len(self.examples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        line = self.examples[idx]
        encoding = self.tokenizer(
            line[0], line[1],
            add_special_tokens=True,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True
        )
        item = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids']
        }
        if self.task == "domain_classif":
            if self.has_labels and len(line) == 3:
                item['labels'] = self.domain_dict.get(line[-1], -1)
        else:  # seq_pair_classif
            if self.has_labels and len(line) == 3:
                item['labels'] = self.sentiment_dict.get(line[2], -1)
        return item


def mnli_preprocess(file_dir, domains):
    # Preprocess training set
    train_path = os.path.join(file_dir, "MNLI/MNLI/train.tsv")
    dev_matched_path = os.path.join(file_dir, "MNLI/MNLI/dev_matched.tsv")
    dev_mismatched_path = os.path.join(file_dir, "MNLI/MNLI/dev_mismatched.tsv")
    test_matched_path = os.path.join(file_dir, "MNLI/MNLI/test_matched.tsv")
    test_mismatched_path = os.path.join(file_dir, "MNLI/MNLI/test_mismatched.tsv")


    # Process training data
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = f.readlines()

    # Process dev sets
    with open(dev_matched_path, "r", encoding="utf-8") as f:
        dev_matched_data = f.readlines()
    with open(dev_mismatched_path, "r", encoding="utf-8") as f:
        dev_mismatched_data = f.readlines()

    # Process test sets (unlabeled)
    with open(test_matched_path, "r", encoding="utf-8") as f:
        test_matched_data = f.readlines()
    with open(test_mismatched_path, "r", encoding="utf-8") as f:
        test_mismatched_data = f.readlines()


    # Process datasets
    for domain in domains:
        # Training data (full set)
        train_examples = []
        for idx, data in enumerate(train_data):
            if idx == 0:  # skip header
                continue
            ex = data.strip().split('\t')
            if ex[3] == domain:
                train_examples.append((ex[-4], ex[-3], ex[-1]))

        with open(os.path.join(file_dir, f"{domain}.train.tmp"), "w") as f:
            json.dump({"domain": domain, "data": train_examples}, f)


        for dev_data, suffix in [(dev_matched_data, 'dev_matched'),
                                 (dev_mismatched_data, 'dev_mismatched')]:
            dev_examples = []
            for idx, data in enumerate(dev_data):
                if idx == 0:  # skip header
                    continue
                ex = data.strip().split('\t')
                if ex[3] == domain:
                    dev_examples.append((ex[-8], ex[-7], ex[-1]))

            with open(os.path.join(file_dir, f"{domain}.{suffix}.tmp"), "w") as f:
                json.dump({"domain": domain, "data": dev_examples}, f)


        # Test sets (unlabeled)
        for test_data, suffix in [(test_matched_data, 'test_matched'),
                                  (test_mismatched_data, 'test_mismatched')]:
            test_examples = []
            for idx, data in enumerate(test_data):
                if idx == 0:  # skip header
                    continue
                ex = data.strip().split('\t')
                if ex[3] == domain:
                    test_examples.append((ex[-2], ex[-1]))

            with open(os.path.join(file_dir, f"{domain}.{suffix}.tmp"), "w") as f:
                json.dump({"domain": domain, "data": test_examples}, f)



if __name__ == '__main__':
    # Test the updated functionality
    # TODO: add tests for train, dev, and predict
    domains = ['telephone', 'oup']
    sentiments = ["entailment", "contradiction", "neutral"]
    file_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'

    mnli_preprocess(file_dir, domains)

    # Test training set (should have labels)
    train_dataset = MNLIDataset(
        file_dir,
        ['telephone'],
        {'telephone': 0},
        {"entailment": 0, "neutral": 1, "contradiction": 2},
        'train',
        'seq_pair_classif'
    )
    print(f"Training set size: {len(train_dataset)}")
    print("First training item:", train_dataset[0])


    # Test validation set
    val_dataset = MNLIDataset(
        file_dir,
        ['oup'],
        {'oup': 0},
        {"entailment": 0, "neutral": 1, "contradiction": 2},
        'dev_mismatched',
        'seq_pair_classif'
    )
    print(f"Validation set size: {len(val_dataset)}")
    print("First validation item:", val_dataset[0])

    # Test predict set (unlabeled)
    predict_dataset_m = MNLIDataset(
        file_dir,
        domains,
        {'telephone': 0, 'oup':1},
        {"entailment": 0, "neutral": 1, "contradiction": 2},
        'test_matched',
        'seq_pair_classif',
        has_labels=False
    )
    predict_dataset_mm = MNLIDataset(
        file_dir,
        domains,
        {'telephone': 0, 'oup':1},
        {"entailment": 0, "neutral": 1, "contradiction": 2},
        'test_mismatched',
        'seq_pair_classif',
        has_labels=False
    )
    predict_dataset = ConcatDataset([predict_dataset_m, predict_dataset_mm])
    print(f"Test set size: {len(predict_dataset)}")
    print("First test item:", predict_dataset[0])
