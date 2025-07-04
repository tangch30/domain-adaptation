import os, math
import json # vs jsonline (???)
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import AutoTokenizer


# convert data sources to torch.utils.data.IterableDatasets format
# The code is developed with the assistance of DeepSeek

def collate_fn(batch):
    """Custom collate function to handle batching of dictionary items"""
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    token_type_ids = torch.stack([torch.tensor(item['token_type_ids']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
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

    def prepare_data(self):
        mnli_preprocess(self.data_dir, self.domains, stage='fit')
        mnli_preprocess(self.data_dir, self.domains, stage='predict')


    def setup(self, stage, task='seq_pair_classif', domain=None, ratio=None):
        # for domain classification, load all domains
        self.stage = stage
        self.task = task

        # Handle ``all'' domains
        target_domains = self.domains if domain == "all" else [domain]

        if task == 'seq_pair_classif':
            if stage == 'fit':
                suffix = '.train_part'
                self.train = MNLIDataset(self.data_dir, target_domains, self.domain_dict,
                                         self.sentiment_dict, suffix, task, ratio=ratio)
                suffix = '.valid_part'
                self.valid = MNLIDataset(self.data_dir, target_domains, self.domain_dict,
                                         self.sentiment_dict, suffix, task, ratio=ratio)
            elif stage == 'predict':
                suffix = '.dev'
                self.dev = MNLIDataset(self.data_dir, target_domains, self.domain_dict,
                                         self.sentiment_dict, suffix, task, ratio=ratio)

        elif task == 'domain_classif':
            if stage == 'fit':
                self.train_dict = dict()
                self.valid_dict = dict()
                for domain in self.domains:
                    suffix = '.train_part'
                    self.train_dict[domain] = MNLIDataset(self.data_dir,
                                                          domain, self.domain_dict,
                                                          self.sentiment_dict,
                                                          suffix, task, ratio=ratio)
                    suffix = '.valid_part'
                    self.valid_dict[domain] = MNLIDataset(self.data_dir, domain, self.domain_dict,
                                         self.sentiment_dict, suffix, task, ratio=ratio)
            elif stage == 'predict':
                self.dev_dict = dict()
                suffix = '.dev'
                for domain in self.domains:
                    self.dev_dict[domain] = MNLIDataset(self.data_dir, domain, self.domain_dict,
                                         self.sentiment_dict, suffix, task, ratio=ratio)


    def train_dataloader(self):
        assert self.stage == 'fit'
        if self.task == 'seq_pair_classif':
            return DataLoader(self.train,
                              batch_size=self.batch_size,
                              collate_fn=collate_fn,
                              shuffle=True
                              )
        else:
            train_dataloaders = dict()
            for domain in self.domains:
                train_dataloaders[domain] = DataLoader(
                    self.train_dict[domain],
                    batch_size=self.batch_size,
                    collate_fn=collate_fn
                )
            return train_dataloaders


    def val_dataloader(self):
        assert self.stage == 'fit'
        if self.task == 'seq_pair_classif':
            return DataLoader(
                self.valid,
                batch_size=self.batch_size,
                collate_fn=collate_fn
            )
        else:
            valid_dataloaders = dict()
            for domain in self.domains:
                valid_dataloaders[domain] = DataLoader(
                    self.valid_dict[domain],
                    batch_size=self.batch_size,
                    collate_fn=collate_fn
                )
            return valid_dataloaders


    def predict_dataloader(self):
        assert self.stage == 'predict'
        if self.task == 'seq_pair_classif':
            return DataLoader(
                self.dev,
                batch_size=self.batch_size,
                collate_fn=collate_fn
            )
        else:
            dev_dataloaders = dict()
            for domain in self.domains:
                dev_dataloaders[domain] = DataLoader(
                    self.dev_dict[domain],
                    batch_size=self.batch_size,
                    collate_fn=collate_fn
                )
            return dev_dataloaders


class MNLIDataset(Dataset):
    def __init__(self, file_dir, domains, domain_dict, sentiment_dict, suffix, task, ratio=None):
        self.file_dir = file_dir
        self.domains = domains if isinstance(domains, list) else [domains]  # Ensure list

        self.tokenizer = AutoTokenizer.from_pretrained(r'C:\Users\tangc\PycharmProjects\domain-adaptation\bert-base-uncased-cache')
        self.suffix = suffix
        self.task = task
        self.domain_dict = domain_dict
        self.sentiment_dict = sentiment_dict
        self.length = None

        examples = []
        for domain in self.domains:  # Iterate through all specified domains
            filename = f"{domain}{suffix}.tmp"
            file_path = os.path.join(self.file_dir, filename)
            if os.path.exists(file_path):  # Check if file exists
                with open(file_path, "r") as f:
                    json_dict = json.load(f)
                domain_examples = json_dict["data"]
                if ratio is not None:
                    length = math.floor(len(domain_examples) * ratio)
                    domain_examples = domain_examples[:length]
                examples.extend(domain_examples)

        self.examples = examples
        self.length = len(examples)



    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        line = self.examples[idx]
        assert self.task in ["domain_classif", "seq_pair_classif"]
        if self.task == "domain_classif":
            encoding = self.tokenizer(line[0], line[1],
                                        add_special_tokens=True,
                                        padding='max_length',
                                        max_length=self.tokenizer.model_max_length)
            target = self.domain_dict.get(self.domain, -1)
        else:
            # sequence pair classification
            encoding = self.tokenizer(line[0], line[1],
                                    add_special_tokens=True,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length)
            target = self.sentiment_dict.get(line[2], -1)

        #if target == -1:
        #    continue

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            'labels': target,
        }


def mnli_preprocess(file_dir, domains, stage='fit'):
    # dump each domain to temporary files
    # TODO: should I shuffle
    # TODO: tokenization into unicode (see original BERT implementation)
    if stage == 'fit':
        with open(os.path.join(file_dir, "MNLI/MNLI/train.tsv"), "r", encoding="utf-8") as f:
            all_data = f.readlines()
        for domain in domains:
            examples = list()
            for idx, data in enumerate(all_data):
                if idx == 0: # skip header
                    continue
                ex = data.strip().split('\t')
                if ex[3] == domain:
                    examples.append((ex[-4], ex[-3], ex[-1]))
            # Split and extract actual data
            train_subset, valid_subset = random_split(dataset=examples, lengths=[0.7, 0.3])
            train = [examples[i] for i in train_subset.indices]
            valid = [examples[i] for i in valid_subset.indices]

            print("Domain {} | # of train examples {} | valid examples {}".format(
                domain, len(train), len(valid))
            )
            with open(os.path.join(file_dir, domain+".train_part.tmp"), "w") as f:
                json.dump({"domain": domain, "data": train}, f)
            with open(os.path.join(file_dir, domain+".valid_part.tmp"), "w") as f:
                json.dump({"domain": domain, "data": valid}, f)


    elif stage == 'predict':
        ### dev
        with open(os.path.join(file_dir, "MNLI/MNLI/dev_matched.tsv"), "r", encoding="utf-8") as f:
            matched_data = f.readlines()
        with open(os.path.join(file_dir, "MNLI/MNLI/dev_mismatched.tsv"), "r", encoding="utf-8") as f:
            mismatched_data = f.readlines()

        for domain in domains:
            # TODO: check sample processing
            examples = list()
            for dev_data in [matched_data, mismatched_data]:
                for idx, data in enumerate(dev_data):
                    if idx == 0:
                        continue
                    ex = data.strip().split('\t')
                    if ex[3] == domain:
                        examples.append((ex[-8], ex[-7], ex[-1]))
            with open(os.path.join(file_dir, domain + ".dev.tmp"), "w") as f:
                json.dump({"domain": domain, "data": examples}, f)



if __name__ == '__main__':
    # test mnli_preprocess
    #mnli_preprocess(r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets',
    #                ['telephone', 'government', 'travel'])
    # Test dataset
    domain = "telephone"
    task = "seq_pair_classif"
    dataset = MNLIDataset(
        r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets',
        domain,
        {domain: 0},
        {"entailment":0, "neutral":1, "contradiction":2},
        '.train_part',
        task
    )

    # Print first 3 items
    for i, item in enumerate(dataset):
        if i >= 3:
            break
        print(f"Item {i}:")
        print("Input item type: ", type(item))
        print("Input id type: ", type(item["input_ids"]))
        #print(f"  Input IDs: {item['input_ids'].shape}")
        print(f"  Labels: {item['labels']}")

    # Test collate
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    for batch in loader:
        print("Batch:")
        for k, v in batch.items():
            print(f"  {k}: {v.shape}")
        break

    # Test labels
    dataset = MNLIDataset(
        r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets',
        ['telephone'],
        {'telephone': 0},
        {"entailment": 0, "neutral": 1, "contradiction": 2},
        '.train_part',
        'seq_pair_classif'
    )

    for i, item in enumerate(dataset):
        #print(i, item)

        if item['labels'] not in [0, 1, 2]:
            print(f"Bad example at index {i}:")
            print(f"  Label: {item['labels'].item()}")
            print(f"  Content: {item}")
            break






