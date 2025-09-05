import os, math
import json, random
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from transformers import AutoTokenizer


def input_handler(batch, device='cuda'):
    batch_x = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    labels = batch['labels'].to(device)
    return batch_x, labels


def collate_fn(batch):
    """Custom collate function to handle batching of dictionary items"""
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    token_type_ids = torch.stack([torch.tensor(item['token_type_ids']) for item in batch])
    indices = [item['idx'] for item in batch]  # Collect indices

    # Handle labeled vs unlabeled data
    if 'labels' in batch[0]:
        labels = torch.stack([torch.tensor(item['labels']) for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
            'indices': indices,
        }
    else:
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'indices': indices
        }


class DomainSampler(Sampler):
    def __init__(self, dataset, domain_probs, epoch_length=None, seed=None):
        self.dataset = dataset
        self.domain_probs = domain_probs
        self.epoch_length = epoch_length if epoch_length is not None else len(dataset)
        self.seed = seed

        # Group indices by domain
        self.domain_to_indices = {}
        for idx in range(len(dataset)):
            domain_idx = dataset.domain_indices[idx]
            if domain_idx not in self.domain_to_indices:
                self.domain_to_indices[domain_idx] = []
            self.domain_to_indices[domain_idx].append(idx)

    def __iter__(self):
        # Create deterministic RNG if seed is set
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)

        indices = []
        for _ in range(self.epoch_length):
            # Sample domain using PyTorch RNG
            domain_idx = torch.multinomial(
                torch.tensor(self.domain_probs),
                1,
                generator=rng
            ).item()

            # Sample random index from domain
            domain_indices = self.domain_to_indices[domain_idx]
            idx_in_domain = torch.randint(
                0,
                len(domain_indices),
                (1,),
                generator=rng
            ).item()

            indices.append(domain_indices[idx_in_domain])
        return iter(indices)

    def __len__(self):
        return self.epoch_length


class MNLIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, domains, sentiments, val_size=1000):
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
        self.sampler_seed = None

        # Initialize lambda attributes
        self.train_sampler_lambda = None
        self.val_sampler_lambda = None
        self.predict_sampler_lambda = None

        self.val_size = val_size

    def set_sampler_seed(self, seed):
        """Set seed for reproducible sampling"""
        self.sampler_seed = seed


    def set_sampler_lambda(self, lambda_vector, loader_type='train', size=None):
        if loader_type == 'train':
            self.train_sampler_lambda = lambda_vector
        elif loader_type == 'val':
            self.val_sampler_lambda = lambda_vector
            # Store custom validation size if provided
            if size is not None:
                self.val_size = size
        elif loader_type == 'predict':
            self.predict_sampler_lambda = lambda_vector
        else:
            raise ValueError("loader_type must be 'train', 'val', or 'predict'")


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
                # Validation set (currently only dev_matched)
                self.valid = MNLIDataset(
                    self.data_dir, target_domains, self.domain_dict,
                    self.sentiment_dict, ['dev_matched'], task
                )
                print(f'Setup stage {stage} | train size {len(self.train)} | validation size {len(self.valid)}')

            elif stage == 'predict':  # Unlabeled data
                self.predict = MNLIDataset(
                    self.data_dir, target_domains, self.domain_dict,
                    self.sentiment_dict, ['test_matched'], task, has_labels=False
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
                self.valid = MNLIDataset(
                    self.data_dir, target_domains, self.domain_dict,
                    self.sentiment_dict, ['dev_matched'], task
                )
                print(f'Domain Classif: stage {stage} | train size {len(self.train)} | valid size {len(self.valid)}')

            elif stage == 'predict':  # Test sets
                self.predict = MNLIDataset(
                    self.data_dir, target_domains, self.domain_dict,
                    self.sentiment_dict, ['test_matched'], task, has_labels=False
                )
                print(f'Domain Classif: stage {stage} | predict size {len(self.predict)}')


    def _create_dataloader(self, dataset, sampler_lambda, is_training=False):
        if sampler_lambda is not None:
            # Validate lambda length matches number of target domains
            if len(sampler_lambda) != len(self.domains):
                raise ValueError(f"lambda length ({len(sampler_lambda)}) must match number of domains ({len(self.domains)})")

            epoch_length = self.val_size if not is_training else None
            sampler = DomainSampler(
                dataset,
                sampler_lambda,
                epoch_length=epoch_length,
                seed=self.sampler_seed
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
                #generator=torch.Generator().manual_seed(self.sampler_seed) if self.sampler_seed else None
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                shuffle=is_training,
                generator=torch.Generator().manual_seed(self.sampler_seed) if self.sampler_seed else None
            )


    # Then in the specific dataloader methods:
    def train_dataloader(self):
        assert self.stage == 'fit'
        return self._create_dataloader(
            self.train,
            self.train_sampler_lambda,
            is_training=True
        )

    def val_dataloader(self):
        assert self.stage == 'fit'
        return self._create_dataloader(
            self.valid,
            self.val_sampler_lambda,
            is_training=False
        )

    def predict_dataloader(self):
        assert self.stage == 'predict'
        return self._create_dataloader(
            self.predict,
            self.predict_sampler_lambda,
            is_training=False
        )


    def get_dataloader(self, domain, loader_type='train', is_training=False):
        """
        Get DataLoader for a specific domain or all domains (if domain == 'all') and loader type ('train', 'val', or 'predict').

        Args:
            domain (str): The domain to load data for (or 'all' to load all domains).
            loader_type (str): The type of DataLoader ('train', 'val', or 'predict').

        Returns:
            DataLoader: A DataLoader for the specified domain and loader type.
        """
        assert loader_type in ['train', 'val', 'predict'], "loader_type must be 'train', 'val', or 'predict'"

        # Determine target domains based on the 'domain' argument
        target_domains = self.domains if domain == 'all' else [domain]

        # Print which domains are being loaded
        print(f'Loading datasets for target domains: {target_domains}')

        # Load the correct dataset based on the loader type
        if loader_type == 'train':
            dataset = MNLIDataset(
                self.data_dir, target_domains, self.domain_dict,
                self.sentiment_dict, 'train', 'seq_pair_classif'
            )
        elif loader_type == 'val':
            dataset = MNLIDataset(
                self.data_dir, target_domains, self.domain_dict,
                self.sentiment_dict, 'dev_matched', 'seq_pair_classif'
            )
        else:  # loader_type == 'predict'
            dataset = MNLIDataset(
                self.data_dir, target_domains, self.domain_dict,
                self.sentiment_dict, 'test_matched', 'seq_pair_classif', has_labels=False
            )

        # Step 2: Set the sampler lambda for the domain if needed (only for 'train' and 'all' domain)
        if loader_type == 'train' and self.train_sampler_lambda is not None and domain == 'all':
            sampler_lambda = self.train_sampler_lambda
        elif loader_type == 'val' and self.val_sampler_lambda is not None:
            sampler_lambda = self.val_sampler_lambda
        else:
            sampler_lambda = None  # No sampling for prediction or default behavior

        # Step 3: Create and return the appropriate DataLoader
        return self._create_dataloader(dataset, sampler_lambda, is_training=is_training)


class MNLIDataset(Dataset):
    def __init__(self, file_dir, domains, domain_dict, sentiment_dict,
                 suffixes, task, ratio=None, has_labels=True):
        self.file_dir = file_dir
        self.domains = domains if isinstance(domains, list) else [domains]
        self.tokenizer = AutoTokenizer.from_pretrained(
            r'C:\Users\tangc\PycharmProjects\domain-adaptation\bert-base-uncased-cache'
        )
        self.suffixes = [suffixes] if isinstance(suffixes, str) else suffixes
        self.task = task
        self.domain_dict = domain_dict
        self.sentiment_dict = sentiment_dict
        self.has_labels = has_labels
        self.examples = []
        self.domain_indices = []  # Store local domain indices
        self.length = 0

        # Create local domain mapping for the current target domains
        self.local_domain_dict = {d: idx for idx, d in enumerate(self.domains)}

        for domain in self.domains:
            for suffix in self.suffixes:
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
                            if len(example) == 3:  # Only if label exists
                                example[-1] = domain

                    # Add examples and their local domain indices
                    local_domain_idx = self.local_domain_dict[domain]
                    self.examples.extend(domain_examples)
                    self.domain_indices.extend([local_domain_idx] * len(domain_examples))

        self.length = len(self.examples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        line = self.examples[idx]

        # Handle unlabeled test data (only 2 elements)
        if len(line) == 2:
            premise, hypothesis = line
            label = None
        else:
            premise, hypothesis, label = line

        encoding = self.tokenizer(
            premise, hypothesis,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True
        )
        item = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            "idx": idx
        }
        if self.task == "domain_classif":
            if self.has_labels and label is not None:
                item['labels'] = self.domain_dict.get(label, -1)
        else:  # seq_pair_classif
            if self.has_labels and label is not None:
                item['labels'] = self.sentiment_dict.get(label, -1)
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

    # Initialize datamodule
    domains = ["telephone", "slate", "government", "fiction", "travel"]
    dm = MNLIDataModule(file_dir, domains, sentiments)
    # Set seed for reproducibility
    dm.set_sampler_seed(42)

    # First run
    dm.setup('fit', domain="all")
    loader1 = dm.train_dataloader()

    # Second run (same seed)
    dm.setup('fit', domain="all")  # Re-initialize dataset
    loader2 = dm.train_dataloader()

    # Batches will be in identical order
    count = 0
    for b1, b2 in zip(loader1, loader2):
        if count >=10:
            break
        assert torch.equal(b1['input_ids'], b2['input_ids'])
        count += 1

    # Initialize with default validation size
    dm = MNLIDataModule(file_dir, domains, sentiments, val_size=500)

    # Or set custom lambda and size later
    dm.set_sampler_lambda(
        lambda_vector=[0.3, 0.3, 0.1, 0.1, 0.2],
        loader_type='val',
        size=300  # Override default size
    )

    # Setup and validation will now use 300 samples
    dm.setup('fit', domain="all")
    val_loader = dm.val_dataloader()
    count = 0
    for _ in val_loader:
        print(count)
        count += 1
