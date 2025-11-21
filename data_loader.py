import os, math
import json, random
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, random_split
from transformers import AutoTokenizer


# partly coded by DeepSeek

#TODO: train with DomainSampler (multiple epochs), val_size (?)

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


def collate_fn_digits(batch):
    """
        Ensure all images are 3x32x32.
        batch: list of (image, label)
    """
    imgs = [item["inputs"] for item in batch]
    labels = [item["labels"] for item in batch]
    new_imgs = []
    for img in imgs:
        # Ensure 3 channels
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)  # replicate grayscale -> RGB

        # Ensure 32x32 size
        if img.size(1) != 32 or img.size(2) != 32:
            img = F.interpolate(img.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False)
            img = img.squeeze(0)

        new_imgs.append(img)

    imgs = torch.stack(new_imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels


def get_stat(dataset):
    X, labels = list(), list()
    for idx in range(len(dataset)):
        X.append(dataset[idx][0])
        labels.append(dataset[idx][1])
    X = torch.stack(X)
    d_mean, d_std = list(), list()

    for c in range(X.size()[1]):
        d_mean.append(torch.mean(X[:, c, :, :]))
        d_std.append(torch.std(X[:, c, :, :]))

    return torch.tensor(d_mean), torch.tensor(d_std)


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, domain_idx, task="digit_classif", preprocess=False):
        self.dataset = dataset
        self.domain_idx = domain_idx
        self.task = task
        self.preprocess_flag = preprocess

        if preprocess:
            self.d_mean, self.d_std = get_stat(dataset)
            # ---- SVHN normalization parameters ----
            # Precomputed from full SVHN train set
            # svhn_mean = (0.4377, 0.4438, 0.4728)
            # svhn_std = (0.1980, 0.2010, 0.1970)
            print(f"Dataset statistics: mean {self.d_mean} std {self.d_std}")


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        # perform a shift and scale operation
        if self.preprocess_flag:
            x = (x - self.d_mean.view(-1, 1, 1)) / self.d_std.view(-1, 1, 1)

        if self.task == "digit_classif":
            return {
                    "inputs": x,
                    "labels": torch.tensor(y, dtype=torch.long),
                    "domain_index": torch.tensor(self.domain_idx)
                    }

        elif self.task == "domain_classif":
            return {
                    "inputs": x,
                    "labels": torch.tensor(self.domain_idx, dtype=torch.long),
                    "domain_index": torch.tensor(self.domain_idx)
                    }
        else:
            raise ValueError(f"Unknown task: {self.task}")


class DomainSampler(Sampler):
    def __init__(self, dataset, domain_probs, max_samples=None, seed=None):
        self.dataset = dataset
        self.domain_probs = domain_probs
        self.epoch_length = max_samples if max_samples is not None else len(dataset)
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


class DigitDataModule(pl.LightningDataModule):
    def __init__(self, val_size=1000):
        super().__init__()
        self.domains = ["svhn", "mnist", "usps"]
        self.train_sampler_lambda = None
        self.val_sampler_lambda = None
        self.predict_sampler_lambda = None


    def set_sampler_config(self, lambda_vector, loader_type='train', size=None):
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


    def prepare_data(self):
        self.data_dict = dict()

        self.data_dict['svhn:nt'] = torchvision.datasets.SVHN("./", 'train', download=True)
        self.data_dict['svhn:nte'] = torchvision.datasets.SVHN("./", 'extra', download=True)
        self.data_dict['svhn:t'] = torchvision.datasets.SVHN("./", 'test', download=True)
        self.data_dict['mnist:nt'] = torchvision.datasets.MNIST("./", True, download=True)
        self.data_dict['mnist:t'] = torchvision.datasets.MNIST("./", False, download=True)
        self.data_dict['usps:nt'] = torchvision.datasets.USPS("./", True, download=True)
        self.data_dict['usps:t'] = torchvision.datasets.USPS("./", False, download=True)

        # Get statistics
        curr_domain = None
        data_statistics = dict()
        for dataname in self.data_dict.keys():
            domain_name = dataname.split(':')[0]
            print(f"Processing {domain_name}")
            dataset = self.data_dict[dataname]
            if curr_domain is None or curr_domain != domain_name:
                if curr_domain is not None:
                    if f"{curr_domain}:size" not in data_statistics:
                        data_statistics[f"{curr_domain}:size"] = 0

                    if f"{curr_domain}:labels" not in data_statistics:
                        data_statistics[f"{curr_domain}:labels"] = set()

                    data_statistics[f"{curr_domain}:size"] = num_examples
                    data_statistics[f"{curr_domain}:labels"] = labels_set

                curr_domain = domain_name
                num_examples = 0
                labels_set = set()

            for idx in range(len(dataset)):
                num_examples += 1
                labels_set.add(dataset[idx][1])

        data_statistics[f"{curr_domain}:size"] = num_examples
        data_statistics[f"{curr_domain}:labels"] = labels_set

        print("============== Dataset statistics ===============")
        for key, value in data_statistics.items():
            print(key)
            print(value)


    def setup(self, stage, task='digit_classif', domain=None, val_ratio=0.2):
        self.stage = stage
        self.task = task
        assert domain is not None, "Please provide a domain(s)"

        target_domains = self.domains if domain == "all" else domain.split("-")
        for domain in target_domains:
            print(f"Setup dataset, target domain: {domain}")

        transform = torchvision.transforms.ToTensor()
        domain_datasets = {}

        for d in target_domains:
            domain_idx = self.domains.index(d)

            # load raw datasets
            if d == "svhn":
                full_train = DigitDataset(
                    torchvision.datasets.SVHN("./", split="train", transform=transform),
                    domain_idx,
                    preprocess=True,
                )
                # extra_train = DigitDataset(
                #         torchvision.datasets.SVHN("./", split="extra", transform=transform),
                #         domain_idx,
                #         preprocess=True,
                #     )
                test = DigitDataset(
                    torchvision.datasets.SVHN("./", split="test", transform=transform),
                    domain_idx,
                    preprocess=True
                )

            elif d == "mnist":
                full_train = DigitDataset(
                    torchvision.datasets.MNIST("./", train=True, transform=transform),
                    domain_idx
                )
                test = DigitDataset(
                    torchvision.datasets.MNIST("./", train=False, transform=transform),
                    domain_idx
                )

            elif d == "usps":
                full_train = DigitDataset(
                    torchvision.datasets.USPS("./", train=True, transform=transform),
                    domain_idx
                )
                test = DigitDataset(
                    torchvision.datasets.USPS("./", train=False, transform=transform),
                    domain_idx
                )

            # split into train and validation
            n_val = int(len(full_train) * val_ratio)
            n_train = len(full_train) - n_val
            train, val = random_split(full_train, [n_train, n_val])

            # if d == "svhn":
            #     #TODO: Currently train and extra train are preprocessed separately
            #     train = ConcatDataset([train, extra_train])

            # store separately for each domain
            domain_datasets[f"{d}:train"] = train
            domain_datasets[f"{d}:val"] = val
            domain_datasets[f"{d}:test"] = test

        for k in domain_datasets.keys():
            print(f"Processed domain {k.split(':')[0]}, type {k.split(':')[1]}, size {len(domain_datasets[k])}")

        self.datasets = domain_datasets

        if stage == "fit":
            self.train = ConcatDataset([domain_datasets[f"{d}:train"] for d in target_domains])
            self.valid = ConcatDataset([domain_datasets[f"{d}:val"] for d in target_domains])
            print(f"Task {task} | stage {stage} | "
                  f"train size {len(self.train)} | validation size {len(self.valid)}")

        elif stage == "predict":
            self.predict = ConcatDataset([domain_datasets[f"{d}:test"] for d in target_domains])
            print(f"Task {task} | stage {stage} | predict size {len(self.predict)}")


    def _create_dataloader(self, dataset, sampler_lambda, is_training=False, seed=None, max_samples=None):
        if sampler_lambda is not None:
            sampler = DomainSampler(dataset, sampler_lambda, max_samples=max_samples, seed=seed)
            return DataLoader(dataset, collate_fn=collate_fn_digits,
                              batch_size=self.batch_size, sampler=sampler)
        else:
            return DataLoader(dataset, collate_fn=collate_fn_digits,
                              batch_size=self.batch_size, shuffle=is_training)


    def train_dataloader(self, seed=None, max_samples=None):
        return self._create_dataloader(
            self.train,
            self.train_sampler_lambda,
            is_training=True,
            seed=seed,
            max_samples=max_samples
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.valid,
            self.val_sampler_lambda,
            is_training=False
        )

    def predict_dataloader(self):
        return self._create_dataloader(
            self.predict,
            self.predict_sampler_lambda,
            is_training=False
        )

    def get_dataloader(self, domain, loader_type='train', seed=None, is_training=False):
        assert loader_type in ['train', 'val', 'test']
        key = f"{domain}:{loader_type}"
        dataset = self.datasets[key]
        sampler_lambda = None
        if loader_type == 'train':
            sampler_lambda = self.train_sampler_lambda
        elif loader_type == 'val':
            sampler_lambda = self.val_sampler_lambda
        elif loader_type == 'predict':
            sampler_lambda = self.predict_sampler_lambda
        return self._create_dataloader(dataset, sampler_lambda, seed=seed, is_training=is_training)


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

        # Initialize lambda attributes
        self.train_sampler_lambda = None
        self.val_sampler_lambda = None
        self.predict_sampler_lambda = None

        self.val_size = val_size


    def set_sampler_config(self, lambda_vector, loader_type='train', size=None):
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
        # TODO: What is this ?
        self.predict_suffix = suffix

    def prepare_data(self):
        mnli_preprocess(self.data_dir, self.domains)

    def setup(self, stage, task='seq_pair_classif', domain=None):
        self.stage = stage
        self.task = task
        assert domain is not None, "Please provide a domain(s)"
        target_domains = self.domains if domain == "all" else domain.split("-")
        print(f'Setup datasets, target domains: {target_domains}')
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
            print(f'Task {task} | Setup stage {stage} | train size {len(self.train)} | validation size {len(self.valid)}')

        elif stage == 'predict':  # Unlabeled data
            self.predict = MNLIDataset(
                self.data_dir, target_domains, self.domain_dict,
                self.sentiment_dict, ['test_matched'], task, has_labels=False
            )
            print(f'Task {task} | Setup stage {stage} | predict size {len(self.predict)}')


    def _create_dataloader(self, dataset, sampler_lambda, is_training=False, seed=None, max_samples=None):
        if sampler_lambda is not None:
            # Validate lambda length matches number of processed domains in dataset
            if len(sampler_lambda) != len(dataset.domains):
                raise ValueError(f"lambda length ({len(sampler_lambda)}) must match number of domains ({len(dataset.domains)}) in dataset")

            sampler = DomainSampler(
                dataset,
                sampler_lambda,
                max_samples=max_samples,
                seed=seed
            )
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=collate_fn
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                shuffle=is_training,
                generator=torch.Generator().manual_seed(seed) if seed is not None else None
            )


    # Then in the specific dataloader methods:
    def train_dataloader(self, seed=None, max_samples=None):
        assert self.stage == 'fit'
        return self._create_dataloader(
            self.train,
            self.train_sampler_lambda,
            is_training=True,
            seed=seed,
            max_samples=max_samples
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

        # Determine target domains based on the 'domain' argument (could be a subset of available domains)
        target_domains = self.domains if domain == 'all' else domain.split("-")

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
        self.suffixes = [suffixes] if isinstance(suffixes, str) else suffixes #TODO
        self.task = task
        self.domain_dict = domain_dict
        self.sentiment_dict = sentiment_dict
        self.has_labels = has_labels
        self.examples = []
        self.domain_indices = []  # Store local domain indices
        self.length = 0

        # Create local domain mapping for the current target domains in this dataset instance
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
                        domain_examples_new = list()
                        for idx, example in enumerate(domain_examples):
                            new_example = (example[0], example[1], domain)
                            domain_examples_new.append(new_example)
                        domain_examples = domain_examples_new

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
            if label is not None:
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
                train_examples.append((ex[-4], ex[-3], ex[-1])) # Sentence A, Sentence B, label

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
                # store labeled data by domain
                json.dump({"domain": domain, "data": test_examples}, f)



if __name__ == '__main__':
    # # Test the updated functionality
    # # TODO: add tests for train, dev, and predict
    # domains = ['telephone', 'oup']
    # sentiments = ["entailment", "contradiction", "neutral"]
    # file_dir = r'C:\Users\tangc\PycharmProjects\domain-adaptation\datasets'
    #
    # mnli_preprocess(file_dir, domains)
    #
    # # Test training set (should have labels)
    # train_dataset = MNLIDataset(
    #     file_dir,
    #     ['telephone'],
    #     {'telephone': 0},
    #     {"entailment": 0, "neutral": 1, "contradiction": 2},
    #     'train',
    #     'seq_pair_classif'
    # )
    # print(f"Training set size: {len(train_dataset)}")
    # print("First training item:", train_dataset[0])
    #
    # # Test validation set
    # val_dataset = MNLIDataset(
    #     file_dir,
    #     ['oup'],
    #     {'oup': 0},
    #     {"entailment": 0, "neutral": 1, "contradiction": 2},
    #     'dev_mismatched',
    #     'seq_pair_classif'
    # )
    # print(f"Validation set size: {len(val_dataset)}")
    # print("First validation item:", val_dataset[0])
    #
    # # Test predict set (unlabeled)
    # predict_dataset_m = MNLIDataset(
    #     file_dir,
    #     domains,
    #     {'telephone': 0, 'oup':1},
    #     {"entailment": 0, "neutral": 1, "contradiction": 2},
    #     'test_matched',
    #     'seq_pair_classif',
    #     has_labels=False
    # )
    #
    # predict_dataset_mm = MNLIDataset(
    #     file_dir,
    #     domains,
    #     {'telephone': 0, 'oup':1},
    #     {"entailment": 0, "neutral": 1, "contradiction": 2},
    #     'test_mismatched',
    #     'seq_pair_classif',
    #     has_labels=False
    # )
    #
    # predict_dataset = ConcatDataset([predict_dataset_m, predict_dataset_mm])
    # print(f"Test set size: {len(predict_dataset)}")
    # print("First test item:", predict_dataset[0])
    #
    # # Initialize datamodule
    # domains = ["telephone", "slate", "government", "fiction", "travel"]
    # dm = MNLIDataModule(file_dir, domains, sentiments)
    #
    # # First run
    # dm.setup('fit', domain="all")
    # loader1 = dm.train_dataloader()
    #
    # # Second run (same seed)
    # dm.setup('fit', domain="all")  # Re-initialize dataset
    # loader2 = dm.train_dataloader()
    #
    # # Batches will be in identical order
    # count = 0
    # for b1, b2 in zip(loader1, loader2):
    #     if count >=10:
    #         break
    #     assert torch.equal(b1['input_ids'], b2['input_ids'])
    #     count += 1
    #
    # # Initialize with default validation size
    # dm = MNLIDataModule(file_dir, domains, sentiments, val_size=500)
    #
    # # Or set custom lambda and size later
    # dm.set_sampler_lambda(
    #     lambda_vector=[0.3, 0.3, 0.1, 0.1, 0.2],
    #     loader_type='val',
    #     size=300  # Override default size
    # )
    #
    # # Setup and validation will now use 300 samples
    # dm.setup('fit', domain="all")
    # val_loader = dm.val_dataloader()
    # count = 0
    # for _ in val_loader:
    #     print(count)
    #     count += 1

    ddm = DigitDataModule()
    ddm.prepare_data()
    ddm.reset_batch_size(8)
    ddm.setup('fit', task='digit_classif', domain="all", val_ratio=0.2)
    dataloader = ddm.get_dataloader("svhn", loader_type='train', is_training=True)
    print(f"batch size {ddm.batch_size}, number of batches {len(dataloader)}")
    print("A sample of batch")
    for idx, batch in enumerate(dataloader):
        if idx > 0:
            break
        print(batch[0].size())
