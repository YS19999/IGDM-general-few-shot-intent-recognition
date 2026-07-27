import datetime
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def tprint(s):
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), s),
        flush=True)


def to_tensor(data, cuda):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():

        data[key] = torch.from_numpy(data[key])
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def _get_datapath_metatest(args):
    if args.shot == 0:
        test_path = f"../intent_data/test/{args.dataset}/{args.dataset}_test.json"
        return test_path
    else:
        train_path = f"../intent_data/test/{args.dataset}/{args.dataset}_{args.shot}_shot_train.json"
        val_path = f"../intent_data/test/{args.dataset}/{args.dataset}_val.json"
        test_path = f"../intent_data/test/{args.dataset}/{args.dataset}_test.json"

        return train_path, val_path, test_path


def _get_datapath_metatrain(args):
    train_path = f"../intent_data/train/train.json"
    val_path = f"../intent_data/train/val.json"
    test_path = f"../intent_data/train/test.json"

    return train_path, val_path, test_path


def _load_data(datapath):
    label = {}
    text_len = []
    with open(datapath, 'r', encoding='utf-8', errors='ignore') as f:
        dataset = json.load(f)
        data = []
        for row in dataset["data"]:
            if row['intent'] not in label:
                label[row['intent']] = 1
            else:
                label[row['intent']] += 1

            item = {
                'intent': row['intent'],
                'text': row['text']
            }
            text_len.append(len(row['text'].split(" ")))
            data.append(item)

        tprint('Class balance:')
        print(label)
        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data


class TextDataset(Dataset):
    def __init__(self, dataset):
        super(TextDataset, self).__init__()

        self.data = []
        self.label = []

        for line in dataset:
            self.data.append(line["text"])
            if "_" in line["intent"]:
                la = " ".join(line["intent"].split("_")).lower()
            else:
                la = line["intent"].lower()
            self.label.append(la)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]
        label = self.label[item]

        return text, label

instructions = [
    "Predict the intent of the input query. Intent is the main topic or purpose of a query.\nYou need to select the most suitable intent from: ",
    "Identify the key purpose of the input question. Intent is defined as the main goal behind the query.\nChoose the best matching intent from: ",
    "Classify the intent of the given utterance. Intent signifies the central theme or aim.\nPick the correct intent category from: ",
    "Assess the main topic of the query to derive its intent. Intent is the fundamental purpose.\nChoose from the following intents: ",
    "Extract the intent from the input text. Intent is the essence of the query.\nIdentify and select the relevant intent from: ",
    "Recognize the intent behind the user's input. Intent indicates the key subject or objective.\nSelect the appropriate intent from: ",
    "Infer the intent of the query based on its content. Intent is the main focus.\nPick the best intent option from: ",
    "Categorize the query into its intent type. Intent is the primary category representing the purpose.\nChoose from the intent list: ",
    "Analyze the query to find its core intent. Intent is the driving force behind the request.\nSelect from the provided intents: ",
    "Detect the intent of the user's request. Intent is the central idea.\nOpt for the most fitting intent from: ",
]

class BatchTextCall_Train(object):
    def __init__(self, args, tokenizer, support_data):
        self.args = args
        self.tokenizer = tokenizer
        self.support_data = support_data

    def train_instruction(self, text, label, la_strs, label2id):

        new_text = []
        new_label = []
        text_instructions = []
        # 构造少样本支持和查询数据
        # 构建训练数据: 指令 + 候选集 + 支持集 (shot) + 查询集
        for (te, la) in zip(text, label):

            random.shuffle(la_strs)
            la_strs1 = ", ".join(la_strs)

            spt_data = random.sample(self.support_data[la], k=self.args.shot)
            support = []
            for te in spt_data:
                support.append(f"Support: {te}\nIntent: {label2id[la]}")
            support = "\n".join(support)

            tem_instructing = random.choice(instructions)
            instruct_text = tem_instructing + la_strs1 + "\n" + support + "\n"
            text_instructions.append({"text": instruct_text})

            query_text = f"Query: {te}" + "\nIntent: "
            new_text.append({"text": query_text})
            new_label.append({"text": label2id[la]})

        return new_text, new_label, text_instructions

    def tokenization(self, data):

        for e in data:
            tokens = self.tokenizer(e["text"], return_tensors='pt')
            e['input_ids'], e['attention_mask'] = tokens['input_ids'][0].numpy(), tokens['attention_mask'][0].numpy()

        text_len = np.array([len(e['input_ids']) for e in data])
        max_text_len = max(text_len)

        text = np.zeros([len(data), max_text_len], dtype=np.int64)
        text_mask = np.zeros([len(data), max_text_len], dtype=np.int64)

        for i in range(len(data)):
            text[i, :len(data[i]['input_ids'])] = data[i]['input_ids']
            text_mask[i, :len(data[i]['attention_mask'])] = data[i]['attention_mask']

        new_data = {
            'input_ids': text,
            'attention_mask': text_mask
        }

        return new_data

    def __call__(self, batch):

        batch_text = [item[0] for item in batch]
        batch_label = [item[1] for item in batch]
        # 标签序列化处理
        label2id = {}
        id2label = {}
        unique_label = []
        for la in batch_label:
            if la not in unique_label:
                unique_label.append(la)
        random.shuffle(unique_label)
        la_strs = []
        for i, la in enumerate(unique_label):
            label2id[la] = str(i + 1)
            id2label[str(i + 1)] = la
            la_strs.append(f"{i + 1}='{la}'")

        new_text, new_label, text_instructions = self.train_instruction(batch_text, batch_label, la_strs, label2id)

        new_text = self.tokenization(new_text)
        new_text = to_tensor(new_text, self.args.cuda)
        new_label = self.tokenization(new_label)
        new_label = to_tensor(new_label, self.args.cuda)
        text_instructions = self.tokenization(text_instructions)
        text_instructions = to_tensor(text_instructions, self.args.cuda)

        return new_text, text_instructions, new_label, id2label


class BatchTextCall_Test(object):
    def __init__(self, args, tokenizer, support_data):
        self.args = args
        self.tokenizer = tokenizer
        self.support_data = support_data

    def test_instruction(self, unknown_text, unknown_label, la_strs, label2id):

        # 构建少样本测试
        quy_data = []
        quy_label = []
        for (te, la) in zip(unknown_text, unknown_label):

            random.shuffle(la_strs)
            la_strs1 = ", ".join(la_strs)

            quy_label.append({"text": label2id[la]})

            instructing = random.choice(instructions)
            if self.args.shot == 0:
                instruct_text = instructing + la_strs1 + "\n" + f"Query: {te}\nIntent: "
            else:
                spt_data = random.sample(self.support_data[la], k=self.args.shot)
                support = [f"Support: {st}\nIntent: {label2id[la]}" for st in spt_data]
                support = "\n".join(support)
                instruct_text = instructing + la_strs1 + "\n" + support + "\n" + f"Query: {te}\nIntent: "
            quy_data.append({"text": instruct_text})

        return quy_data, quy_label

    def tokenization(self, data):

        for e in data:
            tokens = self.tokenizer(e["text"], return_tensors='pt')
            e['input_ids'], e['attention_mask'] = tokens['input_ids'][0].numpy(), tokens['attention_mask'][0].numpy()

        text_len = np.array([len(e['input_ids']) for e in data])
        max_text_len = max(text_len)

        text = np.zeros([len(data), max_text_len], dtype=np.int64)
        text_mask = np.zeros([len(data), max_text_len], dtype=np.int64)

        for i in range(len(data)):
            text[i, :len(data[i]['input_ids'])] = data[i]['input_ids']
            text_mask[i, :len(data[i]['attention_mask'])] = data[i]['attention_mask']

        new_data = {
            'input_ids': text,
            'attention_mask': text_mask
        }

        return new_data

    def __call__(self, batch):

            batch_text = [item[0] for item in batch]
            batch_label = [item[1] for item in batch]
            # 标签序列化处理
            label2id = {}
            id2label = {}
            unique_label = []
            for la in batch_label:
                if la not in unique_label:
                    unique_label.append(la)
            random.shuffle(unique_label)
            la_strs = []
            for i, la in enumerate(unique_label):
                label2id[la] = str(i + 1)
                id2label[str(i + 1)] = la
                la_strs.append(f"{i + 1}='{la}'")

            quy_data, quy_label = self.test_instruction(batch_text, batch_label, la_strs, label2id)

            quy_data = self.tokenization(quy_data)
            quy_data = to_tensor(quy_data, self.args.cuda)
            quy_label = self.tokenization(quy_label)
            quy_label = to_tensor(quy_label, self.args.cuda)

            return quy_data, quy_label, id2label


def _get_dataloader(args, tokenizer, meta_train=False):

    if meta_train:

        train_path, val_path, test_path = _get_datapath_metatrain(args)
        train_data = _load_data(train_path)
        val_data = _load_data(val_path)
        test_data = _load_data(test_path)
        tprint('#train {} #valid {} #test {}'.format(len(train_data), len(val_data), len(test_data)))

        meta_data = train_data + val_data
        data_dict = {}
        for raw in meta_data:
            la = " ".join(raw["intent"].split("_")).lower()
            if la not in data_dict:
                data_dict[la] = [raw["text"]]
            else:
                data_dict[la].append(raw["text"])
        support_data = {}
        query_data = []
        for key, item in data_dict.items():
            random.shuffle(item)
            support_data[key] = item[:args.shot]
            for sent in item[args.shot:]:
                query_data.append({"text": sent, "intent": key})

        train_dataset = TextDataset(query_data)
        val_dataset = TextDataset(val_data)
        test_dataset = TextDataset(test_data)

        text_dataset_call = BatchTextCall_Train(args, tokenizer, support_data)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                      collate_fn=text_dataset_call)
        val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
                                    collate_fn=text_dataset_call)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                     collate_fn=text_dataset_call)

        return train_dataloader, val_dataloader, test_dataloader

    else:

        if args.shot == 0:
            test_path = _get_datapath_metatest(args)
            test_data = _load_data(test_path)
            tprint('#query {}'.format(len(test_data)))
            test_dataset = TextDataset(test_data)

            text_dataset_call_test = BatchTextCall_Test(args, tokenizer, None)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                         collate_fn=text_dataset_call_test)
        else:
            train_path, val_path, test_path = _get_datapath_metatest(args)
            train_data = _load_data(train_path)
            test_data = _load_data(test_path)
            tprint('#support {}  #query {}'.format(len(train_data), len(test_data)))

            support_data = {}
            for raw in train_data:
                raw["intent"] = " ".join(raw["intent"].split("_")).lower()
                if raw["intent"] not in support_data:
                    support_data[raw["intent"]] = [raw["text"]]
                else:
                    support_data[raw["intent"]].append(raw["text"])

            test_dataset = TextDataset(test_data)

            text_dataset_call_test = BatchTextCall_Test(args, tokenizer, support_data)
            test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                         collate_fn=text_dataset_call_test)
        return test_dataloader
