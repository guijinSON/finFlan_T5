import torch 
from torch.utils.data import Dataset, DataLoader
       
class Seq2SeqDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = self.tgt[idx]

        return {
            'src':src, 'tgt':tgt
        }


class Seq2SeqBatchGenerator:
    def __init__(self, 
                 tokenizer
                 ):
        
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        src = [item['src'] for item in batch]
        tgt = [item['tgt'] for item in batch]

        src_tokenized = self.tokenize(src)
        tgt_tokenized = self.tokenize(tgt)

        return {
            'src_input_ids': src_tokenized.input_ids, 
            'src_attention_mask': src_tokenized.attention_mask,
            'tgt_input_ids': tgt_tokenized.input_ids,
            'tgt_attention_mask': tgt_tokenized.attention_mask,
            }

    def tokenize(self,input_str):
        return  self.tokenizer.batch_encode_plus(input_str, 
                                                    padding='longest', 
                                                    max_length=512,
                                                    truncation=True, 
                                                    return_tensors='pt')


def get_dataloader(dataset, batch_generator, batch_size=4, shuffle=True):
    data_loader = DataLoader(dataset, 
                              batch_size=batch_size, 
                              shuffle=shuffle, 
                              collate_fn=batch_generator,
                              drop_last=True,
                              num_workers=4)
    return data_loader
