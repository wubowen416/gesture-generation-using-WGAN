import os
import codecs
import json
import numpy as np
import torch

from transformers import BertJapaneseTokenizer, BertModel
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")


def process_word_vec(words):
    """
    Process recongized words as dict to bert embeddings
    Args:
        words - list of dict {'word', 'start_time', 'end_time'}
    Returns:
        features - embedded result mathcing the length of word_list
    """
    num_words = [] # num of words in each recognized chunk
    tokens = {
        'input_ids': torch.LongTensor([[2]]), # 2 - 'sos'
        'token_type_ids': torch.LongTensor([[0]]),
        'attention_mask': torch.LongTensor([[1]])
    }

    for i, word in enumerate(words):

        bert_tokens = tokenizer(word['word'], return_tensors='pt')

        tokens['input_ids'] = torch.cat([tokens['input_ids'], bert_tokens['input_ids'][:, 1:-1]], dim=-1) 
        tokens['token_type_ids'] = torch.cat([tokens['token_type_ids'], bert_tokens['token_type_ids'][:, 1:-1]], dim=-1) 
        tokens['attention_mask'] = torch.cat([tokens['attention_mask'], bert_tokens['attention_mask'][:, 1:-1]], dim=-1) 

        num_words.append(bert_tokens['input_ids'].size(-1) - 2)

    tokens['input_ids'] = torch.cat([tokens['input_ids'], torch.LongTensor([[3]])], dim=-1) # 3 - 'eof'
    tokens['token_type_ids'] = torch.cat([tokens['token_type_ids'], torch.LongTensor([[0]])], dim=-1)
    tokens['attention_mask'] = torch.cat([tokens['attention_mask'], torch.LongTensor([[1]])], dim=-1)

    embeddings = model(**tokens)['last_hidden_state'].detach().numpy()[0] # features for each tokenized word

    features = [] # features for each recongized chunk, may contain several words
    start_word_index = 0
    for i, num_word in enumerate(num_words):
        end_word_index = start_word_index + num_word
        if i == 0:
            end_word_index += 1 # include 'sos'
        if i == len(num_words) - 1:
            end_word_index += 1  # include 'eos
        # Average features for mulitple tokens in one word
        feature = embeddings[start_word_index:end_word_index]
        feature = np.mean(feature, axis=0)
        # print(tokens['input_ids'][0][start_word_index:end_word_index])
        features.append(feature)
        start_word_index = end_word_index
    features = np.stack(features)

    assert len(features) == len(words), "lenght of features must match length of words"
    return features

def frame_encoding(audio_filename):
    """
    Args:
        audio_filename - file path of audio file
    Returns:
        Frame encoded word embedding
    """

    # Search for text file using audio_filename
    text_filename = os.path.basename(audio_filename).replace('audio', 'text').replace('.wav', '.json')
    text_filename = os.path.join('data/takekuchi/source/text', text_filename)

    if not os.path.exists(text_filename):
        return None

    # Load transcriptions
    with codecs.open(text_filename, 'r', 'utf8') as f:
        words = json.load(f)['word_list']

    # Frame encoding
    fps = 20
    embedding_dim = 768
    bert_word_vecs = process_word_vec(words)
    

    word_features = []
    ct = 0.0
    for i, (word_info, bert_word_vec) in enumerate(zip(words, bert_word_vecs)):

        st = word_info['start_time']
        et = word_info['end_time']

        # Silence between words
        if st != ct:
            num_frames = int((st - ct) * fps)
            word_feature = np.zeros((num_frames, embedding_dim))
            word_features.append(word_feature)

        # Encode word
        if et == st:
            num_frames = 1
        else:
            num_frames = np.round((et - st) * fps)
        word_feature = np.repeat(bert_word_vec[np.newaxis, :], num_frames, axis=0)
        word_features.append(word_feature)
        ct = et

    word_features = np.concatenate(word_features, axis=0)
    return word_features