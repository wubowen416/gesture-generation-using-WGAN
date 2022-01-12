import os
import codecs
import json
import numpy as np

from transformers import BertJapaneseTokenizer, BertModel
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")


def get_word_vec(word):
    """
    Args:
        word - a single word
    Returns:
        embedding - shape [1, dim]
    """
    tokens = tokenizer([word], return_tensors='pt')
    embeddings = model(**tokens)['pooler_output'].detach().numpy()
    return embeddings

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

    word_vecs = []
    ct = 0.0
    for i, word_info in enumerate(words):

        word = word_info['word']
        st = word_info['start_time']
        et = word_info['end_time']

        # Silence between words
        if st != ct:
            num_frames = int((st - ct) * fps)
            word_vec = np.zeros((num_frames, embedding_dim))
            word_vecs.append(word_vec)

        # Encode word
        if et == st:
            num_frames = 1
        else:
            num_frames = np.round((et - st) * fps)
        word_vec = np.repeat(get_word_vec(word), num_frames, axis=0)
        word_vecs.append(word_vec)
        ct = et

    word_vec = np.concatenate(word_vecs, axis=0)
    return word_vec