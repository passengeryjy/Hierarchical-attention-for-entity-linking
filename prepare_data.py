import argparse
import json
from tqdm import tqdm


#读取表征文件
def read_vec(args):
    with open(args.ent_vec,"r") as f:
        ent_vec = f.readlines()
    with open(args.word_info,"r", encoding='utf-8') as f:
        word_info = f.readlines()
    with open(args.stop_word,"r") as f:
        stop_words = [line.strip() for line in f]
    word_vectors = []
    ent_vectors = []
    word2id = {}
    ent2id = {}
    for line in tqdm(word_info, desc="Reading word vectors"):
        #拆分数据
        parts = line.strip().split('\t')
        if len(parts) == 3:
            word = parts[0]
            frequency = parts[1]
            vector = [float(x) for x in parts[2].split(',')]
            tensor_vec = vector
            word_vectors.append(tensor_vec)
            word2id[word] = len(word2id)
    for line in tqdm(ent_vec, desc="Reading entity vectors"):
        spl = line.strip().split('\t')
        ent_id = spl[0]
        ent_vector = [float(x) for x in spl[1].split(',')]
        tensor_ent_vec = ent_vector
        if ent_id not in ent2id:
            ent_vectors.append(tensor_ent_vec)
            ent2id[ent_id] = len(ent2id)
    print("Read {} words and {} entities".format(len(word_vectors), len(ent_vectors)))

    return word_vectors, ent_vectors, word2id, ent2id, stop_words
def process(file, word_vec, ent_vec, word2id, ent2id, stop_words):
    '''
    doc_token_emb：[bs,num_token,emb_size]
    doc_idx: [bs, idx_doc]
    sent_order: [bs, num_sent] 句子原顺序
    length_sent: [bs, num_sent] 句子长度
    length_doc: [bs, len_doc]
    men_sent_idx: [bs, num_men]
    men_ent_idx: [bs, num_men]
    men_cen_emb: [bs, num_men, num_cen, emb_size]
    '''
    #存放模型所需特征
    features = []
    data_file = file
    with open(data_file, "r") as f:
        data = json.load(f)
    documents = data['documents']
    for sample in tqdm(documents, desc="Processing doc"):
        #存放句子和句子顺序
        sent_word = []
        sent_order = []
        doc_token = []
        #存放句子长度
        # sentence_lengths = []
        #存放待链接实体所在句子索引和gold链接实体
        men_sent_idx = []
        men_ent_idx = []
        #存放候选实体id,
        can_ent_ids = []

        doc_id = sample['id']
        doc_text = sample['document']
        mention = sample['mentions']

        # 使用句号分割句子
        sentences = doc_text.split('.')
        # 去除空字符串
        tmp = [sentence.strip() for sentence in sentences if sentence.strip()]
        length_doc = 0
        for sen in tmp:
            word = sen.split(' ')
            #停用词过滤
            # filter_words = [w for w in word if w not in stop_words]
            # sent_word.append(filter_words)
            for w in range(len(word)):
                if word[w] not in word2id:
                    continue
                else:
                    word[w] = word2id[word[w]]
            new_word = [x for x in word if isinstance(x, int)]
            sent_word.append(new_word)
            # doc_token.extend(word)

        # 统计每个句子的长度和文档长度
        sentence_lengths = [len(sent_word[id]) for id in range(len(sent_word))]
        length_doc = sum(sentence_lengths)

        for mid, men in enumerate(mention):
            # 分割候选实体
            can_ent = men["candidates"].split("\t")
            can_ent = [x for x in can_ent if x != '']
            if len(can_ent) == 2:
                continue
            else:
                # 读取待链接实体的句子索引和gold链接实体
                men_sent_idx.append(men['sent_index'])
                men_ent_idx.append(men['gold_index']-1)
                # 提取待链接实体的所有候选实体ID
                can_ent_id = can_ent[1::2]
                for c in range(len(can_ent_id)):
                    if can_ent_id[c] not in ent2id:
                        continue
                    else:
                        can_ent_id[c] = ent2id[can_ent_id[c]]
                new_can_ent_id = [x for x in can_ent_id if isinstance(x, int)]
                can_ent_ids.append(new_can_ent_id) #[num_men, num_can_ent]


        feature = {
            'doc_token': sent_word, #全文token, [num_sent, len_sent]
            'doc_id': doc_id, #文档id
            # 'sent_order': sent_order, #句子顺序列表
            # 'length_sent': sentence_lengths, #句子长度记录
            # 'length_doc': length_doc,  # 文档长度
            'men_sent_idx': men_sent_idx,   # 待链接实体所在句子索引
            'men_ent_idx': men_ent_idx,   # 待链接实体的gold实体在候选实体中索引
            'can_ent_idx': can_ent_ids,  # 候选实体id
            # 'word_vec': word_vec,
            'ent_vec': ent_vec
        }
        features.append(feature)
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default='dataset/documents_train.json')
    parser.add_argument("--test_data", type=str, default='dataset/documents_test.json')
    parser.add_argument("--ent_vec", type=str, default='dataset/ent_vec.txt')
    parser.add_argument("--word_info", type=str, default='dataset/word_info.txt')
    parser.add_argument("--stop_word", type=str, default='dataset/stopword.txt')
    # parser.add_argument("output", type=str, default="sentences.pkl")
    # parser.add_argument("--nb_splits",type=int, default=5)
    args = parser.parse_args()
    #读取表征文件
    process(args)