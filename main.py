import os
import argparse
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tabulate import tabulate
import numpy as np
import time
from model import HAN
from prepare_data import process, read_vec

#数据打包
def collate_batch(batch):
    # max_len = max([len(f["doc_token"]) for f in batch])
    doc_token = [f["doc_token"] for f in batch]
    #待链接实体所在句子索引，[bs, num_men]
    men_sent_idx = [f["men_sent_idx"] for f in batch]
    #待链接实体的真实实体在候选实体中索引，[bs, num_men]
    men_ent_idx = [f["men_ent_idx"] for f in batch]
    #候选实体ID，[bs, num_men, num_can_ent]
    can_ent_idx = [f["can_ent_idx"] for f in batch]
    # output = (doc_token, men_sent_idx, men_ent_idx, can_ent_idx)
    # r_t = torch.Tensor(rating).long()
    # list_doc = doc_token
    #对每个batch的文档按句子数降序排序
    sorted_doc = sorted([(len(doc), doc_idx, doc) for doc_idx, doc in enumerate(doc_token)], reverse=True)
    length_doc, doc_idx, ordered_list_doc = zip(*sorted_doc)
    length_doc = list(length_doc)
    # 计算文档的最大句子数max_sents
    max_sents = length_doc[0]

    #对每个batch里面的文档按照重排顺序进行重新打包
    # doc_token = [doc_token[x] for x in doc_idx]
    men_sent_idx = [men_sent_idx[x] for x in doc_idx]
    men_ent_idx = [men_ent_idx[x] for x in doc_idx]
    can_ent_idx = [can_ent_idx[x] for x in doc_idx]
    can_ent_index = can_ent_idx
    # 对每篇文档中句子按长度降序排序，这里是所有句子信息
    stat = sorted([(len(s), doc_idx, s_idx, s) for doc_idx, doc in enumerate(ordered_list_doc) for s_idx, s in enumerate(doc)],
                  reverse=True)
    # 保存句子的最大单词数max_words
    max_words = stat[0][0]

    length_sen = []
    #把一个batch中所有文档的句子填充到batch_t张量
    batch_t = torch.zeros(len(stat), max_words).long()
    sent_order = torch.zeros(len(ordered_list_doc), max_sents).long().fill_(0)  # (bs*num_sent)
    #待链接实体文档索引，句子索引/每batch
    num_men = []

    for i, s in enumerate(stat):
        #这里的s[1]为重排后的文档顺序了
        sent_order[s[1], s[2]] = i + 1  # i+1 because 0 is for empty. [文档索引，句子索引], 每个batch中的句子的排序索引
        batch_t[i, 0:len(s[3])] = torch.LongTensor(s[3])
        length_sen.append(s[0])
    # for i in range(len(men_sent_idx)):
    #     for j in range(len(men_sent_idx[i])):
    #         men_sent_idx[i][j] = sent_order[i, men_sent_idx[i][j]]
    #         num_men.append((i, men_sent_idx[i][j]))

    # num_men_tensor = torch.tensor(num_men, dtype=torch.long)
    #对候选实体进行处理,最大链接实体和最大候选实体数
    max_num_men = max(len(x) for x in men_sent_idx)
    max_num_can_ent = max(len(can) for x in can_ent_idx for can in x)
    padded_can_ent_idx = []
    for men in can_ent_idx:
        padded_men = []
        for link_ent in men:
            # 对候选实体进行补齐
            padded_link_ent = link_ent + [0] * (max_num_can_ent - len(link_ent))
            padded_men.append(padded_link_ent)
        # 对待链接实体进行补齐
        padded_men += [[0] * max_num_can_ent] * (max_num_men - len(men))
        padded_can_ent_idx.append(padded_men)
    can_ent_idx = torch.tensor(padded_can_ent_idx, dtype=torch.long)
    output = (batch_t, sent_order, length_sen, length_doc, men_sent_idx, men_ent_idx, can_ent_idx, can_ent_index)
    # return batch_t, sent_order, length_sen, length_doc, doc_token
    return output

def train(epoch, net, dataset, device, msg="val/test", optimize=False, optimizer=None, criterion=None):

    if optimize:
        net.train()
    else:
        net.eval()

    epoch_loss = 0
    acc_all = 0
    with tqdm(total=len(dataset),desc=msg) as pbar:
        for iteration, batch in enumerate(dataset):
            inputs = {
                "batch_doc": batch[0].to(device),
                "sent_order": batch[1].to(device),
                "length_sent": batch[2],
                "length_doc": batch[3],
                "men_sent_idx": batch[4],
                "men_ent_idx": batch[5], #[bs, num_men(每个待链接实体对应真实实体在候选实体中索引)]
                "can_ent_idx": batch[6].to(device), #[bs, num_men(待链接实体数量), num_can_ent(候选实体数量)]
                "can_ent_index": batch[7],
            }
            #输出为每个链接实体的候选实体得分，[bs, num_men, num_ent]，填充位值为0
            output = net(**inputs)

            #计算损失和准确率
            #记录每个待链接实体的最高分候选实体id
            #把待链接实体的正确实体和最高得分展平，方便后续计算
            #一个batch中所有待链接实体的真实实体相对索引
            flat_men_ent_id = [item for sublist in batch[5] for item in sublist]
            #记录一个batch中所有待链接实体的最高得分的候选实体相对索引
            score_best_ent_id = []
            #存放真实链接实体得分，一维列表，存放所有待链接实体的真实实体得分
            score_true_ent = []
            #存放候选实体中除真实实体外的最高得分，长度和待链接实体个数一致
            score_can_ent = []
            #存放所有非真是实体的得分
            non_true_scores_all = []
            for bs in range(len(batch[7])):
                for men_idx in range(len(batch[7][bs])):
                    scores = output[bs][men_idx]
                    # 记录最高得分候选实体的索引
                    score_best_ent_id.append(np.argmax((scores.cpu().detach().numpy())))
                    true_id = batch[5][bs][men_idx]
                    # 获取真实实体得分
                    score_true_ent.append(scores[true_id].item())
                    # 删除真实实体得分
                    non_true_scores = torch.cat([scores[:true_id], scores[true_id + 1:]])
                    score_can_ent.append(torch.max(non_true_scores).item())
                    non_true_scores_all.append(non_true_scores)

            assert len(flat_men_ent_id) == len(score_best_ent_id)
            #计算准确率
            correct_predictions = sum(1 for true_id, pred_id in zip(flat_men_ent_id, score_best_ent_id) if true_id == pred_id)
            total_predictions = len(flat_men_ent_id)
            acc = correct_predictions / total_predictions
            acc_all += acc

            #计算max-marg损失，这里我选了候选实体中最高得分和真实链接实体得分之间计算损失
            # target = [1.0] * len(score_true_ent)
            # score_true_ent_tensor = torch.tensor(score_true_ent, dtype=torch.float32, requires_grad=True, device=device)
            # score_can_ent_tensor = torch.tensor(score_can_ent, dtype=torch.float32, requires_grad=True, device=device)
            # assert len(score_true_ent_tensor) == len(score_can_ent_tensor)
            # target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
            # loss = criterion(score_true_ent_tensor, score_can_ent_tensor, target_tensor)

            # 计算每个真实链接实体和每个候选实体的损失，并求和
            true_scores = []
            non_true_scores = []
            for i in range(len(score_true_ent)):
                true_score = score_true_ent[i]
                non_true_scores_i = non_true_scores_all[i]
                # 过滤掉 -0 和 0 的值
                non_true_scores_i = non_true_scores_i[(non_true_scores_i != 0.0)
                                                      & (non_true_scores_i != -0.0)]
                true_scores.extend([true_score] * len(non_true_scores_i))
                non_true_scores.extend(non_true_scores_i)

            true_scores_tensor = torch.tensor(true_scores, dtype=torch.float32, requires_grad=True, device=device)
            non_true_scores_tensor = torch.tensor(non_true_scores, dtype=torch.float32, requires_grad=True,
                                                  device=device)
            target_tensor = torch.ones_like(true_scores_tensor, device=device)
            loss = criterion(true_scores_tensor, non_true_scores_tensor, target_tensor)

            epoch_loss += loss.item()

            if optimize:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.update(1)
            pbar.set_postfix({"acc":(acc_all/(iteration+1)), "Max-Margin":epoch_loss/(iteration+1)})

    avg_loss = epoch_loss / len(dataset)
    avg_acc = acc_all / len(dataset)
    # print(f"===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.4f}, Avg. accuracy: {avg_acc * 100:.2f}")

    return avg_acc, avg_loss

def main(args):
    print(32*"-"+"\nEntity Link:\n" + 32*"-")
    #数据处理部分
    word_vectors, ent_vectors, word2id, ent2id, stop_words = read_vec(args)
    train_set = process(args.train_data, word_vectors, ent_vectors, word2id, ent2id, stop_words)
    test_set = process(args.test_data, word_vectors, ent_vectors, word2id, ent2id, stop_words)
    dataloader = DataLoader(train_set, batch_size=args.b_size, shuffle=True, collate_fn=collate_batch)
    dataloader_test = DataLoader(test_set, batch_size=args.b_size, shuffle=False, collate_fn=collate_batch)

    #使用max-margin loss
    criterion = torch.nn.MarginRankingLoss(margin=0.1, reduction="mean")

    device = torch.device("cuda" if args.cuda else "cpu")

    #实例化模型
    net = HAN(ntoken=len(word_vectors), nent=len(ent_vectors), emb_size=args.emb_size, hid_size=args.hid_size)
    #加载预训练词向量
    net.set_emb_tensor(torch.FloatTensor(word_vectors))
    net.set_ent_tensor(torch.FloatTensor(ent_vectors))
    if args.cuda:
        net.to(device)

    #如果只想测试
    if args.load and args.load != "none":
        net.load_state_dict(torch.load(args.load))
        test_acc, _ = train(0, net, dataloader_test, device, msg="Test", criterion=criterion)
        print("Test Accuracy:", test_acc)
        return 0

    #使用随机梯度下降进行优化
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)
    best_score = -1

    #保存日志
    log_data = []
    headers = ["Epoch", "Train Loss", "Train Acc", "Test Loss", "Test Acc"]
    with open(args.log_file, "w") as f:
        f.write(tabulate([headers], tablefmt="grid") + "\n")
    test_acc_score = []
    start_time = time.time()  # 记录训练开始时间
    #开训
    for epoch in range(1, args.epochs + 1):

        print("\n-------EPOCH {}-------".format(epoch))
        train_acc, train_loss = train(epoch, net, dataloader, device, msg="training", optimize=True, optimizer=optimizer, criterion=criterion)

        #测试
        test_acc, test_loss = train(epoch, net, dataloader_test, device, msg="Evaluation", criterion=criterion)
        if test_acc > best_score:
            test_acc_score.append(test_acc)
            print("best model saved as {}".format(args.save))
            save_dir = os.path.dirname(args.save)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(net.state_dict(), args.save)

        #保存日志
        log_entry = [epoch, f"{train_loss:.4f}", f"{train_acc * 100:.2f}%", f"{test_loss:.4f}", f"{test_acc * 100:.2f}%"]
        log_data.append(log_entry)
        with open(args.log_file, "a") as f:
            f.write(tabulate([log_entry], tablefmt="grid") + "\n")
        #打印当前 epoch 的日志
        print(tabulate([log_entry], headers=headers, tablefmt="grid"))
    end_time = time.time()  # 记录训练结束时间
    print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")
    print("Best test accuracy:", max(test_acc_score))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hierarchical Attention Networks for Document Classification')
    parser.add_argument("--emb_size", type=int, default=300)
    parser.add_argument("--hid_size", type=int, default=150)
    parser.add_argument("--epochs", type=int,default=500)
    parser.add_argument("--clip_grad", type=float,default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--b_size", type=int, default=4)
    parser.add_argument("--save", type=str, default="saved_model/model.pth")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--log_file", type=str, default="saved_model/log.txt")
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument("--train_data", type=str, default='dataset/documents_train.json')
    parser.add_argument("--test_data", type=str, default='dataset/documents_test.json')
    parser.add_argument("--ent_vec", type=str, default='dataset/ent_vec.txt')
    parser.add_argument("--word_info", type=str, default='dataset/word_info.txt')
    parser.add_argument("--stop_word", type=str, default='dataset/stopword.txt')
    args = parser.parse_args()

    main(args)
