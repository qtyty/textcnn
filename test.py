import torch
from config import parse_config
from data_loader import DataBatchIterator
from sklearn.metrics import classification_report


def test_textcnn_model(model, test_data, config):
    model.eval()
    test_data_iter = iter(test_data)
    predict = torch.LongTensor([])
    label = torch.LongTensor([])
    for idx, batch in enumerate(test_data_iter):
        model.zero_grad()
        ground_truth = batch.label
        # batch_first = False
        outputs = model(batch.sent)
        result = torch.max(outputs, 1)[1]
        predict = torch.cat((predict, result), 0)
        label = torch.cat((label, batch.label), 0)
    return predict, label


def main():
    # 读配置文件
    config = parse_config()
    # 载入训练集合
    train_data = DataBatchIterator(
        config=config,
        is_train=True,
        dataset="train",
        batch_size=config.batch_size,
        shuffle=True)
    train_data.load()

    vocab = train_data.vocab

    # 载入测试集合
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size)
    test_data.set_vocab(vocab)
    test_data.load()

    # 测试时
    checkpoint = torch.load(config.save_model + ".pt",
                            map_location=config.device)
    model = checkpoint

    # model = build_textcnn_model(
    #     vocab, config, train=True)
    predict,label=test_textcnn_model(model,test_data,config)
    print(classification_report(label, predict))


if __name__ == "__main__":
    main()