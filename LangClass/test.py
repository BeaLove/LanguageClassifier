import argparse

import torch
from dataloader import SentenceData
from tqdm import tqdm


def test(checkpoint, data_dir):
    if torch.cuda.is_available():
        cuda = True
        device = "cuda:0"
        print("using cuda")
    else:
        cuda = False
        device = 'cpu'
        print("using cpu")

    test_set = SentenceData(data_dir)
    test_data = torch.utils.data.DataLoader(test_set, num_workers=2, batch_size=8)
    if not cuda:
        model = torch.load(checkpoint, map_location=torch.device("cpu"))
    else:
        model = torch.load(checkpoint)
    model = model.to(device)
    batch_i = tqdm(test_data, desc="Testing")
    correct = 0
    total = len(test_set)
    for sample in batch_i:
        x, y = sample
        x = x.to(device)
        y = y.to(device)
        output = model.forward(x)
        prediction = torch.argmax(output, dim=1)
        compare = [1 if prediction[i] == y[i] else 0 for i in range(len(prediction))]
        correct += sum(compare)
        accuracy = correct/total
    return accuracy


def parse_args(argv=None):
    del argv
    parser = argparse.ArgumentParser(description="test the language classifier")
    parser.add_argument('--model_checkpoint', dest='checkpoint', type=str, help="full path to checkpoint file including directory")
    parser.add_argument('--test_data_dir', dest='data_dir', type=str, help='directory of test data')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    accuracy = test(checkpoint=args.checkpoint, data_dir=args.data_dir)
    print(accuracy)