import random
import pandas as pd

def create_dataset(dataset):
    file_name_0 = "dataset/" + dataset + ".0"
    file_name_1 = "dataset/" + dataset + ".1"
    datas = []
    ground_truth = []
    dataset_0 = open(file_name_0, "r")
    for sentence in dataset_0:
        sentence = sentence.rstrip("\n")
        words = sentence.split(" ")
        if len(words) >= 5:
            datas.append(sentence)
            ground_truth.append(0)
    dataset_1 = open(file_name_1, "r")
    for sentence in dataset_1:
        sentence = sentence.rstrip("\n")
        words = sentence.split(" ")
        if len(words) >= 5:
            datas.append(sentence)
            ground_truth.append(1)
    final_dataset = list(zip(datas, ground_truth))
    random.shuffle(final_dataset)
    random.shuffle(final_dataset)
    random.shuffle(final_dataset)
    return zip(*final_dataset)

if __name__ == "__main__":
    print("Training set is creating...")
    train_x, train_y = create_dataset("train")
    train_x, train_y = list(train_x), list(train_y)
    print("Training set creation is over.")
    print("Training set size: " + str(len(train_x)))
    print("-------------------------")
    print("Test set is creating...")
    test_x, test_y = create_dataset("test")
    test_x, test_y = list(test_x), list(test_y)
    print("Test set creation is over.")
    print("Test set size: " + str(len(test_x)))
    print("-------------------------")
    print("Dev set is creating...")
    dev_x, dev_y = create_dataset("dev")
    dev_x, dev_y = list(dev_x), list(dev_y)
    print("Dev set creation is over.")
    print("Dev set size: " + str(len(dev_x)))
    print("-------------------------")
    print("CSV files are creating...")
    train_temp = {'Sentences': train_x, 'Labels': train_y}
    train_set = pd.DataFrame(data=train_temp)
    train_set.to_csv("train.csv", index=False)
    print("train.csv creation is over.")
    dev_temp = {'Sentences': dev_x, 'Labels': dev_y}
    dev_set = pd.DataFrame(data=dev_temp)
    dev_set.to_csv("dev.csv", index=False)
    print("dev.csv creation is over.")
    test_temp = {'Sentences': test_x, 'Labels': test_y}
    test_set = pd.DataFrame(data=test_temp)
    test_set.to_csv("test.csv", index=False)
    print("test.csv creation is over.")
    print("-------------------------")
