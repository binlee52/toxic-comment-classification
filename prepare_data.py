from datasets import load_dataset
import os
import os.path as osp
import pandas as pd

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

if __name__ == "__main__":

    # data를 저장할 ROOT 경로 생성
    ROOT = "./data"
    makedirs(ROOT)

    train_path = osp.join(ROOT, "kor_hate_train.csv")
    test_path = osp.join(ROOT, "kor_hate_test.csv")
    origin_path = osp.join(ROOT, "kor_hate_origin.csv")

    print("Check if the file exists.")
    # 데이터가 존재하는지 확인
    if not osp.exists(origin_path):
        print("preparing data...")

        # data 다운로드
        if not osp.exists(train_path) or osp.exists(test_path):
            dataset = load_dataset("kor_hate")
            trainset = dataset["train"]
            trainset.to_csv(train_path, index=False)
            testset = dataset["test"]
            testset.to_csv(test_path, index=False)

        trainset = pd.read_csv(train_path)
        testset = pd.read_csv(test_path)        
        dataset = pd.concat((trainset, testset))
        dataset = dataset.drop(labels=["contain_gender_bias", "bias"], axis=1)
        dataset = dataset.rename(columns={"hate":"labels"})
        dataset.to_csv(origin_path, index=False)
    
    if osp.exists(train_path):
        os.remove(train_path)
    if osp.exists(test_path):
        os.remove(test_path)

    print("prepared data.")