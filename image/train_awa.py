from awa_trainer import ModelTrainer
from data_utils import get_data_snapshot
import numpy as np
from argparse import ArgumentParser

dataroot = './image/datasets/awa'

def do_experiment(embedding, distance, split, topk):
    top1 = {}
    top3 = {}
    top5 = {}

    for dist in distance:
        for emb in embedding:
            top1[f'{emb}-{dist}'] = []
            top3[f'{emb}-{dist}'] = []
            top5[f'{emb}-{dist}'] = []
            print('+++++++++++++++++++++++++++++++++++++')
            print('+++++++++++++++++++++++++++++++++++++')
            print(f'Embedding: {emb}-{dist}')
            for s in split:
                print(f'|Split: {s}')
                data_snapshot = get_data_snapshot(dataroot=dataroot, dataset='awa', attribute_embedding=emb, split=s, bs=128)
                for k in topk:
                    print(f' |TopK: {k}')
                    trainer = ModelTrainer(data_snapshot, gpu=False, topk=k, distance=dist)
                    r = trainer.run_model(1)
                    if k == 1:
                        top1[f'{emb}-{dist}'].append(r)
                    elif k == 3:
                        top3[f'{emb}-{dist}'].append(r)
                    elif k == 5:
                        top5[f'{emb}-{dist}'].append(r)

    return top1, top3, top5

def print_summary(embedding, distance, top1, top3, top5):
    for embs in embedding:
        for dist in distance:
            emb = f'{embs}-{dist}'
            print(emb)
            if len(top1[emb]) > 0:
                print('top1')
                bin_mean = np.array(top1[emb]).T.mean(axis=1)
                bin_std = np.array(top1[emb]).T.std(axis=1)
                print(f'Support: {bin_mean[0]:.4} ± {bin_std[0]:.2}')
                print(f'Novel: {bin_mean[1]:.4} ± {bin_std[1]:.2}')
                print(f'Mix: {bin_mean[2]:.4} ± {bin_std[2]:.2}')
                print(f'Test: {bin_mean[3]:.4} ± {bin_std[3]:.2}')
            
            if len(top3[emb]) > 0:
                print('top3')
                bin_mean = np.array(top3[emb]).T.mean(axis=1)
                bin_std = np.array(top3[emb]).T.std(axis=1)
                print(f'Support: {bin_mean[0]:.4} ± {bin_std[0]:.2}')
                print(f'Novel: {bin_mean[1]:.4} ± {bin_std[1]:.2}')
                print(f'Mix: {bin_mean[2]:.4} ± {bin_std[2]:.2}')
                print(f'Test: {bin_mean[3]:.4} ± {bin_std[3]:.2}')

            if len(top5[emb]) > 0:
                print('top5')
                bin_mean = np.array(top5[emb]).T.mean(axis=1)
                bin_std = np.array(top5[emb]).T.std(axis=1)
                print(f'Support: {bin_mean[0]:.4} ± {bin_std[0]:.2}')
                print(f'Novel: {bin_mean[1]:.4} ± {bin_std[1]:.2}')
                print(f'Mix: {bin_mean[2]:.4} ± {bin_std[2]:.2}')
                print(f'Test: {bin_mean[3]:.4} ± {bin_std[3]:.2}')


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--embedding", dest="embedding", help="embedding type", type=str, default='label')
    parser.add_argument("-d", "--distance", dest="distance", type=str, default='cos')
    args = parser.parse_args()

    embedding = []
    distance = []

    embedding.append(args.embedding)
    distance.append(args.distance)

    split = [1,2,3]
    topk = [1,3,5]
    top1, top3, top5 = do_experiment(embedding, distance, split, topk)
    print_summary(embedding, distance, top1, top3, top5)

if __name__ == "__main__":
    main()