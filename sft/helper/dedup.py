import json
import time
import Levenshtein
from multiprocessing import Pool, cpu_count
import os
import sys
import numpy as np

dup_ratio = 0.05

def write_json(data, filename):
    for d in data:
        del d['dedup_id']
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i]['dedup_id'] = i
    return data

def get_lev_sim(d1, d2):
    lev_text = Levenshtein.distance(d1['text'], d2['text']) / max(len(d1['text']), len(d2['text']))
    lev_answer = Levenshtein.distance(d1['answer'], d2['answer']) / max(len(d1['answer']), len(d2['answer']))
    return (1-lev_text) * (1-lev_answer)

def top_percent_indices(A, ratio=dup_ratio):
    flattened_A = A.flatten()
    sorted_indices = np.argsort(flattened_A)
    top_percent_count = int(ratio * 2 * A.shape[0])
    top_percent_indices = sorted_indices[-top_percent_count:]
    top_percent_pairs = np.unravel_index(top_percent_indices, A.shape)
    top_percent_pairs = list(zip(top_percent_pairs[0], top_percent_pairs[1]))
    return top_percent_pairs

def compare_with_dedup_list(d, data):
    adj_list = []
    for d2 in data:
        if d['dedup_id'] == d2['dedup_id']:
            adj_list.append(0)
            continue
        adj_list.append(get_lev_sim(d, d2))
    return d['dedup_id'], adj_list

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def deduplicate_data(data_path, result_path):
    print(f'Deduplicating {data_path}, the output will be saved to {result_path}')
    data = read_json(data_path)
    start_time = time.time()
    num_cores = min(100, cpu_count())
    dup_edges = []

    with Pool(num_cores) as pool:
        results = pool.starmap(compare_with_dedup_list, [(d, data) for d in data])
    results = sorted(results, key=lambda x: x[0])
    result_matrix = np.array([x[1] for x in results])
    dup_edges = top_percent_indices(result_matrix)

    uf = UnionFind(len(data))
    for x, y in dup_edges:
        uf.union(x, y)

    unique_data_dict = {}
    for i in range(len(data)):
        rep = uf.find(i)
        if rep not in unique_data_dict:
            unique_data_dict[rep] = data[i]
    unique_data = list(unique_data_dict.values())

    end_time = time.time()

    print(f"Execution Done, Total time {(end_time - start_time) * 1000:.3f} ms")
    print(f"Num of Total Samples: {len(data)}")
    print(f"Num of Unique Samples: {len(unique_data)}")
    
    write_json(unique_data, result_path)
    return len(data), len(unique_data)

def main(source_root, target_root):
    os.makedirs(target_root, exist_ok=True)
    with open(os.path.join(target_root, 'dedup.out'), 'w', encoding='utf-8') as f:
        sys.stdout = f
        start_time = time.time()
        total_len, unique_len = 0, 0
        domains =  [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
        for task in domains:
            os.makedirs(os.path.join(target_root, task), exist_ok=True)
            for filename in os.listdir(os.path.join(source_root, task)):
                data_path = os.path.join(source_root, task, filename)
                result_path = os.path.join(target_root, task, filename)
                result = deduplicate_data(data_path, result_path)
                total_len += result[0]
                unique_len += result[1]
        end_time = time.time()
        print(f"Total Execution Done, Total time {end_time - start_time:.3f} seconds")
        print(f"Total Num of Samples: {total_len}")
        print(f"Total Num of Deduplicated Samples: {unique_len}")

if __name__ == '__main__':
    main('results', 'deduped_results')
