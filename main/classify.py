import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def calculate_distances(existing_folder, new_folder):
    for existing_file in os.listdir(existing_folder):
        existing_path = os.path.join(existing_folder, existing_file)
        existing_data = np.load(existing_path)

        for new_file in os.listdir(new_folder):
            new_path = os.path.join(new_folder, new_file)
            new_data = np.load(new_path)

            # 임베딩끼리 거리 구하기
            existing_emb = existing_data[0, :]
            new_emb = new_data[0, :]

            split_num1 = int(len(existing_emb) / 512)
            split_num2 = int(len(new_emb) / 512)

            if split_num2 == 1:
                split_new_emb = new_emb
            else:
                split_new_emb = np.array_split(new_emb, split_num2)

            split_existing_emb = np.array_split(existing_emb, split_num1)

            for i in range(split_num1):
                emb1 = split_existing_emb[i].reshape(1, -1)
                for j in range(split_num2):
                    if split_num2 == 1:
                        emb2 = split_new_emb.reshape(1, -1)
                    else:
                        emb2 = split_new_emb[j].reshape(1, -1)
                    distance_emb = euclidean_distances(emb1, emb2)
                    # print(f"emb_distance : {distance_emb}")

            # 랜드마크 거리 구하기
            existing_land = existing_data[1, :]
            new_land = new_data[1, :]

            no_zero_existing_land = np.nonzero(existing_land)
            no_zero_new_land = np.nonzero(new_land)

            existing_land = existing_land[no_zero_existing_land]
            new_land = new_land[no_zero_new_land]

            land_split_num1 = len(existing_land) // 10
            land_split_num2 = len(new_land) // 10

            if land_split_num2 == 1:
                split_new_land = new_land
            else:
                split_new_land = np.array_split(new_land, land_split_num2)

            split_existing_land = np.array_split(existing_land, land_split_num1)

            for i in range(land_split_num1):
                landmark1 = split_existing_land[i].reshape(10, -1)
                if land_split_num2 == 1:
                    landmark2 = split_new_land.reshape(10, -1)
                else:
                    for j in range(land_split_num2):
                        landmark2 = split_new_land[j].reshape(10, -1)


                test_land_parts = []
                base_land_parts = []
                for j in range(0, len(landmark1), 2):
                    test_land_parts.append(np.array([landmark1[j], landmark1[j+1]]).reshape(1, -1))
                    base_land_parts.append(np.array([landmark2[j], landmark2[j+1]]).reshape(1, -1))

                euclidean_dists = []
                for j in range(len(test_land_parts)):
                    dist = euclidean_distances(test_land_parts[j], base_land_parts[j])
                    euclidean_dists.append(dist)
                euclidean_dists = np.array(euclidean_dists).reshape(1, -1)
                # print(f"land_distance: {euclidean_dists}")

                if distance_emb < 0.9 and np.max(euclidean_dists) < 100:
                    print(f"New file {new_file} matches existing file {existing_file}")
