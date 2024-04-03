# embedding.npy 합치기
# 이거 헷갈리네 차원이 늘어나면 안돼?왜?
# concat?

import os
import numpy as np


# 같은 사람(앞 숫자 네자리)끼리 모아서 dict 생성
def collect_same_person(input_path, output_folder):
    # 파일 이름을 네 자리 숫자로 그룹화하기 위한 딕셔너리 생성
    file_groups = {}

    # 폴더 내의 모든 파일 탐색
    for root, _, files in os.walk(input_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.endswith('.npy'):
                prefix = file_name[:4]
                # 해당 그룹에 파일 추가
                if prefix in file_groups:
                    file_groups[prefix].append(file_path)
                else:
                    file_groups[prefix] = [file_path]

    combine_vecs(file_groups, output_folder)

# 위에서 만든 dict를 이용해서 하나의 npy 파일 만들기
def combine_vecs(file_groups, output_folder):
    for prefix, files in file_groups.items():
        combined_data = []
        for file_path in files:
            print(file_path)
            data = np.load(file_path, allow_pickle=True)
            combined_data.append(data)
        # 그룹 내의 모든 데이터 결합
        # print(combined_data)
        combined_data = np.concatenate(combined_data, axis=0)
        
        # 결합된 데이터를 새 파일로 저장
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file_path = os.path.join(output_folder, f'{prefix}_combined.npy')
        np.save(output_file_path, combined_data)



# emb와 landmark 하나로 합치기
def merge_vec_files(folder1, folder2, output_folder):
   # 폴더 내의 파일 목록 가져오기
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    
    # 폴더1과 폴더2의 공통 파일 이름 찾기
    common_names = set(files1) & set(files2)
    
    # 공통 파일 이름을 가진 파일을 읽고 병합하여 저장
    for file_name in common_names:
        if file_name.endswith('.npy'):
            file_path1 = os.path.join(folder1, file_name)
            file_path2 = os.path.join(folder2, file_name)
            data1 = np.load(file_path1).reshape(1,-1)
            data2 = np.load(file_path2, allow_pickle=True).reshape(1,-1)

            # 두 배열의 길이를 맞추고 모자라는 부분은 0으로 채웠음 이거 물어보기
            resized_data2 = np.zeros_like(data1)
            resized_data2[:, :data2.shape[1]] = data2

            # 데이터 병합
            merged_data = np.concatenate([data1, resized_data2])
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            # 병합된 데이터 저장
            output_path = os.path.join(output_folder, file_name)
            np.save(output_path, merged_data)

