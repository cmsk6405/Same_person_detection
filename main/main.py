from image_crop import yolo_crop
from embedding import resnet_embedding
from landmark import mtcnn_landmark, fan_lanmark
from merge_verctors import collect_same_person, merge_vec_files
from classfy import calculate_distances

import argparse

def main(cfg):
    #이미지 크롭
    train = cfg.get("train_setting")
    test = cfg.get("test_setting")

    def make_data(type):
        yolo_crop(type.get("image_path"), type.get("crop_save_path"))

        #크롭한 이미지로 임베딩
        resnet_embedding(type.get("crop_save_path"), type.get("emb_vecs_save"))
        collect_same_person(type.get("emb_vecs_save"), type.get("emb_comb_save"))

        #크롭한 이미지로 랜드마크
        mtcnn_landmark(type.get("crop_save_path"), type.get("mtcnn_vecs_save"))
        # fan_landmark(type.get("crop_save_path"), type.get("mtcnn_vecs_save"))
        collect_same_person(type.get("mtcnn_vecs_save"), type.get("lm_comb_save"))

        #위 두개 합치기
        merge_vec_files(type.get("emb_comb_save"), type.get("lm_comb_save"), type.get("merge_vecs_save"))
    
    make_data(train)
    make_data(test)

    calculate_distances(train.get("merge_vecs_save"), test.get("merge_vecs_save"))



def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./configs.py", type=str, help="configuration file")
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exec(open(args.config).read())

    main(config) # config.py 파일을 통해서 저장경로를 설정