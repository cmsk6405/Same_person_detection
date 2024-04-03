
config = {
    # train(base)로 구축할 데이터들을 저장할 경로
    "train_setting":{
        "image_path": "./images/train",
        "crop_save_path": ".images/train/crop_images",

        "emb_vecs_save" : "./vectors/train/embed_vecs",
        "mtcnn_vecs_save" : "./vectors/train/lm_mtcnn_vecs",
        "fan_vecs_save" : "./vectors/train/lm_fan_vecs",

        "emb_comb_save" : "./vectors/train/emb_comb", 
        "lm_comb_save" : "./vectors/train/lm_comb",
        "merge_vecs_save" : "./vectors/train/merge_vecs"
    },

    # test(new)데이터들을 저장할 경로
    "test_setting":{
        "image_path": "./images/test",
        "crop_save_path": ".images/test/crop_images",

        "emb_vecs_save" : "./vectors/test/embed_vecs",
        "mtcnn_vecs_save" : "./vectors/test/lm_mtcnn_vecs",
        "fan_vecs_save" : "./vectors/test/lm_fan_vecs",

        "emb_comb_save" : "./vectors/test/emb_comb", 
        "lm_comb_save" : "./vectors/test/lm_comb",
        "merge_vecs_save" : "./vectors/test/merge_vecs"
    }
}