#dataset loader
import json
import os
import h5py
import glob
import numpy as np


def read_dataset(process, args, w2i):
    #data file path
    fld_path = os.path.join(args.path_folder,args.type)
    filename = 'IR_'+process+'_'+args.type.lower()

    #data 
    with open(os.path.join(fld_path,filename+'.json')) as json_data:
        data = json.load(json_data)

    for key, val in data.items():
        word_d, word_c, img_list, target_ind, img_id   = val['dialog'], val['caption'], val['img_list'], val['target'], val['target_img_id']
        stack_d=[]
        for i, sen in enumerate(word_d):
            sen = sen[0].lower().strip().split(" ")
            stack_d += sen
        word_c = word_c.lower().strip().split(" ")
        if args.dialog:
            yield([w2i[x] for x in stack_d],img_list,target_ind,img_id)

        if args.caption:
            yield([w2i[x] for x in word_c],img_list,target_ind,img_id)

        if args.combine:
           word = stack_d+word_c
           yield([w2i[x] for x in word],img_list,target_ind,img_id)


def minibatch(data, batch_size=32):
   for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


def get_img_feat(image_list):
   img_m = [img_features[visual_feat_mapping[str(i)]] for i in image_list]
   # img_m = torch.from_numpy(np.array(img_m))

   return img_m


def preprocess(batch):
    """ Add zero-padding to a batch. """
    # add zero-padding to make all sequences equally long
    seqs = [example.word for example in batch]
    max_length = max(map(len, seqs))
    seqs = [seq + [PAD] * (max_length - len(seq)) for seq in seqs]

    img_feat = [get_img_feat(example.img_list) for example in batch]
    img_feat = np.array(img_feat)

    tags = [example.img_ind for example in batch]

    return seqs, img_feat, tags


def get_image_features(args):
    #Loading img_features
    path_to_h5_file = glob.glob(args.img_feat+"/*.h5")[0]
    img_features = np.asarray(h5py.File(path_to_h5_file, 'r')['img_features'])
    return img_features

def get_visual_feature_mapping(args):
    path_to_json_file = glob.glob(args.img_feat+"/*.json")[0]
    with open(path_to_json_file, 'r') as f:
        visual_feat_map = json.load(f)['IR_imgid2id']

    return visual_feat_map
