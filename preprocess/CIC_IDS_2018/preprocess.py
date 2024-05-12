import pymongo
from scapy.all import PcapReader
from scapy.compat import bytes_encode, hex_bytes, bytes_hex
from scapy.layers import l2

import os
import random
import time
import tensorflow as tf
import ujson

def str_to_pkt(string):
    return l2.Ether(hex_bytes(string))

def reset_addr(pkt):
    """
        匿名化MAC地址和IP地址
    """
    assert pkt.haslayer('Ether') and pkt.haslayer('IP')
    pkt.getlayer('Ether').dst = "00:00:00:00:00:00"
    pkt.getlayer('Ether').src = "00:00:00:00:00:00"
    pkt.getlayer('IP').src = "0.0.0.0"
    pkt.getlayer('IP').dst = "0.0.0.0"
    return pkt

def _shuffle(arr):
    random.shuffle(arr)
    random.shuffle(arr)
    random.shuffle(arr)
    return arr


def final_data(new_db='mixed', raw_db='PacketInString'):
    
    train_ids, valid_ids, test_ids = [], [], []
    #train_ids_us, train_ids_us_os = [], []

    client = pymongo.MongoClient(host = 'mongodb://localhost:27017/')
    raw_db = client.get_database(raw_db)
    new_db = client.get_database(new_db)

    print(f'==> shuffling by cols...')
    col_names = raw_db.list_collection_names()
    for name in col_names:
        col = raw_db.get_collection(name)
        cur_ids = []
        for bs in col.find(no_cursor_timeout=True):
            cur_ids.append((name, bs['_id']))

        cur_ids = _shuffle(cur_ids)
        len1 = int(len(cur_ids) * 0.3)  # 613
        len2 = int(len(cur_ids) * 0.5)

        train_ids.extend(cur_ids[:len1])
        valid_ids.extend(cur_ids[len1:len2])
        test_ids.extend(cur_ids[len2:])

        #label = name.lower()
        # assert label in label_maps
        # if label in oversample_dict:
        #     train_ids_us.extend(cur_ids[:len1])
        #     new_ids = []
        #     repeat_times = oversample_dict[label]
        #     for pair in cur_ids[:len1]:
        #         cnt = 0
        #         while cnt < repeat_times:
        #             new_ids.append(pair)
        #             cnt += 1
        #     train_ids_us_os.extend(new_ids)
        # elif label in undersample_dict:
        #     len_ = len1 // undersample_dict[label]
        #     train_ids_us.extend(cur_ids[:len_])
        #     train_ids_us_os.extend(cur_ids[:len_])
        # else:
        #     train_ids_us.extend(cur_ids[:len1])
        #     train_ids_us_os.extend(cur_ids[:len1])

    print(' ===== train ===== ')
    train_ids = _shuffle(train_ids)
    train_col = new_db.get_collection('train')
    _insert_by_id(train_col, train_ids, raw_db)

    # print(' ===== train_undersample ===== ')
    # train_ids_us = _shuffle(train_ids_us)
    # train_us_col = new_db.get_collection('train_us')
    # _insert_by_id(train_us_col, train_ids_us, raw_db)

    # print(' ===== train_undersample_oversample ==== ')
    # train_ids_us_os = _shuffle(train_ids_us_os)
    # train_us_os_col = new_db.get_collection('train_us_os')
    # _insert_by_id(train_us_os_col, train_ids_us_os, raw_db)

    print(' ===== validation ===== ')
    valid_ids = _shuffle(valid_ids)
    valid_col = new_db.get_collection('valid')
    _insert_by_id(valid_col, valid_ids, raw_db)

    print(' ===== test ===== ')
    test_ids = _shuffle(test_ids)
    test_col = new_db.get_collection('test')
    _insert_by_id(test_col, test_ids, raw_db)


def _insert_by_id(target_col, pairs, original_db):
    #print(f"{str(target_col)}")
    for pair in pairs:
        name, bid = pair
        bson_targets = original_db.get_collection(name).find({'_id': bid}, no_cursor_timeout=True)
        tmp = [x for x in bson_targets]
        assert len(tmp) == 1
        sample = tmp[0]
        sample.pop('_id')
        target_col.insert_one(sample)

if __name__ == "__main__":
    final_data()