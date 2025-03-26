import json
from tqdm import tqdm
import random
from collections import defaultdict
import pandas as pd

from src.utils import *


class PreDataPreparation:

    def __init__(self):
        self.review_file_path = "./data/original_data/review.json"
        self.user_file_path = "./data/original_data/user_filtered.json"
        self.item_file_path = "./data/original_data/business.json"
        self.review_with_features_file = "./data/original_data/reviews_pickle.pickle"
        self.final_pre_data_file_path = "./data/final_pre_data.json"
        self.data_maps_file_path = "./data/data_maps.json"
        self.max_date = '2019-12-31 00:00:00'
        self.min_date = '2019-01-01 00:00:00'
        self.user_core = 5
        self.item_core = 5
        self.negative_sampling_size = 50


    def _get_review_with_features(self):
        raw_data = load_pickle(self.review_with_features_file)
        review_with_features = {}
        for i in raw_data:
            review_with_features[(i['user'], i['item'])] = i
        return review_with_features


    def _get_user_data(self, data_maps):
        data_dct = {}
        lines = open(self.user_file_path).readlines()
        for line in tqdm(lines):
            user = json.loads(line.strip())
            user_id = user['user_id']
            user_desc = user['name']
            if user_id in data_maps['user2id']:
                data_dct[user_id] = {'user_desc': user_desc}
        return data_dct


    def _get_item_data(self):
        data_dct = {}
        lines = open(self.item_file_path).readlines()
        for line in tqdm(lines):
            item = json.loads(line.strip())
            item_id = item['business_id']
            item_name = item['name']
            item_city = item['city']
            item_state = item['state']
            item_desc = item_name + "_" + item_city + "_" + item_state
            data_dct[item_id] = {'item_desc': item_desc}
        return data_dct


    def _get_user_item_interactions(self, review_data):
        user_item_interaction = {}
        for data in review_data:
            user, item, time, rating, user_review, review_feature, review_explanation = data
            user_item_interaction[user] = user_item_interaction.get(user, []) + [(item, time, rating, user_review, review_feature, review_explanation)]
        for user, item_time in user_item_interaction.items():
            user_item_interaction[user] = sorted(user_item_interaction[user], key=lambda x: x[1])
        return user_item_interaction


    def _get_review_data(self):
        review_data = []
        rating_score = 0.0
        review_with_features = self._get_review_with_features()
        lines = open(self.review_file_path).readlines()

        for line in tqdm(lines):
            review = json.loads(line.strip())
            user = review['user_id']
            item = review['business_id']
            rating = review['stars']
            date = review['date']
            user_review = review['text']
            if date < self.min_date or date > self.max_date or float(rating) <= rating_score:
                continue
            time = date.replace('-','').replace(':','').replace(' ','')

            review_feature, review_explanation = "", ""
            rev_exp_data = review_with_features.get((user, item), {})
            if 'sentence' in rev_exp_data.keys():
                select_random_idx = random.randint(0, len(rev_exp_data['sentence'])-1)
                review_feature = rev_exp_data['sentence'][select_random_idx][0]
                review_explanation = rev_exp_data['sentence'][select_random_idx][2]

            review_data.append((user, item, int(time), rating, user_review, review_feature, review_explanation))

        return review_data


    def _check_kcore(self, user_item_interaction):
        user_count = defaultdict(int)
        item_count = defaultdict(int)
        for user, items in user_item_interaction.items():
            items = [i[0] for i in items]
            for item in items:
                user_count[user] += 1
                item_count[item] += 1
        for user, cnt in user_count.items():
            if cnt < self.user_core:
                return user_count, item_count, False
        for item, cnt in item_count.items():
            if cnt < self.item_core:
                return user_count, item_count, False
        return user_count, item_count, True


    def _filter_kcore(self, user_item_interaction):
        user_count, item_count, iskcore = self._check_kcore(user_item_interaction)
        while not iskcore:
            for user, cnt in user_count.items():
                if user_count[user] < 5:
                    user_item_interaction.pop(user)
                else:
                    updated_item_lst = []
                    for item in user_item_interaction[user]:
                        # print(item[0])
                        if item_count[item[0]]>=5:
                            updated_item_lst.append(item)
                    user_item_interaction[user] = updated_item_lst
            user_count, item_count, iskcore = self._check_kcore(user_item_interaction)
        return user_item_interaction


    def _get_mappings(self, user_item_interaction):
        user2id = {}
        id2user = {}
        item2id = {}
        id2item = {}
        user_cnt = 1
        item_cnt = 1

        for user, item_lst in user_item_interaction.items():
            item_lst = [i[0] for i in item_lst]
            if user not in user2id:
                user2id[user] = "user_" + str(user_cnt)
                id2user["user_" + str(user_cnt)] = user
                user_cnt += 1

            items_ids_lst = []
            for item in item_lst:
                if item not in item2id:
                    item2id[item] = "item_" + str(item_cnt)
                    id2item["item_" + str(item_cnt)] = item
                    item_cnt += 1
                items_ids_lst.append(item2id[item])

        data_maps = {
            'user2id': user2id,
            'id2user': id2user,
            'item2id': item2id,
            'id2item': id2item
        }
        return data_maps


    def pre_data_preparation(self):
        review_data = self._get_review_data()
        print("Total review data count: ", len(review_data))

        user_item_interaction = self._get_user_item_interactions(review_data)
        print("Total length of user_item_interaction: ", len(user_item_interaction))

        user_item_interaction = self._filter_kcore(user_item_interaction)
        print("Total items satisfying Kscore: ", len(user_item_interaction))

        data_maps = self._get_mappings(user_item_interaction)
        with open(self.data_maps_file_path, "w") as file:
            json.dump(data_maps, file, indent=4)
        print("Saved data maps")

        user_data_dct = self._get_user_data(data_maps)
        item_data_dct = self._get_item_data()

        final_df = pd.DataFrame({
            'user_id1': list(user_item_interaction.keys()),
            'item_id_list1': [[j[0] for j in i] for i in user_item_interaction.values()],
            'visit_date_list': [[j[1] for j in i] for i in user_item_interaction.values()],
            'rating_list': [[j[2] for j in i] for i in user_item_interaction.values()],
            'review_list': [[j[3] for j in i] for i in user_item_interaction.values()],
            'review_feature_list': [[j[4] for j in i] for i in user_item_interaction.values()],
            'review_explanation_list': [[j[5] for j in i] for i in user_item_interaction.values()],
        })
        final_df['user_id'] = final_df['user_id1'].map(data_maps['user2id'])
        final_df['item_id_list'] = final_df['item_id_list1'].map(lambda x: [data_maps['item2id'][i] for i in x])
        final_df['user_desc'] = final_df['user_id1'].map(lambda x: user_data_dct[x]['user_desc'])
        final_df['item_title_list'] = final_df['item_id_list1'].map(lambda x: [item_data_dct[i]['item_desc'] for i in x])
        # final_df['neg_item_id_list'] = final_df['item_id_list'].map(
        #     lambda x: random.sample(
        #         list(set(data_maps['item2id'].values()) - set(x)),
        #         self.negative_sampling_size
        #     )
        # )

        # make sure that all the list columns have same lengths
        cols = ['item_id_list1', 'visit_date_list', 'rating_list', 'review_list', 'review_feature_list',
                'review_explanation_list', 'item_id_list', 'item_title_list']
        for i in range(len(cols) - 1):
            ser = (final_df[cols[i]].map(len) == final_df[cols[i + 1]].map(len))
            print(ser[ser == False].shape)

        final_df.to_json(self.final_pre_data_file_path, orient="records", default_handler=str)