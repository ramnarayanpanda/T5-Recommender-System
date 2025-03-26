import random
import pandas as pd
from time import time 
from tqdm import tqdm
import concurrent.futures
from sklearn.model_selection import train_test_split

from src.utils import *
from src.data_templates import tasks


class DataPreparation:

    def __init__(self):
        self.final_pre_data_file_path = "./data/final_pre_data.json"
        self.data_maps_file_path = "./data/data_maps.json"
        self.data_maps = None
        self.sequential_error_cnt = 0
        self.review_error_cnt = 0
        self.traditional_error_cnt = 0
        self.explanation_error_cnt = 0
        self.rating_error_cnt = 0
        self.num_threads = 8
        self.train_data_path = "./data/train_data.json"
        self.test_data_path = "./data/test_data.json"


    def data_preparation(self):
        final_data_lst = []

        with open(self.data_maps_file_path, "r") as f:
            self.data_maps = json.load(f)
        whole_data = pd.read_json(self.final_pre_data_file_path).to_dict(orient="records")
        # whole_data = whole_data[:100]

        functions_to_run = [
            self._rating_data_preparation,
            self._sequential_data_preparation,
            self._explanation_data_preparation,
            self._review_data_preparation,
            self._traditional_data_preparation
        ]
        for func in functions_to_run:
            prev_len = len(final_data_lst)
            t1 = time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(func, data_dct) for data_dct in whole_data]
                results = []
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(whole_data), desc=f"Running {func.__name__}"):
                    results.extend(future.result())
                final_data_lst.extend(results)
            t2 = time()
            print(f"{func.__name__}: Time: {t2 - t1:.2f}s, Length: {len(final_data_lst) - prev_len}")

        print(self.sequential_error_cnt, self.review_error_cnt, self.traditional_error_cnt, self.explanation_error_cnt, self.rating_error_cnt)
        print(f"Total data length: {len(final_data_lst)}")

        df = pd.DataFrame(final_data_lst, columns=['task_desc', 'inp_text', 'out_text', 'metric'])
        train, test = train_test_split(df, test_size=0.2, stratify=df['task_desc'])
        train.to_json(self.train_data_path, orient="records", default_handler=str)
        test.to_json(self.test_data_path, orient="records", default_handler=str)


    def _traditional_data_preparation(self, data_dct):
        data_dct = {
            'user_id': data_dct['user_id'],
            'user_desc': data_dct['user_desc'],
            'item_id_list': data_dct['item_id_list'],
            'item_title_list': data_dct['item_title_list'],
        }
        flattened_dct = flatten_dict(data_dct)
        data_lst = []

        try:
            for sub_dct in flattened_dct:
                random_keys = random.sample(list(tasks['traditional'].keys()), 3)
                for sub_task_key in random_keys:
                    if 0 <= int(sub_task_key) <= 15:
                        choice_bet_yes_no = max(0, min(100, random.gauss(mu=50, sigma=20)))
                        if choice_bet_yes_no >= 50:
                            random_idx = random.choice(range(0, len(data_dct['item_id_list']) - 1))
                            sub_dct['candidate_item_id'] = data_dct['item_id_list'][random_idx]
                            sub_dct['candidate_item_title'] = data_dct['item_title_list'][random_idx]
                            out_text = "yes"
                        else:
                            random_item_id = random.choice(list(set(self.data_maps['id2item'].keys()) - set(data_dct['item_id_list'])))
                            sub_dct['candidate_item_id'] = random_item_id
                            sub_dct['candidate_item_title'] = self.data_maps['id2item'][random_item_id]
                            out_text = "no"
                    else:
                        random_pos_to_add_the_target_item = random.randint(0, 49)
                        candidate_item_id_list = random.sample(list(set(self.data_maps['id2item'].keys()) - set(data_dct['item_id_list'])), 50)
                        sub_dct['target_item_id'] = sub_dct['item_id_list']
                        candidate_item_id_list.insert(random_pos_to_add_the_target_item, sub_dct['target_item_id'])
                        sub_dct['candiate_item_id_list'] = "{" + "--".join(candidate_item_id_list) + "}"
                        out_text = tasks['traditional'][sub_task_key][1].format(**sub_dct)

                    # print(sub_dct)
                    inp_text = tasks['traditional'][sub_task_key][0].format(**sub_dct)
                    metric = tasks['traditional'][sub_task_key][2]
                    data_lst.append(['traditional_' + sub_task_key, inp_text, out_text, metric])
        except Exception as e:
            print(str(e))
            self.traditional_error_cnt += 1

        return data_lst


    def _review_data_preparation(self, data_dct):
        data_dct = {
            'user_id': data_dct['user_id'],
            'user_desc': data_dct['user_desc'],
            'review_body': data_dct['review_list'],
            'rating': data_dct['rating_list'],
        }
        flattened_dct = flatten_dict(data_dct)
        data_lst = []

        try:
            for sub_dct in flattened_dct:
                if sub_dct['review_body'] == "":
                    continue
                random_keys = random.sample(list(tasks['review'].keys()), 2)
                for sub_task_key in random_keys:
                    inp_text = tasks['review'][sub_task_key][0].format(**sub_dct)
                    out_text = tasks['review'][sub_task_key][1].format(**sub_dct)
                    metric = tasks['review'][sub_task_key][2]
                    data_lst.append(['review_' + sub_task_key, inp_text, out_text, metric])
        except Exception as e:
            print(str(e))
            self.review_error_cnt += 1

        return data_lst


    def _explanation_data_preparation(self, data_dct):
        data_dct = {
            'user_id': data_dct['user_id'],
            'user_desc': data_dct['user_desc'],
            'item_id': data_dct['item_id_list'],
            'item_title': data_dct['item_title_list'],
            'rating': data_dct['rating_list'],
            'explanation': data_dct['review_explanation_list'],
            'feature': data_dct['review_feature_list']
        }
        flattened_dct = flatten_dict(data_dct)
        data_lst = []

        try:
            for sub_dct in flattened_dct:
                if sub_dct['explanation'] == "" or sub_dct['feature'] == "":
                    continue
                random_keys = random.sample(list(tasks['explanation'].keys()), 3)
                for sub_task_key in random_keys:
                    inp_text = tasks['explanation'][sub_task_key][0].format(**sub_dct)
                    out_text = tasks['explanation'][sub_task_key][1].format(**sub_dct)
                    metric = tasks['explanation'][sub_task_key][2]
                    data_lst.append(['explanation_' + sub_task_key, inp_text, out_text, metric])
        except Exception as e:
            print(str(e))
            self.explanation_error_cnt += 1

        return data_lst


    def _sequential_data_preparation(self, data_dct):
        data_dct = {
            'user_id': data_dct['user_id'],
            'user_desc': data_dct['user_desc'],
            'item_id_list': data_dct['item_id_list'],
            'item_title_list': data_dct['item_title_list'],
            'original_item_id_list': data_dct['item_id_list'],
            'original_item_title_list': data_dct['item_title_list']
        }

        # out of all the items visited by the user, we will select 80% to 90% as the visit history, next item will be the target
        # incase where we need candidates, we will use same 80% to 90% as visit history, next item + another randomly selected 50 items for candidates
        random_keys = random.sample(list(tasks['sequential'].keys()), 5)
        data_lst = []
        # random_keys = tasks['sequential'].keys()

        try:
            for sub_task_key in random_keys:
                min_frac, max_frac = 0.7, 0.95
                min_size = int(min_frac * len(data_dct['original_item_id_list']))
                max_size = int(max_frac * len(data_dct['original_item_id_list']))
                selected_size = random.randint(min_size, max_size)

                data_dct['item_id_list'] = "{" + "--".join(data_dct['original_item_id_list'][:selected_size]) + "}"
                data_dct['item_title_list'] = "{" + "--".join(data_dct['original_item_title_list'][:selected_size]) + "}"
                data_dct['target_item_id'] = data_dct['original_item_id_list'][selected_size]
                data_dct['target_item_title'] = data_dct['original_item_title_list'][selected_size]

                random_pos_to_add_the_target_item = random.randint(0, 49)
                candidate_item_id_list = random.sample(list(set(self.data_maps['id2item'].keys()) - set(data_dct['original_item_id_list'])), 50)
                candidate_item_id_list.insert(random_pos_to_add_the_target_item, data_dct['target_item_id'])
                data_dct['candiate_item_id_list'] = "{" + "--".join(candidate_item_id_list) + "}"

                inp_text = tasks['sequential'][sub_task_key][0].format(**data_dct)
                if 24 <= int(sub_task_key) <= 29:
                    choice_bet_yes_no = max(0, min(100, random.gauss(mu=50, sigma=20)))
                    if choice_bet_yes_no > 50:
                        out_text = "yes"
                    else:
                        data_dct['target_item_id'] = random.choice(list(set(self.data_maps['id2item'].keys()) - set(data_dct['original_item_id_list'])))
                        out_text = "no"
                else:
                    out_text = tasks['sequential'][sub_task_key][1].format(**data_dct)
                metric = tasks['sequential'][sub_task_key][2]
                data_lst.append(['sequential_' + sub_task_key, inp_text, out_text, metric])

        except Exception as e:
            print(str(e))
            self.sequential_error_cnt += 1

        return data_lst


    def _rating_data_preparation(self, data_dct):
        data_dct = {
            'user_id': data_dct['user_id'],
            'user_desc': data_dct['user_desc'],
            'item_id': data_dct['item_id_list'],
            'item_title': data_dct['item_title_list'],
            'rating': data_dct['rating_list']
        }
        flattened_dct = flatten_dict(data_dct)
        # for each user item intercation, we will randomly select 2 tasks and apply them on the data to generate input and ouput texts
        # if you can train on more data, then increase this to 5 may be

        data_lst = []
        for sub_dct in flattened_dct:
            random_keys = random.sample(list(tasks['rating'].keys()), 2)
            out_text = ""
            for sub_task_key in random_keys:
                if 0 <= int(sub_task_key) <= 10:
                    out_text = tasks['rating'][sub_task_key][1].format(**sub_dct)
                elif 11 <= int(sub_task_key) <= 14:
                    choice_bet_yes_no = max(0, min(100, random.gauss(mu=50, sigma=20)))
                    if choice_bet_yes_no > 50:
                        out_text = "yes"
                    else:
                        sub_dct['rating'] = random.choice([i for i in range(0, 5) if i != sub_dct['rating']])
                        out_text = "no"
                elif 15 <= int(sub_task_key) <= 18:
                    out_text = "like" if sub_dct['rating'] >= 4 else "dislike"
                inp_text = tasks['rating'][sub_task_key][0].format(**sub_dct)
                metric = tasks['rating'][sub_task_key][2]
                data_lst.append(['rating_' + sub_task_key, inp_text, out_text, metric])

        return data_lst