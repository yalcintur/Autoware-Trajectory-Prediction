import os
import random
import numpy as np
import torch
from ..submodules.model import Model
from ..submodules.autoware_feeder import Feeder
from datetime import datetime
import itertools
import glob
from scipy import spatial
import pickle


class GripPredictor:
    def __init__(self):
        self.CUDA_VISIBLE_DEVICES = '0'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.CUDA_VISIBLE_DEVICES

        self.max_x = 1.
        self.max_y = 1.
        self.batch_size_train = 64
        self.batch_size_val = 32
        self.batch_size_test = 1
        self.total_epoch = 10
        self.base_lr = 0.01
        self.lr_decay_epoch = 5
        self.dev = 'cuda:0'

        self.test_result_file = 'prediction_result.txt'

        self.criterion = torch.nn.SmoothL1Loss()

        self.history_frames = 6
        self.future_frames = 10
        self.total_frames = self.history_frames + self.future_frames
        self.max_num_object = 120
        self.neighbor_distance = 10
        self.total_feature_dimension = 13



        self.seed_torch()

    def seed_torch(self, seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def text_data_extract(self, data):
        lines = data.strip().split('\n')

        # Extracting data in the desired format
        extracted_data = []

        for line in lines:
            values = line.split()

            # Extracting the relevant columns based on their index
            # Note: Python indices start at 0
            frame_data = [
                float(values[0]),  # frame_id
                float(values[1]),  # object_id
                float(values[2]),  # object_type
                float(values[3]),  # position_x
                float(values[4]),  # position_y
                float(values[5]),  # position_z
                float(values[6]),  # object_length
                float(values[7]),  # object_width
                float(values[8]),  # object_height
                float(values[9]),   # heading
                float(values[10]),  # heading
                float(values[11]),  # heading
            ]
            extracted_data.append(frame_data)

        return extracted_data

    def get_frame_instance_dict(self, data):
        '''
        Read raw data from files and return a dictionary: 
                {frame_id: 
                        {object_id: 
                                # 10 features
                                [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading]
                        }
                }
        '''

        now_dict = {}
        for row in data:
            # print(row[0])
            # instance = {row[1]:row[2:]}
            n_dict = now_dict.get(row[0], {})
            # print(n_dict.keys())
            n_dict[row[1]] = row  # [2:]
            # n_dict.append(instance)
            # now_dict[]
            now_dict[row[0]] = n_dict

        return now_dict

    def process_data(self, pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last):
        # object_id appears at the last observed frame
        visible_object_id_list = list(pra_now_dict[pra_observed_last].keys())
        # number of current observed objects
        num_visible_object = len(visible_object_id_list)

        # compute the mean values of x and y for zero-centralization.
        visible_object_value = np.array(
            list(pra_now_dict[pra_observed_last].values()))
        xy = visible_object_value[:, 3:5].astype(float)
        mean_xy = np.zeros_like(visible_object_value[0], dtype=float)
        m_xy = np.mean(xy, axis=0)
        mean_xy[3:5] = m_xy

        # compute distance between any pair of two objects
        dist_xy = spatial.distance.cdist(xy, xy)
        # if their distance is less than $neighbor_distance, we regard them are neighbors.
        neighbor_matrix = np.zeros((self.max_num_object, self.max_num_object))
        neighbor_matrix[:num_visible_object, :num_visible_object] = (
            dist_xy < self.neighbor_distance).astype(int)

        now_all_object_id = set([val for x in range(
            pra_start_ind, pra_end_ind) for val in pra_now_dict[x].keys()])
        non_visible_object_id_list = list(
            now_all_object_id - set(visible_object_id_list))
        num_non_visible_object = len(non_visible_object_id_list)

        # for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
        object_feature_list = []
        # non_visible_object_feature_list = []
        for frame_ind in range(pra_start_ind, pra_end_ind):
            # we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1]
            # -mean_xy is used to zero_centralize data
            # now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
            now_frame_feature_dict = {obj_id: (list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] if obj_id in visible_object_id_list else list(
                pra_now_dict[frame_ind][obj_id]-mean_xy)+[0]) for obj_id in pra_now_dict[frame_ind]}
            # if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
            if frame_ind == 24590:
                print(now_frame_feature_dict.keys())
                print(len(now_frame_feature_dict))
                print(len(now_frame_feature_dict[1.0]))
                print(visible_object_id_list+non_visible_object_id_list)
                print(self.total_feature_dimension)

            now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(
                self.total_feature_dimension)) for vis_id in visible_object_id_list+non_visible_object_id_list])
            object_feature_list.append(now_frame_feature)

        # object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
        object_feature_list = np.array(object_feature_list)

        # object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
        object_frame_feature = np.zeros(
            (self.max_num_object, pra_end_ind-pra_start_ind, self.total_feature_dimension))

        # np.transpose(object_feature_list, (1,0,2))
        object_frame_feature[:num_visible_object +
                             num_non_visible_object] = np.transpose(object_feature_list, (1, 0, 2))

        return object_frame_feature, neighbor_matrix, m_xy

    def generate_test_data(self, frame_list):
        print("GENERATE TEST DATA START")
        now_dict = self.get_frame_instance_dict(frame_list)

        frame_id_set = sorted(set(now_dict.keys()))
        print(frame_id_set)
        all_feature_list = []
        all_adjacency_list = []
        all_mean_list = []
        # get all start frame id
        #start_frame_id_list = frame_id_set[::self.history_frames]
        start_frame_id_list = frame_id_set[::]
        # start_frame_id_list = frame_id_set
        print(start_frame_id_list)
        for ind, start_ind in enumerate(start_frame_id_list):
            print(ind)
            print("Len frame id list", len(start_frame_id_list))
            print("len history frame", self.history_frames)
            print(len(start_frame_id_list) - self.history_frames)
            if ind > len(start_frame_id_list) - self.history_frames:
                break
            print(start_ind)
            start_ind = int(start_ind)
            end_ind = int(start_ind + self.history_frames)
            observed_last = start_ind + self.history_frames - 1
            # print(start_ind, end_ind)
            object_frame_feature, neighbor_matrix, mean_xy = self.process_data(
                now_dict, start_ind, end_ind, observed_last)

            all_feature_list.append(object_frame_feature)
            all_adjacency_list.append(neighbor_matrix)
            all_mean_list.append(mean_xy)

        print(all_feature_list)
        # (N, V, T, C) --> (N, C, T, V)
        all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
        all_adjacency_list = np.array(all_adjacency_list)
        all_mean_list = np.array(all_mean_list)
        # print(all_feature_list.shape, all_adjacency_list.shape)
        return all_feature_list, all_adjacency_list, all_mean_list

    def generate_data(self, datab):
        print("GENERATE DATA START")
        all_data = []
        all_adjacency = []
        all_mean_xy = []

        now_data, now_adjacency, now_mean_xy = self.generate_test_data(datab)
        all_data.extend(now_data)
        all_adjacency.extend(now_adjacency)
        all_mean_xy.extend(now_mean_xy)

        all_data = np.array(all_data)  # (N, C, T, V)=(5010, 11, 12, 70) Train
        all_adjacency = np.array(all_adjacency)  # (5010, 70, 70) Train
        all_mean_xy = np.array(all_mean_xy)  # (5010, 2) Train

        # Train (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
        # Test (N, C, T, V)=(415, 11, 6, 70), (415, 70, 70), (415, 2)
        print(np.shape(all_data), np.shape(
            all_adjacency), np.shape(all_mean_xy))

        return pickle.dumps([all_data, all_adjacency, all_mean_xy])

    def my_print(self, pra_content):
        with open(self.log_file, 'a') as writer:
            print(pra_content)
            writer.write(pra_content+'\n')

    def my_load_model(self, pra_model, pra_path):
        print("LOAD MODEL START")

        checkpoint = torch.load(pra_path)
        pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
        print('Successfull loaded from {}'.format(pra_path))
        return pra_model

    def data_loader(self, pra_path, pra_batch_size=128, pra_shuffle=False, pra_drop_last=False, train_val_test='train'):
        print("DATA LOADER START")

        graph_args = {'max_hop': 2, 'num_node': 120}
        feeder = Feeder(data_path=pra_path, graph_args=graph_args,
                        train_val_test=train_val_test)
        loader = torch.utils.data.DataLoader(
            dataset=feeder,
            batch_size=pra_batch_size,
            shuffle=pra_shuffle,
            drop_last=pra_drop_last,
            num_workers=0,
        )
        return loader

    def preprocess_data(self, pra_data, pra_rescale_xy):
        # pra_data: (N, C, T, V)
        # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
        feature_id = [3, 4, 9, 10]
        ori_data = pra_data[:, feature_id].detach()
        data = ori_data.detach().clone()

        new_mask = (data[:, :2, 1:] != 0) * (data[:, :2, :-1] != 0)
        data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]
                           ).float() * new_mask.float()
        data[:, :2, 0] = 0

        # # small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
        object_type = pra_data[:, 2:3]

        data = data.float().to(self.dev)
        ori_data = ori_data.float().to(self.dev)
        object_type = object_type.to(self.dev)  # type
        data[:, :2] = data[:, :2] / pra_rescale_xy

        return data, ori_data, object_type

    def test_model(self, pra_model, pra_data_loader):
        # pra_model.to(dev)
        print("TEST MODEL START")
        pra_model.eval()
        rescale_xy = torch.ones((1, 2, 1, 1)).to(self.dev)
        rescale_xy[:, 0] = self.max_x
        rescale_xy[:, 1] = self.max_y
        all_overall_sum_list = []
        all_overall_num_list = []
        final_result = []
        with open(self.test_result_file, 'w') as writer:
            # train model using training data
            for iteration, (ori_data, A, mean_xy) in enumerate(pra_data_loader):
                # data: (N, C, T, V)
                # C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
                data, no_norm_loc_data, _ = self.preprocess_data(
                    ori_data, rescale_xy)

                # (N, C, T, V)=(N, 4, 6, 120)
                input_data = data[:, :, :self.history_frames, :]
                output_mask = data[:, -1, -1, :]  # (N, V)=(N, 120)
                # print(data.shape, A.shape, mean_xy.shape, input_data.shape)

                ori_output_last_loc = no_norm_loc_data[:, :2,
                                                       self.history_frames-1:self.history_frames, :]

                A = A.float().to(self.dev)
                predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=self.future_frames,
                                      pra_teacher_forcing_ratio=0, pra_teacher_location=None)  # (N, C, T, V)=(N, 2, 6, 120)
                predicted = predicted * rescale_xy

                for ind in range(1, predicted.shape[-2]):
                    predicted[:, :, ind] = torch.sum(
                        predicted[:, :, ind-1:ind+1], dim=-2)
                predicted += ori_output_last_loc

                now_pred = predicted.detach().cpu().numpy()  # (N, C, T, V)=(N, 2, 6, 120)
                now_mean_xy = mean_xy.detach().cpu().numpy()  # (N, 2)
                now_ori_data = ori_data.detach().cpu().numpy()  # (N, C, T, V)=(N, 11, 6, 120)
                now_mask = now_ori_data[:, -1, -1, :]  # (N, V)

                now_pred = np.transpose(now_pred, (0, 2, 3, 1))  # (N, T, V, 2)
                now_ori_data = np.transpose(
                    now_ori_data, (0, 2, 3, 1))  # (N, T, V, 11)

                for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, now_ori_data, now_mask):
                    # (6, 120, 2), (2,), (6, 120, 11), (120, )
                    num_object = np.sum(n_mask).astype(int)

                    # only use the last time of original data for ids (frame_id, object_id, object_type)
                    # (6, 120, 11) -> (num_object, 3)
                    n_dat = n_data[-1, :num_object, :3].astype(int)
                    for time_ind, n_pre in enumerate(n_pred[:, :num_object], start=1):
                        # (120, 2) -> (n, 2)
                        # print(n_dat.shape, n_pre.shape)
                        for info, pred in zip(n_dat, n_pre+n_mean_xy):
                            information = info.copy()
                            information[0] = information[0] + time_ind
                            result = ' '.join(information.astype(
                                str)) + ' ' + ' '.join(pred.astype(str)) + '\n'

                            writer.write(result)
                            final_result.append(result)

        return final_result
    
    def run_test(self, pra_model, pra_data_path):
        loader_test = self.data_loader(pra_data_path, pra_batch_size=self.batch_size_test,
                                       pra_shuffle=False, pra_drop_last=False, train_val_test='test')
        
        final_result = self.test_model(pra_model, loader_test)
        print(final_result)
        return final_result
