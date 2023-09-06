import rclpy
from rclpy.node import Node
from typing import Any, Dict, List, Union
import uuid
import tf2_ros
from geometry_msgs.msg import Vector3

import time
from tf2_msgs.msg import TFMessage

from .submodules.grip_predictor import GripPredictor
from .submodules.model import Model

# Import (relevant) Autoware MSG Types:
from autoware_auto_perception_msgs.msg import (
    DetectedObject,
    DetectedObjects,
    ObjectClassification,
    TrackedObject,
    TrackedObjects,
    PredictedObjects,
    PredictedObject
)

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray


class DataPredictor(Node):

    def __init__(self) -> None:
        super().__init__('data_predictor')

        self.timeframelist = set()

        # Subscribers [Perception Topics]:
        self.perception_detection_sub = self.create_subscription(
            DetectedObjects, "/perception/object_recognition/detection/objects",
            self.store_predict_publish_callback,
            1)

        self.perception_tracking_sub = self.create_subscription(
            TrackedObjects, "/perception/object_recognition/tracking/objects",
            self.store_predict_publish_callback,
            1)

        self.tf_move_sub = self.create_subscription(
            TFMessage, "/tf",
            self.store_predict_publish_callback,
            1)

        # Publisher [Prediction Topic]:
        self.prediction_pub = self.create_publisher(
            PredictedObjects, "prediction", 10)

        self.marker_pub = self.create_publisher(
            MarkerArray, 'path_markers', 1)

        # Lists to store received/subscribed Perception-Objects [DetectedObjects/TrackedObjects]:
        self.detectedObjs_lst = list()
        self.trackedObjs_lst = list()
        self.tf_lst = list()

    # Callback Function executed each time 'TrackedObjects' or 'DetectedObjects' message is received:
    def store_predict_publish_callback(self, received_msg) -> None:

        # Store received message in the 'detectedObjs_lst' or 'trackedObjs_lst' list (depending on message type):
        self.store_msg(received_msg)

        # Compute Prediction (see 'prediction'-function) and publish results: Every 6 seconds
        if len(self.timeframelist) >= 6:
            predObjects_list, marker_array = self.prediction(received_msg=received_msg)

            self.marker_pub.publish(marker_array)

            for predObjects in predObjects_list:
                self.publish(predObjects)

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    def object_classification_to_category_name(self, object_classification) -> str:
        """https://github.com/tier4/autoware_auto_msgs/blob/tier4/main/autoware_auto_perception_msgs/msg/ObjectClassification.idl"""
        cls_to_cat: Dict[int, str] = {
            0: "unknown",
            1: "car",
            2: "truck",
            3: "bus",
            4: "trailer",
            5: "motorcycle",
            6: "bicycle",
            7: "pedestrian",
        }

        return cls_to_cat.get(object_classification, "unknown")

    def parse_perception_objects(self, msg) -> List[Dict[str, Any]]:
        """https://github.com/tier4/autoware_auto_msgs/tree/tier4/main/autoware_auto_perception_msgs
        Args:
            msg (autoware_auto_perception_msgs.msg.DetectedObjects): autoware detection msg (.core/.universe)

        Returns:
            List[Dict[str, Any]]: dict format
        """
        assert isinstance(
            msg, (DetectedObjects, TrackedObjects)
        ), f"Invalid object message type: {type(msg)}"

        scene_annotation_list: List[Dict[str, Any]] = []
        for obj in msg.objects:
            obj: Union[DetectedObject, TrackedObject]
            pose = obj.kinematics.centroid_position

            position: Dict[str, Any] = {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
            }

            label_dict: Dict[str, Any] = {
                "object_id": int(obj.object_id),
                "time": int(msg.header.stamp.sec),
                "attribute_names": [],  # not available
                "three_d_bbox": {
                    "translation": position,
                },
                "num_lidar_pts": 1,
                "num_radar_pts": 0,
            }
            if obj.object_id != 0:
                scene_annotation_list.append(label_dict)

        return scene_annotation_list

    # Function to store received messages:

    def store_msg(self, received_msg) -> None:

        msgType = type(received_msg)

        if (msgType is DetectedObjects):
            self.detectedObjs_lst.append(received_msg)
            self.get_logger().info("Received 'detected' objects.")

        elif (msgType is TrackedObjects):
            self.trackedObjs_lst.append(received_msg)
            self.timeframelist.add(received_msg.header.stamp.sec)
            self.get_logger().info("Received 'tracked' objects.")
            self.get_logger().info(str(received_msg.header.stamp.sec))

        elif (msgType is TFMessage):
            self.tf_lst.append(received_msg)
            self.get_logger().info("Received TF message.")
            self.timeframelist.add(received_msg.transforms[0].header.stamp.sec)
            self.get_logger().info(str(len(self.timeframelist)))

        else:
            self.get_logger().warning("Unspecified message type observed:" + str(msgType))

    # Function to publish computed predictions (to 'prediction'-topic):
    def publish(self, pred_msg) -> None:

        # ...
        self.prediction_pub.publish(pred_msg)
        self.get_logger().info("Published predicted objects.")

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------
    # self.timeframelist.add(received_msg.transforms[0].header.stamp.sec)
    # [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, object_width, object_height, heading]

    # Function to compute the predictions (based on the received/stored messages):

    def extract_data(self, text_data) -> List[int]:

        extracted_data = []

        for data in text_data:
            lines = data.strip().split('\n')

            for line in lines:
                values = line.split()

                # Extracting the relevant columns based on their index
                # Note: Python indices start at 0
                frame_data = [
                    float(values[0]),  # frame_id 1
                    float(values[1]),  # object_id 2
                    float(values[2]),  # object_type 3
                    float(values[3]),  # position_x 4
                    float(values[4]),  # position_y 5
                ]
                extracted_data.append(frame_data)

        return extracted_data
    
    def create_marker(self, obj_id, predicted_object, header):

        marker = Marker()
        marker.header = header
        marker.id = obj_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose = predicted_object.kinematics.initial_pose.pose
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker
    
    def prediction(self, received_msg=None) -> PredictedObjects:

        list_for_pred = []
        message_type = type(received_msg)

        if (message_type == DetectedObjects or message_type == TrackedObjects):
            header = received_msg.header
        elif message_type == TFMessage:
            header = received_msg.transforms[0].header

        header.frame_id = "map"
        
        for tracked_objects in self.trackedObjs_lst:
            decoded_objects = self.parse_perception_objects(tracked_objects)

            for decoded_object in decoded_objects:
                obj_his = []
                if (decoded_object["object_id"]) == 0:
                    continue
                obj_his.append(float(decoded_object["time"]))  # 1
                obj_his.append(float(decoded_object["object_id"]))  # 2
                obj_his.append(float(1))  # 3
                obj_his.append(
                    float(decoded_object["three_d_bbox"]["translation"]["x"]))  # 4
                obj_his.append(
                    float(decoded_object["three_d_bbox"]["translation"]["y"]))  # 5
                obj_his.append(
                    float(decoded_object["three_d_bbox"]["translation"]["z"]))  # 6
                obj_his.append(float(5))  # 7
                obj_his.append(float(2))  # 8
                obj_his.append(float(2))  # 9
                obj_his.append(float(2))  # 10
                obj_his.append(float(2))  # 11
                obj_his.append(float(2))  # 12
                list_for_pred.append(obj_his)

        for tf in self.tf_lst:
            obj_his = []
            obj_his.append(float(tf.transforms[0].header.stamp.sec))  # 1
            obj_his.append(float(1))  # 2
            obj_his.append(float(1))  # 3
            obj_his.append(
                float(tf.transforms[0].transform.translation.x))  # 4
            obj_his.append(
                float(tf.transforms[0].transform.translation.y))  # 5
            obj_his.append(
                float(tf.transforms[0].transform.translation.z))  # 6
            obj_his.append(float(5))  # 7
            obj_his.append(float(2))  # 8
            obj_his.append(float(2))  # 9
            obj_his.append(float(2))  # 10
            obj_his.append(float(2))  # 11
            obj_his.append(float(2))  # 12
            list_for_pred.append(obj_his)

        grip_pred = GripPredictor()

        pretrained_model_path = '/home/gorgor/model_trained2.pt'

        graph_args = {'max_hop': 2, 'num_node': 120}
        model = Model(in_channels=4, graph_args=graph_args,
                      edge_importance_weighting=True)
        model.to(grip_pred.dev)
        feed_data = grip_pred.generate_data(list_for_pred)
        model = grip_pred.my_load_model(model, pretrained_model_path)
        final_result = grip_pred.run_test(model, feed_data)

        predicted_data = self.extract_data(final_result)
        print(predicted_data)
        pred_objects_dict = {}
        pred_objects_list = []
        
        marker_array = MarkerArray()
        unique_rank_id = 1
        for predicted in predicted_data:
            time_stamp = int(predicted[0])
            if time_stamp not in pred_objects_dict:
                pred_objects_dict[time_stamp] = []
            pred_object_future = PredictedObject()
            pred_object_future.object_id = int(predicted[1])
            pred_object_future.existence_probability = 1.0
            pred_object_future.kinematics.initial_pose.pose.position.x = predicted[3]
            pred_object_future.kinematics.initial_pose.pose.position.y = predicted[4]
            pred_objects_dict[time_stamp].append(pred_object_future)
            marker = self.create_marker(unique_rank_id, pred_object_future, header)
            unique_rank_id += 1
            marker_array.markers.append(marker)

        for key in pred_objects_dict.keys():
            pred_objects = PredictedObjects()
            pred_objects.header = header
            for pred_object in pred_objects_dict[key]:
                pred_objects.objects.append(pred_object)
            pred_objects_list.append(pred_objects)

        self.timeframelist.clear()

        return pred_objects_list, marker_array

    # ---------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------

    #  object_id: 0
    #  existence_probability: 0.0
    #  classification: []
    #  kinematics:
    #  initial_pose:
    # object_id: 3
    # existence_probability: 0.0
    #  classification: []
    #  kinematics:
    #    initial_pose:
    #     pose:
    #       position:
    #         x: 0.0
    #         y: 0.0
    #         z: 0.0
    #       orientation:
    #         x: 0.0
    #         y: 0.0
    #         z: 0.0
    #         w: 1.0

    # self.get_logger().info(str(received_msg.transforms[0].transform.translation.x))

    # ---------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------


def main(args=None):

    # ...
    rclpy.init(args=args)
    data_predictor = DataPredictor()

    # ...
    rclpy.spin(data_predictor)

    # ...
    data_predictor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
