import rclpy
from rclpy.node import Node

# Import (relevant) Autoware MSG Types:
from autoware_auto_perception_msgs.msg import DetectedObjects
from autoware_auto_perception_msgs.msg import TrackedObjects

from autoware_auto_perception_msgs.msg import PredictedObjects
from autoware_auto_perception_msgs.msg import PredictedObject


class DataPredictor(Node):

    def __init__(self) -> None:
        super().__init__('data_predictor')

        # Subscribers [Perception Topics]:
        self.perception_detection_sub = self.create_subscription(
            DetectedObjects, "/perception/object_recognition/detection/objects", 
            self.store_predict_publish_callback, 
            1)

        self.perception_tracking_sub = self.create_subscription(
            TrackedObjects, "/perception/object_recognition/tracking/objects", 
            self.store_predict_publish_callback, 
            1)

        # Publisher [Prediction Topic]:
        self.prediction_pub = self.create_publisher(
            PredictedObjects, "prediction", 1)

        # Lists to store received/subscribed Perception-Objects [DetectedObjects/TrackedObjects]:
        self.detectedObjs_lst = list()
        self.trackedObjs_lst = list()

    # Callback Function executed each time 'TrackedObjects' or 'DetectedObjects' message is received:
    def store_predict_publish_callback(self, received_msg) -> None:

        # Store received message in the 'detectedObjs_lst' or 'trackedObjs_lst' list (depending on message type):
        self.store_msg(received_msg)

        # Compute Prediction (see 'prediction'-function) and publish results:
        predObjects = self.prediction(received_msg = received_msg)
        self.publish(predObjects)

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    # Function to store received messages:
    def store_msg(self, received_msg) -> None:

        msgType = type(received_msg) 

        if (msgType is DetectedObjects):
            self.detectedObjs_lst.append(received_msg)
            self.get_logger().info("Received 'detected' objects.")

        elif (msgType is TrackedObjects):
            self.trackedObjs_lst.append(received_msg)
            self.get_logger().info("Received 'tracked' objects.")

        else:
            self.get_logger().warning("Unspecified message type observed:" + str(msgType))

    # Function to publish computed predictions (to 'prediction'-topic):
    def publish(self, pred_msg) -> None:

        # ...
        self.prediction_pub.publish(pred_msg)
        self.get_logger().info("Published predicted objects.")

    # -----------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    # Function to compute the predictions (based on the received/stored messages):
    def prediction(self, received_msg = None) -> PredictedObjects:

        # TODO: <YOUR CODE>
        pred_objects = PredictedObjects()
        pass

        # ...
        return pred_objects


# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# ...
def main(args = None):

    # ...
    rclpy.init(args = args)
    data_predictor = DataPredictor()

    # ...
    rclpy.spin(data_predictor)

    # ...
    data_predictor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

