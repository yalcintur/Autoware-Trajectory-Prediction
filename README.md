### Autoware Trajectory Prediction and Visualization with GRIP++

---

**Description**:

This repository provides tools for trajectory prediction and visualization within the Autoware ecosystem, integrating seamlessly with GRIP++. Using state-of-the-art prediction algorithms, this project facilitates accurate forecasting of dynamic agents in the vicinity of an autonomous vehicle, bolstered by an intuitive visualization interface provided by GRIP++.

---

**Features**:

1. **Trajectory Prediction**: Offers advanced predictive modeling to forecast movement paths of detected and tracked objects using GRIP++.
2. **Integration with Autoware Messages**: Seamlessly accepts and emits messages of types `autoware_auto_perception_msgs/msg/TrackedObjects`, `/tf`, and `autoware_auto_perception_msgs/msg/PredictedObjects`.
3. **Visual Interface with Autoware**: Makes use of Autoware's robust graphical tools to provide clear visualizations of predicted trajectories alongside real-time sensor data.
4. **Extensibility**: Designed with flexibility in mind, allowing users to easily modify and enhance prediction algorithms by simply altering the `prediction` function in the `DataPredictor` class.

---

**Usage**:

1. **Environment Setup**:
   ```
   ade start
   ade enter
   ```
2. **Building and Setting Up ROS2 Package**:
   ```
   colcon build --symlink-install
   source install/local_setup.zsh
   ```

3. **Running Predictor Node**:
   ```
   ros2 run predictor_template predictor
   ```

4. **Playing Back Recorded Data**:
   ```
   ros2 bag play <path to 'tracking_data_for_prediction.bag' directory>
   ```

5. **Listening to Published Predictions** *(Optional)*:
   ```
   ros2 topic echo prediction
   ```

---

**Modification**:

To enhance or modify the prediction algorithm, inject your code into the `prediction` function of the `DataPredictor` class located in the `data_predictor.py` file within the `src/predictor_template/predictor_template` directory. Afterwards, execute the aforementioned ROS2 package building steps.

For a deep dive into the structure of message types, consult the [autoware documentation](https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/classautoware_1_1perception_1_1tracking_1_1_tracked_object.html) or the [specified repository](https://github.com/tier4/autoware_auto_msgs/tree/tier4/main/autoware_auto_perception_msgs/msg).

---

**Dependencies**:

- [Autoware](https://autowarefoundation.gitlab.io/autoware.auto/AutowareAuto/installation-ade.html) with ADE
- Grip++
- ROS2 Foxy

---

**Contributions**:

We welcome contributions! Please follow our contribution guidelines to ensure a smooth collaboration process.
![Screenshot from 2023-09-06 15-58-32](https://github.com/yalcintur/Autoware-Trajectory-Prediction/assets/42304303/ff40f8c2-aedd-4f0b-b91f-86013b00e985)


[![Demo Video]([https://img.youtube.com/vi/VID/0.jpg](https://github.com/yalcintur/Autoware-Trajectory-Prediction/assets/42304303/ff40f8c2-aedd-4f0b-b91f-86013b00e985))](https://www.youtube.com/watch?v=__JSOTbNtgE)

This project marries the prowess of Autoware's trajectory prediction with the visualization brilliance of Grip++, promising a comprehensive solution for autonomous vehicle applications that demand both accuracy and clarity in trajectory forecasting.
