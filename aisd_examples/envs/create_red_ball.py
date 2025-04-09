import math
import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2

class CreateRedBallEnv(gym.Env):
    def __init__(self, render_mode=None):
        # Observation: integer pixel value from 0 (left) to 640 (right)
        self.observation_space = spaces.Discrete(641)  # 0 to 640 inclusive
        # Action: Discrete integer representing an x coordinate between 0 and 640.
        self.action_space = spaces.Discrete(641)
        self.render_mode = render_mode

        # Initialize ROS 2 and create the RedBall node.
        rclpy.init()
        self.redball = RedBall()

        # Initialize state variables:
        self.state = 320  # Start with the ball “in the center”
        self.step_count = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        # Handle seeding if necessary.
        super().reset(seed=seed)
        # Reset episode step count.
        self.step_count = 0
        # Reset state to current redball position (or default to 320).
        self.state = self.redball.redball_position if self.redball.redball_position is not None else 320
        return self.state, {}

    def step(self, action):
        # Convert the action (an integer 0-640) to a Twist command.
        twist = Twist()
        # Compute angular velocity: map 320 to 0 and scale to ±π/2.
        twist.angular.z = (action - 320) / 320 * (math.pi / 2)
        # Publish the Twist message.
        self.redball.twist_publisher.publish(twist)

        # Call spin_once to process callbacks.
        rclpy.spin_once(self.redball)
        # Wait until the robot has finished turning. (This flag should be updated by a subscriber.)
        while not self.redball.create3_is_stopped:
            rclpy.spin_once(self.redball)
        
        # Get new observation from the redball node.
        observation = self.redball.redball_position if self.redball.redball_position is not None else 320
        self.state = observation

        # Compute reward: here, a simple function that gives higher reward when the ball is near the center.
        reward = -abs(observation - 320)

        # Increment the step counter.
        self.step_count += 1
        terminated = (self.step_count >= self.max_steps)
        truncated = False
        info = {"step": self.step_count}
        return observation, reward, terminated, truncated, info

    def render(self):
        # No human rendering is provided.
        pass

    def close(self):
        # Clean up the ROS node and shutdown ROS 2.
        self.redball.destroy_node()
        rclpy.shutdown()


class RedBall(Node):
    """
    A ROS 2 Node that processes camera images to determine the red ball's x-axis position,
    publishes annotated images on the 'target_redball' topic, and sends Twist messages.
    """
    def __init__(self):
        super().__init__('redball')
        # Subscription to the simulated camera topic.
        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning

        # CvBridge for image conversion.
        self.br = CvBridge()
        # Publisher for an annotated image.
        self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
        # Publisher for Twist messages (for robot control).
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # The current x-axis pixel value of the red ball in the image.
        self.redball_position = 320  # Default to center.
        # Flag indicating whether the Create 3 robot has stopped moving.
        self.create3_is_stopped = True  # Initially stopped; in a real implementation, this would be updated.

    def listener_callback(self, msg):
        # Convert ROS Image message to an OpenCV image.
        frame = self.br.imgmsg_to_cv2(msg)

        # Convert image to HSV.
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define color bounds for the red ball.
        bright_red_lower_bounds = (110, 100, 100)
        bright_red_upper_bounds = (130, 255, 255)
        # Create a mask.
        bright_red_mask = cv2.inRange(hsv_conv_img, bright_red_lower_bounds, bright_red_upper_bounds)
        # Blur the mask.
        blurred_mask = cv2.GaussianBlur(bright_red_mask, (9, 9), 3, 3)

        # Morphological operations to remove noise.
        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        eroded_mask = cv2.erode(blurred_mask, erode_element)
        dilated_mask = cv2.dilate(eroded_mask, dilate_element)

        # Detect circles using HoughCircles.
        detected_circles = cv2.HoughCircles(dilated_mask, cv2.HOUGH_GRADIENT, 1, 150,
                                            param1=100, param2=20, minRadius=2, maxRadius=2000)
        the_circle = None
        if detected_circles is not None:
            # Assume the first detected circle corresponds to the red ball.
            for circle in detected_circles[0, :]:
                # Draw the circle onto the frame.
                circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]),
                                           (0, 255, 0), thickness=3)
                the_circle = (int(circle[0]), int(circle[1]))
            # Update the red ball observation as the x-axis coordinate.
            self.redball_position = int(the_circle[0])
            # Publish the annotated image.
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
        else:
            # If no red ball is detected, default the observation to center.
            self.get_logger().info('no ball detected')
            self.redball_position = 320

        # Here, you would update self.create3_is_stopped based on feedback (e.g. subscribing to a stop_status topic).
        # For this candidate, we assume the robot is stopped once the callback processes the image.
        self.create3_is_stopped = True

