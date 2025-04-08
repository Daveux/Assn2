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
        # Set up arbitrary discrete observation and action spaces.
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(5)
        rclpy.init()
        self.redball = RedBall()
        
        # Initialize the state.
        self.state = 0

    def reset(self, seed=None, options=None):
        # Call the parent reset to handle seeding.
        super().reset(seed=seed)
        
        # Choose an arbitrary state from the observation space.
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        # Choose another arbitrary state.
        self.state = self.observation_space.sample()

        rclpy.spin_once(self.redball)
        
        # Provide an arbitrary reward (here, a fixed reward of 1).
        reward = 1
        
        # Set termination and truncation arbitrarily.
        terminated = False
        truncated = False
        
        # Use an empty dict for info.
        info = {}
        #print(f"Successfully connected to step \n Action taken: {action}, New state: {self.state}, Reward: {reward}")
        return self.state, reward, terminated, truncated, info

    def render(self):
        # Render does nothing for now.
        pass

    def close(self):
        # Clean up the environment.
        self.redball.destroy_node()
        rclpy.shutdown()       

class RedBall(Node):
  """
  A Node to analyse red balls in images and publish the results
  """
  def __init__(self):
    super().__init__('redball')
    self.subscription = self.create_subscription(
      Image,
      'custom_ns/camera1/image_raw',
      self.listener_callback,
      10)
    self.subscription # prevent unused variable warning

    # A converter between ROS and OpenCV images
    self.br = CvBridge()
    self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
    self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

  def listener_callback(self, msg):
    frame = self.br.imgmsg_to_cv2(msg)

    # convert image to BGR format (red ball becomes blue)
    hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    bright_red_lower_bounds = (110, 100, 100)
    bright_red_upper_bounds = (130, 255, 255)
    bright_red_mask = cv2.inRange(hsv_conv_img, bright_red_lower_bounds, bright_red_upper_bounds)

    blurred_mask = cv2.GaussianBlur(bright_red_mask,(9,9),3,3)
# some morphological operations (closing) to remove small blobs
    erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    eroded_mask = cv2.erode(blurred_mask,erode_element)
    dilated_mask = cv2.dilate(eroded_mask,dilate_element)

    # on the color-masked, blurred and morphed image I apply the cv2.HoughCircles-method to detect circle-shaped objects
    detected_circles = cv2.HoughCircles(dilated_mask, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=20, minRadius=2, maxRadius=2000)
    the_circle = None
    if detected_circles is not None:
        for circle in detected_circles[0, :]:
            circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0),thickness=3)
            the_circle = (int(circle[0]), int(circle[1]))
        self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
    else:
        self.get_logger().info('no ball detected')
