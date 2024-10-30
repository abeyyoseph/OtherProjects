import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/abey/Desktop/Repos/OtherProjects/ROS2_Husky_UGV/install/husky_nav'
