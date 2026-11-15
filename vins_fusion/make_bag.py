"""EuRoC ASL -> chronologically ordered ROS1 bag; run in Melodic Python 2."""
import csv
import heapq
import os
import sys
import cv2
import rosbag
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Imu


def rows(root, sensor):
    with open(os.path.join(root, 'mav0', sensor, 'data.csv')) as handle:
        for row in csv.reader(line for line in handle if not line.startswith('#')):
            if row:
                yield int(row[0]), sensor, row[1:]


def main():
    root, output = sys.argv[1:]
    streams = [iter(rows(root, name)) for name in ['cam0', 'cam1', 'imu0']]
    queue = []
    for index, stream in enumerate(streams):
        t, sensor, values = next(stream)
        heapq.heappush(queue, (t, index, sensor, values))
    bridge = CvBridge()
    temporary = output + '.partial'
    count = 0
    with rosbag.Bag(temporary, 'w') as bag:
        while queue:
            t, index, sensor, values = heapq.heappop(queue)
            stamp = rospy.Time(t // 1000000000, t % 1000000000)
            if sensor == 'imu0':
                message = Imu()
                v = list(map(float, values))
                message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = v[:3]
                message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = v[3:6]
                message.orientation_covariance[0] = -1
                topic = '/imu0'
            else:
                path = os.path.join(root, 'mav0', sensor, 'data', values[0].strip())
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError('Cannot read ' + path)
                message = bridge.cv2_to_imgmsg(image, encoding='mono8')
                topic = '/' + sensor + '/image_raw'
            message.header.stamp = stamp
            message.header.frame_id = sensor
            bag.write(topic, message, stamp)
            count += 1
            if count % 10000 == 0:
                print('Wrote %d sensor messages' % count)
                sys.stdout.flush()
            try:
                t, sensor, values = next(streams[index])
                heapq.heappush(queue, (t, index, sensor, values))
            except StopIteration:
                pass
    os.rename(temporary, output)
    print('Wrote %d messages to %s' % (count, output))


if __name__ == '__main__':
    main()
