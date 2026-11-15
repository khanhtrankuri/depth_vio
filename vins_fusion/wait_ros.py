"""Wait for ROS master / all VINS sensor subscriptions (ROS Melodic Python 2)."""
import sys
import time
import rosgraph

deadline = time.time() + 60
while time.time() < deadline:
    try:
        state = rosgraph.Master('/integration_wait').getSystemState()
        subscribers = dict(state[1])
        if sys.argv[1] == 'master' or all(subscribers.get(t) for t in
                ['/cam0/image_raw', '/cam1/image_raw', '/imu0']):
            sys.exit(0)
    except Exception:
        pass
    time.sleep(0.2)
raise SystemExit('Timed out waiting for ROS ' + sys.argv[1])
