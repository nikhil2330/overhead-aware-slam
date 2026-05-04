import os
import signal
import subprocess
import time

import rclpy

from sensor_fusion_nodes.teleop_snapshot import TeleopSnapshot


def stop_launch_process(proc):
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=10.0)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5.0)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass


def main():
    launch_log = '/tmp/miniature_waffle_capture_world.log'
    launch_cmd = [
        'ros2',
        'launch',
        'sensor_fusion_run',
        'capture_world.launch.py',
    ]

    launch_proc = None

    with open(launch_log, 'w', encoding='utf-8') as log_file:
        launch_proc = subprocess.Popen(
            launch_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        print(f'Started Gazebo world in background. Log: {launch_log}')
        time.sleep(4.0)

        if launch_proc.poll() is not None:
            print('Gazebo launch exited early. Check the log above.')
            return

        rclpy.init()
        node = TeleopSnapshot()

        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            stop_launch_process(launch_proc)


if __name__ == '__main__':
    main()
