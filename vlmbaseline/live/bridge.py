"""ROS 2 node: RTAB-Map localization in a saved map -> the user's pose in the export's world frame -> the demo server.

Runs where ROS 2 is (the hydra-dev container), needs only rclpy, numpy and the standard library; posecore.py next to
it (or the vlmbaseline package) does the pose:

    python3 bridge.py --align lab_03_align.json --scene lab_03 [--server http://127.0.0.1:8765] [--log poses.jsonl]
                      [--raw raw.jsonl] [--no-gate]

Inputs: /rtabmap/odom (odometry, every frame), /rtabmap/localization_pose (published when the frame was matched to the
saved map) and /rtabmap/info (which map node it matched); see posecore.py for the pose and the jump check.
Every 1 / rate s (default 2 Hz) it POSTs {scene, pose: [x, y, yaw], localized, uncertain, since_loc_s, node, ...} to
/api/live and writes the same line to --log. --raw keeps the inputs (odometry, localizations, matched nodes) so the
pose can be recomputed offline with other settings (live/replay_raw.py).
"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.request

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import Info

try:
    from vlmbaseline.live.posecore import PoseCore, T_of, flat
except ImportError:  # copied next to this file in the container
    from posecore import PoseCore, T_of, flat


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class Bridge(Node):
    def __init__(self, a):
        super().__init__("vlm_pose_bridge")
        al = json.load(open(a.align))
        self.scene, self.server, self.rate = a.scene or al["scene"], a.server, a.rate
        self.core = PoseCore(al, gate=not a.no_gate)
        self.log = open(a.log, "a") if a.log else None
        self.raw = open(a.raw, "a") if a.raw else None
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/rtabmap/odom", self.on_odom, qos)
        self.create_subscription(PoseWithCovarianceStamped, "/rtabmap/localization_pose", self.on_loc, qos)
        self.create_subscription(Info, "/rtabmap/info", self.on_info, qos)
        self.create_timer(1.0 / self.rate, self.tick)
        self.busy = False

    def _raw(self, **kw):
        if self.raw:
            self.raw.write(json.dumps(kw) + "\n")

    def on_odom(self, m):
        t = ns(m.header.stamp)
        if m.pose.covariance[0] >= 9999 or (m.pose.pose.orientation.w == 0 and m.pose.pose.orientation.z == 0):
            self.core.on_odom(t, None, lost=True)  # odometry lost / reset
            self._raw(k="lost", t=t)
            return
        T = T_of(m.pose.pose)
        self.core.on_odom(t, T)
        self._raw(k="o", t=t, T=flat(T))

    def on_loc(self, m):
        t = ns(m.header.stamp)
        T = T_of(m.pose.pose)
        r = self.core.on_loc(t, T)
        self._raw(k="l", t=t, T=flat(T), r=r)
        if r == "rejected":
            self.get_logger().info(f"localization {self.core.n_rejected} far from odometry, not taken yet")

    def on_info(self, m):
        nid = m.loop_closure_id or m.proximity_detection_id
        self.core.on_info(ns(m.header.stamp), nid)
        if nid > 0:
            self._raw(k="i", t=ns(m.header.stamp), n=int(nid))

    def tick(self):
        rec = self.core.pose()
        if rec is None:
            return
        rec = {"scene": self.scene, **rec}
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        if self.raw:
            self.raw.flush()
        if self.server and not self.busy:
            self.busy = True
            threading.Thread(target=self._post, args=(rec,), daemon=True).start()

    def _post(self, rec):
        try:
            req = urllib.request.Request(self.server.rstrip("/") + "/api/live", data=json.dumps(rec).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=1.0).read()
        except Exception as e:  # the server may be down; keep running
            self.get_logger().warn(f"post failed: {e}", throttle_duration_sec=10.0)
        finally:
            self.busy = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", required=True)
    ap.add_argument("--scene")
    ap.add_argument("--server", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--raw", default="", help="keep the inputs (for live/replay_raw.py)")
    ap.add_argument("--no-gate", action="store_true", help="take every localization at once (no jump check)")
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=["bridge"] + ros_args)
    node = Bridge(a)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
