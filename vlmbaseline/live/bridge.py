"""ROS 2 node: RTAB-Map localization in a saved map -> the user's pose in the export's world frame -> the demo server.

Runs where ROS 2 is (the hydra-dev container), needs only rclpy, numpy and the standard library:

    python3 bridge.py --align lab_03_align.json --scene lab_03 [--server http://127.0.0.1:8765] [--log poses.jsonl]

Inputs: /rtabmap/odom (odometry, every frame), /rtabmap/localization_pose (published when the frame was matched to the
saved map) and /rtabmap/info (which map node it matched). map_T_odom is updated on every localization; between them
the pose follows odometry. The pose in the world frame uses the correction of the matched node's pass (align.py):
of the map nodes within NEAR_M of the pose, the one whose id is closest to the last matched node (the same pass: a
place passed twice can differ by ~2 m between the saved map and the export).
Every 1 / rate s (default 2 Hz) it POSTs {scene, pose: [x, y, yaw], localized, since_loc_s, node} to /api/live
and writes the same line to --log.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import threading
import time
import urllib.request

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import Info

NEAR_M = 4.0  # the correction comes from a map node this close to the pose


def T_of(p):
    q, t = p.orientation, p.position
    w, x, y, z = q.w, q.x, q.y, q.z
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class Bridge(Node):
    def __init__(self, a):
        super().__init__("vlm_pose_bridge")
        al = json.load(open(a.align))
        self.scene, self.server, self.rate = a.scene or al["scene"], a.server, a.rate
        self.B_S = np.array(al["body_T_sensor"])
        self.ids = np.array([n["id"] for n in al["nodes"]])
        self.pos = np.array([n["pos"] for n in al["nodes"]])
        self.corr = np.array([n["corr"] for n in al["nodes"]])
        self.log = open(a.log, "a") if a.log else None
        self.odom_t, self.odom_T = [], []  # recent odometry by stamp (to pair with a localization)
        self.map_T_odom, self.loc_ns, self.node = None, None, None
        self.n_loc = 0
        self.matched_once = False  # a frame was matched to the saved map (before that the pose is only a guess)
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, "/rtabmap/odom", self.on_odom, qos)
        self.create_subscription(PoseWithCovarianceStamped, "/rtabmap/localization_pose", self.on_loc, qos)
        self.create_subscription(Info, "/rtabmap/info", self.on_info, qos)
        self.create_timer(1.0 / self.rate, self.tick)
        self.busy = False

    def on_odom(self, m):
        if m.pose.covariance[0] >= 9999 or (m.pose.pose.orientation.w == 0 and m.pose.pose.orientation.z == 0):
            return  # odometry lost / reset
        self.odom_t.append(ns(m.header.stamp))
        self.odom_T.append(T_of(m.pose.pose))
        if len(self.odom_t) > 600:
            del self.odom_t[:100], self.odom_T[:100]

    def _odom_at(self, t):
        i = bisect.bisect_left(self.odom_t, t)
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.odom_t)), key=lambda j: abs(self.odom_t[j] - t),
                   default=None)
        if best is None or abs(self.odom_t[best] - t) > 100_000_000:
            return None
        return self.odom_T[best]

    def on_loc(self, m):
        t = ns(m.header.stamp)
        O = self._odom_at(t)
        if O is None:
            return
        self.map_T_odom = T_of(m.pose.pose) @ np.linalg.inv(O)
        self.loc_ns = t
        self.n_loc += 1

    def on_info(self, m):
        nid = m.loop_closure_id or m.proximity_detection_id
        if nid > 0:
            self.node = nid
            self.matched_once = True

    def tick(self):
        if not self.odom_t:
            return
        t = self.odom_t[-1]
        ok = self.map_T_odom is not None and self.matched_once
        rec = {"scene": self.scene, "stamp_ns": t, "localized": ok, "n_loc": self.n_loc}
        if ok:
            M = self.map_T_odom @ self.odom_T[-1]
            p = M[:3, 3]
            d = np.linalg.norm(self.pos - p, axis=1)
            k = None
            if self.node is not None:  # the map node of the last match, or the closest id (same pass) near here
                near = d < NEAR_M
                if near.any():
                    k = int(np.argmin(np.where(near, np.abs(self.ids - self.node), np.inf)))
            if k is None:
                k = int(np.argmin(d))
            S = self.corr[k] @ M @ self.B_S
            f = S[:3, 2]
            rec.update(pose=[round(float(S[0, 3]), 3), round(float(S[1, 3]), 3), round(math.atan2(f[1], f[0]), 4)],
                       since_loc_s=round((t - self.loc_ns) / 1e9, 2), node=int(self.ids[k]), matched=self.node,
                       map_T=[round(float(v), 4) for v in M[:3].ravel()])
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
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
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=["bridge"] + ros_args)
    node = Bridge(a)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
