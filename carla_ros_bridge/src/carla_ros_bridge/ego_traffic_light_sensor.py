#!/usr/bin/env python

from collections import defaultdict
from typing import Dict

import carla
from carla_msgs.msg import CarlaEgoTrafficLightInfo, CarlaTrafficLightStatus
from ros_compatibility.node import CompatibleNode
from ros_compatibility.qos import DurabilityPolicy, QoSProfile

from carla_ros_bridge.actor import Actor
from carla_ros_bridge.ego_vehicle import EgoVehicle
from carla_ros_bridge.pseudo_actor import PseudoActor
from carla_ros_bridge.traffic import TrafficLight

# Publish info no less frequently than this.
_PUBLISH_INTERVAL_SECONDS = 5.0

class StoplineInfo:
    """An util class to keep track of a traffic light and its stop location."""

    def __init__(self, traffic_light_actor: TrafficLight, approaching_road_id: int, approaching_lane_id: int, stop_location: carla.Location):
        self.traffic_light_actor = traffic_light_actor
        self.stop_location = stop_location
        self.approaching_road_id = approaching_road_id
        self.approaching_lane_id = approaching_lane_id


def create_stop_line_info_map(node: CompatibleNode, actor_list: Dict[int, Actor], search_distance: float = 2.0):
    """ Return a dictionary of stop line info.

        Args:
            node: ROS node wrapper for logging purposes
            actor_list: A dictionary of spawn actors. Use this to search for traffic light actors
            search_distance: the approximate distance where to get the previous waypoints
        Return:
            A map(road_id, lane_id -> StopLineInfo)
    """
    results = defaultdict(dict)
    for actor in actor_list.values():
        if not isinstance(actor, TrafficLight):
            continue

        carla_actor = actor.carla_actor
        affected_points = carla_actor.get_affected_lane_waypoints()
        if len(affected_points) == 0:
            node.logerr(f"Unable to find any affected points for traffic light id: {carla_actor.id}")
            continue

        for point in affected_points:
            if not point.is_junction:
                node.logerr(
                    f"The affected waypoint from traffic light is not in a junction. Traffic_light_id: {carla_actor.id}, road_id: {point.road_id}, lane_id: {point.lane_id}"
                )
                continue

            # Go to previous waypoints to get the road id and lane id on which the ego will be approaching.
            previous_points = point.previous(search_distance)
            if len(previous_points) == 0:
                node.logerr(
                    f"Unable to find previous points for traffic_light_id: {carla_actor.id}, road_id: {point.road_id}, lane_id: {point.lane_id}"
                )
                continue

            # This stop location is roughly at the stop line. It depends on how the map is created in RoadRunner.
            stop_location = point.transform.location
            p = previous_points[0]
            results[p.road_id][p.lane_id] = StoplineInfo(actor, p.road_id, p.lane_id, stop_location)

    return results


def get_stop_line_info(stop_line_info: Dict[int,Dict[int, StoplineInfo]], road_id: int, lane_id: int):
    """Given road_id and lane_id, find and return a StopLineInfo object if any."""
    if stop_line_info is None:
        return None
    if road_id in stop_line_info and lane_id in stop_line_info[road_id]:
        return stop_line_info[road_id][lane_id]
    return None


class EgoTrafficLightSensor(PseudoActor):
    """A sensor to publish CarlaEgoTrafficLightInfo"""

    def __init__(self, uid: int, name: str, parent: Actor, node: CompatibleNode, actor_list: Dict[int, Actor], carla_map: carla.Map):
        super(EgoTrafficLightSensor, self).__init__(uid=uid, name=name, parent=parent, node=node)

        self.pub = node.new_publisher(
            CarlaEgoTrafficLightInfo,
            self.get_topic_prefix() + "/info",
            qos_profile=QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )

        self._info_published_at = None

        self.map = carla_map
        self.map_name = self.map.name
        self.ego = None
        self.actor_list = actor_list
        self.stop_line_info_map = None
        self.cur_sli = None

        self.msg = CarlaEgoTrafficLightInfo()
        self.msg.traffic_light_status = CarlaTrafficLightStatus()
        self.msg.inside_intersection = False

    def destroy(self):
        super(EgoTrafficLightSensor, self).destroy()
        self.node.destroy_publisher(self.pub)
        self.actor_list = None
        self.stop_line_info_map = None
        self.node.loginfo("Destroy EgoTrafficLightSensor")

    @staticmethod
    def get_blueprint_name():
        return "sensor.pseudo.ego_traffic_light"

    def update(self, frame: int, timestamp: float):
        try:
            if self.map is None:
                self.node.logwarn("Carla Map is not assgined")
                return

            if self.stop_line_info_map is None or self.map_name != self.map.name:
                self.stop_line_info_map = create_stop_line_info_map(self.node, self.actor_list)
                self.map_name = self.map.name
                self.ego = self.get_ego(self.actor_list)

            if self.ego is None:
                self.node.logwarn("Unable to find ego.")
                return

            ego_location = self.ego.get_location()
            wp = self.map.get_waypoint(ego_location)
            sli = get_stop_line_info(self.stop_line_info_map, wp.road_id, wp.lane_id)

            inside_intersection = wp.is_junction
            cur_status = self.cur_sli.traffic_light_actor.get_status() if self.cur_sli else None
            new_status = sli.traffic_light_actor.get_status() if sli else None

            if (self.msg.inside_intersection != inside_intersection
                or self.has_traffic_light_status_changes(cur_status, new_status)
                or self._info_published_at is None
                or timestamp - self._info_published_at > _PUBLISH_INTERVAL_SECONDS
            ):
                self.calculate_and_publish_data(ego_location, sli, inside_intersection, timestamp)
                self._info_published_at = timestamp
        except Exception as e:
            self.node.loginfo("Error: {}".format(e))

    def calculate_and_publish_data(self, ego_location: carla.Location, new_sli: StoplineInfo, inside_intersection: bool, timestamp: float):
        """Publish CarlaEgoTrafficLightInfo message."""
        if inside_intersection:
            self.msg.distance_to_stopline = -1.0
            if self.cur_sli:
                self.msg.traffic_light_status = self.cur_sli.traffic_light_actor.get_status()
        else:
            if new_sli:
                self.msg.distance_to_stopline = ego_location.distance(new_sli.stop_location)
                self.msg.traffic_light_status = new_sli.traffic_light_actor.get_status()
            else:
                self.msg.distance_to_stopline = -1.0
                self.msg.traffic_light_status = CarlaTrafficLightStatus()
            # Store new StopLineInfo.
            self.cur_sli = new_sli

        self.msg.ego_id = self.ego.id
        self.msg.inside_intersection = inside_intersection
        self.msg.header = self.get_msg_header(timestamp=timestamp)
        self.pub.publish(self.msg)

    def get_ego(self, actor_list: Dict[int, Actor]):
        """Return a CarlaActor representing the ego car."""
        for actor in actor_list.values():
            if isinstance(actor, EgoVehicle):
                return actor.carla_actor
        return None

    def has_traffic_light_status_changes(self, cur_traffic_light_status: CarlaTrafficLightStatus, new_traffic_light_status: CarlaTrafficLightStatus):
        """Return True if the traffic light status (either traffic light id or state) has been changed; False otherwise."""
        if cur_traffic_light_status is None or new_traffic_light_status is None:
            # Don't detect changes when one of the status is None.
            return False

        return cur_traffic_light_status != new_traffic_light_status
