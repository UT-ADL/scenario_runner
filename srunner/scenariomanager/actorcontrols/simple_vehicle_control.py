#!/usr/bin/env python

# Copyright (c) 2020-2021 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an example control for vehicles which
does not use CARLA's vehicle engine.

Limitations:
- Does not respect any traffic regulation: speed limit, priorities, etc.
- Can only consider obstacles in forward facing reaching (i.e. in tight corners obstacles may be ignored).
"""

from distutils.util import strtobool
import math

import carla

from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.actorcontrols.visualizer import Visualizer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


class SimpleVehicleControl(BasicControl):

    """
    Controller class for vehicles derived from BasicControl.

    The controller directly sets velocities in CARLA, therefore bypassing
    CARLA's vehicle engine. This allows a very precise speed control, but comes
    with limitations during cornering.

    In addition, the controller can consider blocking obstacles, which are
    classified as dynamic (i.e. vehicles, bikes, pedestrians). Activation of this
    features is controlled by passing proper arguments to the class constructor.
    The collision detection uses CARLA's obstacle sensor (sensor.other.obstacle),
    which checks for obstacles in the direct forward channel of the vehicle, i.e.
    there are limitation with sideways obstacles and while cornering.

    The controller can also respect red traffic lights, and use a given deceleration
    value for more realistic behavior. Both behaviors are activated via the corresponding
    controller arguments. With comfort_acceleration / comfort_deceleration the speed changes
    at a comfortable rate, braking harder only when an obstacle or a red light needs it.

    Args:
        actor (carla.Actor): Vehicle actor that should be controlled.
        args (dictionary): Dictonary of (key, value) arguments to be used by the controller.
                           May include: (consider_obstacles, true/false)     - Enable consideration of obstacles
                                        (proximity_threshold, distance)      - Distance in front of actor in which
                                                                               obstacles are considered
                                        (obstacle_margin, distance)          - Gap kept to an obstacle: the speed
                                                                               is matched this far before it
                                                                               (default 0 = at the obstacle)
                                        (obstacle_path_width, distance)      - Only obstacles whose bounding
                                                                               box (XY) comes within this
                                                                               distance of the actor's own path
                                                                               (its remaining points) count; the
                                                                               distance from the path to the
                                                                               obstacle's box, 0 if they overlap
                                                                               (default: every obstacle the sensor
                                                                               sees, straight ahead)
                                        (waypoint_reached_threshold, distance) - Distance between actor and waypoint
                                                                               determining reached or not
                                        (consider_trafficlights, true/false) - Enable consideration of traffic lights
                                        (max_deceleration, float)            - Use a reasonable deceleration value for
                                                                               this vehicle
                                        (max_acceleration, float)            - Use a reasonable acceleration value for
                                                                               this vehicle
                                        (comfort_deceleration, float)        - Braking rate [m/s2]: the speed falls
                                                                               by at most this much per second, or
                                                                               by what an obstacle or a red light
                                                                               needs (see _set_new_velocity); a tick
                                                                               needing more than
                                                                               MAX_NEEDED_DECELERATION is stock.
                                                                               Wins over max_deceleration; not on
                                                                               a timed path
                                        (comfort_acceleration, float)        - The speed rises by at most this much
                                                                               per second [m/s2]. Wins over
                                                                               max_acceleration; not on a timed
                                                                               path
                                        (attach_camera, true/false)          - Attach OpenCV display to actor
                                                                               (useful for debugging)

    Attributes:

        _generated_waypoint_list (list of carla.Transform): List of target waypoints the actor
            should travel along. A waypoint here is of type carla.Transform!
            Defaults to [].
        _last_update (float): Last time step the update function (tick()) was called.
            Defaults to None.
        _consider_obstacles (boolean): Enable/Disable consideration of obstacles
            Defaults to False.
        _proximity_threshold (float): Distance in front of actor in which obstacles are considered
            Defaults to infinity.
        _waypoint_reached_threshold(float): Distance between actor and waypoint determining reached or not
            Defaults to 4.0 meters.
        _cv_image (CV Image): Contains the OpenCV image, in case a debug camera is attached to the actor
            Defaults to None.
        _camera (sensor.camera.rgb): Debug camera attached to actor
            Defaults to None.
        _obstacle_sensor (sensor.other.obstacle): Obstacle sensor attached to actor
            Defaults to None.
        _obstacle_distance (float): Distance of the closest obstacle returned by the obstacle sensor
            Defaults to infinity.
        _obstacle_actor (carla.Actor): Closest obstacle returned by the obstacle sensor
            Defaults to None.
    """

    # With comfort_deceleration: the hardest braking [m/s2] an obstacle or a red light may ask for, about a full stop
    # on dry asphalt. A tick that needs more (an obstacle first sensed too close, within obstacle_margin, a red light
    # seen too late) is stock -- its instant stop or target -- so the car avoids whatever stock avoids.
    MAX_NEEDED_DECELERATION = 9.0
    # With comfort_deceleration the car brakes onto its path's last point: the point counts as reached this close
    # (m), and the car covers the last bit at this crawl (m/s) -- braked to 0 short of the threshold it would stand
    # there and never complete its path; from the crawl the final stop is gentle.
    _END_REACHED_M = 0.5
    _END_CREEP_MS = 0.3
    # s; the longest time one call may move the speed by under the comfort limits: a longer gap since the last
    # command (a car that stood at a reached goal, then got a new path) would otherwise allow the jump they prevent.
    _MAX_RATE_STEP_S = 0.25
    # m; a red light's stop waypoint counts as this actor's stop line only this far beside its heading line (the
    # light's other lanes are a lane width or more away; the nearest one is taken).
    _STOP_LINE_LATERAL_M = 5.0
    # m/s; the comfort limits step from the last commanded speed while the measured one stays this close to it
    # (physics takes a little drag off each tick; stepping from the measured speed added that drag to every step, and
    # a drag above comfort_acceleration kept the car below its target for good). Further off (blocked, teleported):
    # the measured speed.
    _FOLLOW_TOLERANCE_MPS = 1.0
    # m; with comfort_deceleration the speed behind an obstacle aims this far outside obstacle_margin: aimed at the
    # margin itself a car following at the obstacle's speed sat on its edge, and any rounding there meant "within the
    # margin: stop" (a saw-tooth every 1.5 s in a 20 Hz simulation).
    _FOLLOW_BUFFER_M = 0.5

    def __init__(self, actor, args=None):
        super(SimpleVehicleControl, self).__init__(actor)
        self._generated_waypoint_list = []
        self._last_update = None
        self._consider_traffic_lights = False
        self._consider_obstacles = False
        self._proximity_threshold = float('inf')
        self._obstacle_margin = 0.0
        self._obstacle_path_width = None
        self._waypoint_reached_threshold = 4.0
        self._max_deceleration = None
        self._max_acceleration = None
        self._comfort_deceleration = None
        self._comfort_acceleration = None

        self._obstacle_sensor = None
        self._obstacle_distance = float('inf')
        self._obstacle_actor = None
        # Read once per actor (actor.bounding_box is a blocking RPC on CARLA 0.9.16, ~11 ms): obstacle boxes by
        # actor id, red-light stop points by light id, and this actor's half length.
        self._obstacle_boxes = {}
        self._stop_lines = {}
        self._half_length = None
        self._commanded_speed = None    # m/s; the speed of the last velocity _set_new_velocity set

        self._visualizer = None

        self._brake_lights_active = False

        if args and 'consider_obstacles' in args and strtobool(args['consider_obstacles']):
            self._consider_obstacles = strtobool(args['consider_obstacles'])
            bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.other.obstacle')
            bp.set_attribute('distance', '250')
            if args and 'proximity_threshold' in args:
                self._proximity_threshold = float(args['proximity_threshold'])
                bp.set_attribute('distance', str(max(float(args['proximity_threshold']), 250)))
            bp.set_attribute('hit_radius', '1')
            bp.set_attribute('only_dynamics', 'True')
            self._half_length = self._actor.bounding_box.extent.x
            self._obstacle_sensor = CarlaDataProvider.get_world().spawn_actor(
                bp, carla.Transform(carla.Location(x=self._half_length, z=1.0)), attach_to=self._actor)
            self._obstacle_sensor.listen(lambda event: self._on_obstacle(event))  # pylint: disable=unnecessary-lambda

        if args and 'obstacle_margin' in args:
            # The stock braking reaches the obstacle's speed AT it (0.5 dv^2 / d): bumper to bumper. VSE's export
            # asks for a gap (11-23: an NPC following a stopped ego stopped 0-0.3 m from its bumper).
            self._obstacle_margin = max(float(args['obstacle_margin']), 0.0)

        if args and 'obstacle_path_width' in args:
            # The obstacle sensor is a straight channel: it also brakes for traffic CROSSING in front, which the
            # car's own (curving) path never meets -- 11-23 two NPC bikes stopped for the ego crossing their junction
            # and the ego waited for them for good. VSE Play's NPC agent looks along its own route instead.
            # Measured to the obstacle's box, not its centre: 11-24 a bus whose centre was 2.6 m off a bike's path
            # (its side 0.6 m) did not count and the bike squeezed past its back corner with 0.2-0.6 m to spare.
            self._obstacle_path_width = max(float(args['obstacle_path_width']), 0.0)

        if args and 'consider_trafficlights' in args and strtobool(args['consider_trafficlights']):
            self._consider_traffic_lights = strtobool(args['consider_trafficlights'])

        if args and 'waypoint_reached_threshold' in args:
            self._waypoint_reached_threshold = float(args['waypoint_reached_threshold'])

        if args and 'max_deceleration' in args:
            self._max_deceleration = float(args['max_deceleration'])

        if args and 'max_acceleration' in args:
            self._max_acceleration = float(args['max_acceleration'])

        # The velocity is set directly: 11-24 VSE's NPC bikes went 13 -> 0.4 -> 13 m/s from one 0.2 s sample to the
        # next behind a slowing bus (0 within the margin, full speed again as soon as the bus was no slower). The
        # stock max_deceleration is a hard cap (no harder braking even to avoid a crash), so these are separate.
        if args and 'comfort_deceleration' in args:
            self._comfort_deceleration = max(float(args['comfort_deceleration']), 0.0)

        if args and 'comfort_acceleration' in args:
            self._comfort_acceleration = max(float(args['comfort_acceleration']), 0.0)

        if args and 'attach_camera' in args and strtobool(args['attach_camera']):
            self._visualizer = Visualizer(self._actor)

    def _obstacle_box(self, actor):
        """
        Return the actor's bounding box as (x, y offset from the actor, half length, half width, yaw offset [deg]),
        read once per actor id: actor.bounding_box is a blocking RPC on CARLA 0.9.16 (~11 ms), too slow per tick
        """
        box = self._obstacle_boxes.get(actor.id)
        if box is None:
            try:
                bbox = actor.bounding_box
                box = (bbox.location.x, bbox.location.y, bbox.extent.x, bbox.extent.y,
                       getattr(getattr(bbox, 'rotation', None), 'yaw', 0.0))
            except (AttributeError, RuntimeError) as exc:
                # No box to read: measure to its centre, as before the box was used.
                print("simple_vehicle_control: no bounding box for obstacle {} ({}), using its centre".format(
                    actor.id, exc))
                box = (0.0, 0.0, 0.0, 0.0, 0.0)
            self._obstacle_boxes[actor.id] = box
        return box

    def _obstacle_on_path(self):
        """
        True when the sensed obstacle counts: no obstacle_path_width set (stock), no path to compare with, or the
        obstacle's bounding box (XY, turned by its yaw) within obstacle_path_width of the actor's remaining path
        (from the actor through its points)
        """
        if self._obstacle_path_width is None or self._obstacle_actor is None or not self._waypoints:
            return True
        try:
            box_x, box_y, half_length, half_width, box_yaw = self._obstacle_box(self._obstacle_actor)
            # The obstacle's own snapshot (no RPC): CarlaDataProvider only caches the actors it tracks, and says
            # "not found" every tick for the others (e.g. traffic not spawned by the scenario).
            other = self._obstacle_actor.get_transform()
            here = CarlaDataProvider.get_location(self._actor)
        except RuntimeError:
            return True
        if other is None or here is None:
            return True
        yaw = math.radians(other.rotation.yaw)
        centre_x = other.location.x + box_x * math.cos(yaw) - box_y * math.sin(yaw)
        centre_y = other.location.y + box_x * math.sin(yaw) + box_y * math.cos(yaw)
        yaw = math.radians(other.rotation.yaw + box_yaw)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        # The path in the box's frame (box centre at the origin, its length along x)
        points = []
        for x, y in [(here.x, here.y)] + [(wp.location.x, wp.location.y) for wp in self._waypoints]:
            dx, dy = x - centre_x, y - centre_y
            points.append((dx * cos_yaw + dy * sin_yaw, -dx * sin_yaw + dy * cos_yaw))
        for start, end in zip(points, points[1:]):
            if _segment_box_distance(start, end, half_length, half_width) <= self._obstacle_path_width:
                return True
        return False

    def _own_half_length(self):
        """
        Return half the actor's length [m] (its bounding box, read once); 0 when it has none
        """
        if self._half_length is None:
            try:
                self._half_length = float(self._actor.bounding_box.extent.x)
            except (AttributeError, RuntimeError) as exc:
                print("simple_vehicle_control: no bounding box for {} ({}), stop lines measured from its "
                      "centre".format(getattr(self._actor, 'id', '?'), exc))
                self._half_length = 0.0
        return self._half_length

    def _stop_line_gap(self):
        """
        Return the distance [m] from the actor's front to the stop line of the traffic light it is at (the light's
        stop waypoint ahead of its centre, the one nearest its heading line), or None when there is none
        """
        try:
            light = self._actor.get_traffic_light()
            here = CarlaDataProvider.get_transform(self._actor)
            if light is not None and light.id not in self._stop_lines:
                # get_stop_waypoints walks the map from the light's trigger volume: once per light.
                self._stop_lines[light.id] = [(wp.transform.location.x, wp.transform.location.y)
                                              for wp in light.get_stop_waypoints()]
        except RuntimeError as exc:
            print("simple_vehicle_control: no stop line for the red light ({}), braking as hard as "
                  "allowed".format(exc))
            return None
        if light is None or here is None:
            return None
        yaw = math.radians(here.rotation.yaw)
        forward_x, forward_y = math.cos(yaw), math.sin(yaw)
        ahead = []      # (beside, along) of each stop point ahead
        for x, y in self._stop_lines[light.id]:
            dx, dy = x - here.location.x, y - here.location.y
            along, beside = dx * forward_x + dy * forward_y, abs(dy * forward_x - dx * forward_y)
            if along > 0.0 and beside <= self._STOP_LINE_LATERAL_M:
                ahead.append((beside, along))
        if not ahead:
            return None
        return min(ahead)[1] - self._own_half_length()

    def _red_light_deceleration(self, speed):
        """
        Return the deceleration [m/s2] that stops the actor's front at the stop line of the red light it is at;
        infinite when no line is left ahead of its front
        """
        if speed <= 0.0:
            return 0.0
        gap = self._stop_line_gap()
        if gap is None or gap <= 0.0:
            return float('inf')
        return speed * speed / (2.0 * gap)

    def _on_obstacle(self, event):
        """
        Callback for the obstacle sensor

        Sets _obstacle_distance and _obstacle_actor according to the closest obstacle
        found by the sensor.
        """
        if not event:
            return
        self._obstacle_distance = event.distance
        self._obstacle_actor = event.other_actor

    def reset(self):
        """
        Reset the controller
        """

        if self._visualizer:
            self._visualizer.reset()
        if self._obstacle_sensor:
            self._obstacle_sensor.destroy()
            self._obstacle_sensor = None
        if self._actor and self._actor.is_alive:
            self._actor = None

    def run_step(self):
        """
        Execute on tick of the controller's control loop

        If _waypoints are provided, the vehicle moves towards the next waypoint
        with the given _target_speed, until reaching the final waypoint. Upon reaching
        the final waypoint, _reached_goal is set to True.

        If _waypoints is empty, the vehicle moves in its current direction with
        the given _target_speed.

        For further details see :func:`_set_new_velocity`
        """

        if self._reached_goal:
            # Reached the goal, so stop
            velocity = carla.Vector3D(0, 0, 0)
            self._actor.set_target_velocity(velocity)
            return

        if self._visualizer:
            self._visualizer.render()

        self._reached_goal = False

        if not self._waypoints:
            # No waypoints are provided, so we have to create a list of waypoints internally
            # get next waypoints from map, to avoid leaving the road
            self._reached_goal = False

            map_wp = None
            if not self._generated_waypoint_list:
                map_wp = CarlaDataProvider.get_map().get_waypoint(CarlaDataProvider.get_location(self._actor))
            else:
                map_wp = CarlaDataProvider.get_map().get_waypoint(self._generated_waypoint_list[-1].location)
            while map_wp is not None and len(self._generated_waypoint_list) < 50:
                map_wps = map_wp.next(2.0)
                if map_wps:
                    self._generated_waypoint_list.append(map_wps[0].transform)
                    map_wp = map_wps[0]
                else:
                    break

            # Remove all waypoints that are too close to the vehicle
            while (self._generated_waypoint_list and
                   self._generated_waypoint_list[0].location.distance(self._actor.get_location()) < 0.5):
                self._generated_waypoint_list = self._generated_waypoint_list[1:]

            if not self._generated_waypoint_list:
                # A dead end (the lane has no successor) or no road: nowhere to drive, so hold
                # still. Indexing the empty list raised IndexError and ended the whole run (e.g. an
                # NPC parked at a lane end while it waits for its trigger).
                self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
                self._actor.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                self._last_update = GameTime.get_time()
                return

            direction_norm = self._set_new_velocity(self._offset_waypoint(self._generated_waypoint_list[0]))
            if direction_norm < 2.0:
                self._generated_waypoint_list = self._generated_waypoint_list[1:]
        else:
            # When changing from "free" driving without pre-defined waypoints to a defined route with waypoints
            # it may happen that the first few waypoints are too close to the ego vehicle for obtaining a
            # reasonable control command. Therefore, we drop these waypoints first.
            # Measured flat: a point straight above the vehicle (a path that starts at a spawn
            # point placed above the road) is as unreachable as a near one, and steering at it
            # divides by an XY distance of zero.
            while self._waypoints and self._xy_distance(self._waypoints[0].location) < 0.5:
                self._waypoints = self._waypoints[1:]

            self._reached_goal = False
            if not self._waypoints:
                self._reached_goal = True
                if self._timed_path:
                    # On the last point of a timed path: stop there, not one tick past it.
                    self._actor.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                direction_norm = self._set_new_velocity(self._offset_waypoint(self._waypoints[0]))
                reached_threshold = self._waypoint_reached_threshold
                if len(self._waypoints) == 1 and (self._timed_path or (
                        self._comfort_deceleration is not None and self._comfort_limited())):
                    # The last point of a timed path is due at its time: dropped at the reach
                    # threshold (4 m) it stopped the vehicle that far short and early, as much as
                    # 1.8 s on a 2 m/s path. The path's speed brings the vehicle there on time.
                    # With comfort_deceleration the car brakes onto the last point (_set_new_velocity)
                    # and stops on it, not 4 m short at the speed it had there.
                    reached_threshold = min(reached_threshold, self._END_REACHED_M)
                if direction_norm < reached_threshold:
                    self._waypoints = self._waypoints[1:]
                    if not self._waypoints:
                        self._reached_goal = True

    def _remaining_path_m(self):
        """
        Return the flat (XY) length of the path still to drive: to the next waypoint, then along the rest
        """
        total = self._xy_distance(self._waypoints[0].location)
        for a, b in zip(self._waypoints, self._waypoints[1:]):
            total += math.hypot(b.location.x - a.location.x, b.location.y - a.location.y)
        return total

    def _xy_distance(self, location):
        """
        Return the flat (XY) distance from the actor to location
        """
        actor_location = self._actor.get_location()
        return math.hypot(location.x - actor_location.x, location.y - actor_location.y)

    def _offset_waypoint(self, transform):
        """
        Given a transform (which should be the position of a waypoint), displaces it to the side,
        according to a given offset

        Args:
            transform (carla.Transform): Transform to be moved

        returns:
            offset_location (carla.Transform): Moved transform
        """
        if self._offset == 0:
            offset_location = transform.location
        else:
            right_vector = transform.get_right_vector()
            offset_location = transform.location + carla.Location(x=self._offset*right_vector.x,
                                                                  y=self._offset*right_vector.y)

        return offset_location

    def _comfort_limited(self):
        """
        True when comfort_acceleration or comfort_deceleration is set and the path is not timed (a timed path's speed
        comes from its vertex times: limiting it would put the car behind schedule)
        """
        return not self._timed_path and (self._comfort_acceleration is not None or
                                         self._comfort_deceleration is not None)

    def _base_speed(self, current_speed):
        """
        Return the speed the comfort limits step from: the last commanded speed while current_speed stays within
        _FOLLOW_TOLERANCE_MPS of it, else (blocked, teleported, the first call) current_speed
        """
        if self._commanded_speed is None or abs(current_speed - self._commanded_speed) > self._FOLLOW_TOLERANCE_MPS:
            return current_speed
        return self._commanded_speed

    def _counted_obstacle(self):
        """
        Return (distance past obstacle_margin, its speed) of the sensed obstacle that counts (consider_obstacles,
        within proximity_threshold and on the path), its speed None within the margin; None when there is none
        """
        if not self._consider_obstacles:
            return None
        if not (self._obstacle_distance < self._proximity_threshold and self._obstacle_on_path()):
            return None
        distance = max(self._obstacle_distance - self._obstacle_margin, 0)
        if distance <= 0:
            return distance, None
        return distance, math.sqrt(
            self._obstacle_actor.get_velocity().x**2 + self._obstacle_actor.get_velocity().y**2)

    def _set_new_velocity(self, next_location):
        """
        Calculate and set the new actor veloctiy given the current actor
        location and the _next_location_

        If _consider_obstacles is true, the speed is adapted according to the closest
        obstacle in front of the actor, if it is within the _proximity_threshold distance.
        If _consider_trafficlights is true, the vehicle will enforce a stop at a red
        traffic light.
        If _max_deceleration is set, the vehicle will reduce its speed according to the
        given deceleration value.
        If _comfort_deceleration is set, the speed falls by at most max(_comfort_deceleration, needed)
        per second, needed being what the obstacle asks for (its speed matched obstacle_margin before
        it: 0.5 dv^2 / d) or the red light (stopped at its stop line: 0.5 v^2 / d); behind an obstacle
        it is also kept at most the obstacle's speed plus sqrt(2 * _comfort_deceleration *
        (d - _FOLLOW_BUFFER_M)), what that braking still matches just outside the margin.
        If _comfort_acceleration is set, the speed rises by at most that much per second.
        The comfort limits step from the last commanded speed (see _base_speed), win over the max_*
        limit of their direction and do not apply on a timed path (its vertex times set the speed), nor
        in a tick that needs more than MAX_NEEDED_DECELERATION (within the margin, an obstacle first
        sensed too close, a red light too late): that tick is stock.
        If the vehicle reduces its speed, braking lights will be activated.

        Args:
            next_location (carla.Location): Next target location of the actor

        returns:
            direction (carla.Vector3D): Length of direction vector of the actor
        """

        current_time = GameTime.get_time()
        target_speed = self._target_speed

        if not self._last_update:
            self._last_update = current_time

        current_speed = math.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

        # What the obstacle and a red light ask for (from the measured speeds); only comfort_deceleration reads it.
        obstacle = self._counted_obstacle()
        red_light = (self._consider_traffic_lights and self._actor.is_at_traffic_light() and
                     self._actor.get_traffic_light_state() == carla.TrafficLightState.Red)
        acceleration = None         # the stock obstacle law's (negative) acceleration, when it brakes
        needed_deceleration = 0.0   # m/s2
        if obstacle is not None:
            distance, current_speed_other = obstacle
            if distance <= 0:
                needed_deceleration = float('inf')      # within the margin: no distance left to brake in
            elif current_speed_other < current_speed:
                acceleration = -0.5 * (current_speed - current_speed_other)**2 / distance
                needed_deceleration = -acceleration
        comfort = self._comfort_limited()
        if comfort and red_light and self._comfort_deceleration is not None:
            needed_deceleration = max(needed_deceleration, self._red_light_deceleration(current_speed))
        end_m = None    # m left to the path's end mark (comfort_deceleration brakes onto the last point)
        if comfort and self._comfort_deceleration is not None and self._waypoints:
            end_m = max(self._remaining_path_m() - self._END_REACHED_M, 0.0)
            if current_speed > self._END_CREEP_MS:
                # what stops it at the end mark from here; following the comfort curve tick by tick runs late
                needed_deceleration = max(needed_deceleration, current_speed ** 2 / (2.0 * max(end_m, 0.05)))
        if needed_deceleration > self.MAX_NEEDED_DECELERATION:
            comfort = False         # more than the comfort limits may brake: this tick is stock
        if comfort and end_m is not None:
            # Brake onto the path's last point at comfort_deceleration (stock stopped dead on reaching it: a car
            # behind then had to stop at once -- the Tartu lap's second motorbike, Allan 2026-11-24).
            # It reaches the point (within the 0.5 m end threshold) at a crawl, then stops.
            target_speed = min(target_speed, max(math.sqrt(2.0 * self._comfort_deceleration * end_m),
                                                 self._END_CREEP_MS))
        # The speed this call steps from: the measured one (stock), the commanded one under the comfort limits
        speed = self._base_speed(current_speed) if comfort else current_speed
        # s; a gap longer than _MAX_RATE_STEP_S since the last command moves the speed by that much only
        comfort_step = max(min(current_time - self._last_update, self._MAX_RATE_STEP_S), 0.0)

        if obstacle is not None:
            # If distance is less than the proximity threshold, adapt velocity
            distance, current_speed_other = obstacle
            if distance > 0:
                if acceleration is not None:
                    target_speed = max(acceleration * (current_time - self._last_update) + speed, 0)
                if comfort and self._comfort_deceleration is not None:
                    # No faster than comfort_deceleration still brings down to the obstacle's speed just outside
                    # the margin. Without it the target went back to full speed whenever the car was no faster
                    # than the obstacle, overshot and stopped within the margin: a saw-tooth behind a bus.
                    target_speed = min(target_speed, current_speed_other + math.sqrt(
                        2.0 * self._comfort_deceleration * max(distance - self._FOLLOW_BUFFER_M, 0.0)))
            else:
                target_speed = 0

        if red_light:
            target_speed = 0

        if target_speed < speed:
            if not self._brake_lights_active:
                self._brake_lights_active = True
                light_state = self._actor.get_light_state()
                light_state |= carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if comfort and self._comfort_deceleration is not None:
                # needed_deceleration is at most MAX_NEEDED_DECELERATION here (else the tick is stock)
                target_speed = max(target_speed,
                                   speed - comfort_step * max(self._comfort_deceleration, needed_deceleration))
            elif self._max_deceleration is not None:
                target_speed = max(target_speed, current_speed - (current_time -
                                                                  self._last_update) * self._max_deceleration)
        else:
            if self._brake_lights_active:
                self._brake_lights_active = False
                light_state = self._actor.get_light_state()
                light_state &= ~carla.VehicleLightState.Brake
                self._actor.set_light_state(carla.VehicleLightState(light_state))
            if comfort and self._comfort_acceleration is not None:
                # Below 0.5 m/s the vehicle may not properly accelerate: bumped to 0.5 m/s if the target allows.
                target_speed = max(min(target_speed, speed + comfort_step * self._comfort_acceleration),
                                   min(0.5, target_speed))
            elif self._max_acceleration is not None:
                tmp_speed = min(target_speed, current_speed + (current_time -
                                                               self._last_update) * self._max_acceleration)
                # If the tmp_speed is < 0.5 the vehicle may not properly accelerate.
                # Therefore, we bump the speed to 0.5 m/s if target_speed allows.
                target_speed = max(tmp_speed, min(0.5, target_speed))

        # set new linear velocity
        velocity = carla.Vector3D(0, 0, 0)
        direction = next_location - CarlaDataProvider.get_location(self._actor)
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)
        if direction_norm < 1e-3:
            # The target is where the vehicle is (CarlaDataProvider's location of this tick can
            # differ from the live one run_step measured): count it as reached instead of
            # dividing by zero.
            self._actor.set_target_velocity(velocity)
            self._commanded_speed = 0.0
            self._last_update = current_time
            return direction_norm
        velocity.x = direction.x / direction_norm * target_speed
        velocity.y = direction.y / direction_norm * target_speed

        self._actor.set_target_velocity(velocity)
        self._commanded_speed = target_speed

        # set new angular velocity
        current_yaw = CarlaDataProvider.get_transform(self._actor).rotation.yaw
        # When we have a waypoint list, use the direction between the waypoints to calculate the heading (change)
        # otherwise use the waypoint heading directly
        if self._waypoints:
            delta_yaw = math.degrees(math.atan2(direction.y, direction.x)) - current_yaw
        else:
            new_yaw = CarlaDataProvider.get_map().get_waypoint(next_location).transform.rotation.yaw
            delta_yaw = new_yaw - current_yaw

        if math.fabs(delta_yaw) > 360:
            delta_yaw = delta_yaw % 360

        if delta_yaw > 180:
            delta_yaw = delta_yaw - 360
        elif delta_yaw < -180:
            delta_yaw = delta_yaw + 360

        angular_velocity = carla.Vector3D(0, 0, 0)
        if target_speed == 0:
            angular_velocity.z = 0
        else:
            angular_velocity.z = delta_yaw / (direction_norm / target_speed)
        self._actor.set_target_angular_velocity(angular_velocity)

        self._last_update = current_time

        return direction_norm


def _segment_box_distance(start, end, half_length, half_width):
    """
    Return the XY distance from the segment start-end ((x, y) each, in the box's frame) to the box
    |x| <= half_length, |y| <= half_width; 0 when they overlap
    """
    (ax, ay), (bx, by) = start, end
    dx, dy = bx - ax, by - ay
    # Overlap: clip the segment to the box (Liang-Barsky); something left = they meet.
    t_in, t_out = 0.0, 1.0
    for p, q in ((-dx, ax + half_length), (dx, half_length - ax), (-dy, ay + half_width), (dy, half_width - ay)):
        if p == 0.0:
            if q < 0.0:
                break
        elif p < 0.0:
            t_in = max(t_in, q / p)
        else:
            t_out = min(t_out, q / p)
        if t_in > t_out:
            break
    else:
        return 0.0
    # Apart: the closest pair has a segment end or a box corner in it.
    best = min(math.hypot(max(abs(x) - half_length, 0.0), max(abs(y) - half_width, 0.0))
               for x, y in ((ax, ay), (bx, by)))
    seg2 = dx * dx + dy * dy
    for corner_x in (-half_length, half_length):
        for corner_y in (-half_width, half_width):
            f = 0.0 if seg2 <= 1e-12 else max(0.0, min(1.0, ((corner_x - ax) * dx + (corner_y - ay) * dy) / seg2))
            best = min(best, math.hypot(ax + f * dx - corner_x, ay + f * dy - corner_y))
    return best
