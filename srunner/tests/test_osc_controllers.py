#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests (pytest, no CARLA server) for the OpenSCENARIO actor controllers.

Pinned contracts:
- pedestrian_control: a path point within 0.5 m in XY of the walker is skipped (VSE's first vertex
  is the spawn), and the walking direction is the flat unit vector towards the next point, never
  NaN, whatever the height difference.
- simple_vehicle_control without waypoints (free driving, e.g. an NPC waiting for its trigger) holds
  still at a dead end instead of raising IndexError, and drives on where the lane goes on.
- simple_vehicle_control's obstacle_margin: the speed is matched that far before the obstacle (stopped
  within it), without it at the obstacle as before;
- obstacle_path_width: an obstacle counts only within that XY distance of the car's remaining path (a car crossing
  beside the path does not stop it; one on the path does); without the property every sensed obstacle counts.
"""

import math

import carla
import pytest

from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl
from srunner.scenariomanager.actorcontrols.simple_vehicle_control import SimpleVehicleControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class _FakeWalker(object):
    is_alive = True

    def __init__(self, location, yaw=0.0):
        self._transform = carla.Transform(location, carla.Rotation(yaw=yaw))
        self.controls = []

    def get_location(self):
        return self._transform.location

    def get_transform(self):
        return self._transform

    def get_control(self):
        return carla.WalkerControl()

    def apply_control(self, control):
        self.controls.append(control)


def _walker_control(walker, waypoints, speed=1.4):
    # carla.Walker cannot be made without a server: skip the constructor's type check.
    control = PedestrianControl.__new__(PedestrianControl)
    control._actor = walker
    control.update_target_speed(speed)
    control.update_waypoints(waypoints)
    return control


def _direction(walker):
    direction = walker.controls[-1].direction
    return (direction.x, direction.y, direction.z)


def test_walker_path_starting_at_its_spawn_walks_to_the_next_point():
    walker = _FakeWalker(carla.Location(x=10.0, y=5.0, z=1.0))
    control = _walker_control(walker, [carla.Transform(carla.Location(x=10.0, y=5.0, z=0.1)),  # the spawn
                                       carla.Transform(carla.Location(x=20.0, y=5.0, z=0.0))])
    control.run_step()  # used to apply the direction (nan, nan, -inf)
    assert _direction(walker) == pytest.approx((1.0, 0.0, 0.0))
    assert len(control._waypoints) == 1
    assert control.check_reached_waypoint_goal() is False


def test_walker_direction_is_flat_and_unit_whatever_the_height():
    walker = _FakeWalker(carla.Location(x=0.0, y=0.0, z=1.0))
    control = _walker_control(walker, [carla.Transform(carla.Location(x=3.0, y=4.0, z=-1.0))])
    control.run_step()
    x, y, z = _direction(walker)
    assert (x, y, z) == pytest.approx((0.6, 0.8, 0.0))
    assert math.hypot(x, y) == pytest.approx(1.0)


def test_walker_path_of_only_its_own_spot_is_reached():
    walker = _FakeWalker(carla.Location(x=10.0, y=5.0, z=1.0), yaw=90.0)
    control = _walker_control(walker, [carla.Transform(carla.Location(x=10.2, y=5.1, z=0.0))])
    control.run_step()
    assert control.check_reached_waypoint_goal() is True
    assert not any(math.isnan(value) for value in _direction(walker))
    assert _direction(walker) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)  # walks on, facing its yaw


class _FakeLaneWaypoint(object):
    """A map waypoint on a lane that goes on for 'ahead' metres, then ends."""

    def __init__(self, x, ahead):
        self.transform = carla.Transform(carla.Location(x=x, y=0.0, z=0.0))
        self._ahead = ahead

    def next(self, distance):
        if self._ahead < distance:
            return []
        return [_FakeLaneWaypoint(self.transform.location.x + distance, self._ahead - distance)]


class _FakeLaneMap(object):
    def __init__(self, lane_end_x):
        self._lane_end_x = lane_end_x

    def get_waypoint(self, location):
        return _FakeLaneWaypoint(location.x, self._lane_end_x - location.x)


class _FakeVehicle(object):
    def __init__(self, x):
        self._location = carla.Location(x=x, y=0.0, z=0.0)
        self.velocities = []

    def get_location(self):
        return self._location

    def get_velocity(self):
        return carla.Vector3D()

    def get_transform(self):
        return carla.Transform(self._location)

    def set_target_velocity(self, velocity):
        self.velocities.append(velocity)

    def set_target_angular_velocity(self, velocity):
        pass

    def get_light_state(self):
        return carla.VehicleLightState.NONE

    def set_light_state(self, state):
        pass


def _free_driving(monkeypatch, vehicle_x, lane_end_x, speed):
    vehicle = _FakeVehicle(vehicle_x)
    monkeypatch.setattr(CarlaDataProvider, "get_map", staticmethod(lambda world=None: _FakeLaneMap(lane_end_x)))
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    control = SimpleVehicleControl(vehicle)
    control.update_target_speed(speed)
    return vehicle, control


@pytest.mark.parametrize("speed", [0.0, 5.0])
def test_free_driving_vehicle_holds_still_at_a_dead_end(monkeypatch, speed):
    vehicle, control = _free_driving(monkeypatch, vehicle_x=100.0, lane_end_x=101.0, speed=speed)
    control.run_step()  # used to raise IndexError: no lane within 2 m ahead
    assert (vehicle.velocities[-1].x, vehicle.velocities[-1].y) == (0.0, 0.0)
    assert control.check_reached_waypoint_goal() is False


def test_free_driving_vehicle_drives_on_where_the_lane_goes_on(monkeypatch):
    vehicle, control = _free_driving(monkeypatch, vehicle_x=0.0, lane_end_x=100.0, speed=5.0)
    control.run_step()
    assert vehicle.velocities[-1].x == pytest.approx(5.0)


class _MovingVehicle(_FakeVehicle):
    def __init__(self, x, speed):
        super(_MovingVehicle, self).__init__(x)
        self._speed = speed

    def get_velocity(self):
        return carla.Vector3D(x=self._speed)


class _StoppedCar(object):
    def get_velocity(self):
        return carla.Vector3D()


def _obstacle_step(monkeypatch, args, obstacle_distance, speed=8.0):
    vehicle = _MovingVehicle(0.0, speed)
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    monkeypatch.setattr(CarlaDataProvider, "get_map",
                        staticmethod(lambda world=None: _FakeLaneMap(1000.0)))
    control = SimpleVehicleControl(vehicle, args)
    control.update_target_speed(speed)
    # What consider_obstacles' sensor would report (spawning it needs a server).
    control._consider_obstacles = True
    control._proximity_threshold = 30.0
    control._obstacle_distance = obstacle_distance
    control._obstacle_actor = _StoppedCar()
    control._last_update = 0.01     # not 0: a falsy value restarts the clock (dt 0)
    monkeypatch.setattr("srunner.scenariomanager.actorcontrols.simple_vehicle_control.GameTime.get_time",
                        staticmethod(lambda: 0.05))
    control._set_new_velocity(carla.Location(x=50.0))
    return vehicle.velocities[-1].x


def test_obstacle_margin_stops_the_car_that_far_before_a_stopped_car(monkeypatch):
    assert _obstacle_step(monkeypatch, {'obstacle_margin': '2.5'}, obstacle_distance=2.4) == 0.0
    assert _obstacle_step(monkeypatch, {'obstacle_margin': '2.5'}, obstacle_distance=2.5) == 0.0


def test_obstacle_margin_brakes_harder_than_without(monkeypatch):
    with_margin = _obstacle_step(monkeypatch, {'obstacle_margin': '2.5'}, obstacle_distance=6.0)
    without = _obstacle_step(monkeypatch, None, obstacle_distance=6.0)
    # 0.5 * 8^2 / d over the 0.04 s since the last update: d = 3.5 m vs 6 m.
    assert with_margin == pytest.approx(8.0 - 0.04 * 0.5 * 64 / 3.5)
    assert without == pytest.approx(8.0 - 0.04 * 0.5 * 64 / 6.0)


class _CrossingCar(object):
    def __init__(self, x, y):
        self._location = carla.Location(x=x, y=y, z=0.0)

    def get_velocity(self):
        return carla.Vector3D()

    def get_location(self):
        return self._location


def _path_step(monkeypatch, args, obstacle):
    vehicle = _MovingVehicle(0.0, 8.0)
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    monkeypatch.setattr(CarlaDataProvider, "get_map", staticmethod(lambda world=None: _FakeLaneMap(1000.0)))
    control = SimpleVehicleControl(vehicle, args)
    control.update_target_speed(8.0)
    # The path: straight on to x=20, then a 90 deg turn left to (20, 30).
    control._waypoints = [carla.Transform(carla.Location(x=20.0)), carla.Transform(carla.Location(x=20.0, y=30.0))]
    control._consider_obstacles = True
    control._proximity_threshold = 30.0
    control._obstacle_distance = 6.0
    control._obstacle_actor = obstacle
    control._last_update = 0.01
    monkeypatch.setattr("srunner.scenariomanager.actorcontrols.simple_vehicle_control.GameTime.get_time",
                        staticmethod(lambda: 0.05))
    control._set_new_velocity(carla.Location(x=20.0))
    return vehicle.velocities[-1].x


def test_obstacle_beside_the_path_does_not_stop_the_car_with_a_path_width(monkeypatch):
    beside = _CrossingCar(6.0, -5.0)                       # 5 m to the right of the straight leg
    assert _path_step(monkeypatch, {'obstacle_path_width': '2.5'}, beside) == pytest.approx(8.0)
    assert _path_step(monkeypatch, None, beside) < 8.0     # stock: every sensed obstacle counts


def test_obstacle_on_the_path_still_stops_the_car(monkeypatch):
    on_leg = _CrossingCar(6.0, 1.0)                        # 1 m off the straight leg
    after_turn = _CrossingCar(21.0, 15.0)                  # on the leg after the turn
    assert _path_step(monkeypatch, {'obstacle_path_width': '2.5'}, on_leg) < 8.0
    assert _path_step(monkeypatch, {'obstacle_path_width': '2.5'}, after_turn) < 8.0
