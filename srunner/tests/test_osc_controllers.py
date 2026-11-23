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
