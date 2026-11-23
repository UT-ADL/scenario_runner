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
- obstacle_path_width: an obstacle counts only when its bounding box (XY, turned by its yaw, its offset from the
  actor included) comes within that distance of the car's remaining path (a car crossing beside the path does not
  stop it; one on the path does, and so does a long bus whose centre is beyond the width but whose side is not);
  without the property every sensed obstacle counts. The box is read once per obstacle (a blocking RPC on 0.9.16).
- comfort_acceleration / comfort_deceleration (m/s2): the commanded speed moves toward the target by at most
  comfort_acceleration * dt up and max(comfort_deceleration, needed) * dt down, needed = what matches the obstacle's
  speed obstacle_margin before it, or stops the car's front at a red light's stop line; stepping from the last
  commanded speed (a little physics drag per tick does not hold the car back); a gap since the last command counts
  as at most 0.25 s. A tick needing more than 9 m/s2 (within the margin, an obstacle first sensed too close, a red
  light too late) is stock: a cut-in 6 m ahead at 13 m/s is avoided as stock avoids it. A one-tick drop of the target
  moves the speed a little, a bike behind a slowing bus follows it smoothly; comfort_* wins over the stock max_* of
  its direction. Without comfort_* the commands are those of the controller before it (0ceb118) byte for byte, with
  or without the stock max_deceleration / max_acceleration.
"""

import hashlib
import math
import types

import carla
import pytest

from srunner.scenariomanager.actorcontrols.pedestrian_control import PedestrianControl
from srunner.scenariomanager.actorcontrols.simple_vehicle_control import SimpleVehicleControl, _segment_box_distance
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


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


class _Box(object):
    """An obstacle: a bounding box at a pose (a car by default), counting its bounding_box reads."""

    _ids = [100]

    def __init__(self, x, y, yaw=0.0, half_length=2.3, half_width=1.0, box_x=0.0, speed=0.0):
        self._ids[0] += 1
        self.id = self._ids[0]
        self.x, self.y, self.yaw, self.speed = x, y, yaw, speed
        self._box = carla.BoundingBox(carla.Location(x=box_x), carla.Vector3D(half_length, half_width, 1.0))
        self.box_reads = 0

    @property
    def bounding_box(self):
        self.box_reads += 1
        return self._box

    def get_transform(self):
        return carla.Transform(carla.Location(x=self.x, y=self.y), carla.Rotation(yaw=self.yaw))

    def get_location(self):
        return carla.Location(x=self.x, y=self.y)

    def get_velocity(self):
        return carla.Vector3D(x=self.speed)


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
    beside = _Box(6.0, -5.0)                               # 5 m to the right of the straight leg (its side 4 m)
    assert _path_step(monkeypatch, {'obstacle_path_width': '2.5'}, beside) == pytest.approx(8.0)
    assert _path_step(monkeypatch, None, beside) < 8.0     # stock: every sensed obstacle counts


def test_obstacle_on_the_path_still_stops_the_car(monkeypatch):
    on_leg = _Box(6.0, 1.0)                                # 1 m off the straight leg
    after_turn = _Box(21.0, 15.0)                          # on the leg after the turn
    assert _path_step(monkeypatch, {'obstacle_path_width': '2.5'}, on_leg) < 8.0
    assert _path_step(monkeypatch, {'obstacle_path_width': '2.5'}, after_turn) < 8.0


# --- obstacle_path_width: the distance from the path to the obstacle's box ----------------------------------------

def test_segment_box_distance():
    def distance(ax, ay, bx, by):       # to the box |x| <= 2, |y| <= 1
        return _segment_box_distance((ax, ay), (bx, by), 2.0, 1.0)

    assert distance(-10.0, 0.0, 10.0, 0.0) == 0.0                   # through it
    assert distance(-0.5, 0.2, 0.5, -0.2) == 0.0                    # inside it
    assert distance(1.0, 0.5, 1.0, 0.5) == 0.0                      # a point inside it
    assert distance(1.0, 0.0, 10.0, 5.0) == 0.0                     # one end inside it
    assert distance(-3.0, 3.0, 3.0, -3.0) == 0.0                    # across a corner
    assert distance(-10.0, 3.0, 10.0, 3.0) == pytest.approx(2.0)    # beside the long side
    assert distance(5.0, -10.0, 5.0, 10.0) == pytest.approx(3.0)    # beside the short side
    assert distance(4.0, 3.0, 10.0, 9.0) == pytest.approx(math.hypot(2.0, 2.0))     # an end nearest
    assert distance(0.0, 4.0, 4.0, 0.0) == pytest.approx(1.0 / math.sqrt(2.0))      # a corner nearest
    assert distance(2.0, 3.0, 2.0, 3.0) == pytest.approx(2.0)       # a point outside


@pytest.fixture
def path_control(monkeypatch):
    """A controller on a straight path along x (0 -> 40) with obstacle_path_width 1.5 m (VSE's NPC value)."""
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    control = SimpleVehicleControl(_FakeVehicle(0.0), {'obstacle_path_width': '1.5'})
    control._waypoints = [carla.Transform(carla.Location(x=40.0))]
    return control


def _counts(control, obstacle):
    control._obstacle_actor = obstacle
    return control._obstacle_on_path()


def test_a_bus_beside_the_path_counts_by_its_side_not_its_centre(path_control):
    # 11-24 Tartu lap: a bus's centre 2.6 m off a bike's path, its side 0.6 m; measured to the centre it did not
    # count and the bike squeezed past its back corner.
    assert _counts(path_control, _Box(15.0, 2.6, half_length=6.0, half_width=2.0)) is True
    assert _counts(path_control, _Box(15.0, 2.6, half_length=0.0, half_width=0.0)) is False   # its centre alone


def test_a_car_whose_box_stays_beyond_the_width_does_not_count(path_control):
    assert _counts(path_control, _Box(15.0, 2.6)) is False                  # its side 1.6 m off
    assert _counts(path_control, _Box(15.0, -2.6)) is False                 # the other side
    assert _counts(path_control, _Box(15.0, 2.4)) is True                   # 1.4 m
    assert _counts(path_control, _Box(44.0, 0.0)) is False                 # its back 1.7 m past the path's end
    assert _counts(path_control, _Box(43.5, 0.0)) is True                   # 1.2 m past it


def test_a_turned_box_is_measured_along_its_own_axes(path_control):
    # A 12 m bus across the road (yaw 90) 7 m beside the path reaches to 1 m off it; along the road, 5.75 m off.
    assert _counts(path_control, _Box(15.0, 7.0, yaw=90.0, half_length=6.0, half_width=1.25)) is True
    assert _counts(path_control, _Box(15.0, 7.0, yaw=0.0, half_length=6.0, half_width=1.25)) is False
    # At 45 deg a 6 x 2 m box 5 m beside the path reaches down to 5 - 2 * sqrt(2) = 2.17 m off it.
    for yaw in (45.0, -45.0, 135.0):
        turned = _Box(15.0, 5.0, yaw=yaw, half_length=3.0, half_width=1.0)
        path_control._obstacle_path_width = 2.2
        assert _counts(path_control, turned) is True
        path_control._obstacle_path_width = 2.1
        assert _counts(path_control, turned) is False


def test_the_box_offset_from_the_actor_origin_is_included(path_control):
    # Origin 4.5 m beside the path, facing away from it, its box centred 2 m behind the origin: the box reaches
    # 0.5 m from the path (without the offset it would stop 2.5 m off).
    assert _counts(path_control, _Box(15.0, 4.5, yaw=90.0, half_length=2.0, box_x=-2.0)) is True
    assert _counts(path_control, _Box(15.0, 4.5, yaw=90.0, half_length=2.0)) is False


def test_the_obstacle_box_is_read_once_per_actor(path_control):
    bus, car = _Box(15.0, 2.6, half_length=6.0, half_width=2.0), _Box(15.0, 2.4)
    for _ in range(5):
        _counts(path_control, bus)
        _counts(path_control, car)
    assert (bus.box_reads, car.box_reads) == (1, 1)


# --- max_acceleration / max_deceleration ----------------------------------------------------------------------------

_COMFORT = {'obstacle_margin': '2.5', 'comfort_acceleration': '4.0', 'comfort_deceleration': '2.5'}   # VSE's NPCs


class _ScriptedCar(object):
    """A vehicle that keeps `keep` of its target velocity (physics drag) and moves by it; logs every command."""

    id = 11
    is_alive = True

    def __init__(self, keep=0.98, speed=0.0):
        self.location = carla.Location()
        self.velocity = carla.Vector3D(x=speed)
        self.keep = keep
        self.at_light = False
        self.light_state = carla.TrafficLightState.Green
        self.lights = carla.VehicleLightState.NONE
        self.log = []
        self.speeds = []

    def get_location(self):
        return carla.Location(self.location.x, self.location.y, 0.0)

    def get_velocity(self):
        return self.velocity

    def get_transform(self):
        return carla.Transform(self.get_location())

    def set_target_velocity(self, velocity):
        self.log.append(('v', velocity.x, velocity.y))
        self.speeds.append(math.hypot(velocity.x, velocity.y))
        self.velocity = carla.Vector3D(velocity.x * self.keep, velocity.y * self.keep, 0.0)

    def set_target_angular_velocity(self, velocity):
        self.log.append(('w', velocity.z))

    def get_light_state(self):
        return self.lights

    def set_light_state(self, state):
        self.lights = state
        self.log.append(('l', int(state)))

    def is_at_traffic_light(self):
        return self.at_light

    def get_traffic_light_state(self):
        return self.light_state


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    GameTime.restart()
    yield
    GameTime.restart()


def _drive_ticks(args, script, ticks, car, target, waypoints, tick=0.05):
    """Run ticks of run_step; script(i, control) sets each tick's readings before it. Return the controller."""
    GameTime.restart()      # frames count from 1 again
    control = SimpleVehicleControl(car, args)
    control.update_target_speed(target)
    control.update_waypoints(waypoints)
    control._consider_obstacles = True      # what the sensor would report (spawning it needs a server)
    control._proximity_threshold = 30.0
    for i in range(ticks):
        GameTime.on_carla_tick(types.SimpleNamespace(frame=i + 1, delta_seconds=tick, elapsed_seconds=i * tick))
        script(i, control)
        control.run_step()
        car.location = car.location + carla.Location(x=car.velocity.x * tick, y=car.velocity.y * tick)
    return control


def _cruise(args, script, ticks=40, keep=1.0, speed=13.0, start=13.0):
    """Commanded speeds of a car on a straight path, target speed `speed`, moving at `start`."""
    car = _ScriptedCar(keep=keep, speed=start)
    _drive_ticks(args, script, ticks, car, speed, [carla.Transform(carla.Location(x=2000.0))])
    return car.speeds


def _steps(speeds):
    return [b - a for a, b in zip(speeds, speeds[1:])]


def test_a_one_tick_drop_of_the_target_moves_the_speed_a_little(provider):
    def script(i, control):
        control.update_target_speed(0.0 if i == 10 else 13.0)

    stock = _cruise({'obstacle_margin': '2.5'}, script)
    assert min(stock) == 0.0 and stock[-1] == pytest.approx(13.0)          # 13 -> 0 -> 13
    comfort = _cruise(_COMFORT, script)
    assert min(comfort) == pytest.approx(13.0 - 2.5 * 0.05, abs=1e-4)      # down by comfort_deceleration * dt
    assert max(abs(step) for step in _steps(comfort)) <= 2.5 * 0.05 + 1e-4
    assert comfort[-1] == pytest.approx(13.0)


def test_an_obstacle_within_the_margin_stops_the_car_as_stock_then_it_starts_gently(provider):
    stopped = _Box(30.0, 0.0)

    def script(i, control):     # the sensor sees it 2 m ahead (the margin is 2.5) for one tick, then out of range
        control._obstacle_actor, control._obstacle_distance = stopped, (2.0 if i == 10 else 100.0)

    stock = _cruise({'obstacle_margin': '2.5'}, script, ticks=100)
    assert stock[10] == 0.0 and stock[11] == pytest.approx(13.0)           # the lap's stop-go
    comfort = _cruise(_COMFORT, script, ticks=100)
    assert comfort[10] == 0.0                                               # no distance left: the stock stop
    assert comfort[11] == pytest.approx(0.5)                                # then the stock 0.5 m/s start ...
    assert max(_steps(comfort[11:])) <= 4.0 * 0.05 + 1e-4                   # ... and comfort_acceleration
    assert comfort[-1] == pytest.approx(13.0)


@pytest.mark.parametrize("gap, speed_after", [
    (25.0, 10.0 - 0.04 * 2.0),      # 0.5 * 10^2 / 25: gentler than comfort_deceleration, as the stock law asks
    (10.0, 10.0 - 0.04 * 5.0),      # harder than comfort_deceleration: what is needed
    (2.0, 10.0 - 0.04 * 25.0),      # 25 needed, more than 9: the stock law's target
    (-0.5, 0.0),                    # within the margin: the stock stop
])
def test_braking_for_an_obstacle_goes_past_comfort_deceleration_and_beyond_9_is_stock(monkeypatch, gap, speed_after):
    speed = _obstacle_step(monkeypatch, _COMFORT, obstacle_distance=2.5 + gap, speed=10.0)
    assert speed == pytest.approx(speed_after)                              # 0.04 s since the last update
    if gap < 5.0:
        assert speed == _obstacle_step(monkeypatch, {'obstacle_margin': '2.5'}, 2.5 + gap, speed=10.0)


@pytest.mark.parametrize("bus_speed", [0.0, 2.8])
def test_a_cut_in_too_close_for_the_comfort_limits_is_avoided_as_stock_avoids_it(provider, bus_speed):
    # 13 m/s, a bus first sensed 6 m ahead (margin 2.5): 0.5 * (13 - v)^2 / 3.5 = 15-24 m/s2 needed, more than the
    # 9 the comfort limits allow -- with them the bike drove into it; stock brakes harder.
    def run(args):
        bike = _ScriptedCar(keep=1.0, speed=13.0)
        bus = _Box(1000.0, 0.0, half_length=6.0, half_width=1.3, speed=bus_speed)
        gaps = []

        def script(i, control):
            control._proximity_threshold = 250.0
            if i == 5:          # cuts in: its back 6 m ahead of the bike's front
                bus.x = bike.location.x + 1.0 + 6.0 + 6.0
            if i >= 5:
                gaps.append((bus.x - 6.0) - (bike.location.x + 1.0))
                control._obstacle_actor, control._obstacle_distance = bus, gaps[-1]
            bus.x += bus.speed * 0.05

        _drive_ticks(dict(args, obstacle_path_width='1.5'), script, 200, bike, 13.0,
                     [carla.Transform(carla.Location(x=5000.0))])
        return bike.speeds, gaps

    stock, stock_gaps = run({'obstacle_margin': '2.5'})
    speeds, gaps = run(_COMFORT)
    assert min(stock_gaps) > 0.0
    assert min(gaps) > 0.0 and min(gaps) >= min(stock_gaps) - 0.05        # not into it: as stock
    assert speeds[5] == stock[5]                                            # the first tick is stock
    if bus_speed == 0.0:
        assert speeds[-1] == 0.0                                            # it stops


def test_comfort_properties_win_over_the_stock_limits_of_their_direction(provider):
    def script(i, control):
        control.update_target_speed(0.0 if 10 <= i < 20 else 13.0)

    both = _cruise({'max_deceleration': '1.0', 'comfort_deceleration': '2.5',
                    'max_acceleration': '1.0', 'comfort_acceleration': '4.0'}, script)
    assert min(_steps(both)) == pytest.approx(-2.5 * 0.05, abs=1e-4)       # comfort_deceleration, not 1.0
    assert max(_steps(both)) == pytest.approx(4.0 * 0.05, abs=1e-4)        # comfort_acceleration, not 1.0
    mixed = _cruise({'max_deceleration': '1.0', 'comfort_acceleration': '4.0'}, script)
    assert min(_steps(mixed)) == pytest.approx(-1.0 * 0.05, abs=1e-4)      # the stock max_deceleration
    assert max(_steps(mixed)) == pytest.approx(4.0 * 0.05, abs=1e-4)
    # The stock direction steps from the measured speed, as stock: 3 % drag taken off 13 m/s, then 1.0 m/s2.
    dragged = _cruise({'max_deceleration': '1.0', 'comfort_acceleration': '4.0'}, script, keep=0.97)
    assert dragged[10] == pytest.approx(13.0 * 0.97 - 1.0 * 0.05, abs=1e-4)


class _Light(object):
    def __init__(self, stops):
        self.id = 5
        self._stops = stops
        self.stop_reads = 0

    def get_stop_waypoints(self):
        self.stop_reads += 1
        return [types.SimpleNamespace(transform=carla.Transform(carla.Location(x=x, y=y))) for x, y in self._stops]


class _AtRedLight(_MovingVehicle):
    """A vehicle 4 m long at x = 0 (heading +x), inside a red light's trigger volume."""

    def __init__(self, speed, light):
        super(_AtRedLight, self).__init__(0.0, speed)
        self._light = light
        self.box_reads = 0

    @property
    def bounding_box(self):
        self.box_reads += 1
        return carla.BoundingBox(carla.Location(), carla.Vector3D(2.0, 1.0, 1.0))

    def is_at_traffic_light(self):
        return True

    def get_traffic_light_state(self):
        return carla.TrafficLightState.Red

    def get_traffic_light(self):
        return self._light


def _red_light_step(monkeypatch, stops, args, calls=1):
    vehicle = _AtRedLight(10.0, _Light(stops))
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    monkeypatch.setattr(CarlaDataProvider, "get_map", staticmethod(lambda world=None: _FakeLaneMap(1000.0)))
    monkeypatch.setattr("srunner.scenariomanager.actorcontrols.simple_vehicle_control.GameTime.get_time",
                        staticmethod(lambda: 0.05))
    control = SimpleVehicleControl(vehicle, dict(args, consider_trafficlights='true'))
    control.update_target_speed(10.0)
    for _ in range(calls):
        control._last_update = 0.01
        control._commanded_speed = None
        control._set_new_velocity(carla.Location(x=50.0))
    return vehicle.velocities[-1].x, vehicle


@pytest.mark.parametrize("stops, speed_after", [
    ([(12.0, 0.0)], 10.0 - 0.04 * 5.0),         # the front 10 m before the line: 10^2 / (2 * 10)
    ([(14.0, 3.5), (12.0, 0.2)], 10.0 - 0.04 * 5.0),    # the line in this lane, not the next lane's
    ([(42.0, 0.0)], 10.0 - 0.04 * 2.5),         # 40 m: 1.25 needed, comfort_deceleration brakes harder
    ([(6.0, 0.0)], 0.0),                        # 4 m: 12.5 needed, more than 9: the stock stop
    ([(1.5, 0.0)], 0.0),                        # the front past the line: the stock stop
    ([(-5.0, 0.0)], 0.0),                       # no line ahead: the stock stop
])
def test_a_red_light_brakes_to_stop_at_its_stop_line(monkeypatch, stops, speed_after):
    speed, _vehicle = _red_light_step(monkeypatch, stops, _COMFORT)
    assert speed == pytest.approx(speed_after)
    stock, _vehicle = _red_light_step(monkeypatch, stops, {})
    assert stock == 0.0


def test_a_red_lights_stop_line_and_the_own_box_are_read_once(monkeypatch):
    _speed, vehicle = _red_light_step(monkeypatch, [(12.0, 0.0)], _COMFORT, calls=5)
    assert (vehicle._light.stop_reads, vehicle.box_reads) == (1, 1)


def test_a_long_gap_since_the_last_command_moves_the_speed_by_a_quarter_second_only(monkeypatch):
    vehicle = _MovingVehicle(0.0, 0.0)          # standing, e.g. at a reached goal, then a new path at 13 m/s
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    monkeypatch.setattr(CarlaDataProvider, "get_map", staticmethod(lambda world=None: _FakeLaneMap(1000.0)))
    monkeypatch.setattr("srunner.scenariomanager.actorcontrols.simple_vehicle_control.GameTime.get_time",
                        staticmethod(lambda: 10.0))
    control = SimpleVehicleControl(vehicle, _COMFORT)
    control.update_target_speed(13.0)
    control._last_update = 0.01
    control._set_new_velocity(carla.Location(x=50.0))
    assert vehicle.velocities[-1].x == pytest.approx(0.25 * 4.0)
    stock = SimpleVehicleControl(vehicle, {'max_acceleration': '4.0'})    # the stock limit: the whole gap
    stock.update_target_speed(13.0)
    stock._last_update = 0.01
    stock._set_new_velocity(carla.Location(x=50.0))
    assert vehicle.velocities[-1].x == pytest.approx(13.0)


def test_the_limits_step_from_the_commanded_speed_so_drag_does_not_hold_the_car_back(provider):
    # Physics takes 3 % off each tick (7.8 m/s2 at 13 m/s, more than comfort_acceleration): stepping from the
    # measured speed never got past about 6.7 m/s.
    speeds = _cruise(_COMFORT, lambda i, control: None, ticks=80, keep=0.97, start=0.0)
    assert speeds[-1] == pytest.approx(13.0)
    assert speeds.index(max(speeds)) <= int((13.0 / 4.0 + 0.2) / 0.05)
    assert max(_steps(speeds)) <= 0.5 + 1e-6                               # the first step: the stock 0.5 m/s bump


@pytest.mark.parametrize("keep", [1.0, 0.97])
def test_a_bike_behind_a_slowing_bus_follows_it_smoothly(provider, keep):
    # 11-24 Tartu lap: bikes at 50 km/h behind a bus slowing to 10 km/h jerked 13 -> 0.4 -> 13 m/s. The bus's side
    # is 0.6 m off the bike's path (its centre 1.9 m), the bike's front 1 m ahead of its centre.
    def run(args):
        bike = _ScriptedCar(keep=keep, speed=13.0)
        bus = _Box(60.0, 1.9, half_length=6.0, half_width=1.3, speed=8.0)
        gaps = []

        def script(i, control):
            # The sensor reports the gap of the last physics step; then both move on (the bike after run_step).
            gaps.append((bus.x - 6.0) - (bike.location.x + 1.0))
            control._obstacle_actor, control._obstacle_distance = bus, gaps[-1]
            control._proximity_threshold = 250.0
            # The bus's own controller wobbles its speed a little (a perfectly steady one hides the stop-go).
            bus.speed = (8.0 if i < 100 else max(2.8, 8.0 - (i - 100) * 0.05 * 1.5)) + 0.1 * math.sin(i * 0.7)
            bus.x += bus.speed * 0.05

        _drive_ticks(dict(args, obstacle_path_width='1.5'), script, 600, bike, 13.0,
                     [carla.Transform(carla.Location(x=5000.0))])
        return bike.speeds, gaps

    stock, _gaps = run({'obstacle_margin': '2.5'})
    assert max(abs(step) for step in _steps(stock[300:])) > 10.0             # stop-go
    speeds, gaps = run(_COMFORT)
    assert max(abs(step) for step in _steps(speeds)) <= 2.5 * 0.05 + 1e-4  # never harder than comfort_deceleration
    assert min(gaps) >= 2.4                                                 # obstacle_margin 2.5
    actual = [speed * keep for speed in speeds[-100:]]                     # the bike's speed after the drag
    assert sum(actual) / len(actual) == pytest.approx(2.8, abs=0.1)         # the bus's


# --- without comfort_*: the commands of the controller before them, stock max_* or not ------------------------------

def _scripted_commands(args, tick=0.05):
    """
    Drive a scripted run (a path with a turn, target speed changes, an obstacle closing in and inside the margin, a
    red light) and return the log of every command (velocity, angular velocity, light state)
    """
    car = _ScriptedCar()
    obstacle = _Box(0.0, 0.0)

    def script(i, control):
        if i == 2:
            control.update_target_speed(10.0)
        elif i == 60:
            control.update_target_speed(4.0)
        elif i == 80:
            control.update_target_speed(12.0)
        if 20 <= i < 40:
            control._obstacle_distance, control._obstacle_actor, obstacle.speed = 24.0 - (i - 20) * 1.3, obstacle, 2.0
        elif 40 <= i < 50:
            control._obstacle_distance, obstacle.speed = 25.0, 15.0
        elif 110 <= i < 115:
            control._obstacle_distance, obstacle.speed = 1.0, 0.0
        elif i >= 115:
            control._obstacle_distance = 100.0
        car.at_light = 90 <= i < 100
        car.light_state = carla.TrafficLightState.Red if car.at_light else carla.TrafficLightState.Green

    _drive_ticks(args, script, 140, car, 0.0, [carla.Transform(carla.Location(x=40.0)),
                                               carla.Transform(carla.Location(x=40.0, y=80.0))], tick=tick)
    return car.log


@pytest.mark.parametrize("stock_limits, count, digest", [
    ({}, 288, '740af7c14d0c564a9c7c1e5b5d758a244a7277d3e5b50f992e0353743529feec'),
    ({'max_deceleration': '2.5'}, 288, '03b7ff67d7d877c668d7a22db5b7c2af51f718dc7fcf6edf7ec3e95810b000e5'),
    ({'max_deceleration': '2.5', 'max_acceleration': '3.0'}, 290,
     'be4d44623d7dc9ca1c4eb6c8ca646838ddd50b1d672931863597ba97f96b903d'),
])
def test_without_comfort_properties_the_commands_are_as_before(provider, stock_limits, count, digest):
    log = _scripted_commands(dict({'obstacle_margin': '2.5', 'consider_trafficlights': 'true'}, **stock_limits))
    text = repr([tuple(round(v, 9) if isinstance(v, float) else v for v in entry) for entry in log])
    # The same runs on the controller of 0ceb118 (before comfort_* and the box distance).
    assert len(log) == count
    assert hashlib.sha256(text.encode()).hexdigest() == digest
