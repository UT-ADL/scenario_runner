#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests (pytest, no CARLA server) for timed paths: FollowTrajectoryAction with Timing, driven by
ChangeActorWaypoints and simple_vehicle_control, simulated in float time.

Pinned contracts:
- GameTime keeps the game time its last tick added;
- a timed path asks for no speed spike when a tick lands just before a vertex time (exact double
  ticks on rows of the tick grid do): no target speed above 1.5 x the path's speed.
"""

import types
import xml.etree.ElementTree as ET

import carla
import py_trees
import pytest

from srunner.scenariomanager.actorcontrols.actor_control import ActorControl
from srunner.scenariomanager.actorcontrols.simple_vehicle_control import SimpleVehicleControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeActorWaypoints
from srunner.scenariomanager.timer import GameTime
from srunner.tools.openscenario_parser import OpenScenarioParser


class _Car(object):
    """A vehicle that moves exactly as its target velocity says, one tick at a time."""

    id = 7
    is_alive = True

    def __init__(self, x, y):
        self.location = carla.Location(x=x, y=y, z=0.0)
        self.velocity = carla.Vector3D()

    def get_location(self):
        return carla.Location(self.location.x, self.location.y, self.location.z)

    def get_velocity(self):
        return self.velocity

    def get_transform(self):
        return carla.Transform(self.get_location())

    def set_target_velocity(self, velocity):
        self.velocity = carla.Vector3D(velocity.x, velocity.y, velocity.z)

    def set_target_angular_velocity(self, velocity):
        pass

    def get_light_state(self):
        return carla.VehicleLightState.NONE

    def set_light_state(self, state):
        pass

    def move(self, seconds):
        self.location = self.location + carla.Location(x=self.velocity.x * seconds, y=self.velocity.y * seconds)


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    saved = OpenScenarioParser.use_carla_coordinate_system
    OpenScenarioParser.use_carla_coordinate_system = True
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    GameTime.restart()
    yield
    GameTime.restart()
    OpenScenarioParser.use_carla_coordinate_system = saved
    py_trees.blackboard.Blackboard().set("ActorsWithController", {}, overwrite=True)


def _drive(rows, tick, extra=1.0):
    """
    Run a timed path of rows (time, x, y) on simple_vehicle_control; return the car, the target
    speed of every tick and the action's end time.
    """
    car = _Car(rows[0][1], rows[0][2])
    control = ActorControl.__new__(ActorControl)
    control.control_instance = SimpleVehicleControl(car)
    py_trees.blackboard.Blackboard().set("ActorsWithController", {car.id: control}, overwrite=True)
    positions = [ET.fromstring('<Position><WorldPosition x="%r" y="%r" z="0"/></Position>' % (x, y))
                 for _, x, y in rows]
    times = [t for t, _, _ in rows]
    action = ChangeActorWaypoints(car, waypoints=[(p, 'shortest') for p in positions], times=times)

    frame = [0]

    def world_tick():
        frame[0] += 1
        GameTime.on_carla_tick(types.SimpleNamespace(frame=frame[0], delta_seconds=tick,
                                                     elapsed_seconds=frame[0] * tick))

    world_tick()
    action.initialise()
    speeds, ended = [], None
    for _ in range(int((times[-1] + extra) / tick)):
        if ended is None:
            if action.update() != py_trees.common.Status.RUNNING:
                ended = GameTime.get_time()
            else:
                speeds.append(control.control_instance._target_speed)
        control.run_step()
        car.move(tick)
        world_tick()
    return car, speeds, ended


def _straight_rows(speed, row_seconds, duration):
    count = int(round(duration / row_seconds))
    return [(round(i * row_seconds, 6), i * row_seconds * speed, 0.0) for i in range(count + 1)]


def test_game_time_keeps_its_last_step():
    GameTime.on_carla_tick(types.SimpleNamespace(frame=10, delta_seconds=0.05, elapsed_seconds=0.5))
    assert GameTime.get_last_step() == 0.05
    GameTime.on_carla_tick(types.SimpleNamespace(frame=13, delta_seconds=0.05, elapsed_seconds=0.65))
    assert GameTime.get_last_step() == pytest.approx(0.15)  # three frames since the last tick
    GameTime.restart()
    assert GameTime.get_last_step() == 0.0


@pytest.mark.parametrize("row_seconds", [0.05, 0.1, 0.5])
@pytest.mark.parametrize("speed", [2.0, 10.0])
def test_timed_path_speed_has_no_spikes(speed, row_seconds):
    rows = _straight_rows(speed, row_seconds, 6.0)
    # Exact double ticks land a rounding error before rows of the tick grid: the case that asked
    # for (error) / 0.001 s, threw the car off and grew to an infinite speed.
    t, near = 0.0, 0
    for _ in range(120):
        t += 0.05
        near += any(0 < row[0] - t < 0.001 for row in rows)
    assert near > 0
    _, speeds, _ = _drive(rows, 0.05)
    assert max(speeds) <= 1.5 * speed
    assert min(speeds[1:-1]) >= 0.5 * speed
