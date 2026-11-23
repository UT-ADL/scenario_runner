#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests (pytest, no CARLA server, fake lights) for the traffic signals of standard OpenSCENARIO
files (VSE's export).

Pinned contracts:
- a standard file's TrafficSignalStateAction name resolves through vse_signal_<name> = "x y z yaw"
  (CARLA-frame trigger-box centre + heading) to EVERY light whose trigger-box centre is within
  0.5 m and heading within 30 deg, nearest first; none raises with the nearest distance; without
  the parameter the old lookup (OpenDRIVE id, id=, pos=) is used; a town file ignores it;
- lowercase lamp words (on;off;off ...) and GREEN/RED/YELLOW/OFF give the CARLA states;
- a standard file holds: is_frozen() is read once before the first change, then each light is
  frozen before set_state; a gone light fails the step and is printed; a town file keeps the
  plain set_state;
- release: setting vse_signal_<name>_release ends that approach; the lights are released only once
  no approach is held (freeze is map-wide): freeze(False) unless the map was frozen before the run,
  then one reset_group per touched group; the scenario's end (OpenScenario.terminate,
  CarlaDataProvider.cleanup) releases whatever is still held, once, and never raises.
"""

import xml.etree.ElementTree as ET

import carla
import py_trees
import pytest

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ChangeParameter, TrafficLightHold,
                                                                      TrafficLightRelease, TrafficLightStateSetter)
from srunner.scenarios.open_scenario import OpenScenario
from srunner.tools.openscenario_parser import OpenScenarioParser

RED, YELLOW, GREEN, OFF = (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow,
                           carla.TrafficLightState.Green, carla.TrafficLightState.Off)


class _Map(object):
    """The map-wide freeze flag (CARLA's freeze is FreezeAllTrafficLights) and every light call, in order."""

    def __init__(self, frozen=False):
        self.frozen = frozen
        self.calls = []


class _FakeLight(object):
    def __init__(self, world, actor_id, x, y, yaw=0.0, box=(0.0, 0.0, 0.0), box_yaw=0.0, opendrive_id=None):
        self._map = world
        self.id = actor_id
        self.type_id = "traffic.traffic_light"
        self.is_alive = True
        self.state = None
        self.group = [self]
        self._opendrive_id = opendrive_id
        self._transform = carla.Transform(carla.Location(x=x, y=y, z=0.0), carla.Rotation(yaw=yaw))
        self.trigger_volume = carla.BoundingBox(carla.Location(*box), carla.Vector3D(1.0, 1.0, 1.0))
        self.trigger_volume.rotation = carla.Rotation(yaw=box_yaw)

    def get_transform(self):
        return self._transform

    def get_opendrive_id(self):
        return self._opendrive_id

    def is_frozen(self):
        self._map.calls.append(("is_frozen", self.id))
        return self._map.frozen

    def freeze(self, freeze):
        self._map.calls.append(("freeze", self.id, freeze))
        self._map.frozen = freeze

    def set_state(self, state):
        self._map.calls.append(("set_state", self.id, state))
        self.state = state

    def get_group_traffic_lights(self):
        return self.group

    def reset_group(self):
        self._map.calls.append(("reset_group", self.id))


class _FakeActorList(list):
    def filter(self, pattern):
        return self


def _group(*lights):
    for light in lights:
        light.group = list(lights)


@pytest.fixture(autouse=True)
def _clean_state():
    """Restore the parser's per-file state and the data provider's light registry after each test."""
    saved = (OpenScenarioParser.standard_semantics, OpenScenarioParser.global_osc_parameters)
    yield
    OpenScenarioParser.standard_semantics, OpenScenarioParser.global_osc_parameters = saved
    OpenScenarioParser._light_identities = None
    CarlaDataProvider._held_lights = {}
    CarlaDataProvider._touched_lights = []
    CarlaDataProvider._lights_frozen_before = None


@pytest.fixture
def world():
    return _Map()


@pytest.fixture
def lights(world, monkeypatch):
    """Approach A: twins 10 and 11 (box centres 0.3 m apart, heading 90); approach B: light 20 2 m away."""
    found = _FakeActorList([
        _FakeLight(world, 10, 100.0, 50.0, yaw=90.0, box=(2.0, 0.0, 0.0), opendrive_id="301"),  # centre (100, 52)
        _FakeLight(world, 11, 100.3, 50.0, yaw=90.0, box=(2.0, 0.0, 0.0), opendrive_id="302"),  # centre (100.3, 52)
        _FakeLight(world, 20, 102.0, 50.0, yaw=90.0, box=(2.0, 0.0, 0.0), opendrive_id="303"),  # centre (102, 52)
    ])
    monkeypatch.setattr(CarlaDataProvider, "get_all_actors", staticmethod(lambda: found))
    return found


def _standard_file(signals=None):
    """Read as a standard file declaring signals {name: "x y z yaw"}."""
    OpenScenarioParser.set_standard_semantics(True)
    OpenScenarioParser.set_global_parameters(
        {"vse_signal_" + name: value for name, value in (signals or {}).items()})


def _ids(found):
    return [light.id for light in found]


# --- lookup ---------------------------------------------------------------------------------------

def test_signal_resolves_every_light_of_its_approach(lights):
    _standard_file({"1123": "100.25 52 0 90"})
    assert _ids(OpenScenarioParser.get_traffic_lights_from_osc_name("1123")) == [11, 10]  # nearest first


def test_signal_centre_is_the_trigger_box_in_the_light_frame(lights):
    # light 20 stands at (102, 50) turned 90 deg: its box 2 m ahead lies at (102, 52), not (104, 50)
    _standard_file({"7": "102 52 0 90"})
    assert _ids(OpenScenarioParser.get_traffic_lights_from_osc_name("7")) == [20]


@pytest.mark.parametrize("heading, found", [
    (90.0, [10]),
    (115.0, [10]),     # 25 deg off
    (125.0, []),       # 35 deg off
    (-270.0, [10]),    # the same heading written the other way round
    (270.0, []),       # the opposite approach
])
def test_signal_heading_rule(lights, heading, found):
    lights[:] = lights[:1]
    _standard_file({"7": "100 52 0 {}".format(heading)})
    if found:
        assert _ids(OpenScenarioParser.get_traffic_lights_from_osc_name("7")) == found
    else:
        with pytest.raises(AttributeError):
            OpenScenarioParser.get_traffic_lights_from_osc_name("7")


def test_box_heading_adds_to_the_light_heading(world, monkeypatch):
    light = _FakeLight(world, 5, 0.0, 0.0, yaw=80.0, box_yaw=10.0)
    monkeypatch.setattr(CarlaDataProvider, "get_all_actors", staticmethod(lambda: _FakeActorList([light])))
    _standard_file({"7": "0 0 0 90"})
    assert _ids(OpenScenarioParser.get_traffic_lights_from_osc_name("7")) == [5]


def test_signal_without_a_light_raises_with_the_nearest_distance(lights):
    _standard_file({"7": "103.5 52 0 90"})
    with pytest.raises(AttributeError, match=r"signal 7 .*nearest trigger box 1\.50 m away"):
        OpenScenarioParser.get_traffic_lights_from_osc_name("7")


def test_signal_on_a_map_without_lights_raises(monkeypatch):
    monkeypatch.setattr(CarlaDataProvider, "get_all_actors", staticmethod(lambda: _FakeActorList([])))
    _standard_file({"7": "0 0 0 0"})
    with pytest.raises(AttributeError, match="no traffic lights"):
        OpenScenarioParser.get_traffic_lights_from_osc_name("7")


def test_malformed_signal_parameter_raises(lights):
    _standard_file({"7": "100 52 90"})
    with pytest.raises(ValueError, match="vse_signal_7"):
        OpenScenarioParser.get_traffic_lights_from_osc_name("7")


def test_condition_name_resolves_to_the_nearest_light(lights):
    _standard_file({"1123": "100.25 52 0 90"})
    assert OpenScenarioParser.get_traffic_light_from_osc_name("1123") is lights[1]


def test_standard_file_without_the_parameter_keeps_the_old_lookup(lights):
    _standard_file()
    assert OpenScenarioParser.get_traffic_lights_from_osc_name("302") == [lights[1]]
    assert OpenScenarioParser.get_traffic_lights_from_osc_name("id=20") == [lights[2]]


def test_town_file_ignores_the_signal_parameter(lights):
    _standard_file({"302": "102 52 0 90"})
    OpenScenarioParser.set_standard_semantics(False)
    assert OpenScenarioParser.get_traffic_light_from_osc_name("302") is lights[1]  # OpenDRIVE id, not light 20


def test_identities_are_swept_once_per_file(lights):
    _standard_file({"1": "100 52 0 90", "2": "102 52 0 90"})
    OpenScenarioParser.get_traffic_lights_from_osc_name("1")
    lights.append(_FakeLight(lights[0]._map, 30, 102.0, 50.0, yaw=90.0, box=(2.0, 0.0, 0.0)))
    assert _ids(OpenScenarioParser.get_traffic_lights_from_osc_name("2")) == [20]
    OpenScenarioParser.set_global_parameters(OpenScenarioParser.global_osc_parameters)  # next file
    assert sorted(_ids(OpenScenarioParser.get_traffic_lights_from_osc_name("2"))) == [20, 30]


# --- actions ----------------------------------------------------------------------------------------

def _state_action(name, state):
    return ET.fromstring(
        '<Action name="step"><GlobalAction><InfrastructureAction><TrafficSignalAction>'
        '<TrafficSignalStateAction name="{}" state="{}"/>'
        '</TrafficSignalAction></InfrastructureAction></GlobalAction></Action>'.format(name, state))


def _parameter_action(ref, value="1"):
    return ET.fromstring(
        '<Action name="release"><GlobalAction><ParameterAction parameterRef="{}">'
        '<SetAction value="{}"/></ParameterAction></GlobalAction></Action>'.format(ref, value))


def _convert(action):
    return OpenScenarioParser.convert_maneuver_to_atomic(action, None, [], {}, None)


@pytest.mark.parametrize("word, state", [
    ("on;off;off", RED),
    ("off;on;off", YELLOW),
    ("off;off;on", GREEN),
    ("off;off;off", OFF),
    ("GREEN", GREEN),
    ("red", RED),
])
def test_lamp_words_in_an_action(lights, word, state):
    _standard_file({"1123": "100.25 52 0 90"})
    atomic = _convert(_state_action("1123", word))
    assert isinstance(atomic, TrafficLightHold)
    assert atomic._state == state


def test_action_holds_every_light_of_the_approach(lights, world):
    _standard_file({"1123": "100.25 52 0 90"})
    atomic = _convert(_state_action("1123", "off;off;on"))
    assert _ids(atomic._lights) == [11, 10]
    assert atomic.update() == py_trees.common.Status.SUCCESS
    assert lights[0].state == GREEN and lights[1].state == GREEN and lights[2].state is None
    assert world.frozen


def test_hold_reads_is_frozen_once_then_freezes_before_each_set(lights, world):
    _standard_file({"1123": "100.25 52 0 90"})
    atomic = _convert(_state_action("1123", "on;off;off"))  # is_frozen is read while the tree is built
    atomic.update()
    _convert(_state_action("1123", "off;off;on")).update()
    assert world.calls == [
        ("is_frozen", 11),
        ("freeze", 11, True), ("set_state", 11, RED), ("freeze", 10, True), ("set_state", 10, RED),
        ("freeze", 11, True), ("set_state", 11, GREEN), ("freeze", 10, True), ("set_state", 10, GREEN),
    ]


def test_hold_on_a_gone_light_fails_and_sets_the_others(lights, world, capsys):
    _standard_file({"1123": "100.25 52 0 90"})
    atomic = _convert(_state_action("1123", "on;off;off"))
    lights[1].is_alive = False
    assert atomic.update() == py_trees.common.Status.FAILURE
    assert lights[0].state == RED and lights[1].state is None
    assert "light(s) 11 gone" in capsys.readouterr().out


def test_standard_file_without_the_parameter_holds_its_one_light(lights, world):
    _standard_file()
    atomic = _convert(_state_action("302", "RED"))
    assert isinstance(atomic, TrafficLightHold) and _ids(atomic._lights) == [11]


def test_town_file_action_keeps_the_plain_setter(lights, world):
    OpenScenarioParser.set_standard_semantics(False)
    atomic = _convert(_state_action("302", "GREEN"))
    assert isinstance(atomic, TrafficLightStateSetter)
    atomic.update()
    assert world.calls == [("set_state", 11, GREEN)]  # no freeze, no is_frozen


def test_release_parameter_builds_a_release_in_a_standard_file_only(lights):
    _standard_file()
    release = _convert(_parameter_action("vse_signal_1123_release"))
    assert isinstance(release, TrafficLightRelease) and release._signal == "1123"
    assert type(_convert(_parameter_action("speed"))) is ChangeParameter
    OpenScenarioParser.set_standard_semantics(False)
    assert type(_convert(_parameter_action("vse_signal_1123_release"))) is ChangeParameter


# --- release ----------------------------------------------------------------------------------------

def _hold(signal, found, state=RED):
    return CarlaDataProvider.hold_traffic_lights(signal, found, state)


def _released(world):
    """The release calls: freeze(False) and reset_group."""
    return [call for call in world.calls if call[0] == "reset_group" or call == ("freeze", call[1], False)]


def test_release_parameter_sets_the_parameter_and_releases(lights, world):
    _standard_file()
    _hold("1123", lights[:2])
    assert _convert(_parameter_action("vse_signal_1123_release")).update() == py_trees.common.Status.SUCCESS
    assert float(CarlaDataProvider.get_osc_global_param_value("vse_signal_1123_release")) == 1.0
    assert not world.frozen


def test_release_of_one_of_two_held_approaches_waits(lights, world, capsys):
    _hold("A", lights[:2])
    _hold("B", lights[2:])
    CarlaDataProvider.release_traffic_light_approach("A")
    assert _released(world) == []
    assert world.frozen
    assert "released when signal(s) B are done" in capsys.readouterr().out


def test_release_of_the_last_approach_unfreezes_and_resets_each_group_once(lights, world):
    _group(lights[0], lights[1])
    _hold("A", lights[:2])
    _hold("B", lights[2:])
    CarlaDataProvider.release_traffic_light_approach("A")
    CarlaDataProvider.release_traffic_light_approach("B")
    assert not world.frozen
    resets = [call for call in world.calls if call[0] == "reset_group"]
    assert resets == [("reset_group", 10), ("reset_group", 20)]
    last_unfreeze = max(i for i, call in enumerate(world.calls) if call == ("freeze", call[1], False))
    first_reset = min(i for i, call in enumerate(world.calls) if call[0] == "reset_group")
    assert last_unfreeze < first_reset


def test_release_keeps_a_map_frozen_before_the_run(lights, world, capsys):
    world.frozen = True
    _hold("A", lights[:2])
    assert "already frozen at start" in capsys.readouterr().out
    CarlaDataProvider.release_traffic_light_approach("A")
    assert world.frozen
    assert ("freeze", 10, False) not in world.calls and ("freeze", 11, False) not in world.calls
    assert [call for call in world.calls if call[0] == "reset_group"] == [("reset_group", 10), ("reset_group", 11)]


def test_approach_held_again_after_a_release_is_released_again(lights, world):
    _hold("A", lights[:1])
    CarlaDataProvider.release_traffic_light_approach("A")
    _hold("A", lights[:1], GREEN)
    assert world.frozen
    CarlaDataProvider.release_traffic_light_approach("A")
    assert not world.frozen
    assert [call for call in world.calls if call[0] == "is_frozen"] == [("is_frozen", 10)]
    assert [call for call in world.calls if call[0] == "reset_group"] == [("reset_group", 10)] * 2


def _scenario():
    scenario = OpenScenario.__new__(OpenScenario)
    scenario.scenario_tree = py_trees.behaviours.Success(name="tree")
    return scenario


def test_scenario_end_releases_approaches_still_held(lights, world):
    _hold("A", lights[:2])
    _hold("B", lights[2:])
    CarlaDataProvider.release_traffic_light_approach("A")
    _scenario().terminate()
    assert not world.frozen
    assert [call for call in world.calls if call[0] == "reset_group"] == [
        ("reset_group", 10), ("reset_group", 11), ("reset_group", 20)]
    assert CarlaDataProvider._lights_frozen_before is None  # the next run reads it again


@pytest.fixture
def provider_state(monkeypatch):
    """Let CarlaDataProvider.cleanup() run on an empty actor pool; every class attribute comes back after."""
    saved = {key: value for key, value in vars(CarlaDataProvider).items()
             if key.startswith("_") and not key.startswith("__") and not isinstance(value, staticmethod)}
    monkeypatch.setattr(CarlaDataProvider, "_carla_actor_pool", {})
    monkeypatch.setattr(CarlaDataProvider, "_client", None)
    yield
    for key, value in saved.items():
        setattr(CarlaDataProvider, key, value)


def test_scenario_end_releases_once(lights, world, provider_state):
    _hold("A", lights[:1])
    scenario = _scenario()
    scenario.terminate()
    scenario.terminate()
    CarlaDataProvider.cleanup()
    assert [call for call in world.calls if call[0] == "reset_group"] == [("reset_group", 10)]


def test_scenario_end_keeps_a_map_frozen_before_the_run(lights, world):
    world.frozen = True
    _hold("A", lights[:1])
    _scenario().terminate()
    assert world.frozen
    assert ("freeze", 10, False) not in world.calls


def test_data_provider_cleanup_releases_held_lights(lights, world, provider_state):
    _hold("A", lights[:2])
    CarlaDataProvider.cleanup()
    assert not world.frozen
    assert len([call for call in world.calls if call[0] == "reset_group"]) == 2


def test_release_on_a_gone_server_prints_and_does_not_raise(lights, world, capsys):
    _hold("A", lights[:1])

    def _timeout(freeze):
        raise RuntimeError("time-out of 10000ms while waiting for the simulator")
    lights[0].freeze = _timeout
    _scenario().terminate()
    assert "release failed (time-out" in capsys.readouterr().out
    assert CarlaDataProvider._touched_lights == []


def test_release_skips_gone_lights(lights, world):
    _hold("A", lights[:2])
    lights[0].is_alive = False
    CarlaDataProvider.release_all_traffic_lights()
    assert ("reset_group", 10) not in world.calls and ("reset_group", 11) in world.calls
