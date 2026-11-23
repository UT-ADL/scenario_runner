#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests (pytest, no CARLA server) for reading standard OpenSCENARIO files such as VSE's exports.

Pinned contracts:
- a LogicFile that is a road file of the loaded map plays on the loaded map (nothing is loaded or
  generated, the file need not exist); other LogicFiles keep the old path;
- standard_semantics is on for a road-file (.xodr) LogicFile and off for a CARLA town name; with it
  TimeReference None gives an untimed path and a condition's delay runs after the condition holds,
  without it both keep ScenarioRunner's old reading;
- the first Init TeleportAction is the start pose;
- new waypoints clear a controller's "goal reached";
- every Criterion of atomic_criteria resolves as criteria_<Name>;
- lights resolve by id=, pos= (CARLA frame) and OpenDRIVE signal id; lamp-word states are accepted;
- a refused spawn is printed with the actor's role and CARLA's reason, and its maneuver groups are
  skipped.
"""

import xml.etree.ElementTree as ET

import carla
import pytest

from srunner.scenarioconfigs import openscenario_configuration as osc_config
from srunner.scenarioconfigs.openscenario_configuration import (OpenScenarioConfiguration,
                                                                road_file_map_name,
                                                                road_file_plays_on)
from srunner.scenariomanager.actorcontrols.basic_control import BasicControl
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.open_scenario import OpenScenario
from srunner.tools.openscenario_parser import OpenScenarioParser


@pytest.fixture(autouse=True)
def _parser_state():
    """Restore the parser's per-file class state after each test."""
    saved = (OpenScenarioParser.standard_semantics, OpenScenarioParser.use_carla_coordinate_system)
    yield
    OpenScenarioParser.standard_semantics, OpenScenarioParser.use_carla_coordinate_system = saved


# --- LogicFile: road file of the loaded map ------------------------------------------------------

def test_road_file_map_name():
    assert road_file_map_name("Town01.xodr") == "Town01"
    assert road_file_map_name("/a/b/tartu_demo_mesh.xodr") == "tartu_demo_mesh"
    assert road_file_map_name("Town01") == "Town01"


@pytest.mark.parametrize("loaded, wanted, plays", [
    ("Town01", "Town01", True),
    ("Town01_Opt", "Town01", True),
    ("Town10HD_Opt", "Town10HD", True),
    ("Town10HD_Opt", "Town10", True),
    ("tartu_demo", "tartu_demo_mesh", True),
    ("tartu_demo_mesh", "tartu_demo", True),
    ("Town02", "Town01", False),
    ("Town1", "Town10HD", False),
    ("tartu_large", "tartu_demo", False),
    ("", "Town01", False),
])
def test_road_file_plays_on(loaded, wanted, plays):
    assert road_file_plays_on(loaded, wanted) is plays


class _FakeMap(object):
    def __init__(self, name):
        self.name = "Carla/Maps/" + name

    def to_opendrive(self):
        return "<OpenDRIVE>loaded</OpenDRIVE>"


class _FakeWorld(object):
    def __init__(self, name):
        self._map = _FakeMap(name)

    def get_map(self):
        return self._map

    def get_settings(self):
        return carla.WorldSettings()


class _FakeClient(object):
    def __init__(self, name):
        self.world = _FakeWorld(name)
        self.calls = []

    def get_world(self):
        return self.world

    def load_world(self, *args, **kwargs):
        self.calls.append("load_world")
        raise RuntimeError("test: load_world")

    def generate_opendrive_world(self, *args, **kwargs):
        self.calls.append("generate_opendrive_world")
        raise RuntimeError("test: generate_opendrive_world")


def _town_config(tmp_path, logic_file, loaded):
    cfg = OpenScenarioConfiguration.__new__(OpenScenarioConfiguration)
    cfg.xml_tree = ET.ElementTree(ET.fromstring(
        '<OpenSCENARIO><RoadNetwork><LogicFile filepath="%s"/></RoadNetwork></OpenSCENARIO>'
        % logic_file))
    cfg.filename = str(tmp_path / "scene.xosc")
    cfg.client = _FakeClient(loaded)
    cfg.logger = osc_config.logging.getLogger("test")
    return cfg


def test_road_file_of_loaded_map_plays_there(tmp_path, monkeypatch, capsys):
    worlds = []
    monkeypatch.setattr(CarlaDataProvider, "set_world", staticmethod(worlds.append))
    cfg = _town_config(tmp_path, "tartu_demo_mesh.xodr", "tartu_demo")  # no .xodr file on disk
    cfg._set_carla_town()
    assert cfg.town == "tartu_demo"
    assert cfg.client.calls == []
    assert worlds == [cfg.client.world]
    assert OpenScenarioParser.standard_semantics is True
    assert "plays on the loaded map tartu_demo" in capsys.readouterr().out


def test_road_file_of_another_map_keeps_the_old_path(tmp_path, monkeypatch):
    monkeypatch.setattr(CarlaDataProvider, "set_world", staticmethod(lambda world: None))
    (tmp_path / "Town02.xodr").write_text("<OpenDRIVE>file</OpenDRIVE>")
    cfg = _town_config(tmp_path, "Town02.xodr", "Town01")
    with pytest.raises(RuntimeError):
        cfg._set_carla_town()
    assert cfg.client.calls == ["generate_opendrive_world"]


def test_town_name_keeps_old_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(CarlaDataProvider, "set_world", staticmethod(lambda world: None))
    OpenScenarioParser.set_standard_semantics(True)
    cfg = _town_config(tmp_path, "Town01", "Town01")
    cfg._set_carla_town()
    assert cfg.town == "Town01"
    assert cfg.client.calls == []
    assert OpenScenarioParser.standard_semantics is False


# --- FollowTrajectoryAction TimeReference --------------------------------------------------------

def _trajectory(time_reference, times=("0", "1.5")):
    vertices = "".join(
        '<Vertex%s><Position><WorldPosition x="%d" y="0" z="0"/></Position></Vertex>'
        % (' time="%s"' % t if t is not None else "", i * 5) for i, t in enumerate(times))
    return ET.fromstring(
        '<FollowTrajectoryAction><Trajectory name="t" closed="false"><Shape><Polyline>%s'
        '</Polyline></Shape></Trajectory><TimeReference>%s</TimeReference>'
        '<TrajectoryFollowingMode followingMode="position"/></FollowTrajectoryAction>'
        % (vertices, time_reference))


def test_time_reference_none_is_untimed_in_a_standard_file():
    OpenScenarioParser.set_standard_semantics(True)
    waypoints, times = OpenScenarioParser.get_trajectory(_trajectory("<None/>"), {})
    assert len(waypoints) == 2 and times is None


def test_untimed_vertices_need_no_time():
    OpenScenarioParser.set_standard_semantics(True)
    waypoints, times = OpenScenarioParser.get_trajectory(_trajectory("<None/>", (None, None)), {})
    assert len(waypoints) == 2 and times is None


def test_timing_keeps_the_vertex_times():
    OpenScenarioParser.set_standard_semantics(True)
    timing = '<Timing domainAbsoluteRelative="relative" scale="1" offset="0"/>'
    _, times = OpenScenarioParser.get_trajectory(_trajectory(timing), {})
    assert times == [0.0, 1.5]


def test_time_reference_none_keeps_times_in_a_town_file():
    OpenScenarioParser.set_standard_semantics(False)
    _, times = OpenScenarioParser.get_trajectory(_trajectory("<None/>"), {})
    assert times == [0.0, 1.5]


# --- condition delay -----------------------------------------------------------------------------

_DELAYED = ('<Condition name="c" delay="5" conditionEdge="none"><ByValueCondition>'
            '<SimulationTimeCondition value="1" rule="greaterThan"/></ByValueCondition></Condition>')


def test_delay_runs_after_the_condition_in_a_standard_file():
    OpenScenarioParser.set_standard_semantics(True)
    node = OpenScenarioParser.convert_condition_to_atomic(ET.fromstring(_DELAYED), [])
    assert [type(c) for c in node.children][1] is TimeOut
    assert not isinstance(node.children[0], TimeOut)


def test_delay_runs_first_in_a_town_file():
    OpenScenarioParser.set_standard_semantics(False)
    node = OpenScenarioParser.convert_condition_to_atomic(ET.fromstring(_DELAYED), [])
    assert isinstance(node.children[0], TimeOut)
    assert not isinstance(node.children[1], TimeOut)


# --- Init TeleportAction -------------------------------------------------------------------------

_INIT = ('<Init><Actions><Private entityRef="car">'
         '<PrivateAction><TeleportAction><Position><WorldPosition x="1" y="2" z="0"/></Position>'
         '</TeleportAction></PrivateAction>'
         '<PrivateAction><RoutingAction><AssignRouteAction><Route name="r" closed="false">'
         '<Waypoint routeStrategy="shortest"><Position><WorldPosition x="9" y="9" z="0"/>'
         '</Position></Waypoint></Route></AssignRouteAction></RoutingAction></PrivateAction>'
         '</Private></Actions></Init>')


def _init_config(init_xml):
    cfg = OpenScenarioConfiguration.__new__(OpenScenarioConfiguration)
    cfg.init = ET.fromstring(init_xml)
    cfg.other_actors, cfg.ego_vehicles = [], []
    cfg.logger = osc_config.logging.getLogger("test")
    return cfg


def test_actor_starts_at_its_init_teleport():
    OpenScenarioParser.set_standard_semantics(False)
    transform = _init_config(_INIT)._get_actor_transform("car")
    assert (transform.location.x, transform.location.y) == (1.0, -2.0)


def test_actor_without_teleport_keeps_the_last_position():
    init = _INIT.replace("TeleportAction>", "XTeleportAction>").replace(
        "<XTeleportAction>", "<AcquirePositionAction>").replace(
        "</XTeleportAction>", "</AcquirePositionAction>")
    init = init.replace("<PrivateAction><AcquirePositionAction>",
                        "<PrivateAction><RoutingAction><AcquirePositionAction>").replace(
        "</AcquirePositionAction></PrivateAction>", "</AcquirePositionAction></RoutingAction></PrivateAction>")
    transform = _init_config(init)._get_actor_transform("car")
    assert transform.location.x == 9.0


# --- a second path -------------------------------------------------------------------------------

def test_new_waypoints_clear_reached_goal():
    control = BasicControl(None)
    control._reached_goal = True
    control.update_waypoints([carla.Transform()])
    assert control.check_reached_waypoint_goal() is False


def test_empty_waypoints_leave_reached_goal():
    control = BasicControl(None)
    control._reached_goal = True
    control.update_waypoints([])
    assert control.check_reached_waypoint_goal() is True


# --- criteria names ------------------------------------------------------------------------------

def test_every_criterion_resolves_by_name():
    condition = ET.fromstring(
        '<Condition name="criteria_ScenarioTimeoutTest" delay="0" conditionEdge="rising">'
        '<ByValueCondition><ParameterCondition parameterRef="x" value="5" rule="lessThan"/>'
        '</ByValueCondition></Condition>')
    node = OpenScenarioParser.convert_condition_to_atomic(condition, [None])
    assert type(node.children[0]).__name__ == "ScenarioTimeoutTest"


def test_unknown_criterion_still_raises():
    condition = ET.fromstring(
        '<Condition name="criteria_NoSuchTest" delay="0" conditionEdge="rising">'
        '<ByValueCondition><ParameterCondition parameterRef="" value="" rule="lessThan"/>'
        '</ByValueCondition></Condition>')
    with pytest.raises(AttributeError):
        OpenScenarioParser.convert_condition_to_atomic(condition, [None])


# --- traffic lights ------------------------------------------------------------------------------

class _FakeLight(object):
    def __init__(self, actor_id, opendrive_id, x, y):
        self.id = actor_id
        self._opendrive_id = opendrive_id
        self._transform = carla.Transform(carla.Location(x=x, y=y, z=0.0))

    def get_opendrive_id(self):
        return self._opendrive_id

    def get_transform(self):
        return self._transform


class _FakeActorList(list):
    def filter(self, pattern):
        return self


@pytest.fixture
def _lights(monkeypatch):
    lights = _FakeActorList([_FakeLight(10, "301", 5.0, -7.0), _FakeLight(11, "302", 50.0, 1.0)])
    monkeypatch.setattr(CarlaDataProvider, "get_all_actors", staticmethod(lambda: lights))
    return lights


def test_light_by_opendrive_id(_lights):
    assert OpenScenarioParser.get_traffic_light_from_osc_name("302") is _lights[1]


def test_light_by_actor_id(_lights):
    assert OpenScenarioParser.get_traffic_light_from_osc_name("id=10") is _lights[0]


def test_light_by_carla_position(_lights):
    assert OpenScenarioParser.get_traffic_light_from_osc_name("pos=5.5,-7.5") is _lights[0]


def test_unknown_light_raises(_lights):
    with pytest.raises(AttributeError):
        OpenScenarioParser.get_traffic_light_from_osc_name("999")


@pytest.mark.parametrize("word, state", [
    ("ON;OFF;OFF", carla.TrafficLightState.Red),
    ("OFF;ON;OFF", carla.TrafficLightState.Yellow),
    ("OFF;OFF;ON", carla.TrafficLightState.Green),
    ("OFF;OFF;OFF", carla.TrafficLightState.Off),
    ("GREEN", carla.TrafficLightState.Green),
])
def test_lamp_words(word, state):
    assert OpenScenarioParser.tl_states[word] == state


# --- failed spawns -------------------------------------------------------------------------------

class _Response(object):
    def __init__(self, actor_id, error):
        self.actor_id = actor_id
        self.error = error


class _Blueprint(object):
    def __init__(self, bp_id):
        self.id = bp_id

    def has_tag(self, tag):
        return self.id.startswith(tag)


class _SpawnedActor(object):
    def __init__(self, actor_id, role):
        self.id = actor_id
        self.attributes = {"role_name": role}


def test_refused_spawn_is_named(monkeypatch, capsys):
    from srunner.scenarioconfigs.scenario_configuration import ActorConfigurationData

    spawned = _SpawnedActor(7, "car_ok")
    world = type("World", (), {"get_actors": lambda self, ids: [spawned],
                               "get_settings": lambda self: carla.WorldSettings()})()
    client = type("Client", (), {"apply_batch_sync": lambda self, batch, tick: [
        _Response(7, ""), _Response(0, "Spawn failed because of collision at spawn position")]})()
    monkeypatch.setattr(CarlaDataProvider, "_client", client)
    monkeypatch.setattr(CarlaDataProvider, "_world", world)
    monkeypatch.setattr(CarlaDataProvider, "generate_spawn_points", staticmethod(lambda: None))
    monkeypatch.setattr(CarlaDataProvider, "create_blueprint",
                        staticmethod(lambda model, *a, **k: _Blueprint(model)))
    monkeypatch.setattr(CarlaDataProvider, "register_actor", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(carla.command, "SpawnActor",
                        lambda bp, tf: type("Cmd", (), {"then": lambda self, c: self})())
    monkeypatch.setattr(carla.command, "SetAutopilot", lambda *a: None)
    monkeypatch.setattr(carla.command, "FutureActor", None)
    configs = [ActorConfigurationData("vehicle.a", carla.Transform(carla.Location(1, 2, 0)), "car_ok"),
               ActorConfigurationData("vehicle.b", carla.Transform(carla.Location(3, 4, 0)), "car_bad")]
    actors = CarlaDataProvider.request_new_actors(configs, tick=False)
    assert actors == [spawned]
    out = capsys.readouterr().out
    assert "Spawn failed: car_bad (vehicle.b) at (3.0, 4.0, 0.2): Spawn failed because of collision" in out


def test_maneuver_group_of_a_missing_actor_is_skipped(capsys):
    scenario = OpenScenario.__new__(OpenScenario)
    scenario._missing_actors = {"car_bad"}
    group = ET.fromstring('<ManeuverGroup name="car_badSequence"><Actors selectTriggeringEntities="false">'
                          '<EntityRef entityRef="car_bad"/></Actors></ManeuverGroup>')
    assert scenario._skip_maneuver_group(group) is True
    assert "Skipping maneuver group car_badSequence: car_bad not spawned" in capsys.readouterr().out
    kept = ET.fromstring('<ManeuverGroup name="g"><Actors selectTriggeringEntities="false">'
                         '<EntityRef entityRef="car_ok"/></Actors></ManeuverGroup>')
    assert scenario._skip_maneuver_group(kept) is False
    no_actor = ET.fromstring('<ManeuverGroup name="lights"><Actors selectTriggeringEntities="false"/>'
                             '</ManeuverGroup>')
    assert scenario._skip_maneuver_group(no_actor) is False


# --- simple_vehicle_control: a path starting above the vehicle -----------------------------------

class _FakeVehicle(object):
    def __init__(self, location):
        self._location = location
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


def test_point_straight_above_the_vehicle_is_skipped(monkeypatch):
    from srunner.scenariomanager.actorcontrols.simple_vehicle_control import SimpleVehicleControl

    vehicle = _FakeVehicle(carla.Location(x=10.0, y=5.0, z=0.0))
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: actor.get_location()))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    control = SimpleVehicleControl(vehicle)
    control.update_target_speed(5.0)
    control.update_waypoints([carla.Transform(carla.Location(x=10.0, y=5.0, z=3.0)),  # the spawn, 3 m up
                              carla.Transform(carla.Location(x=30.0, y=5.0, z=0.0))])
    control.run_step()  # used to raise ZeroDivisionError
    assert len(control._waypoints) == 1
    assert vehicle.velocities[-1].x == pytest.approx(5.0)


def test_target_at_the_cached_location_counts_as_reached(monkeypatch):
    from srunner.scenariomanager.actorcontrols.simple_vehicle_control import SimpleVehicleControl

    vehicle = _FakeVehicle(carla.Location(x=10.0, y=5.0, z=0.0))
    cached = carla.Location(x=30.0, y=5.0, z=0.0)  # the data provider's location of this tick
    monkeypatch.setattr(CarlaDataProvider, "get_location", staticmethod(lambda actor: cached))
    monkeypatch.setattr(CarlaDataProvider, "get_transform", staticmethod(lambda actor: actor.get_transform()))
    control = SimpleVehicleControl(vehicle)
    control.update_target_speed(5.0)
    control.update_waypoints([carla.Transform(carla.Location(x=30.0, y=5.0, z=0.0))])
    control.run_step()  # used to raise ZeroDivisionError
    assert control.check_reached_waypoint_goal() is True
    assert vehicle.velocities[-1].x == 0.0


# --- Storyboard StopTrigger ----------------------------------------------------------------------

_STORYBOARD_STOP = """
<Storyboard><StopTrigger>
  <ConditionGroup><Condition name="criteria_CollisionTest" delay="0" conditionEdge="rising"><ByValueCondition>
    <ParameterCondition parameterRef="" value="" rule="lessThan"/></ByValueCondition></Condition></ConditionGroup>
  <ConditionGroup><Condition name="ScenarioTimeout" delay="0" conditionEdge="rising"><ByValueCondition>
    <SimulationTimeCondition value="1800" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup>
  <ConditionGroup><Condition name="DurationReached" delay="0" conditionEdge="rising"><ByValueCondition>
    <SimulationTimeCondition value="60" rule="greaterThan"/></ByValueCondition></Condition></ConditionGroup>
</StopTrigger></Storyboard>
"""


def _scenario_with_storyboard(xml_text):
    scenario = OpenScenario.__new__(OpenScenario)
    scenario.config = type("Config", (), {"storyboard": ET.fromstring(xml_text)})()
    scenario.other_actors, scenario.ego_vehicles = [], []
    return scenario


def test_storyboard_stop_trigger_counts_in_a_standard_file():
    OpenScenarioParser.set_standard_semantics(True)
    stop = _scenario_with_storyboard(_STORYBOARD_STOP)._create_storyboard_stop_trigger()
    assert len(stop.children) == 2  # the criteria-only group stays a test criterion


def test_storyboard_stop_trigger_is_ignored_in_a_town_file():
    OpenScenarioParser.set_standard_semantics(False)
    assert _scenario_with_storyboard(_STORYBOARD_STOP)._create_storyboard_stop_trigger() is None
