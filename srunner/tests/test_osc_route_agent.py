#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests (pytest, no CARLA server, no ROS) for handing an OpenSCENARIO ego path to an agent.

Pinned contracts:
- --agent with --openscenario needs --waitForEgo; without --agent, $VSE_XOSC_AGENT names it there;
  only route mode forces --sync;
- the ego's path is its first FollowTrajectoryAction / AssignRouteAction (Init first, then the
  stories), e.g. VSE's ego_vehicleTrajectoryAction;
- the agent gets that path in CARLA's frame as its global plan; no path = no agent;
- every point of that plan survives the real downsample_route that Autoware Mini's
  carla_minimal_agent runs on it (its downsampling_interval, 2-5 points, near and far apart);
- the manager runs a route agent's run_step every tick and never applies its control.
"""

import argparse
import sys
import types
import xml.etree.ElementTree as ET

import carla
import pytest

import scenario_runner as sr
from srunner.scenariomanager.scenario_manager import ScenarioManager
from srunner.tools.openscenario_parser import OpenScenarioParser
from srunner.tools.route_manipulation import downsample_route


def _args(**kwargs):
    base = dict(openscenario=None, waitForEgo=False, agent=None, scenario=None, route=None,
                openscenario2=None, sync=False)
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_env_names_the_agent_for_openscenario_with_outside_ego():
    args = _args(openscenario="a.xosc", waitForEgo=True)
    assert sr.resolve_agent_arguments(args, {"VSE_XOSC_AGENT": "/x/carla_minimal_agent.py"})
    assert args.agent == "/x/carla_minimal_agent.py"
    assert args.sync is False


def test_env_is_ignored_without_wait_for_ego():
    args = _args(openscenario="a.xosc")
    assert sr.resolve_agent_arguments(args, {"VSE_XOSC_AGENT": "/x/agent.py"})
    assert args.agent is None


def test_agent_with_openscenario_needs_wait_for_ego():
    assert not sr.resolve_agent_arguments(_args(openscenario="a.xosc", agent="a.py"), {})
    assert sr.resolve_agent_arguments(_args(openscenario="a.xosc", agent="a.py", waitForEgo=True), {})


def test_route_mode_agent_still_forces_sync():
    args = _args(route="r.xml", agent="a.py")
    assert sr.resolve_agent_arguments(args, {})
    assert args.sync is True


def test_agent_with_scenario_is_refused():
    assert not sr.resolve_agent_arguments(_args(scenario="FollowLeadingVehicle", agent="a.py"), {})


# A storyboard shaped as VSE writes it: the ego's path in EgoAct, an NPC path elsewhere.
_STORYBOARD = """
<Storyboard>
  <Init><Actions>
    <Private entityRef="ego_vehicle"><PrivateAction><TeleportAction><Position>
      <WorldPosition x="10" y="5" z="0.5" h="0"/></Position></TeleportAction></PrivateAction></Private>
  </Actions></Init>
  <Story name="MyStory">
    <Act name="Behavior"><ManeuverGroup maximumExecutionCount="1" name="adversary_1Sequence">
      <Actors selectTriggeringEntities="false"><EntityRef entityRef="adversary_1"/></Actors>
      <Maneuver name="m"><Event name="adversary_1Drive" priority="overwrite">
        <Action name="adversary_1TrajectoryAction"><PrivateAction><RoutingAction><FollowTrajectoryAction>
          <Trajectory name="t" closed="false"><Shape><Polyline>
            <Vertex time="0"><Position><WorldPosition x="99" y="99" z="0"/></Position></Vertex>
            <Vertex time="1"><Position><WorldPosition x="98" y="99" z="0"/></Position></Vertex>
          </Polyline></Shape></Trajectory><TimeReference><None/></TimeReference>
          <TrajectoryFollowingMode followingMode="position"/></FollowTrajectoryAction></RoutingAction></PrivateAction>
        </Action></Event></Maneuver></ManeuverGroup></Act>
    <Act name="EgoAct"><ManeuverGroup maximumExecutionCount="1" name="ego_vehicleSequence">
      <Actors selectTriggeringEntities="false"><EntityRef entityRef="ego_vehicle"/></Actors>
      <Maneuver name="ego_vehicleManeuver"><Event name="ego_vehicleRoute" priority="overwrite">
        <Action name="ego_vehicleTrajectoryAction"><PrivateAction><RoutingAction><FollowTrajectoryAction>
          <Trajectory name="ego_vehicleTrajectory" closed="false"><Shape><Polyline>
            <Vertex time="0"><Position><WorldPosition x="10" y="5" z="0.5" h="0"/></Position></Vertex>
            <Vertex time="2"><Position><WorldPosition x="30" y="5" z="0.5" h="1.5708"/></Position></Vertex>
            <Vertex time="4"><Position><WorldPosition x="30" y="25" z="0.5" h="1.5708"/></Position></Vertex>
          </Polyline></Shape></Trajectory><TimeReference><None/></TimeReference>
          <TrajectoryFollowingMode followingMode="position"/></FollowTrajectoryAction></RoutingAction></PrivateAction>
        </Action></Event></Maneuver></ManeuverGroup></Act>
  </Story>
</Storyboard>
"""


@pytest.fixture(autouse=True)
def _parser_state():
    saved = (OpenScenarioParser.standard_semantics, OpenScenarioParser.use_carla_coordinate_system)
    OpenScenarioParser.standard_semantics = True
    OpenScenarioParser.use_carla_coordinate_system = False
    yield
    OpenScenarioParser.standard_semantics, OpenScenarioParser.use_carla_coordinate_system = saved


def test_ego_path_is_found_in_the_story():
    path = OpenScenarioParser.get_actor_path(ET.fromstring(_STORYBOARD), "ego_vehicle", {})
    xs = [float(p.find("WorldPosition").get("x")) for p in path]
    assert xs == [10.0, 30.0, 30.0]


def test_actor_without_path_has_none():
    assert OpenScenarioParser.get_actor_path(ET.fromstring(_STORYBOARD), "nobody", {}) == []


class _RecordingAgent(object):
    """Stands in for carla_minimal_agent: records what it is handed."""

    instances = []

    def __init__(self, config):
        self.plan = None
        self.steps = []
        _RecordingAgent.instances.append(self)

    def set_global_plan(self, gps, world_coord):
        self.plan = (gps, world_coord)

    def run_step(self, input_data, timestamp):
        self.steps.append((input_data, timestamp))
        return carla.VehicleControl(throttle=1.0)


def _runner_with_agent(monkeypatch):
    module = types.ModuleType("recording_agent")
    module.RecordingAgent = _RecordingAgent
    runner = sr.ScenarioRunner.__new__(sr.ScenarioRunner)
    runner.module_agent = module
    runner._args = _args(openscenario="a.xosc", waitForEgo=True, agent="recording_agent.py")
    runner._args.agentConfig = ""
    runner.ego_vehicles = []
    runner.world = None
    runner.agent_instance = None
    monkeypatch.setattr(sr, "_get_latlon_ref", lambda world: (58.0, 26.0))
    return runner


def _config(storyboard):
    ego = types.SimpleNamespace(rolename="ego_vehicle")
    return types.SimpleNamespace(ego_vehicles=[ego], other_actors=[], catalogs={},
                                 storyboard=ET.fromstring(storyboard))


def test_ego_path_is_handed_to_the_agent_in_carla_frame(monkeypatch, capsys):
    runner = _runner_with_agent(monkeypatch)
    agent = runner._setup_route_agent(_config(_STORYBOARD))
    assert agent is runner.agent_instance
    gps, route = agent.plan
    points = [(round(t.location.x, 3), round(t.location.y, 3), round(t.rotation.yaw, 1)) for t, _ in route]
    assert points == [(10.0, -5.0, -0.0), (30.0, -5.0, -90.0), (30.0, -25.0, -90.0)]
    assert len(gps) == 3 and set(gps[0][0]) == {"lat", "lon", "z"}
    assert "3 points, goal at CARLA (30.0, -25.0)" in capsys.readouterr().out


def _storyboard_with_ego_path(points):
    storyboard = ET.fromstring(_STORYBOARD)
    polyline = storyboard.find(".//Trajectory[@name='ego_vehicleTrajectory']/Shape/Polyline")
    for vertex in list(polyline):
        polyline.remove(vertex)
    for i, (x, y) in enumerate(points):
        polyline.append(ET.fromstring(
            '<Vertex time="%d"><Position><WorldPosition x="%r" y="%r" z="0.5" h="0"/></Position></Vertex>'
            % (i, x, y)))
    return ET.tostring(storyboard, encoding="unicode")


@pytest.mark.parametrize("count", [2, 3, 4, 5, 9])
@pytest.mark.parametrize("spacing", [0.3, 60.0])
@pytest.mark.parametrize("interval", [0, 42, 1000])  # Autoware Mini's carla.yaml: 42
def test_every_point_survives_the_agents_downsampling(monkeypatch, count, spacing, interval):
    runner = _runner_with_agent(monkeypatch)
    points = [(10.0 + i * spacing, 5.0 + (i % 2) * spacing) for i in range(count)]
    agent = runner._setup_route_agent(_config(_storyboard_with_ego_path(points)))
    _, route = agent.plan
    kept = downsample_route(route, interval)  # what carla_minimal_agent.set_global_plan keeps
    assert kept == list(range(count))
    assert [(round(t.location.x, 3), round(-t.location.y, 3)) for t, _ in route] == points


def test_no_ego_path_no_agent(monkeypatch, capsys):
    runner = _runner_with_agent(monkeypatch)
    before = len(_RecordingAgent.instances)
    storyboard = _STORYBOARD.replace('entityRef="ego_vehicle"/></Actors>', 'entityRef="other"/></Actors>')
    assert runner._setup_route_agent(_config(storyboard)) is None
    assert len(_RecordingAgent.instances) == before
    assert "has no path in the file" in capsys.readouterr().out


def test_manager_steps_the_route_agent_without_applying_its_control(monkeypatch):
    agent = _RecordingAgent("")
    ego = types.SimpleNamespace(apply_control=lambda control: pytest.fail("route agent control applied"))
    scenario = types.SimpleNamespace(scenario_tree=types.SimpleNamespace(
        tick_once=lambda: None, status="RUNNING"), ego_vehicles=[ego], other_actors=[])
    manager = ScenarioManager()
    manager.load_scenario(scenario, None, route_agent=agent)
    manager._running = True
    manager._watchdog = types.SimpleNamespace(update=lambda: None, get_status=lambda: True)
    monkeypatch.setattr("srunner.scenariomanager.scenario_manager.CarlaDataProvider.on_carla_tick",
                        staticmethod(lambda: None))
    world_ticks = []
    monkeypatch.setattr("srunner.scenariomanager.scenario_manager.CarlaDataProvider.get_world",
                        staticmethod(lambda: types.SimpleNamespace(tick=lambda: world_ticks.append(1))))
    timestamp = types.SimpleNamespace(elapsed_seconds=1.0, delta_seconds=0.05, frame=1,
                                      platform_timestamp=0.0)
    manager._tick_scenario(timestamp)
    assert len(agent.steps) == 1 and agent.steps[0][0] == {}
    assert world_ticks == []
