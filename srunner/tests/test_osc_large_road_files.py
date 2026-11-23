#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests (pytest, no CARLA server) for runs on large road files (Edgar's tartu_large: 77 MB+).

Pinned contracts:
- CarlaDataProvider builds the global route planner on the first get_global_route_planner() after
  set_world, once per world (it used to build one in every set_world, twice per run);
- ChangeActorWaypoints asks for the planner only for a point that is not 'shortest';
- scenario_runner builds it before the run for an old file (no standard semantics) and leaves it
  to first use for a standard file;
- _get_header_latlon_ref returns what _get_latlon_ref returns, reading only up to the header.
"""

import argparse
import types
import xml.etree.ElementTree as ET

import carla
import py_trees
import pytest
from numpy import random

import scenario_runner as sr
from srunner.scenariomanager import carla_data_provider as cdp
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeActorWaypoints
from srunner.tools.openscenario_parser import OpenScenarioParser
from srunner.tools.route_manipulation import _get_header_latlon_ref, _get_latlon_ref


class _FakeMap(object):
    def __init__(self, xodr="<OpenDRIVE/>"):
        self.name = "Carla/Maps/tartu_large"
        self._xodr = xodr

    def get_spawn_points(self):
        return []

    def to_opendrive(self):
        return self._xodr


class _FakeWorld(object):
    def __init__(self, xodr="<OpenDRIVE/>"):
        self._map = _FakeMap(xodr)

    def get_settings(self):
        return carla.WorldSettings()

    def get_map(self):
        return self._map

    def get_blueprint_library(self):
        return []

    def get_actors(self):
        return types.SimpleNamespace(filter=lambda pattern: [])

    def wait_for_tick(self):
        pass


@pytest.fixture
def planners(monkeypatch):
    """Record every GlobalRoutePlanner built; restore the data provider's state afterwards."""
    for name in ("_world", "_map", "_grp", "_sync_flag", "_blueprint_library", "_spawn_points", "_spawn_index",
                 "_client"):
        monkeypatch.setattr(CarlaDataProvider, name, getattr(CarlaDataProvider, name))
    monkeypatch.setattr(CarlaDataProvider, "_rng", random.RandomState(0))
    built = []

    def make_planner(wmap, resolution):
        built.append((wmap, resolution))
        return object()

    monkeypatch.setattr(cdp, "GlobalRoutePlanner", make_planner)
    return built


def test_route_planner_is_built_on_first_use_once_per_world(planners):
    world = _FakeWorld()
    CarlaDataProvider.set_world(world)
    CarlaDataProvider.set_world(world)  # a run calls it twice
    assert planners == []
    planner = CarlaDataProvider.get_global_route_planner()
    assert CarlaDataProvider.get_global_route_planner() is planner
    assert planners == [(world.get_map(), 2.0)]
    CarlaDataProvider.set_world(_FakeWorld())  # another world: a new planner on its first use
    assert CarlaDataProvider.get_global_route_planner() is not planner
    assert len(planners) == 2


def test_no_route_planner_without_a_world(planners):
    CarlaDataProvider._world = None
    CarlaDataProvider._grp = None
    assert CarlaDataProvider.get_global_route_planner() is None
    assert planners == []


def test_shortest_points_never_ask_for_the_route_planner(monkeypatch):
    saved = OpenScenarioParser.use_carla_coordinate_system
    OpenScenarioParser.use_carla_coordinate_system = True
    monkeypatch.setattr(CarlaDataProvider, "get_global_route_planner",
                        staticmethod(lambda: pytest.fail("route planner asked for 'shortest' points")))
    handed = []
    actor = types.SimpleNamespace(id=3)
    control = types.SimpleNamespace(update_waypoints=lambda route, start_time=None, timed=False: handed.append(route))
    py_trees.blackboard.Blackboard().set("ActorsWithController", {actor.id: control}, overwrite=True)
    try:
        positions = [ET.fromstring('<Position><WorldPosition x="%d" y="0" z="0"/></Position>' % x)
                     for x in (0, 10, 20)]
        ChangeActorWaypoints(actor, waypoints=[(p, 'shortest') for p in positions]).initialise()
    finally:
        OpenScenarioParser.use_carla_coordinate_system = saved
        py_trees.blackboard.Blackboard().set("ActorsWithController", {}, overwrite=True)
    assert [t.location.x for t in handed[0]] == [0.0, 10.0, 20.0]


@pytest.mark.parametrize("standard, builds", [(False, 1), (True, 0)])
def test_runner_builds_the_planner_before_an_old_run_only(planners, monkeypatch, standard, builds):
    monkeypatch.setattr(OpenScenarioParser, "standard_semantics", standard)
    world = _FakeWorld()
    runner = sr.ScenarioRunner.__new__(sr.ScenarioRunner)
    runner._args = argparse.Namespace(reloadWorld=False, waitForEgo=False, sync=False)
    runner.client = types.SimpleNamespace(get_world=lambda: world)
    runner._shutdown_requested = False
    assert runner._load_and_wait_for_world("tartu_large") is True
    assert len(planners) == builds


_HEADER = ('<header revMajor="1" revMinor="4" name="">'
           '<geoReference><![CDATA[+proj=tmerc +lat_0=58.38 +lon_0=26.72 +k=0.9996 +units=m]]></geoReference>'
           '<userData><vectorScene program="RoadRunner"/></userData></header>')
_ROAD = '<road name="r" length="1" id="1" junction="-1"><planView/></road>'


@pytest.mark.parametrize("xodr", [
    '<?xml version="1.0" encoding="UTF-8"?>\n<OpenDRIVE>\n    ' + _HEADER + _ROAD * 50 + '</OpenDRIVE>',
    '<OpenDRIVE><header><geoReference>+proj=merc +lon_0=2.5 +lat_0=41.1</geoReference></header></OpenDRIVE>',
    '<OpenDRIVE><header revMajor="1"/>' + _ROAD + '</OpenDRIVE>',  # no geoReference: CARLA's default
    '<OpenDRIVE><header><geoReference>+proj=utm +zone=35</geoReference></header></OpenDRIVE>',
    '<OpenDRIVE>' + _ROAD + '<userData><geoReference>+lat_0=1</geoReference></userData></OpenDRIVE>',
    '<OpenDRIVE>' + _ROAD + '</OpenDRIVE>',  # no header
])
def test_header_geo_reference_is_the_full_parse(xodr):
    world = _FakeWorld(xodr)
    assert _get_header_latlon_ref(world) == _get_latlon_ref(world)


def test_header_geo_reference_reads_only_the_header():
    # Everything after the header is never parsed (a large road file took seconds and gigabytes).
    world = _FakeWorld('<OpenDRIVE>' + _HEADER + '<road <<< not even XML')
    assert _get_header_latlon_ref(world) == (58.38, 26.72)
