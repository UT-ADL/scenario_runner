#!/usr/bin/env python

import carla
import math
import random
import py_trees
import traceback
import signal
import sys

# Import CARLA BasicAgent for navigation
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ActorTransformSetter
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario

class VehicleFromTheLeft(BasicScenario):
    """
    Scenario with multiple vehicles spawned on the map that follow precisely traced routes
    to fixed destinations, ending when all destinations are reached.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=600):
        """
        Setup scenario parameters with agent-based routing
        """
        # Set world and map
        self._world = world
        self._map = world.get_map()
        self._timeout = timeout
        self._debug_mode = debug_mode

        # Agent storage - now supporting multiple agents
        self.agents = []
        self._agent_vehicles = []
        self._keep_running = True  # Flag to keep scenario running
        
        # Set destination coordinates for both agents
        self._destination_coords = [
            carla.Location(x=-259, y=-55, z=33.5),  # First agent destination
            carla.Location(x=80, y=880, z=38)   # Second agent destination
        ]
        
        # Set spawn coordinates for both agents
        self._spawn_coords = [
            carla.Location(x=-165, y=389, z=37),
            carla.Location(x=167.6, y=476, z=35)                       # Second agent spawn
        ]
        
        # Set speeds for both agents (km/h)
        self._agent_speeds = [50, 40]
        
        # Set colors for route visualization
        self._route_colors = [
            carla.Color(0, 255, 0),  # Green for first agent
            carla.Color(255, 255, 0)  # Yellow for second agent
        ]
        
        # Debug helpers
        self._debug_helper = self._world.debug
        self._route_waypoints = []
        self._last_distances = [None, None]  # For tracking progress of each agent
        
        # Set up signal handling for clean termination
        import signal
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Initialize world provider
        CarlaDataProvider.set_world(self._world)

        # Call parent constructor
        super(VehicleFromTheLeft, self).__init__(
            "VehicleFromTheLeft",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable
        )

        print("\nMultiple Vehicle Scenario Initialized")
        
    def _signal_handler(self, sig, frame):
        """Handle keyboard interrupt"""
        print("\nCtrl+C detected. Cleaning up and exiting...")
        self._cleanup()
        if self.original_sigint:
            signal.signal(signal.SIGINT, self.original_sigint)
            self.original_sigint(sig, frame)
        else:
            sys.exit(0)
            
    def _cleanup(self):
        """Clean up all actors and resources"""
        print("Cleaning up scenario actors...")
        try:
            # Clean up the agent vehicles if they exist
            for vehicle in self._agent_vehicles:
                if vehicle and vehicle.is_alive:
                    try:
                        print(f"Destroying vehicle: {vehicle.id}")
                        vehicle.destroy()
                    except Exception as e:
                        print(f"Error destroying vehicle: {e}")
            
            # Clean up any other actors in the other_actors list
            for actor in self.other_actors:
                if actor and actor.is_alive:
                    try:
                        print(f"Destroying actor: {actor.id}")
                        actor.destroy()
                    except Exception as e:
                        print(f"Error destroying actor: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("Cleanup completed")
            
    def _draw_waypoints(self, waypoints, color, lifetime=60.0):
        """
        Draw waypoints in the CARLA scene
        
        Args:
            waypoints: List of (waypoint, RoadOption) tuples or waypoints
            color: CARLA Color object for the route
            lifetime: How long the markers should stay visible
        """
        print(f"Visualizing {len(waypoints)} waypoints with color {color}")
        
        # Store waypoints for later reference
        self._route_waypoints.append(waypoints)
        
        try:
            # Draw a line for the whole route
            for i in range(len(waypoints) - 1):
                # Get current and next waypoint
                current_wp = waypoints[i][0] if isinstance(waypoints[i], tuple) else waypoints[i]
                next_wp = waypoints[i+1][0] if isinstance(waypoints[i+1], tuple) else waypoints[i+1]
                
                # Get locations
                if hasattr(current_wp, 'transform'):
                    current_loc = current_wp.transform.location
                else:
                    current_loc = current_wp
                    
                if hasattr(next_wp, 'transform'):
                    next_loc = next_wp.transform.location
                else:
                    next_loc = next_wp
                
                # Draw line between waypoints
                self._debug_helper.draw_line(
                    current_loc + carla.Location(z=0.5), 
                    next_loc + carla.Location(z=0.5),
                    thickness=0.2, 
                    color=color, 
                    life_time=lifetime
                )
                
                # Draw point at waypoint
                self._debug_helper.draw_point(
                    current_loc + carla.Location(z=0.5),
                    size=0.1,
                    color=color,
                    life_time=lifetime
                )
            
            # Draw last waypoint
            last_wp = waypoints[-1][0] if isinstance(waypoints[-1], tuple) else waypoints[-1]
            if hasattr(last_wp, 'transform'):
                last_loc = last_wp.transform.location
            else:
                last_loc = last_wp
                
            self._debug_helper.draw_point(
                last_loc + carla.Location(z=0.5),
                size=0.1,
                color=color,
                life_time=lifetime
            )
            
            # Draw destination marker
            self._debug_helper.draw_point(
                last_loc + carla.Location(z=1.0),
                size=1.0,
                color=carla.Color(255, 0, 0),
                life_time=lifetime
            )
            
            print("Route visualization complete")
            
        except Exception as e:
            print(f"Error drawing waypoints: {e}")
            traceback.print_exc()

    def _initialize_actors(self, config):
        """
        Spawn vehicles and set up BasicAgents for precise routing to fixed destinations
        """
        print("\n--- Detailed Actor Spawning and Agent Setup ---")

        # Select vehicle blueprint - always use Lincoln MKZ
        blueprint_library = self._world.get_blueprint_library()
        blueprint = blueprint_library.find('vehicle.lincoln.mkz_2017')
        
        # Spawn all agents
        for i in range(len(self._destination_coords)):
            print(f"\nSetting up agent {i+1}...")
            
            # Get all possible spawn points
            spawn_points = self._map.get_spawn_points()

            # Find spawn point closest to the desired spawn location
            target_location = self._spawn_coords[i]
            closest_spawn = None
            closest_distance = float('inf')
            
            try:
                for spawn_point in spawn_points:
                    distance = spawn_point.location.distance(target_location)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_spawn = spawn_point
                
                # Use the closest spawn point
                spawn_transform = closest_spawn
                print(f"Using fixed spawn point for agent {i+1}: {spawn_transform}")
                print(f"Distance from target spawn: {closest_distance:.2f} meters")
            except Exception as e:
                print(f"Error selecting spawn point for agent {i+1}: {e}")
                continue

            # Spawn the vehicle
            try:
                agent_vehicle = self._world.try_spawn_actor(blueprint, spawn_transform)
                
                if agent_vehicle:
                    print(f"Vehicle {i+1} spawned successfully!")
                    
                    # Create BasicAgent with appropriate speed
                    agent = BasicAgent(agent_vehicle, target_speed=self._agent_speeds[i])
                    
                    # Configure agent behavior
                    agent.ignore_traffic_lights(False)  # Obey traffic lights
                    agent.ignore_stop_signs(False)      # Obey stop signs
                    agent.ignore_vehicles(False)        # React to other vehicles
                    
                    # Find the closest waypoint to our desired destination
                    destination_waypoint = self._map.get_waypoint(
                        self._destination_coords[i],
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving
                    )
                    
                    print(f"Found destination waypoint at {destination_waypoint.transform.location}")
                    print(f"Distance from target destination: {destination_waypoint.transform.location.distance(self._destination_coords[i]):.2f} meters")
                    
                    # Get current location waypoint
                    current_location = agent_vehicle.get_location()
                    current_waypoint = self._map.get_waypoint(
                        current_location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving
                    )
                    
                    # Try to use global planner to trace a route
                    try:
                        # Create a global route between current position and destination
                        global_route = agent._global_planner.trace_route(
                            current_waypoint.transform.location,
                            destination_waypoint.transform.location
                        )
                        
                        # Validate route
                        if not global_route or len(global_route) < 2:
                            print(f"Warning: Generated route for agent {i+1} is too short or invalid")
                            # Fallback to direct destination
                            agent.set_destination(destination_waypoint.transform.location)
                        else:
                            # Set the global plan for the agent
                            agent.set_global_plan(global_route)
                            print(f"Set route with {len(global_route)} waypoints for agent {i+1}")

                            # Print waypoints in XML format for both agents
                            print(f"\nAgent {i+1} Waypoints in XML format:")
                            for wp_tuple in global_route:
                                wp = wp_tuple[0]  # Extract waypoint from tuple
                                loc = wp.transform.location
                                print(f'<position x="{loc.x:.6f}" y="{loc.y:.6f}" z="{loc.z:.6f}"/>')
                            print("End of waypoints\n")
                            
                            # Visualize the route with appropriate color
                            self._draw_waypoints(global_route, self._route_colors[i])
                    except Exception as e:
                        print(f"Error with route planning for agent {i+1}: {e}")
                        # Fallback to direct destination
                        try:
                            agent.set_destination(destination_waypoint.transform.location)
                            print(f"Falling back to direct destination: {destination_waypoint.transform.location}")
                        except Exception as e2:
                            print(f"Error setting fallback destination for agent {i+1}: {e2}")
                            continue
                    
                    # Store agent and vehicle
                    self.agents.append(agent)
                    self._agent_vehicles.append(agent_vehicle)
                    
                    # Register the vehicle
                    self.other_actors.append(agent_vehicle)
                    CarlaDataProvider.register_actor(agent_vehicle)
                else:
                    print(f"Failed to spawn vehicle {i+1}")
            
            except Exception as e:
                print(f"Error spawning vehicle {i+1}: {e}")
                traceback.print_exc()
        
        # Return first vehicle for compatibility with existing code
        return self._agent_vehicles[0] if self._agent_vehicles else None

    def _create_behavior(self):
        """
        Create scenario behavior sequence that keeps the agents running until destinations are reached
        """
        try:
            # Create a parallel behavior
            parallel_behavior = py_trees.composites.Parallel(
                name="VehicleRouteParallel",
                policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
            )
            
            # Add timeout behavior
            timeout_behavior = TimeOut(self._timeout)
            parallel_behavior.add_child(timeout_behavior)
            
            # Create a sequence behavior
            sequence = py_trees.composites.Sequence("DestinationSequence")
            
            # Custom behavior to drive until destination is reached
            class DriveToDestination(py_trees.behaviour.Behaviour):
                def __init__(self, scenario):
                    super().__init__("DriveToDestination")
                    self.scenario = scenario
                    self.running_time = 0
                    self.agent_done = [False] * len(scenario.agents)
                
                def initialise(self):
                    self.running_time = 0
                    # If scenario is no longer running, do not reinitialize
                    if not self.scenario._keep_running:
                        print("Scenario is ending, skipping reinitialization")
                        self._success_returned = True
                        self.agent_done = [True] * len(self.scenario.agents)
                        return
                    
                    self._success_returned = False
                    self.agent_done = [False] * len(self.scenario.agents)
                    print("Initializing drive to destination behavior")
                
                def update(self):
                    try:
                        # Check if all agents are done
                        if all(self.agent_done):
                            print("All destinations reached! Ending scenario.")
                            # Force termination of the scenario
                            self.scenario._keep_running = False
                            # This is critical - must return SUCCESS only once to avoid reinitializing
                            if not hasattr(self, "_success_returned") or not self._success_returned:
                                self._success_returned = True
                                return py_trees.common.Status.SUCCESS
                            else:
                                # After returning SUCCESS once, return FAILURE to prevent reinitialization
                                return py_trees.common.Status.FAILURE
                        
                        # Process each agent
                        for i, (agent, vehicle) in enumerate(zip(self.scenario.agents, self.scenario._agent_vehicles)):
                            # Skip if already done or not valid
                            if self.agent_done[i] or not vehicle or not vehicle.is_alive:
                                continue
                                
                            # Check if destination is reached
                            if agent.done():
                                print(f"Agent {i+1} reached destination!")
                                self.agent_done[i] = True
                                continue
                            
                            # Get current location for status updates
                            current_location = vehicle.get_location()
                            current_velocity = vehicle.get_velocity()
                            speed = current_velocity.length() * 3.6  # km/h
                            
                            # Calculate distance to destination
                            distance_to_destination = current_location.distance(
                                self.scenario._destination_coords[i]
                            )
                            
                            # Apply agent control
                            control = agent.run_step()
                            vehicle.apply_control(control)
                            
                            # Check if vehicle is off-route and try to recover
                            if self.running_time % 50 == 0:
                                # Check if we're getting closer to destination
                                if (self.running_time > 200 and 
                                    self.scenario._last_distances[i] and 
                                    distance_to_destination >= self.scenario._last_distances[i]):
                                    # Vehicle hasn't moved closer to destination in a while
                                    print(f"WARNING: Agent {i+1} not making progress toward destination!")
                                    print(f"Current distance: {distance_to_destination:.2f}, Previous: {self.scenario._last_distances[i]:.2f}")
                                    
                                    # Try to reroute if we're not making progress
                                    if distance_to_destination > self.scenario._last_distances[i] + 5:  # If we've moved significantly away
                                        print(f"Attempting to recalculate route for agent {i+1}...")
                                        try:
                                            # Get current waypoint
                                            current_wp = self.scenario._map.get_waypoint(
                                                current_location,
                                                project_to_road=True,
                                                lane_type=carla.LaneType.Driving
                                            )
                                            
                                            # Get destination waypoint
                                            dest_wp = self.scenario._map.get_waypoint(
                                                self.scenario._destination_coords[i],
                                                project_to_road=True,
                                                lane_type=carla.LaneType.Driving
                                            )
                                            
                                            # Recalculate route
                                            global_route = agent._global_planner.trace_route(
                                                current_wp.transform.location,
                                                dest_wp.transform.location
                                            )
                                            
                                            # Validate and update route
                                            if global_route and len(global_route) > 2:
                                                agent.set_global_plan(global_route)
                                                self.scenario._draw_waypoints(global_route, self.scenario._route_colors[i])
                                                print(f"Route recalculated for agent {i+1} with {len(global_route)} waypoints")
                                        except Exception as e:
                                            print(f"Error recalculating route for agent {i+1}: {e}")
                                
                                # Update last distance
                                self.scenario._last_distances[i] = distance_to_destination
                            
                            # Detailed logging if debug mode is on
                            if self.scenario._debug_mode:
                                print(f"Agent {i+1} Status - Location: {current_location}, Speed: {speed:.2f} km/h")
                                print(f"Distance to destination: {distance_to_destination:.2f} meters")
                            
                            # Periodic status update
                            if self.running_time % 100 == 0:
                                print(f"Agent {i+1} - Distance to destination: {distance_to_destination:.2f} meters, Speed: {speed:.2f} km/h")
                                
                                # Show current next waypoint if we have route data
                                if hasattr(agent, '_waypoints_queue') and agent._waypoints_queue:
                                    next_wp = agent._waypoints_queue[0]
                                    # Visualize current target waypoint
                                    if hasattr(next_wp, 'transform'):
                                        next_loc = next_wp.transform.location
                                    else:
                                        next_loc = next_wp
                                    self.scenario._debug_helper.draw_point(
                                        next_loc + carla.Location(z=1.0),
                                        size=0.2,
                                        color=self.scenario._route_colors[i],
                                        life_time=2.0
                                    )
                        
                        self.running_time += 1
                        if self.running_time % 100 == 0:
                            print(f"Scenario running for {self.running_time} steps")
                            
                        # Continue running until all agents reach their destinations
                        return py_trees.common.Status.RUNNING
                    
                    except Exception as e:
                        print(f"Error in drive to destination: {e}")
                        traceback.print_exc()
                        return py_trees.common.Status.FAILURE
            
            # Add drive to destination behavior to sequence
            drive_behavior = DriveToDestination(self)
            sequence.add_child(drive_behavior)
            
            # Add sequence to parallel behavior
            parallel_behavior.add_child(sequence)
            
            return parallel_behavior
        
        except Exception as e:
            print(f"Error creating behavior: {e}")
            traceback.print_exc()
            return None

    def _create_test_criteria(self):
        """Create scenario test criteria"""
        criteria = []
        
        # Add collision criteria for all vehicles
        for vehicle in self._agent_vehicles:
            if vehicle:
                collision_criterion = CollisionTest(vehicle)
                criteria.append(collision_criterion)
        
        return criteria
        
    def terminate(self):
        """
        Clean up all actors upon termination of the scenario
        """
        print("Scenario is terminating...")
        self._keep_running = False
        self._cleanup()
        super(VehicleFromTheLeft, self).terminate()

def get_available_scenarios():
    """
    Register scenario for ScenarioRunner
    """
    return {"VehicleFromTheLeft": VehicleFromTheLeft}