#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Sensor synchronization example for CARLA

The communication model for the syncronous mode in CARLA sends the snapshot
of the world and the sensors streams in parallel.
We provide this script as an example of how to syncrononize the sensor
data gathering in the client.
To to this, we create a queue that is being filled by every sensor when the
client receives its data and the main loop is blocked until all the sensors
have received its data.
This suppose that all the sensors gather information at every tick. It this is
not the case, the clients needs to take in account at each frame how many
sensors are going to tick at each frame.
"""
import random
import glob
import os
import sys
from queue import Queue
from queue import Empty
import pickle
import carla
from config import *
import time



# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name))
    sensor_data.save_to_disk('_out_tick/%06d.png' % sensor_data.frame)


def main():
    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    f = open(TRAFFIC_LIGHT_INT_PATH,"ab")
    traffic_light_int = []

    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.017 
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # We create the sensor queue in which we keep track of the information
        # already received. This structure is thread safe and can be
        # accessed by all the sensors callback concurrently without problem.
        sensor_queue = Queue()

        # Bluepints for the sensors
        blueprint_library = world.get_blueprint_library()
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        if cam_bp.has_attribute('image_size_x') and cam_bp.has_attribute('image_size_y'):
            cam_bp.set_attribute('image_size_x',IMAGE_SIZE_X)
            cam_bp.set_attribute('image_size_y',IMAGE_SIZE_Y)
        vehicle_bp = blueprint_library.find("vehicle.audi.a2")

        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color',color)

        vehicle_transform = random.choice(world.get_map().get_spawn_points()) 
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        vehicle.set_autopilot(True)

        # We create all the sensors and keep them in a list for convenience.
        sensor_list = []

        cam01_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        cam01 = world.spawn_actor(cam_bp, cam01_transform,attach_to=vehicle)
        cam01.listen(lambda data: sensor_callback(data, sensor_queue, "camera01"))
        sensor_list.append(cam01)

        
        
        # Main loop
        ##while True:
        for _ in range(TICK_COUNT):
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:
                for __ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
                    print(f"vehicle.is_at_traffic_light():   {vehicle.is_at_traffic_light()}")
                    traffic_light_int.append(int(vehicle.is_at_traffic_light()))
            except Empty:
                print("    Some of the sensor information is missed")
        
        pickle.dump(traffic_light_int,f) 
        time.sleep(10)    #最後に保存する画像が欠けないように
    

    finally:
        world.apply_settings(original_settings)
        for sensor in sensor_list:
            sensor.destroy()
            print(f"destroyed {sensor.id}")
        vehicle.destroy()
        print("destroyed vehicle")
        f.close()
        print("closed file")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
