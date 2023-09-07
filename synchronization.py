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
import numpy as np



# Sensor callback.
# This is where you receive the sensor data and
# process it as you liked and the important part is that,
# at the end, it should include an element into the sensor queue.
def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name))
    sensor_data.save_to_disk('%s/%-s:%06d.png' % (IMAGE_PATH,sensor_name,sensor_data.frame))


def main():
    # We start creating the client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    world_map = world.get_map()
    tm = client.get_trafficmanager(8000)

    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(IMAGE_PATH, exist_ok=True)
    label_file = open(LABEL_PATH,"wb")
    label_dic = {"traffic_light":np.array([]),"intersection":np.array([])}

    try:
        # We need to save the settings to be able to recover them at the end
        # of the script to leave the server in the same state that we found it.
        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.017 
        settings.synchronous_mode = True
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

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

        vehicle_bp = blueprint_library.find("vehicle.audi.tt")
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color',color)

        vehicle_transform = random.choice(world_map.get_spawn_points()) 
        vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)
        tm.vehicle_percentage_speed_difference(vehicle, -300.)

        # We create all the sensors and keep them in a list for convenience.
        sensor_list = []

        front_cam_transform = carla.Transform(carla.Location(x=2.0, z=2.0))
        left_cam_transform = carla.Transform(carla.Location(x=2.0, y=-1.0, z=2.0), carla.Rotation(yaw=-80))
        right_cam_transform = carla.Transform(carla.Location(x=2.0, y=1.0, z=2.0), carla.Rotation(yaw=80))

        front_cam = world.spawn_actor(cam_bp, front_cam_transform,attach_to=vehicle)
        front_cam.listen(lambda data:sensor_callback(data, sensor_queue, "front_cam"))
        sensor_list.append(front_cam)

        left_cam = world.spawn_actor(cam_bp, left_cam_transform, attach_to=vehicle)
        left_cam.listen(lambda data:sensor_callback(data, sensor_queue, "left_cam"))
        sensor_list.append(left_cam)

        right_cam = world.spawn_actor(cam_bp, right_cam_transform, attach_to=vehicle)
        right_cam.listen(lambda data:sensor_callback(data, sensor_queue, "right_cam"))
        sensor_list.append(right_cam)

        
        
        # Main loop
        ##while True:
        for _ in range(TICK_COUNT):
            # Tick the server
            world.tick()
            vehicle.set_autopilot(True) #tickの後じゃないとオートパイロットが動かない
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            #for文ですべてのセンサーのデータをキューから取り出している
            #一つでもセンサーのデータが欠けている場合、キューとリストの長さが合わなくなり（len(sensor_queue)<len(sensor_list))
            #エラーが起きる
            try:
                for __ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
                print(f"vehicle.is_at_traffic_light():   {vehicle.is_at_traffic_light()}")
                label_dic["traffic_light"] = np.append(label_dic["traffic_light"],int(vehicle.is_at_traffic_light()))   
                label_dic["intersection"] = np.append(label_dic["intersection"],int(world_map.get_waypoint(vehicle.get_transform().location).is_junction))
            except Empty:
                print("    Some of the sensor information is missed")
        
        pickle.dump(label_dic, label_file) 

        time.sleep(10)    #最後に保存する画像が欠けないように
        for sensor in sensor_list:
            sensor.stop()
    

    finally:
        world.apply_settings(original_settings)
        for sensor in sensor_list:
            sensor.destroy()
            print(f"destroyed {sensor.id}")
        vehicle.destroy()
        print("destroyed vehicle")
        label_file.close()
        print("closed file")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
