#/opt/carla-simulator/PythonAPI/examples/tutorial.pyの内容を写経してる
#所々変えているけどほとんど変わらない（予定）
import sys
import os
import subprocess
import carla
import random
import time

#carlaのPythonAPI/examplwsを参照している
try:
    sys.path.append('/opt/carla-simulator/PythonAPI/examples')
except:
    print("Can't load PythonAPI/example")

'''
以下はsys.pathにcarlaのpathが追加されたか確認できるコード
subprocess.callで、追加されたパス内のファイルを参照できる

print(sys.path)
exmaplesPath = ('ls %s'%(sys.path[-1])).split()
subprocess.call((exmaplesPath))
'''

def main():
    actor_list = []

    try:
        client = carla.Client('localhost',2000)
        client.set_timeout(2.0)

        world = client.get_world()
        blueprintLibrary = world.get_blueprint_library()

        #ブループリントからvhicleを一つ取り出している
        #ブループリントが持っているattributeは　https://carla.readthedocs.io/en/0.9.13/bp_library/ にある
        bp = random.choice(blueprintLibrary.filter('vehicle'))

        #取り出したbp(vehicle)の属性(arrtibute)でcolorであるものを読み込み、修正している
        #以下ではvehicleの色をランダムで決めている
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color',color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(world.get_map().get_spawn_points())

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # Let's add now a "depth" camera attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.
        camera_bp = blueprintLibrary.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)

        # Now we register the function that will be called each time the sensor
        # receives an image. In this example we are saving the image to disk
        # converting the pixels to gray-scale.

        '''
        cc = carla.ColorConverter.LogarithmicDepth
        camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame, cc))
        '''
        camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))
    

        # Oh wait, I don't like the location we gave to the vehicle, I'm going
        # to move it a bit forward.
        location = vehicle.get_location()
        location.x += 40
        vehicle.set_location(location)
        print('moved vehicle to %s' % location)

        # But the city now is probably quite empty, let's add a few more
        # vehicles.
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, 10):
            transform.location.x += 8.0

            bp = random.choice(blueprintLibrary.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)

        time.sleep(30)

    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')



if __name__ == '__main__':
    main()