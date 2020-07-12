import bpy, os, sys
from pathlib import Path
from math import radians
#from tqdm import tqdm
import mathutils

root = sys.argv[-1]
timeout = 10
break_timeout = False
render_w = 1080
render_h = 1080


context = bpy.context
scene = bpy.context.scene

bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

objs = bpy.data.objects
objs.remove(objs["Cube"], do_unlink=True)

light_data = bpy.data.lights.new('sun', 'SUN')
light_obj = bpy.data.objects.new('sun', light_data)
bpy.context.collection.objects.link(light_obj)

# Create the camera
cam_data = bpy.data.cameras.new('camera')
cam = bpy.data.objects.new('camera', cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

scene.camera.location = mathutils.Vector((-1.0850913524627686, 1.3391202688217163, 0.8682854175567627))
scene.camera.rotation_euler = mathutils.Euler((1.0890867710113525, 7.418752829835285e-07, 3.82576060295105), 'XYZ')

scene.display.shading.background_type = 'WORLD'
scene.display.shading.background_color = (1.0, 1.0, 1.0)

for model in Path(root).rglob('*.obj'):
    
    timeout -= 1
    if break_timeout and timeout == 0:
        break

    home = model.parents[0]
    render_file = home / 'model_render.png'
    model = str(model)
    
    # make a new scene with cam and lights linked
    context.window.scene = scene
    bpy.ops.scene.new(type='LINK_COPY')
    
    #import model
    bpy.ops.import_scene.obj(filepath=model, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl")
                           
    print("Rendering ", model)
    context.scene.render.image_settings.file_format = 'PNG'
    context.scene.render.resolution_x = render_w
    context.scene.render.resolution_y = render_h
    context.scene.render.filepath = str(render_file)
    context.scene.render.threads = 8
    context.scene.world.light_settings.use_ambient_occlusion = True
    bpy.ops.render.render(write_still=True)

    for o in bpy.context.scene.objects:
        if o.type != 'CAMERA' and o.type != "LIGHT":
            o.select_set(True)
        else:
            o.select_set(False)

    bpy.ops.object.delete() 