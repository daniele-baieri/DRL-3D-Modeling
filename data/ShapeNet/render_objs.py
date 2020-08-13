import bpy, os, sys
from pathlib import Path
from math import radians, pi
import mathutils

"""
Renders all .obj models found recursively in a root folder
as png images, using Blender's Cycles render engine.

By default, it sets a white background and a top-right camera.

Usage:
blender -b -P render_objs.py -- <path to a folder containing obj models>
"""

root = sys.argv[-1]
root = os.getcwd() + '/' + root
print("Rendering in: " + root)
timeout = 400
break_timeout = True
render_w = 512
render_h = 512
ignore_existing = False


bpy.context.scene.render.engine = 'CYCLES'
#bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 32
#bpy.context.scene.render.film_transparent = True
#bpy.context.scene.render.image_settings.color_mode = 'RGBA'

context = bpy.context
scene = bpy.context.scene

bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0.2, 0.2, 0.2, 1)

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

scene.camera.location = mathutils.Vector((-0.596133, 1.06438, 0.701563))
scene.camera.rotation_euler = mathutils.Euler((58.4 * pi / 180, 0, -151.6 * pi / 180), 'XYZ')

scene.display.shading.background_type = 'WORLD'
scene.display.shading.background_color = (0.5, 0.5, 0.5)


for model in Path(root).rglob('*.obj'):
    
    if break_timeout:
        if timeout == 0:
            break
        timeout -= 1   

    home = model.parents[0]
    render_file = home / 'model_render.png'
    if ignore_existing and os.path.exists(render_file):
        print("Skipping...")
        continue

    model = str(model)
    
    # make a new scene with cam and lights linked
    context.window.scene = scene
    bpy.ops.scene.new(type='LINK_COPY')
    
    #import model
    bpy.ops.import_scene.obj(filepath=model, axis_forward='-Z', axis_up='Y', filter_glob="*.obj")#;*.mtl")
                           
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