# This file tells Blender that this folder is a Python package
# and what to do when the addon is enabled.

bl_info = {
    "name": "Rogue SDF AI",
    "author": "Makan Asnasri",
    "version": (2, 4), # Version bump to be sure
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > Rogue_SDF_AI",
    "description": "High-performance SDF modeling with Geometry Nodes and a real-time shader preview.",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "3D View"
}

# This is the magic part. When Blender enables this addon,
# it will import everything from our "main.py" file.
from . import main

def register():
    main.register()

def unregister():
    main.unregister()