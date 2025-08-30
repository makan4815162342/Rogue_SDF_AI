# Rogue_SDF_AI
<img width="1024" height="1024" alt="20250725_2043_Holographic Futuristic Warrior_remix_01k11947psesytv6q3ne8bbq3t" src="https://github.com/user-attachments/assets/776b8188-b054-4346-bb53-2b387579ac19" />
An SDF Add-On for Blender. Rogue SDF AI is an Add-On initially made to improve SDF Prototyer Add-On but as I worked on it, I created its own SDF Renderer and features, but I have decided to keep both SDF Prototyper Features and add my own new features and new SDF Render Shaders on top of it. I don't know where to begin TBH I added a lot of features to this Add-On, in short, this Add-On is in two sections:

The 3D Geometry with SDF behavior, drastically improved. 

The SDF Shader View, with pure SDF Modeling and tools still (WIP) that I keep add and improve it.

Please try it and let me know about the bugs and help me to improve it.

New Update:

Added the ability to see 3D Geometries outside of the Domain in SDF Shader View to work on the model.

New Update:

Added Performance Control for Ray Marching two slider quality steps and performance scale to make it run faster.

Added Remesh with Voxel and QuadriFlow as a new button for better Retopology.

Added Groove and Pipe as new Blending Types.

Canceled: Added Texture Buffer Object (TBO) support to the shader that enables the Add-On to Add 1024 shapes and even more., paused.


New Update:

Improved and added lots of features, my next attempt will be making a better export with C++ and an SDF Render Engine with C++:

New Changes: Domain Scale, Shape Scale, duplicate and Repeat, Mirror, Radial Symmetry inherit their blending, color, smoothing.

Intersection is not working that way, only with one object.

Added Symmetrize One side to the other.

Added Flip for select or all shapes.

Fixed the Sphere nonuniformly scale problem.

Fixed the Curve and Cone, the Curve now have better controls and profiles.

Greatly improved the Curve Shape, pointy head and tails, scale and radius, smooth and hard shapes...

Fixing Symmetrized, Repeat, duplicate for the Curve and bake symmetry for the Curve.

Curve Duplicate is fixed.

Fixed Repeat for Curve.

Improved the SDF Shader View, it only shows withing the Domain Border and it increased the speed by 10X if zoomed out.

Improved the Symmetrize Model, if you pass the center, it won't crash and it supports Curve and Cylinder Z symmetrize now works.

Enabled Points as mesh view to increase the speed for the 3D Geometry section, don't know if it can speed that much but helps.

Added Roundness for the Cube for SDF Shader View.

Added Thickness for the Cube for SDF Shader View.

Added Bevel for the Cube for SDF Shader View.

Added Pyramid effect for the Cube for SDF Shader View.

Added Twist effect for the Cube for SDF Shader View.

Added Bend effect for the Cube for SDF Shader View.

Added these features to Cylinder too.

Added Capsule to the Sphere, Thickness, Pac Man.

Added Cylinder Effect to the Prism, plus made it Ngon.

Fixed the Symmetry and Flipped as much as I could.

Added the Features to the Torus too, Thickness, Inner and Outer Radius, and Pac Man Cut 360.

Added More shapes with their features to the Curve.

Added Space and Local Orientation to the Curve.

Added Elongation to the Torus and its Curve feature.

Added Custom Points with all of the features, position, radius, local rotation and shape features for the Curve.

Added Paint Blending mode as color for all of the shapes.

Added a better Mesh Export PyMCubes that give us better results.

Added Color Export as Vertex Color from SDF.

Added new Blending modes with new Strength slider Displace, Indent, Relief, Engrave, Tunnel.

Added the Mask Blending Mode.

Added Chamfer Blending type.

Added more selection methods for better exporting.

Added third method of exporting VDB with Color.

Improved the Light Preview.

Added Support for Mac and Linux too.

Added Exporting UVs with Texture Map from Vertex Color.

Added Support for Splats with particles that works as path in hair system too or custom shape.

Added support for baking from a Remeshed object.

Added Radial for Repeat.

The Previous Updates:

Let me explains some of its new features, improvements: The 3D Geometry Section: 1-Scale: I added unform Scale to control the scale of all of the shapes inside the Domain. 2-Automatic Preview: I added a Dynamic Resolution that when you move the SDF Shapes they go to lower resolution to work faster and when stopped they go back to higher resolution, you can control the delay and sensitivity. 3-Manual Preview: You can manually control these resolutions yourself. 4-Decimation: I made a complex Decimation Node from YouTube Tutorials and linked it to the main file; you can control the resolution of the models while modeling to control the resolution better. 5-Toggle Overlays: you can toggle it on and off to see the 3D Models more clearly. 6-The SDF List: I added Mute Shapes, Hide Empties, highlight for the SDF Shapes, Color indicator for select shape (Red) in the SDF List, I fixed and improved the Duplicate, move the shape up and down, delete, clear all and I also added Repeat to create a repeat for the shapes. 7-Enable SDF Shader View: it will take to the SDF Section I will talk about it below soon. 8-The SDF Shapes Settings for 3D Geometries: Most of this section is from the Legacy section of the Add-On I sort them out better and added some new features like Flip Shape along X,Y,Z you can flip your shapes with it. 9-Symmetry Section: it's for SDF Section I will talk about it Below soon. 10-Convert to Mesh: you can Convert your mesh, what you see is what you get. 11-Bake to High Quality Mesh: I will talk about it soon since it's for SDF Section Below. 12-Bake Active Symmetry: it's related to SDF Section, I will talk about it below. 13-Brush-Cube Clipping: This Brush let you clip the shapes or let's say crop them, it can be used as a way of modeling or just similar to Zbrush you temporary hide the rest work on that section then delete the brush to see the rest of the model, it helps for faster modeling. 14-Show Render Options: This is a very helpful part of the features I added for the 3D Geometry to be able to see a high res version of your model in render mode fast, you can select to render from either Camera or Viewport, I recommend the Viewport, you can select it to be Eevee or Cycle, it can be render from the Default shading, Material shading or the Rendered Shading, you can set the resolution up to 5, scale of the image samples for the renderer and you can disable the overlays to see with our without wireframes and such and finally you can render from the viewport or final render for camera. 15-Renameing Option: I also forgot to mention I added a renaming option to avoid problems and better cleaning, I want to add a auto renaming system soon.

The SDF Section: You have to first click the SDF Shader View button, you will be sent to a SDF Preview Render. 1-Max Samples: Front the top this is for the curve mostly, it will help you to have more detailed Curves. 2-Highlight Shape: It will show you the shape in the viewport you selected to help you find it, I will add the subtract version of it too. 3-Preview Light: It will help you to change the lighting for the SDF Shader View, I will improve it more to have more highlight, contrast, and better navigation. 4-Sahpe Settings: 1-it has a renaming option, 2-Shader Operations: I added a Global Tint to change the color of the whole objects, 3-Operations: I added Union with Blend, Subtraction and Intersect so far all of them blend colors hard of soft (I will talk about it more below) Shape Features will be added soon but for now we only can Blend, 4-Color: we can change the color of the shapes, 5-Color Blend: we have hard or soft color that we can blend between the objects, 6-Symmetry: Mirror X,Y,Z and Radial Symmetry, I added these cool features for better modeling. 5-Finlize Mesh: 1-We already talked about the Convert to Mesh, 2-Bake to High Quality Mesh: this section let you export your SDF Shapes to Geometry, it has quite cool features in it, the resolution section should be less than 512 and the scale should be 2, you need to locate and name your Volume Mesh to be saved and it will automatically bring it to the Add-On and give your the Geometry and the rest are optimization and improvement that you can try yourself to get a good shape. 6-Bake Active Symmetry: this button will bake the mirrored or radial symmetry object to actual SDF Shapes to be able to export them. 7-I added a brand-new Curve for the SDF Shapes, remade the Cone, and the rest of the shapes will be improved they will have more features and maybe add more shapes even.

That's it for overviewing the buttons for everything I added so far! there are a lot to be added, and I will keep updating it, please let me know about your thoughts and try it, I will fix the Mac problem, I will add tooltips, and support for Vulkan... Be sure to check out my other Add-Ons as well and join my Discord for support and follow me on X. Here some Screenshot from the Add-On:

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 30 - 06 08 54 55" src="https://github.com/user-attachments/assets/c533dd8e-ee73-41bb-89bb-9ec9116a8448" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 27 - 05 37 50 13" src="https://github.com/user-attachments/assets/6786924c-a048-483d-8cc0-abf495ec241e" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 29 - 03 43 20 36" src="https://github.com/user-attachments/assets/d5e3b8bb-a941-4ecb-9195-b3250e4a18a3" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 29 - 03 43 00 29" src="https://github.com/user-attachments/assets/77f569d9-c631-4448-bbf3-5aba21a222da" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 29 - 03 41 51 74" src="https://github.com/user-attachments/assets/7d36904a-6c6a-4a79-96a7-17cb3e9bc8dc" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 29 - 03 42 17 89" src="https://github.com/user-attachments/assets/a848bda4-0c6c-42fe-a9be-78688f6a74c1" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 26 - 09 35 17 97" src="https://github.com/user-attachments/assets/b0686aa0-c26d-4a4d-ac13-633f19da4d11" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 25 - 04 57 34 59" src="https://github.com/user-attachments/assets/37f34741-8b62-45a9-8526-b765d9132a78" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 23 - 02 28 53 89" src="https://github.com/user-attachments/assets/bb03a8b7-f81e-4985-a16d-1375592b6817" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 23 - 04 01 34 47" src="https://github.com/user-attachments/assets/984b8863-9373-409c-973b-50c1c92fdb79" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 22 - 08 26 20 73" src="https://github.com/user-attachments/assets/0d211187-0b5d-41a8-9e54-f5667842065a" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 20 - 00 33 31 50" src="https://github.com/user-attachments/assets/c89a35b2-d5c9-4509-8651-f74f5d21ce9b" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 20 - 02 12 36 68" src="https://github.com/user-attachments/assets/0c556d60-86a8-4de4-aad2-0ccf3d9fadb1" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 21 - 03 11 20 53" src="https://github.com/user-attachments/assets/5598cd32-b019-425a-b6ea-97a01bc8b055" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 15 - 19 57 31 17" src="https://github.com/user-attachments/assets/904a8293-0c9e-4e64-a140-7fa977471f4a" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 08 07 - 20 14 52 38" src="https://github.com/user-attachments/assets/4e5ba8bb-778d-4305-8ea1-08f6c094dfa9" />

<img width="2560" height="1080" alt="Custom Spline Chain_3" src="https://github.com/user-attachments/assets/e634d95c-32ea-4aa2-af7d-845a9cae5cfc" />

<img width="2560" height="1080" alt="Rogue_SDF_AI_Progress" src="https://github.com/user-attachments/assets/3692a093-d5a8-4c04-b00a-f6a97a9a2253" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 19 - 05 37 55 50" src="https://github.com/user-attachments/assets/b9058ce7-043d-4220-82a0-a4175a1080bf" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 21 - 01 56 35 07" src="https://github.com/user-attachments/assets/7046897c-78e3-48d7-a9b1-a89a23627702" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 21 - 05 34 48 53" src="https://github.com/user-attachments/assets/da49d10f-fb78-4a56-af1d-c61d3250849b" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 22 - 03 47 33 46" src="https://github.com/user-attachments/assets/5a8894cb-3a81-4636-8308-a2f4fe57ac1f" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 22 - 05 14 24 32" src="https://github.com/user-attachments/assets/27edfc4a-b662-45ad-9b0e-3954abf33488" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 22 - 06 18 33 67" src="https://github.com/user-attachments/assets/0d8d6a56-8070-401a-8f82-5161291ba93b" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 22 - 06 23 11 90" src="https://github.com/user-attachments/assets/c4387f6e-d536-4e64-8930-cce6bcdb93c4" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 23 - 08 34 35 71" src="https://github.com/user-attachments/assets/5982c71b-2d0a-48e5-9547-e8932797a86f" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 23 - 09 16 18 79" src="https://github.com/user-attachments/assets/72f81136-9d77-4525-9173-db93a909a144" />

<img width="2560" height="1080" alt="Desktop Screenshot 2025 07 25 - 08 27 18 99" src="https://github.com/user-attachments/assets/0f774d4c-5a70-44e3-adc6-69cd6fcb1144" />


