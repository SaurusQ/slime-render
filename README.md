
# Slime renderer

Slime renderer uses particles with simple behaviors to create complicated patters. The particles leave trails. Based on the picture generated the particles decide where to go on the next frame. The particles are calculated on CUDA kernels.

![](slime.png)

## Dependencies

To complile this program you need to:

- Install g++, make and nvcc
- Download Glad of your liking from glad.dav1d.de:
	- Lan: C/C++
	- Spec: OpenGL
	- Profile: Core 
	- gl: OpenGL version
	- Generate loader: yes
	- Extract it to lib/
- Install GLFW 3
	- On Ubuntu install libglfw3 and libglfw3-dev
- Install glm
	- On Ubuntu install libglm-dev

## Compile with configuration UI

- Install imgui
	- Clone imgui to lib/
	- git clone https://github.com/ocornut/imgui.git
- make gui

## Controls

| **Key** | **Action**                         |
|---------|-------------------------------------|
| ESC     | Close the program                   |
| C       | Run/Stop the simulation             |
| M       | Show configuration UI               |
| S       | Export the image to PNG             |
| F       | Show FPS                            |
| N       | Set next configuration from config file |
| O		  | Print out the current config to json |

The program takes a ``json`` file name as the first parameter, that configures all of the simulation parameters. You can see the structure of the file by printing the config out with ``O``. The ``json`` contains a list of configs that can be cycled with ``N``.