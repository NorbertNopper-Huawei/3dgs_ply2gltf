# 3DGS ply2gltf

How to execute:  

`.\ply2gltf.exe some_3dgs.ply --convert --dump`

Generates two files: `some_3dgs.gltf` and `some_3dgs.bin`.

Using the optional `--convert` flag converts from right-handed z-up to right-handed y-up coordinate system by doing a -90 degree rotation, as defined and required in glTF.  
Otherwise it is assumed that the original data is already right-handed y-up.

Using the optional `--dump` flag is writing back the generated glTF binary buffer to the PLY file `some_3dgs_dump.ply`.

## 3DGS for glTF

- https://github.com/KhronosGroup/glTF/pull/2490

## Build

Please make sure, that the given dependencies are fulfilled and your toolchain does include the given headers from the libraries:

- Dependencies
    - C++20 [capable compiler](https://en.cppreference.com/w/cpp/compiler_support/20)
      - 3rd party libraries
        - [nlohmann JSON for Modern C++](https://github.com/nlohmann/json)
    - Build
      - [CMake](https://cmake.org/)
      - [Ninja](https://ninja-build.org/)

### How to build using MSYS2 and gcc on Windows

1. Install [MSYS2](https://www.msys2.org/) and setup the environment as described on the website.
2. Install packages for building:
  - `pacman -S cmake`
  - `pacman -S mingw-w64-x86_64-ninja`
3. Install libraries:
  - `pacman -S mingw-w64-x86_64-nlohmann-json`
4. Create `build` folder and navigate to this directory.
5. Run `cmake ..` to create the build files.
6. Run `ninja` to build the executable.

## Credits

 - Xin Zhao on the Spherical Harmonics rotation and overall debugging

## References

### Spherical Harmonics

- https://en.wikipedia.org/wiki/Spherical_harmonics

#### Rotating Spherical Harmonics

- https://github.com/andrewwillmott/sh-lib

### 3DGS

- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

#### 3DGS glTF extensions

- https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting

#### 3DGS PLY Format

- https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/

#### 3DGS PLY Viewers

- https://superspl.at/editor (Recommended, as bands and/or rotation can be changed)
  - https://playcanvas.com/viewer/
  - https://github.com/playcanvas/supersplat/issues/289
- https://www.3dgsviewers.com/
- https://antimatter15.com/splat/
