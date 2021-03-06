

                                                     _______
                        ____                       /´  ____/_
       __ ______ ____ _/   /__ _ ______ _____ ____/   /_/___/  __ _____ _____
    ==/  ´ ____/ __   \   ____/ ´ ____/  __  ` __    ___.  /==/  ´  ___/ __   \
   ==/   /´=/   ______/  /==/   /´=/   /==/   /=/   /=/   /==/   /´=/   ______/
  ==/   /==/   /____/   /__/   /==/   /__/   /=/   /=/   /__/   /==/   /_____
 ==/___/===\_______/\______,__/===\_________/_/   /==\______/__/===\________/
                                          /_____,´


/////////////////////////// I N T R O D U C T I O N ///////////////////////////

retrofire  is a project that aims to implement various graphics effects popular
in 90s games and demoscene productions, before the  proliferation  of hardware-
accelerated 3D graphics.  It is written  with  performance in mind  in  modern,
idiomatic Rust.  A specific goal of retrofire is to be usably fast on legacy or
low-power hardware such as Raspberry PIs and outdated PCs.

retrofire is completely platform-agnostic and, as of now, has zero non-optional
third-party dependencies.  Render output is via callbacks or into "naked"  `u8`
buffers.  The project  comes with  examples  that include  interactive programs
targetting ncurses and SDL 2.  A future goal is to add an HTML5 / WASM example.


/////////////////////////////// M O D U L E S /////////////////////////////////

retrofire is composed of the following submodules:

lib
|
'== geom ............................. 3D geometry representation and utilities
|   |-- mesh ....................... Triangle mesh with vertex and face attribs
|   |-- solids .................. Mesh representations of various solid objects
|   '-- bbox ................................... Axis-aligned bounding box type
|
'== math ........................... Utilities such as lerp and approx equality
|   |-- random .............. Pseudo-random number generation and distributions
|   |-- vec .................... 4D vectors and related functions and operators
|   |-- mat .................. 4x4 matrices and related functions and operators
|   '-- transform .............. Functions for creating transformation matrices
|
'== render ............................... Self-contained 3D rendering pipeline
|   |-- hsr ......................... Backface and frustum culling and clipping
|   |-- raster ............................ Scanline rasterization of triangles
|   |-- shade ........................ Shading algorithms and related functions
|   |-- stats ....................... Collecting and printing render statistics
|   '-- vary ............................... Interpolation of vertex attributes
|
'== util ................................... Memory buffers and other utilities
    |-- color ................................ An RGBA color type and utilities
    '-- io .........................................Image file input and output

examples
|
'== curses .......................... Using ncurses to render into the terminal
|   |-- triangle ........................ A triangle bouncing around the screen
|   '-- torus ..................... A rotating torus with a psychedelic shading
|
'== sdl ............................... Using SDL 2 to render into a GUI window
    '-- triangle .................... A Gouraud shaded triangle bouncing around
    '-- teapot ....................... A classic Utah teapot with Phong shading
    '-- checkers ........................ A checkered plane and textured crates


//////////////////  R U N N I N G  T H E  E X A M P L E S  ////////////////////

// ncurses examples

  $ cargo run --release --features=ncurses --example=cursed-tri

A flat-shaded triangle bouncing around the terminal window. Use WASD to control
the speed of the selected vertex. Cycle selected vertex with space.

  $ cargo run --release --features=ncurses --example=cursed-torus

A rotating torus with a psychedelic shifting pattern created with trigonometric
functions. Lovingly Gouraud-shaded with ASCII art characters.

// SDL examples

  $ cargo run --release --features=sdl --example=sdl-tri

A color-shifting, Gouraud-shaded triangle bouncing around the window.

  $ cargo run --release --features="sdl teapot" --example=teapot

A  Phong  shaded, exposure-simulated  rendering  of  the  famous  Utah  teapot.
WASD to move the teapot; left and right arrow keys to rotate the camera.

  $ cargo run --release --features=sdl --example=checkers

Textured crates lying on a checkered floor. Arrow keys and WASD to move around.


/////////////////////////////// L I C E N S E /////////////////////////////////

(C)opyright  2020  Johannes  Dahlström.  retrofire is licensed under  either of

-- Apache License, Version 2.0
   (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)

-- MIT license
   (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

// Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the  Apache-2.0  license, shall
be dual licensed as above, without any additional terms or conditions.