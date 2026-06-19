```
                                                    _______
                        ___                       /´  ____/\
      __ ______ _____ /   /\_ _ ______ _____ ____/   /_/___/\ __ _____ ______
   ==/  ´ ____/ __   \   ____/ ´ ____/  __  ` __    ___,  /==/  ´  ___/ __   \
  ==/   /´=/   ______/  /==/   /´=/   /==/   /=/   /=/   /==/   /´=/   ______/\
 ==/   /==/   /____/   /__/   /==/   /__/   /=/   /=/   /__/   /==/   /______\/
==/___/ ==\_______/\______/__/ ==\________,´_/   /==\______/__/ ==\________/\
==\___\/ ==\______\/\_____\__\/ ==\______/_____,´ /==\_____\___\/==\_______\/
                                         \_____\,´
```

# Retrofire-front

Simple frontends for [`retrofire`][1], managing window creation and basic event
handling.

[1]: https://crates.io/crates/retrofire

## Crate features

* `minifb`: Enables a frontend using the [`minifb`][2] library.
* `sdl2`:   Enables a frontend using the [`sdl2`][3] library.
* `wasm`:   Enables a frontend using WebAssembly and [`wasm-bindgen`][4].
* `stats`:  Enables collection and display of performance statistics.

All features are disabled by default.

[2]: https://crates.io/crates/minifb

[3]: https://crates.io/crates/sdl2

[4]: https://crates.io/crates/wasm-bindgen

## Example

```rust
use core::ops::ControlFlow::*;

use re::core::render::{Model, render, shader};
use re::front::minifb::{Frame, Window};
use re::prelude::*;

fn main() {
    let mut win = Window::builder()
        .title("retrofire//example")
        .build()
        .expect("should create window");

    // Initialize
    let triangle = [
        vertex(pt3(-1.0, 0.6, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(1.0, 0.6, 0.0), rgb(0.0, 0.8, 0.0)),
        vertex(pt3(0.0, -1.2, 0.0), rgb(0.4, 0.4, 1.0)),
    ];
    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &ProjMat3<Model>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f<_>>, _: &_| frag.var.to_color4(),
    );
    let Dims(w, h) = win.dims;
    let to_view = translate((0.0, 0.0, 2.0));
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    // Run the main loop
    win.run(|frame: &mut Frame| {
        // Handle events
        // Automatically quits if window is closed or ESC is pressed

        // Update state
        let to_world = rotate_z(rads(frame.t.as_secs_f32()));

        // Render
        render(
            [tri(0, 1, 2)],
            triangle,
            &shader,
            &to_world.then(&to_view).to().then(&project),
            viewport,
            &mut frame.buf,
            frame.ctx,
        );

        // Returning Break(()) quits the application
        Continue(())
    });
}
```

## License

Copyright 2020-2025 Johannes Dahlström.

Retrofire is licensed under either of:

* Apache License, Version 2.0
  (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)

* MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as defined in
the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
