```
                                                    _______
                       ____                       /´  ____/\
      __ ______ _____ /   /\_ _ ______ _____ ____/   /_/___/\ __ _____ ______
   ==/  ´ ____/ __   \   ____/ ´ ____/  __  ` __    ___,  /==/  ´  ___/ __   \
  ==/   /´=/   ______/  /==/   /´=/   /==/   /=/   /=/   /==/   /´=/   ______/\
 ==/   /==/   /____/   /__/   /==/   /__/   /=/   /=/   /__/   /==/   /______\/
==/___/ ==\_______/\______/__/ ==\________,´_/   /==\______/__/ ==\________/\
==\___\/ ==\______\/\_____\__\/ ==\______/_____,´ /==\_____\___\/==\_______\/
                                         \_____\,´
```

# Retrofire-core

Core functionality of the `retrofire` project.

Includes a math library with strongly typed points, vectors, matrices,
colors, and angles; basic geometry primitives; a software 3D renderer with
customizable shaders; with more to come.

## Crate features

* `std`:
  Makes available items requiring I/O, timekeeping, or any floating-point
  functions not included in `core`. In particular this means trigonometric
  and transcendental functions. Enabled by default. If this feature is
  disabled, the crate only depends on `alloc`.

* `libm`:
  Provides software implementations of floating-point functions via the
  [`libm`](https://crates.io/crates/libm) crate.

* `mm`:
  Provides fast approximate implementations of floating-point functions
  via the [`micromath`](https://crates.io/crates/micromath) crate.

One of the above features must be enabled in order to enable APIs using
trigonometric or transcendental functions.

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
