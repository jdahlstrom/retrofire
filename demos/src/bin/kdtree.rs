use core::ops::ControlFlow::*;
use minifb::{MouseButton, MouseMode};
use re::core::geom::kd::KdTree;
use re::core::math::rand::{DefaultRng, Distrib};
use re::front::minifb::Window;
use re::prelude::*;
use std::time::Instant;

fn main() {
    let mut win = Window::builder()
        .title("retrofire//kdtree")
        .build()
        .expect("should create window");

    // Disable backface culling and depth buffering
    win.ctx = Context {
        face_cull: None,
        //color_clear: None,
        depth_test: None,
        depth_clear: None,
        depth_write: false,
        ..win.ctx
    };

    let bounds = (
        pt2::<_, ()>(10.0, 10.0),
        pt2(win.dims.0 as f32 - 10.0, win.dims.1 as f32 - 10.0),
    );
    let mut tree = KdTree::with_bounds(Vec::with_capacity(1024), bounds);

    let mut mouse_down = false;
    let mut rng = DefaultRng::default();
    win.run(|frame| {
        let rg = (bounds.0..bounds.1);
        tree.points.extend(rg.samples(&mut rng).take(100));

        let start = Instant::now();
        tree.rebuild();
        eprintln!(
            "Tree built in {} usec, points={}, nodes={}",
            start.elapsed().as_micros(),
            tree.points.len(),
            tree.nodes.len()
        );

        let (mx, my) = frame
            .win
            .imp
            .get_mouse_pos(MouseMode::Clamp)
            .unwrap();

        if !mouse_down && frame.win.imp.get_mouse_down(MouseButton::Left) {
            mouse_down = true;

            tree.points.push(pt2(mx, my));

            let start = Instant::now();
            tree.rebuild();
            let us = start.elapsed().as_micros();

            eprintln!(
                "Tree built in {us} usec, points={}, nodes={}",
                tree.points.len(),
                tree.nodes.len()
            );
            /*for n in &tree.nodes {
                eprint!("{} {:pad$}", n.depth, ' ', pad = n.depth as usize * 4);
                if let Some(inner) = n.split {
                    eprintln!(
                        "inner: range={:?}, n={}, bounds={:?}, split={},{:.0}, children={:?}",
                        n.range,
                        n.range.1 - n.range.0,
                        (n.bounds.0.0, n.bounds.1.0),
                        if inner.split_axis == 0 { 'x' } else { 'y' },
                        inner.split_pos,

                        inner.children,
                    );
                } else {
                    eprintln!(
                        "leaf: range={:?}, n={}, bounds={:?}",
                        n.range,
                        n.range.1 - n.range.0,
                        (n.bounds.0.0, n.bounds.1.0),
                    );
                }
            }*/
        } else if !frame.win.imp.get_mouse_down(MouseButton::Left) {
            mouse_down = false;
        }

        let buf = &mut frame.buf.borrow_mut().color_buf.buf;

        // for node in &tree.nodes {
        //     if let Some(inner) = node.split {
        //         if inner.split_axis == 0 {
        //             for y in node.bounds.0.y() as u32..node.bounds.1.y() as u32
        //             {
        //                 buf[[inner.split_pos as u32, y]] = 0x66_66_66_66;
        //             }
        //         } else {
        //             for x in node.bounds.0.x() as u32..node.bounds.1.x() as u32
        //             {
        //                 buf[[x, inner.split_pos as u32]] = 0x66_66_66_66;
        //             }
        //         }
        //     }
        //
        //     if node.bounds.0.x() <= mx
        //         && mx < node.bounds.1.x()
        //         && node.bounds.0.y() <= my
        //         && my < node.bounds.1.y()
        //     {
        //         for x in node.bounds.0.x() as u32..node.bounds.1.x() as u32 {
        //             for y in node.bounds.0.y() as u32..node.bounds.1.y() as u32
        //             {
        //                 buf[[x, y]] += 0x11_11_11_11;
        //             }
        //         }
        //     }
        // }

        for p in &tree.points {
            let x = p.x() as u32;
            let y = p.y() as u32;
            let c = 0x33_33_33_33;
            buf[[x, y]] += c;
            buf[[x + 1, y]] += c;
            buf[[x, y + 1]] += c;
            buf[[x + 1, y + 1]] += c;
        }

        if let Some(n) = tree.nearest(pt2(mx, my)) {
            //eprintln!("Nearest={:?}", n.0);
            buf[[n.x() as u32, n.y() as u32]] = 0xFF_FF_00_00;
            buf[[n.x() as u32 + 2, n.y() as u32]] = 0xFF_FF_00_00;
            buf[[n.x() as u32 - 2, n.y() as u32]] = 0xFF_FF_00_00;
            buf[[n.x() as u32, n.y() as u32 + 2]] = 0xFF_FF_00_00;
            buf[[n.x() as u32, n.y() as u32 - 2]] = 0xFF_FF_00_00;
        }

        Continue(())
    });
}
