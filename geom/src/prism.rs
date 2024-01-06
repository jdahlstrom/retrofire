pub struct Prism(pub Vec<Vec2>);

impl Prism {
    pub fn build(self) -> Mesh<Normal3> {
        let pts = self.0;

        //
        // Sides
        //

        let mut verts = vec![];

        let len = pts.len();
        for i in 0..len {
            let prev = (i.wrapping_sub(1)) % len;
            let next = (i + 1) % len;

            let p = pts[prev];
            let v = pts[i];
            let n = pts[next];

            let d1 = v - p;
            let d2 = n - v;

            let d = d1 + d2;

            let n = vec3(d.y(), 0.0, -d.x()).normalize();

            verts.push(vertex(vec3(v.x(), -0.5, v.y()).to(), n));
            verts.push(vertex(vec3(v.x(), 0.5, v.y()).to(), n));
        }

        let mut faces = vec![];

        let l = verts.len();
        for i in (3..l).step_by(2) {
            faces.push(Tri([i, i - 1, i - 3]));
            faces.push(Tri([i, i - 2, i - 3]));
        }
        faces.push(Tri([1, 0, l - 2]));
        faces.push(Tri([1, l - 1, l - 2]));

        //
        // Bases
        //

        eprint!("Tessellating {}-gon...", len);
        let tris = Self::tess(&pts);
        eprintln!("done, {} triangles in total", tris.len());

        for Tri(vs) in &tris {
            let i = verts.len();
            for v in vs {
                verts.push(vertex(
                    vec3(v.x(), -0.5, v.y()).to(),
                    vec3(0.0, -1.0, 0.0).to(),
                ));
            }
            faces.push(Tri([i, i + 1, i + 2]));

            let i = verts.len();
            for v in vs {
                verts.push(vertex(
                    vec3(v.x(), 0.5, v.y()).to(),
                    vec3(0.0, 1.0, 0.0).to(),
                ));
            }
            faces.push(Tri([i, i + 1, i + 2]));
        }

        Mesh { faces, verts }
    }
}
