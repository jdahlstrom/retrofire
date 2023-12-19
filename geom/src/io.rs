use core::str::from_utf8;

use re::geom::{vertex, Mesh, Tri};
use re::math::space::Real;
use re::math::{vec3, Vec3};
use re::render::tex::{uv, TexCoord};
use re::render::Model;

pub fn read_obj(input: impl IntoIterator<Item = u8>) -> Mesh<()> {
    let mut i = input.into_iter().peekable();

    let mut faces = vec![];
    let mut verts = vec![];
    let mut norms = vec![];
    let mut texcs = vec![];

    while i.peek().is_some() {
        let line: Vec<_> = (&mut i).take_while(|&c| c != b'\n').collect();

        if line.is_empty() || line[0] == b'#' {
            continue;
        }

        let tokens: Vec<&[u8]> = line.split(u8::is_ascii_whitespace).collect();

        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            b"v" => verts.push(parse_vertex(&tokens[1..])),
            b"vn" => norms.push(parse_normal(&tokens[1..])),
            b"vt" => texcs.push(parse_texcoord(&tokens[1..])),

            b"f" => faces.push(parse_face(&tokens[1..])),

            _ => todo!(),
        }
    }

    Mesh {
        faces,
        verts: verts
            .into_iter()
            .map(|pos| vertex(pos, ()))
            .collect(),
    }
}

fn parse_face(f: &[&[u8]]) -> Tri<usize> {
    let a = parse_index(f[0]);
    let b = parse_index(f[1]);
    let c = parse_index(f[2]);
    Tri([a, b, c])
}

fn parse_texcoord(c: &[&[u8]]) -> TexCoord {
    let u: f32 = parse_float(c[0]);
    let v: f32 = parse_float(c[1]);
    uv(u, v)
}

fn parse_normal(n: &[&[u8]]) -> Vec3 {
    parse_vertex(n).to()
}

fn parse_vertex(v: &[&[u8]]) -> Vec3<Real<3, Model>> {
    let x: f32 = parse_float(v[0]);
    let y: f32 = parse_float(v[1]);
    let z: f32 = parse_float(v[1]);
    vec3(x, y, z).to()
}

fn parse_float(s: &[u8]) -> f32 {
    from_utf8(s).unwrap().parse().unwrap()
}

fn parse_index(s: &[u8]) -> usize {
    from_utf8(s).unwrap().parse().unwrap()
}

#[cfg(test)]
mod tests {
    use crate::io::read_obj;

    #[test]
    fn test() {
        let input = br"
# comment
f 0 1 3
  f 0 2 3
#anothercomment
v 0.0 0.0 0.0
v 1.0 0.0 0.0

v 0.0 1.0 0.0
v 1.0 1.0 0.0";

        dbg!(read_obj((*input).into_iter()));
    }
}
