//! Reading and writing Wavefront OBJ files.
//!
//! OBJ is a simple text-based format for storing 3D objects.
//!
//! # Examples
//! ```plain
//! ## Empty lines and lines starting with a hash are ignored.
//!
//! ## A vertex position is defined with 'v' followed by three floats:
//! v 1.23 9.87 -5.67
//! ## Floats can use exponent notation:
//! v 1.23e3 9.87e-1 -5.67e002
//!
//! ## A vertex normal is specified with 'vn' followed by three floats:
//! vn 0.7 0.1 -0.7
//! ## Vertex normals aren't necessary unit length:
//! vn 1.0 -1.0 0.5
//!
//! ## A texture coordinate is specified with 'vt' followed by two floats:
//! vt 0.12 0.56
//!
//! ## Faces are defined with 'f' followed by *one-based* vertex indices.
//! ## In the simplest case vertices only have positions:
//! f 1 2 3
//!
//! ## If vertices have extra attributes, indices are separated with a slash,
//! ## in the order position/texcoord/normal:
//! f 1/5/7 2/4/5 3/5/8
//!
//! ## Texture coordinates can be specified without normals:
//! f 1/5 2/4 3/5
//!
//! ## Normals can be specified without texture coordinates:
//! f 1//7 2//4 3//8
//!
//! ## Faces can have more than three indices:
//! f 1 2 3 4 5
//! ```

use alloc::{vec, vec::Vec};
use core::fmt;
use core::num::{ParseFloatError, ParseIntError};
use core::str::{from_utf8, FromStr, Utf8Error};
#[cfg(feature = "std")]
use std::{
    error,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use re::geom::{vertex, Mesh, Tri};
use re::math::space::Real;
use re::math::{vec3, Vec3, Vector};
use re::render::tex::{uv, TexCoord};
use re::render::Model;
use Error::*;

#[derive(Debug)]
pub enum Error {
    #[cfg(feature = "std")]
    Io(std::io::Error),
    UnsupportedItem,
    UnexpectedEnd,
    InvalidValue,
    OutOfBounds(&'static str, usize),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Indices {
    pos: usize,
    uv: Option<usize>,
    n: Option<usize>,
}

type Result<T> = core::result::Result<T, Error>;

/// Loads a OBJ model from a path into a mesh.
///
/// # Errors
/// TODO
#[cfg(feature = "std")]
pub fn load_obj(path: impl AsRef<Path>) -> Result<Mesh<()>> {
    let r = &mut BufReader::new(File::open(path)?);
    // TODO better error reporting
    read_obj(r.bytes().map_while(|b| b.ok()))
}

pub fn read_obj(src: impl IntoIterator<Item = u8>) -> Result<Mesh<()>> {
    let mut it = src.into_iter().peekable();

    let mut faces = vec![];
    let mut verts = vec![];
    let mut norms = vec![];
    let mut texcs = vec![];

    let mut max_indices = Indices { pos: 0, uv: None, n: None };

    while it.peek().is_some() {
        // TODO Try to avoid collecting
        let line: Vec<_> = (&mut it).take_while(|&c| c != b'\n').collect();

        if line.get(0) == Some(&b'#')
            || line.iter().all(|c| c.is_ascii_whitespace())
        {
            continue;
        }

        // TODO Don't collect, just use the iterator
        let tokens: Vec<&[u8]> = line.split(u8::is_ascii_whitespace).collect();

        if tokens.is_empty() {
            continue;
        }

        let params = tokens.get(1..).ok_or(UnexpectedEnd)?;

        // TODO Doesn't handle leading whitespace
        match tokens[0] {
            b"v" => verts.push(parse_vertex(params)?),
            b"vt" => texcs.push(parse_texcoord(params)?),
            b"vn" => norms.push(parse_normal(params)?),

            b"f" => {
                let tri @ Tri([a, b, c]) = parse_face(params)?;
                let Indices { pos, uv, n } = &mut max_indices;

                *pos = (*pos).max(a.pos).max(b.pos).max(c.pos);
                *uv = (*uv).max(a.uv).max(b.uv).max(c.uv);
                *n = (*n).max(a.n).max(b.n).max(c.n);

                faces.push(tri)
            }

            _ => return Err(UnsupportedItem),
        }
    }

    if max_indices.pos >= verts.len() {
        return Err(OutOfBounds("vertex", max_indices.pos));
    }
    if max_indices.uv.is_some_and(|i| i >= texcs.len()) {
        return Err(OutOfBounds("texcoord", max_indices.uv.unwrap()));
    }
    if max_indices.n.is_some_and(|i| i >= norms.len()) {
        return Err(OutOfBounds("normal", max_indices.n.unwrap()));
    }

    let faces: Vec<_> = faces
        .into_iter()
        .map(|Tri([a, b, c])| Tri([a.pos, b.pos, c.pos]))
        .collect();

    let verts = verts
        .into_iter()
        .map(|pos| vertex(pos, ()))
        .collect();

    Ok(Mesh { faces, verts })
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Io(e) => write!(f, "I/O error: {e}"),
            UnsupportedItem => f.write_str("unsupported item type"),
            UnexpectedEnd => f.write_str("unexpected end of input"),
            InvalidValue => f.write_str("invalid numeric value"),
            OutOfBounds(item, idx) => {
                write!(f, "{item} index out of bounds: {idx}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Io(e)
    }
}
impl From<Utf8Error> for Error {
    fn from(e: Utf8Error) -> Self {
        InvalidValue
    }
}
impl From<ParseFloatError> for Error {
    fn from(e: ParseFloatError) -> Self {
        InvalidValue
    }
}
impl From<ParseIntError> for Error {
    fn from(e: ParseIntError) -> Self {
        InvalidValue
    }
}

fn parse_face(f: &[&[u8]]) -> Result<Tri<Indices>> {
    let a = parse_indices(f[0])?;
    let b = parse_indices(f[1])?;
    let c = parse_indices(f[2])?;
    Ok(Tri([a, b, c]))
}

fn parse_texcoord(c: &[&[u8]]) -> Result<TexCoord> {
    let u: f32 = parse_float(c[0])?;
    let v: f32 = parse_float(c[1])?;
    Ok(uv(u, v))
}

fn parse_normal(n: &[&[u8]]) -> Result<Vec3> {
    parse_vertex(n).map(Vector::to)
}

fn parse_vertex(v: &[&[u8]]) -> Result<Vec3<Real<3, Model>>> {
    let x: f32 = parse_float(v[0])?;
    let y: f32 = parse_float(v[1])?;
    let z: f32 = parse_float(v[2])?;
    Ok(vec3(x, y, z).to())
}

fn parse_float(s: &[u8]) -> Result<f32> {
    Ok(from_utf8(s)?.parse()?)
}

fn parse_indices(s: &[u8]) -> Result<Indices> {
    let mut indices = from_utf8(s)?.split('/');
    let pos = indices
        .next()
        .ok_or(UnexpectedEnd)?
        .parse::<usize>()?;

    let texc = indices.next().map(usize::from_str).transpose()?;

    let norm = indices.next().map(usize::from_str).transpose()?;

    // OBJ has one-based indices
    Ok(Indices {
        pos: pos - 1,
        uv: texc.map(|i| i - 1),
        n: norm.map(|i| i - 1),
    })
}

#[cfg(test)]
mod tests {
    use re::geom::Tri;
    use re::math::vec3;

    use super::*;

    #[test]
    fn test() {
        let input = *br"
# comment
f 1 2 4
f 4 1 3
#anothercomment
v 0.0 0.0 0.0
v 1.0 0.0 0.0

v 0.0 2.0 0.0
v 1.0 2.0 0.0";

        let mesh = read_obj(input.into_iter()).unwrap();

        assert_eq!(mesh.faces, vec![Tri([0, 1, 3]), Tri([3, 0, 2])]);
        assert_eq!(mesh.verts[3].pos, vec3(1.0, 2.0, 0.0).to());
    }

    #[test]
    fn vertex_index_out_of_bounds() {
        let input = *b"f 1 2 3\nv 0.0 0.0 0.0\nv 1.0 1.0 1.0";
        let result = read_obj(input.into_iter());
        assert!(matches!(result, Err(OutOfBounds("vertex", 2))));
    }
    #[test]
    fn texcoord_index_out_of_bounds() {
        let input = *b"f 1/1 1/4 1/2\nv 0.0 0.0 0.0\nvt 0.0 0.0\nvt 0.0 1.0";
        let result = read_obj(input.into_iter());
        assert!(matches!(result, Err(OutOfBounds("texcoord", 1))));
    }
}
