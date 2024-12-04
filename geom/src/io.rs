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

use alloc::{string::String, vec};
use core::{
    fmt,
    num::{ParseFloatError, ParseIntError},
};
#[cfg(feature = "std")]
use std::{
    error,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use re::geom::{mesh::Builder, vertex, Mesh, Normal3, Tri};
use re::math::{
    point::Point3,
    vec::{vec3, Vec3, Vector},
};
use re::render::{
    tex::{uv, TexCoord},
    Model,
};

use Error::*;

#[derive(Debug)]
pub enum Error {
    #[cfg(feature = "std")]
    /// Input/output error during reading from a `Read`.
    Io(std::io::Error),
    /// Unsupported line item identifier
    UnsupportedItem(char),
    /// Unexpected end of line or input.
    UnexpectedEnd,
    /// Invalid integer or floating-point value.
    InvalidValue,
    /// Vertex attribute index referring to nonexistent attribute.
    IndexOutOfBounds(&'static str, usize),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Indices {
    pos: usize,
    uv: Option<usize>,
    n: Option<usize>,
}

pub type Result<T> = core::result::Result<T, Error>;

/// Loads an OBJ model from a path.
///
/// # Errors
/// TODO
#[cfg(feature = "std")]
pub fn load_obj(path: impl AsRef<Path>) -> Result<Builder<()>> {
    let r = &mut BufReader::new(File::open(path)?);
    read_from(r)
}

/// Reads an OBJ format mesh from input.
///
/// # Errors
///
#[cfg(feature = "std")]
pub fn read_from(input: impl Read) -> Result<Builder<()>> {
    let mut io_err = None;
    let res = parse_obj(input.bytes().map_while(|r| match r {
        Err(e) => {
            io_err = Some(e.into());
            None
        }
        Ok(b) => Some(b),
    }));
    if let Some(e) = io_err {
        Err(e)
    } else {
        res
    }
}

/// Parses an OBJ format mesh from an iterator.
pub fn parse_obj(src: impl IntoIterator<Item = u8>) -> Result<Builder<()>> {
    let mut faces = vec![];
    let mut verts = vec![];
    let mut norms = vec![];
    let mut tcrds = vec![];

    let mut max = Indices { pos: 0, uv: None, n: None };

    let mut line = String::new();

    let mut it = src.into_iter().peekable();
    while it.peek().is_some() {
        // Reuse allocation
        line.clear();
        line.extend(
            (&mut it)
                .map(char::from)
                .take_while(|&c| c != '\n'),
        );

        let tokens = &mut line.split_ascii_whitespace();

        let Some(item) = tokens.next() else {
            continue;
        };
        match item.as_bytes() {
            [b'#', ..] => continue, // Skip comment

            b"v" => verts.push(parse_point(tokens)?),
            b"vt" => tcrds.push(parse_texcoord(tokens)?),
            b"vn" => norms.push(parse_normal(tokens)?),

            b"f" => {
                let tri @ Tri([a, b, c]) = parse_face(tokens)?;

                // TODO Ugly :(
                max.pos = max.pos.max(a.pos).max(b.pos).max(c.pos);
                max.uv = max.uv.max(a.uv).max(b.uv).max(c.uv);
                max.n = max.n.max(a.n).max(b.n).max(c.n);

                faces.push(tri)
            }

            // TODO Ignore unsupported lines instead?
            [c, ..] => return Err(UnsupportedItem(*c as char)),
            b"" => unreachable!("empty slices are filtered out"),
        }
    }

    if !verts.is_empty() && max.pos >= verts.len() {
        return Err(IndexOutOfBounds("vertex", max.pos));
    }
    if max.uv.is_some_and(|i| i >= tcrds.len()) {
        return Err(IndexOutOfBounds("texcoord", max.uv.unwrap()));
    }
    if max.n.is_some_and(|i| i >= norms.len()) {
        return Err(IndexOutOfBounds("normal", max.n.unwrap()));
    }

    let faces = faces
        .into_iter()
        .map(|Tri(vs)| Tri(vs.map(|i| i.pos)));

    let verts = verts.into_iter().map(|pos| vertex(pos, ()));

    Ok(Mesh::new(faces, verts).into_builder())
}

fn next<'a>(i: &mut impl Iterator<Item = &'a str>) -> Result<&'a str> {
    i.next().ok_or(UnexpectedEnd)
}

fn parse_face<'a>(
    i: &mut impl Iterator<Item = &'a str>,
) -> Result<Tri<Indices>> {
    let a = parse_indices(next(i)?)?;
    let b = parse_indices(next(i)?)?;
    let c = parse_indices(next(i)?)?;
    Ok(Tri([a, b, c]))
}

fn parse_texcoord<'a>(
    i: &mut impl Iterator<Item = &'a str>,
) -> Result<TexCoord> {
    let u = next(i)?.parse()?;
    let v = next(i)?.parse()?;
    Ok(uv(u, v))
}

fn parse_vector<'a>(i: &mut impl Iterator<Item = &'a str>) -> Result<Vec3<Model>>
{
    let x = next(i)?.parse()?;
    let y = next(i)?.parse()?;
    let z = next(i)?.parse()?;
    Ok(vec3(x, y, z))
}

fn parse_normal<'a>(i: &mut impl Iterator<Item = &'a str>) -> Result<Normal3> {
    parse_vector(i).map(Vector::to)
}

fn parse_point<'a>(
    i: &mut impl Iterator<Item = &'a str>,
) -> Result<Point3<Model>> {
    parse_vector(i).map(Vector::to_pt)
}

fn parse_index(s: &str) -> Result<usize> {
    // OBJ has one-based indices
    Ok(s.parse::<usize>()? - 1)
}

fn parse_indices(param: &str) -> Result<Indices> {
    let indices = &mut param.split('/');

    let pos = parse_index(next(indices)?)?;

    let uv = if let Some(uv) = indices.next() {
        Some(parse_index(uv)?)
    } else {
        None
    };
    let n = if let Some(n) = indices.next() {
        Some(parse_index(n)?)
    } else {
        None
    };
    Ok(Indices { pos, uv, n })
}

//
// Foreign trait impls
//

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Io(e) => write!(f, "I/O error: {e}"),
            UnsupportedItem(c) => {
                write!(f, "unsupported item type '{c}'")
            }
            UnexpectedEnd => f.write_str("unexpected end of input"),
            InvalidValue => f.write_str("invalid numeric value"),
            IndexOutOfBounds(item, idx) => {
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
impl From<ParseFloatError> for Error {
    fn from(_: ParseFloatError) -> Self {
        InvalidValue
    }
}
impl From<ParseIntError> for Error {
    fn from(_: ParseIntError) -> Self {
        InvalidValue
    }
}

#[cfg(test)]
mod tests {
    use re::{geom::Tri, math::point::pt3};

    use super::*;

    #[test]
    fn valid_input() {
        let input = *br"
# comment
f 1 2 4
 f 4 1 3
#anothercomment
    v 0.0 0.0 0.0
v       1.0 0.0 0.0
  # comment with leading whitespace
v 0.0 2.0 0.0
v 1.0 2.0      0.0";

        let mesh = parse_obj(input).unwrap().build();

        assert_eq!(mesh.faces, vec![Tri([0, 1, 3]), Tri([3, 0, 2])]);
        assert_eq!(mesh.verts[3].pos, pt3(1.0, 2.0, 0.0));
    }

    #[test]
    fn empty_input() {
        let mesh = parse_obj(*b"")
            .expect("empty input should be valid")
            .build();
        assert!(mesh.faces.is_empty());
        assert!(mesh.verts.is_empty());
    }

    #[test]
    fn input_only_whitespace() {
        let mesh = parse_obj(*b"   \n     \n\n ")
            .expect("white-space only input should be valid")
            .build();
        assert!(mesh.faces.is_empty());
        assert!(mesh.verts.is_empty());
    }

    #[test]
    fn input_only_comments() {
        let mesh = parse_obj(*b"# comment\n #another comment")
            .expect("comment-only input should be valid")
            .build();
        assert!(mesh.faces.is_empty());
        assert!(mesh.verts.is_empty());
    }

    #[test]
    fn unknown_item() {
        let result = parse_obj(*b"f 1 2 3\nxyz 4 5 6");
        assert!(
            matches!(result, Err(UnsupportedItem('x'))),
            "actual was: {result:?}"
        );
    }

    #[test]
    fn vertex_index_oob() {
        let input = *b"f 1 2 3\nv 0.0 0.0 0.0\nv 1.0 1.0 1.0";
        let result = parse_obj(input);
        assert!(
            matches!(result, Err(IndexOutOfBounds("vertex", 2))),
            "actual was: {result:?}",
        );
    }
    #[test]
    fn texcoord_index_oob() {
        let input = *b"f 1/1 1/4 1/2\nv 0.0 0.0 0.0\nvt 0.0 0.0\nvt 0.0 1.0";
        let result = parse_obj(input);
        assert!(
            matches!(result, Err(IndexOutOfBounds("texcoord", 3))),
            "actual was: {result:?}",
        );
    }

    #[test]
    fn unexpected_end_of_input() {
        let input = *b"f";
        let result = parse_obj(input);
        assert!(matches!(result, Err(UnexpectedEnd)));
    }

    #[test]
    #[cfg(feature = "std")]
    fn io_error() {
        use std::io::ErrorKind;

        struct R(bool);
        impl Read for R {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.0 {
                    self.0 = false;
                    buf.copy_from_slice(b"t");
                    Ok(7)
                } else {
                    Err(ErrorKind::BrokenPipe.into())
                }
            }
        }

        let result = read_from(R(true));

        if let Err(Io(e)) = result {
            assert_eq!(e.kind(), ErrorKind::BrokenPipe);
        } else {
            panic!("result should be Err(Io), was: {result:?}")
        }
    }
}
