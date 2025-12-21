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
//! ## Faces can have more than three indices:
//! f 1 2 3 4 5
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
//! ```

use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::{
    fmt::{self, Display, Formatter},
    num::{ParseFloatError, ParseIntError},
};
#[cfg(feature = "std")]
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use retrofire_core::{
    geom::{Mesh, Normal3, Tri, mesh::Builder, vertex},
    math::{Point3, Vec3, vec3},
    render::{Model, TexCoord, uv},
};

use Error::*;

/// Represents errors that may occur during reading OBJ data.
#[derive(Debug)]
pub enum Error {
    #[cfg(feature = "std")]
    /// An input/output error while reading from a [`Read`].
    Io(std::io::Error),
    /// An item that is not a face, vertex, texture coordinate, or normal.
    UnsupportedItem(Vec<u8>),
    /// Unexpected end of line or input.
    UnexpectedEnd,
    /// An invalid integer or floating-point value.
    InvalidValue,
    /// A vertex attribute index that refers to a nonexistent attribute.
    IndexOutOfBounds(&'static str, usize),
    /// Requested vertex attribute not contained in input
    MissingVertexAttribType(&'static str),
}

#[derive(Default)]
pub struct Obj {
    faces: Vec<Tri<Indices>>,
    coords: Vec<Point3<Model>>,
    norms: Vec<Normal3<Model>>,
    texcs: Vec<TexCoord>,
}

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
struct Indices {
    pos: usize,
    uv: Option<usize>,
    n: Option<usize>,
}

#[derive(Copy, Clone, Debug)]
enum Face {
    Tri([Indices; 3]),
    Quad([Indices; 4]),
}

/// Loads an OBJ model from a path.
///
/// # Errors
/// Returns [`Error`] if I/O or OBJ parsing fails.
#[cfg(feature = "std")]
pub fn load_obj<A>(path: impl AsRef<Path>) -> Result<Builder<A>>
where
    Builder<A>: TryFrom<Obj, Error = Error>,
{
    read_obj(File::open(path)?)
}

/// Reads an OBJ format mesh from input.
///
/// # Errors
/// Returns [`Error`] if I/O or OBJ parsing fails.
#[cfg(feature = "std")]
pub fn read_obj<A>(input: impl Read) -> Result<Builder<A>>
where
    Builder<A>: TryFrom<Obj, Error = Error>,
{
    let input = BufReader::new(input);
    let mut io_res: Result<()> = Ok(());
    let res = parse_obj(input.bytes().map_while(|r| match r {
        Err(e) => {
            io_res = Err(e.into());
            None
        }
        Ok(b) => Some(b),
    }));
    io_res.and(res)
}

/// Parses an OBJ format mesh from an iterator.
///
/// # Errors
/// Returns [`self::Error`][Error] if OBJ parsing fails.
pub fn parse_obj<A>(src: impl IntoIterator<Item = u8>) -> Result<Builder<A>>
where
    Builder<A>: TryFrom<Obj, Error = Error>,
{
    do_parse_obj(&mut src.into_iter())?.try_into()
}

fn do_parse_obj(src: &mut dyn Iterator<Item = u8>) -> Result<Obj> {
    let mut obj = Obj::default();
    let Obj { faces, coords, norms, texcs } = &mut obj;

    let mut max_i = Indices::default();
    let mut line = String::new();

    let mut it = src.peekable();
    while it.peek().is_some() {
        // Reuse allocation
        line.clear();
        line.extend(
            (&mut it)
                .map(char::from)
                .take_while(|&c| c != '\n'),
        );

        let tokens = &mut line.split_ascii_whitespace();
        match tokens.next().unwrap_or("").as_bytes() {
            // Skip empty lines and comments
            | b"" | [b'#', ..]
            // Skip group or material definitions for now
            | b"g" | b"mtllib" | b"usemtl"
            // Skip smoothing group names for now
            | b"s" => continue,

            // Vertex coordinate
            b"v" => coords.push(parse_point(tokens)?),
            // Texture coordinate
            b"vt" => texcs.push(parse_texcoord(tokens)?),
            // Normal vector
            b"vn" => norms.push(parse_normal(tokens)?),

            // Face
            b"f" => {
                let face = parse_face(tokens)?;

                let indices = match &face {
                    Face::Tri(is) => is.as_slice(),
                    Face::Quad(is) => is.as_slice(),
                };

                if max_i.n.is_some() && indices[0].n.is_none() {
                    todo!("return error if not all faces have normals")
                }
                if max_i.uv.is_some() && indices[0].uv.is_none() {
                    todo!("return error if not all faces have texcoords")
                }

                // Keep track of max indices to report error at the end of
                // parsing if there turned out to be out-of-bounds indices
                for i in indices {
                    max_i.pos = max_i.pos.max(i.pos);
                    max_i.uv = max_i.uv.max(i.uv);
                    max_i.n = max_i.n.max(i.n);
                }
                if let [a, b, c] = *indices {
                    faces.push(Tri([a, b, c]));
                } else if let [a, b, c, d] = *indices {
                    faces.push(Tri([a, b, c]));
                    faces.push(Tri([a, c, d]));
                }
            }
            // TODO Ignore unsupported lines instead?
            other => {
                return Err(UnsupportedItem(other.to_vec()));
            }
        }
    }

    if !coords.is_empty() && max_i.pos >= coords.len() {
        return Err(IndexOutOfBounds("vertex", max_i.pos));
    }
    if let Some(uv) = max_i.uv
        && uv >= texcs.len()
    {
        return Err(IndexOutOfBounds("texcoord", uv));
    }
    if let Some(n) = max_i.n
        && n >= norms.len()
    {
        return Err(IndexOutOfBounds("normal", n));
    }
    Ok(obj)
}

impl TryFrom<Obj> for Builder<()> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        o.try_into_with(|_| Some(()))
    }
}
impl TryFrom<Obj> for Builder<Normal3<Model>> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        if o.norms.is_empty() {
            return Err(MissingVertexAttribType("normal"));
        }
        o.try_into_with(|i| i.n.map(|ti| o.norms[ti]))
    }
}
impl TryFrom<Obj> for Builder<TexCoord> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        if o.texcs.is_empty() {
            return Err(MissingVertexAttribType("texcoord"));
        }
        o.try_into_with(|i| i.uv.map(|ti| o.texcs[ti]))
    }
}
impl TryFrom<Obj> for Builder<(Normal3<Model>, TexCoord)> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        if o.norms.is_empty() {
            return Err(MissingVertexAttribType("normal"));
        }
        if o.texcs.is_empty() {
            return Err(MissingVertexAttribType("texcoord"));
        }
        o.try_into_with(|i| {
            i.n.zip(i.uv)
                .map(|(n, uv)| (o.norms[n], o.texcs[uv]))
        })
    }
}

impl Obj {
    fn try_into_with<A>(
        &self,
        mut attr_fn: impl FnMut(&Indices) -> Option<A>,
    ) -> Result<Builder<A>> {
        // HashMap not in alloc :(
        let mut map: BTreeMap<Indices, usize> = BTreeMap::new();

        let mut faces = Vec::new();
        let mut verts = Vec::new();

        for Tri(indices) in &self.faces {
            if indices.iter().any(|i| attr_fn(i).is_none()) {
                return Err(MissingVertexAttribType(""));
            }
            let indices = indices.map(|v| {
                *map.entry(v).or_insert_with(|| {
                    verts.push(vertex(
                        self.coords[v.pos],
                        attr_fn(&v).unwrap(), // TODO
                    ));
                    verts.len() - 1
                })
            });
            faces.push(Tri(indices));
        }
        Ok(Mesh::new(faces, verts).into_builder())
    }
}

fn next<'a>(i: &mut impl Iterator<Item = &'a str>) -> Result<&'a str> {
    i.next().ok_or(UnexpectedEnd)
}

fn parse_face<'a>(i: &mut impl Iterator<Item = &'a str>) -> Result<Face> {
    let a = parse_indices(next(i)?)?;
    let b = parse_indices(next(i)?)?;
    let c = parse_indices(next(i)?)?;
    if let Some(d) = i.next() {
        let d = parse_indices(d)?;
        Ok(Face::Quad([a, b, c, d]))
    } else {
        Ok(Face::Tri([a, b, c]))
    }
}

fn parse_texcoord<'a>(
    i: &mut impl Iterator<Item = &'a str>,
) -> Result<TexCoord> {
    let u = next(i)?.parse()?;
    let v = next(i)?.parse()?;
    Ok(uv(u, v))
}

fn parse_vector<'a>(
    i: &mut impl Iterator<Item = &'a str>,
) -> Result<Vec3<Model>> {
    let x = next(i)?.parse()?;
    let y = next(i)?.parse()?;
    let z = next(i)?.parse()?;
    Ok(vec3(x, y, z))
}

fn parse_normal<'a>(
    i: &mut impl Iterator<Item = &'a str>,
) -> Result<Normal3<Model>> {
    Ok(parse_vector(i)?)
}

fn parse_point<'a>(
    i: &mut impl Iterator<Item = &'a str>,
) -> Result<Point3<Model>> {
    Ok(parse_vector(i)?.to_pt())
}

fn parse_index(s: &str) -> Result<usize> {
    // OBJ has one-based indices
    Ok(s.parse::<usize>()? - 1)
}

fn parse_indices(param: &str) -> Result<Indices> {
    let indices = &mut param.split('/');
    let pos = next(indices).and_then(parse_index)?;
    // Texcoord and normal are optional
    let uv = if let Some(uv) = indices.next() {
        if !uv.is_empty() {
            Some(parse_index(uv)?)
        } else {
            // `1//2`: only position and normal
            None
        }
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

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "std")]
            Io(e) => write!(f, "I/O error: {e}"),
            UnsupportedItem(item) => {
                f.write_str("unsupported item type: '")?;
                f.write_str(String::from_utf8_lossy(item).as_ref())
            }
            UnexpectedEnd => f.write_str("unexpected end of input"),
            InvalidValue => f.write_str("invalid numeric value"),
            IndexOutOfBounds(item, idx) => {
                write!(f, "{item} index out of bounds: {idx}")
            }
            MissingVertexAttribType(s) => {
                f.write_str("missing vertex attribute: ")?;
                f.write_str(s)
            }
        }
    }
}

impl core::error::Error for Error {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        #[cfg(feature = "std")]
        if let Io(e) = self {
            return Some(e);
        }
        None
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
    use super::*;
    use retrofire_core::{
        geom::{Tri, Vertex},
        math::point::pt3,
    };

    #[test]
    fn input_with_whitespace_and_comments() {
        let input = *br"
# comment
f 1 2 4
 f 4 1 3
#anothercomment
    v 0.0 0.0       0.0
v       1.0 0.0 0.0
  # comment with leading whitespace
v 0.0 -2.0 0.0
        v 1 2 3";

        let m = &parse_obj::<()>(input).unwrap().build();

        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.verts.len(), 4);

        assert_eq!(
            m.faces()
                .map(|tri| tri.0.map(|v| v.pos))
                .collect::<Vec<_>>(),
            [
                [pt3(0.0, 0.0, 0.0), pt3(1.0, 0.0, 0.0), pt3(1.0, 2.0, 3.0)],
                [pt3(1.0, 2.0, 3.0), pt3(0.0, 0.0, 0.0), pt3(0.0, -2.0, 0.0)]
            ]
        );
    }

    #[test]
    fn float_formats() {
        let input = *br"
            v 1 -2 +3
            v 1.0 -2.0 +3.0
            v .1 -.2 +.3
            v 1. -2. +3.
            v 1.0e0 -0.2e1  +300.0e-2
            f 1 2 3
            f 3 4 5";
        let mesh: Mesh<()> = parse_obj(input).unwrap().build();
        assert_eq!(mesh.verts[0].pos, pt3(1.0, -2.0, 3.0));
        assert_eq!(mesh.verts[4].pos, pt3(1.0, -2.0, 3.0));
    }

    #[test]
    fn quads() {
        let input = *br"
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 1.0 1.0 0.0
            v 0.0 1.0 0.0
            f 1 2 3 4
        ";
        let mesh: Mesh<()> = parse_obj(input).unwrap().build();

        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(mesh.faces, [Tri([0, 1, 2]), Tri([0, 2, 3])]);
    }

    #[test]
    fn positions_and_normals() {
        let input = *br"
            f 1//1 2//3 4//2
            f 4//3 1//1 3//1

            vn 1.0 0.0 0.0
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 0.0 2.0 0.0
            vn 0.0 1.0 0.0
            v 1.0 2.0 3.0
            vn 0.0 0.0 -1.0";

        let m: Mesh<Normal3<_>> = parse_obj(input).unwrap().build();

        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.verts.len(), 5);

        assert_eq!(
            m.faces()
                .map(|tri| tri.0.map(|&Vertex { pos, attrib: n }| (pos, n)))
                .collect::<Vec<_>>(),
            [
                [
                    (pt3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)),
                    (pt3(1.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0)),
                    (pt3(1.0, 2.0, 3.0), vec3(0.0, 1.0, 0.0)),
                ],
                [
                    (pt3(1.0, 2.0, 3.0), vec3(0.0, 0.0, -1.0)),
                    (pt3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)),
                    (pt3(0.0, 2.0, 0.0), vec3(1.0, 0.0, 0.0))
                ]
            ]
        );
    }

    #[test]
    fn positions_and_texcoords() {
        let input = *br"
            f 1/1 2/3 4/2
            f 4/3 1/2 3/1

            vt 0.0 0.0
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 0.0 2.0 0.0
            vt 0.0 1.0
            v 1.0 2.0 3.0
            vt 1.0 1.0";

        let m: Mesh<TexCoord> = parse_obj(input).unwrap().build();

        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.verts.len(), 6);

        assert_eq!(
            m.faces()
                .map(|tri| tri.0.map(|&Vertex { pos, attrib: uv }| (pos, uv)))
                .collect::<Vec<_>>(),
            [
                [
                    (pt3(0.0, 0.0, 0.0), uv(0.0, 0.0)),
                    (pt3(1.0, 0.0, 0.0), uv(1.0, 1.0)),
                    (pt3(1.0, 2.0, 3.0), uv(0.0, 1.0))
                ],
                [
                    (pt3(1.0, 2.0, 3.0), uv(1.0, 1.0)),
                    (pt3(0.0, 0.0, 0.0), uv(0.0, 1.0)),
                    (pt3(0.0, 2.0, 0.0), uv(0.0, 0.0))
                ]
            ]
        );
    }

    #[test]
    fn positions_texcoords_and_normals() {
        let input = *br"
            f 1/1/1 2/3/2 3/2/2
            f 4/3/2 1/1/1 3/1/3

            vn 1.0 0.0 0.0
            vt 0.0 0.0
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            vn 0.0 1.0 0.0
            v 0.0 2.0 0.0
            vt 1.0 1.0
            v 1.0 2.0 3.0
            vt 0.0 -1.0
            vn 0.0 0.0 1.0";

        let m = &parse_obj::<(Normal3<_>, TexCoord)>(input)
            .unwrap()
            .build();
        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.verts.len(), 5);

        assert_eq!(
            m.faces()
                .map(|tri| tri
                    .0
                    .map(|&Vertex { pos, attrib: (n, uv) }| (pos, n, uv)))
                .collect::<Vec<_>>(),
            [
                [
                    (pt3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), uv(0.0, 0.0)),
                    (pt3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), uv(0.0, -1.0)),
                    (pt3(0.0, 2.0, 0.0), vec3(0.0, 1.0, 0.0), uv(1.0, 1.0))
                ],
                [
                    (pt3(1.0, 2.0, 3.0), vec3(0.0, 1.0, 0.0), uv(0.0, -1.0)),
                    (pt3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), uv(0.0, 0.0)),
                    (pt3(0.0, 2.0, 0.0), vec3(0.0, 0.0, 1.0), uv(0.0, 0.0))
                ]
            ]
        );
    }

    #[test]
    fn empty_input() {
        let mesh: Mesh<()> = parse_obj(*b"")
            .unwrap_or_else(|e| {
                panic!("empty input should be valid, got {e:?}")
            })
            .build();
        assert!(mesh.faces.is_empty());
        assert!(mesh.verts.is_empty());
    }

    #[test]
    fn input_only_whitespace() {
        let mesh: Mesh<()> = parse_obj(*b"   \n     \n\n ")
            .expect("white-space only input should be valid")
            .build();
        assert!(mesh.faces.is_empty());
        assert!(mesh.verts.is_empty());
    }

    #[test]
    fn input_only_comments() {
        let mesh: Mesh<()> = parse_obj(*b"# comment\n #another comment")
            .expect("comment-only input should be valid")
            .build();
        assert!(mesh.faces.is_empty());
        assert!(mesh.verts.is_empty());
    }

    #[test]
    fn unknown_item() {
        let result = parse_obj::<()>(*b"f 1 2 3\nxyz 4 5 6");
        assert!(
            matches!(&result, Err(UnsupportedItem(item)) if item == b"xyz"),
            "actual was: {result:?}"
        );
    }

    #[test]
    fn vertex_index_oob() {
        let input = *b"f 1 2 3\nv 0.0 0.0 0.0\nv 1.0 1.0 1.0";
        let result = parse_obj::<()>(input);
        assert!(
            matches!(result, Err(IndexOutOfBounds("vertex", 2))),
            "actual was: {result:?}",
        );
    }
    #[test]
    fn texcoord_index_oob() {
        let input = *b"f 1/1 1/4 1/2\nv 0.0 0.0 0.0\nvt 0.0 0.0\nvt 0.0 1.0";
        let result = parse_obj::<()>(input);
        assert!(
            matches!(result, Err(IndexOutOfBounds("texcoord", 3))),
            "actual was: {result:?}",
        );
    }

    #[test]
    fn unexpected_end_of_input() {
        let input = *b"f";
        let result = parse_obj::<()>(input);
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
                    buf[..1].copy_from_slice(b"t");
                    Ok(7)
                } else {
                    Err(ErrorKind::BrokenPipe.into())
                }
            }
        }

        let result = read_obj::<()>(R(true));

        if let Err(Io(e)) = result {
            assert_eq!(e.kind(), ErrorKind::BrokenPipe);
        } else {
            panic!("result should be Err(Io), was: {result:?}")
        }
    }
}
