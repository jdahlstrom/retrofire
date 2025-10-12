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
use re::geom::{Mesh, Normal3, Tri, mesh::Builder, vertex};
use re::math::{Point3, Vec3, vec3};
use re::render::{Model, TexCoord, uv};
#[cfg(feature = "std")]
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use ErrorKind::*;

/// Represents errors that may occur during reading OBJ data.
#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub line_no: u32,
}
#[derive(Debug)]
pub enum ErrorKind {
    #[cfg(feature = "std")]
    /// An input/output error while reading from a [`Read`].
    Io(std::io::Error),
    /// An item that is not a face, vertex, texture coordinate, or normal.
    UnsupportedItem(Vec<u8>),
    /// Unexpected end of line or input.
    UnexpectedEnd,
    /// An invalid index value.
    InvalidIndex,
    /// An invalid floating-point value.
    InvalidNumber,
    /// A vertex attribute index that refers to a nonexistent attribute.
    IndexOutOfBounds(&'static str, usize),
    /// Requested vertex attribute not contained in input
    MissingVertexAttribType(&'static str),
}

#[derive(Default)]
pub struct Obj {
    faces: Vec<Tri<Indices>>,
    coords: Vec<Point3<Model>>,
    norms: Vec<Normal3>,
    texcs: Vec<TexCoord>,

    curr_line: String,
    line_no: u32,
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
pub fn load_obj(path: impl AsRef<Path>) -> Result<Builder<()>> {
    read_obj(File::open(path).map_err(|e| Error { line_no: 0, kind: Io(e) })?)
}

/// Reads an OBJ format mesh from input.
///
/// # Errors
/// Returns [`Error`] if I/O or OBJ parsing fails.
#[cfg(feature = "std")]
pub fn read_obj(input: impl Read) -> Result<Builder<()>> {
    let input = BufReader::new(input);
    let mut io_res: Result<()> = Ok(());
    let res = parse_obj(input.bytes().map_while(|r| match r {
        Err(e) => {
            io_res = Err(Error { line_no: 0, kind: e.into() });
            None
        }
        Ok(b) => Some(b),
    }));
    io_res.and(res)
}

/// Parses an OBJ format mesh from an iterator.
///
/// # Errors
/// Returns [`self::Error`] if OBJ parsing fails.
pub fn parse_obj<A>(src: impl IntoIterator<Item = u8>) -> Result<Builder<A>>
where
    Builder<A>: TryFrom<Obj, Error = Error>,
{
    fn parse_obj_(src: &mut dyn Iterator<Item = u8>) -> Result<Obj> {
        let mut obj = Obj::default();

        let mut max_i = Indices::default();
        let mut line = String::new();

        let mut bytes = src.peekable();
        while bytes.peek().is_some() {
            // Reuse allocation
            line.clear();
            line.extend(
                (&mut bytes)
                    .map(char::from)
                    .take_while(|&c| c != '\n'),
            );
            obj.line_no += 1;

            let tokens = &mut line.split_ascii_whitespace();
            match tokens.next().unwrap_or("").as_bytes() {
                // Skip empty lines and comments
                | b"" | [b'#', ..]
                // Skip group or material definitions for now
                | b"g" | b"mtllib" | b"usemtl"
                // Skip smoothing group names for now
                | b"s" => continue,

                // Vertex coordinate
                b"v" => obj.coords.push(obj.parse_vector(tokens)?.to_pt()),
                // Texture coordinate
                b"vt" => obj.texcs.push(obj.parse_texcoord(tokens)?),
                // Normal vector
                b"vn" => obj.norms.push(obj.parse_vector(tokens)?.to()),

                // Face
                b"f" => {
                    let face = obj.parse_face(tokens)?;

                    let indices: &[_] = match &face {
                        Face::Tri(is) => is,
                        Face::Quad(is) => is,
                    };

                    if max_i.n.is_some() && indices[0].n.is_none() {
                        todo!("return error if not all faces have normals")
                    }
                    if max_i.uv.is_some() && indices[0].uv.is_none() {
                        todo!("return error if not all faces have texcoords")
                    }

                    // Keep track of max indices to report error at the end of
                    // parsing if there turned out to be out-of-bounds indices
                    // TODO also record line for error reporting
                    for i in indices {
                        max_i.pos = max_i.pos.max(i.pos);
                        max_i.uv = max_i.uv.max(i.uv);
                        max_i.n = max_i.n.max(i.n);
                    }
                    match *indices {
                        [a, b, c] => obj.faces.push(Tri([a, b, c])),
                        [a, b, c, d] => {
                            obj.faces.push(Tri([a, b, c]));
                            obj.faces.push(Tri([a, c, d]));
                        },
                        _ => unreachable!("source is either Face::Tri or Face::Quad"),
                    };
                }
                // TODO Ignore unsupported lines instead?
                other => {
                    return obj.err(UnsupportedItem(other.to_vec()));
                }
            }
        }

        if !obj.coords.is_empty() && max_i.pos >= obj.coords.len() {
            return obj.err(IndexOutOfBounds("vertex", max_i.pos));
        }
        if let Some(uv) = max_i.uv
            && uv >= obj.texcs.len()
        {
            return obj.err(IndexOutOfBounds("texcoord", uv));
        }
        if let Some(n) = max_i.n
            && n >= obj.norms.len()
        {
            return obj.err(IndexOutOfBounds("normal", n));
        }
        Ok(obj)
    }
    parse_obj_(&mut src.into_iter())?.try_into()
}

impl Obj {
    fn err<T>(&self, kind: ErrorKind) -> Result<T> {
        Err(Error { line_no: self.line_no, kind })
    }
}

impl TryFrom<Obj> for Builder<()> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        o.try_into_with(|_| Some(()))
    }
}
impl TryFrom<Obj> for Builder<Normal3> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        if o.norms.is_empty() {
            return o.err(MissingVertexAttribType("normal"));
        }
        o.try_into_with(|i| i.n.map(|ti| o.norms[ti]))
    }
}
impl TryFrom<Obj> for Builder<TexCoord> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        if o.texcs.is_empty() {
            return o.err(MissingVertexAttribType("texcoord"));
        }
        o.try_into_with(|i| i.uv.map(|ti| o.texcs[ti]))
    }
}
impl TryFrom<Obj> for Builder<(Normal3, TexCoord)> {
    type Error = Error;

    fn try_from(o: Obj) -> Result<Self> {
        if o.norms.is_empty() {
            return o.err(MissingVertexAttribType("normal"));
        }
        if o.texcs.is_empty() {
            return o.err(MissingVertexAttribType("texcoord"));
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
        let mut bld = Builder::default();

        for Tri(indices) in &self.faces {
            if indices.iter().any(|i| attr_fn(i).is_none()) {
                return self.err(MissingVertexAttribType(""));
            }
            let [a, b, c] = indices.map(|v| {
                *map.entry(v).or_insert_with(|| {
                    bld.mesh.verts.push(vertex(
                        self.coords[v.pos],
                        attr_fn(&v).unwrap(), // TODO
                    ));
                    bld.mesh.verts.len() - 1
                })
            });
            bld.push_face(a, b, c)
        }
        Ok(bld)
    }

    fn next<'a>(
        &self,
        i: &mut impl Iterator<Item = &'a str>,
    ) -> Result<&'a str> {
        match i.next() {
            Some(next) => Ok(next),
            None => self.err(UnexpectedEnd),
        }
    }

    fn parse_face<'a>(
        &self,
        i: &mut impl Iterator<Item = &'a str>,
    ) -> Result<Face> {
        let a = self.parse_indices(self.next(i)?)?;
        let b = self.parse_indices(self.next(i)?)?;
        let c = self.parse_indices(self.next(i)?)?;
        if let Some(d) = i.next() {
            let d = self.parse_indices(d)?;
            Ok(Face::Quad([a, b, c, d]))
        } else {
            Ok(Face::Tri([a, b, c]))
        }
    }

    fn parse_texcoord<'a>(
        &self,
        i: &mut impl Iterator<Item = &'a str>,
    ) -> Result<TexCoord> {
        let u = self.parse_num(self.next(i)?)?;
        let v = self.parse_num(self.next(i)?)?;
        Ok(uv(u, v))
    }

    fn parse_vector<'a>(
        &self,
        i: &mut impl Iterator<Item = &'a str>,
    ) -> Result<Vec3<Model>> {
        let x = self.parse_num(self.next(i)?)?;
        let y = self.parse_num(self.next(i)?)?;
        let z = self.parse_num(self.next(i)?)?;
        Ok(vec3(x, y, z))
    }

    fn parse_index(&self, s: &str) -> Result<usize> {
        // OBJ has one-based indices
        s.parse()
            .map(|i: usize| i - 1)
            .or_else(|_| self.err(InvalidIndex))
    }

    fn parse_num(&self, s: &str) -> Result<f32> {
        s.parse().or_else(|_| self.err(InvalidNumber))
    }

    fn parse_indices(&self, param: &str) -> Result<Indices> {
        let indices = &mut param.split('/');
        let pos = self.parse_index(self.next(indices)?)?;

        // Texcoord and normal are optional
        let uv = if let Some(uv) = indices.next()
            && !uv.is_empty()
        {
            Some(self.parse_index(uv)?)
        } else {
            None
        };
        let n = if let Some(n) = indices.next() {
            Some(self.parse_index(n)?)
        } else {
            None
        };
        Ok(Indices { pos, uv, n })
    }
}

//
// Foreign trait impls
//

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "error at line {}: ", self.line_no)?;
        match &self.kind {
            #[cfg(feature = "std")]
            Io(e) => write!(f, "I/O error: {e}"),
            UnsupportedItem(item) => {
                f.write_str("unsupported item type: '")?;
                f.write_str(String::from_utf8_lossy(&item).as_ref())
            }
            UnexpectedEnd => f.write_str("unexpected end of input"),
            InvalidIndex => f.write_str("invalid index value"),
            InvalidNumber => f.write_str("invalid numeric value"),
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
        if let Io(e) = &self.kind {
            return Some(e);
        }
        None
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for ErrorKind {
    fn from(e: std::io::Error) -> Self {
        Io(e)
    }
}
impl From<ParseFloatError> for ErrorKind {
    fn from(_: ParseFloatError) -> Self {
        InvalidNumber
    }
}
impl From<ParseIntError> for ErrorKind {
    fn from(_: ParseIntError) -> Self {
        InvalidIndex
    }
}

#[cfg(test)]
mod tests {
    use re::{geom::Tri, math::point::pt3};

    use super::*;

    fn positions<T>(m: &Mesh<T>) -> Vec<[[f32; 3]; 3]> {
        m.faces
            .iter()
            .map(|Tri(vs)| vs.map(|vi| m.verts[vi].pos.0))
            .collect()
    }
    fn normals(m: &Mesh<Normal3>) -> Vec<[[f32; 3]; 3]> {
        m.faces
            .iter()
            .map(|Tri(vs)| vs.map(|vi| m.verts[vi].attrib.0))
            .collect()
    }

    #[test]
    fn input_with_whitespace_and_comments() {
        let input = *br"
# comment
f 1 2 4
 f 4 1 3
#comment without whitespace after hash
    v 0.0 0.0       0.0
v       1.0 0.0 0.0
  # comment with leading whitespace
v 0.0 -2.0 0.0
        v 1 2 3";

        let m @ Mesh::<()> { faces, verts } =
            &parse_obj::<()>(input).unwrap().build();

        assert_eq!(faces.len(), 2);
        assert_eq!(verts.len(), 4);

        assert_eq!(
            positions(&m),
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, -2.0, 0.0]]
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
    fn positions_texcoords_and_normals() {
        let input = *br"
            f 1/1/1 2/3/2 3/2/2
            f 4/3/2 1/1/1 3/1/3

            vn 1.0 0.0 0.0
            vt 0.0 0.0 0.0
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            vn 1.0 0.0 0.0
            v 0.0 2.0 0.0
            vt 1.0 1.0 1.0
            v 1.0 2.0 3.0
            vt 0.0 -1.0 2.0
            vn 1.0 0.0 0.0";

        let m @ Mesh::<(Normal3, TexCoord)> { faces, verts } =
            &parse_obj(input).unwrap().build();

        assert_eq!(faces.len(), 2);
        assert_eq!(verts.len(), 5);

        assert_eq!(
            positions(&m),
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            ]
        );
        todo!("assert normals and texcoords");
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

        let m: Mesh<Normal3> = parse_obj(input).unwrap().build();

        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.verts.len(), 5);

        assert_eq!(
            positions(&m),
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            ]
        );
        assert_eq!(
            normals(&m),
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
            ]
        );
    }

    #[test]
    fn positions_and_texcoords() {
        let input = *br"
            f 1//1 2//3 4//2
            f 4//3 1//2 3//1

            vn 1.0 0.0 0.0
            v 0.0 0.0 0.0
            v 1.0 0.0 0.0
            v 0.0 2.0 0.0
            vn 0.0 1.0 0.0
            v 1.0 2.0 3.0
            vn 0.0 0.0 -1.0";

        let m: Mesh<TexCoord> = parse_obj(input).unwrap().build();

        assert_eq!(m.faces.len(), 2);
        assert_eq!(m.verts.len(), 6);

        assert_eq!(
            positions(&m),
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            ]
        );
        todo!("assert texcoords");
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
            matches!(&result, Err(Error { line_no: 2, kind: UnsupportedItem(item) }) if item ==
                b"xyz"),
            "actual was: {result:?}"
        );
    }

    #[test]
    fn vertex_index_oob() {
        let input = *b"f 1 2 3\nv 0.0 0.0 0.0\nv 1.0 1.0 1.0";
        let result = parse_obj::<()>(input);
        assert!(
            matches!(
                result,
                Err(Error {
                    line_no: 3,
                    kind: IndexOutOfBounds("vertex", 2)
                })
            ),
            "actual was: {result:?}",
        );
    }
    #[test]
    fn texcoord_index_oob() {
        let input = *b"f 1/1 1/4 1/2\nv 0.0 0.0 0.0\nvt 0.0 0.0\nvt 0.0 1.0";
        let result = parse_obj::<()>(input);
        assert!(
            matches!(
                result,
                Err(Error {
                    line_no: 4, // TODO incorrect
                    kind: IndexOutOfBounds("texcoord", 3)
                })
            ),
            "actual was: {result:?}",
        );
    }

    #[test]
    fn unexpected_end_of_input() {
        let input = *b"f";
        let result = parse_obj::<()>(input);
        assert!(matches!(
            result,
            Err(Error {
                line_no: 1,
                kind: UnexpectedEnd
            })
        ));
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

        let result = read_obj(R(true));

        if let Err(Io(e)) = result {
            assert_eq!(e.kind(), ErrorKind::BrokenPipe);
        } else {
            panic!("result should be Err(Io), was: {result:?}")
        }
    }
}
