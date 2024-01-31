use crate::render::tex;
use crate::util::buf::Buf2;
use crate::util::text::Font;

/// Renders ("bakes") the byte string into a texture.
pub fn bake<T>(s: &[u8], font: &Font<T>) -> tex::Owned<T>
where
    T: Clone + Default,
{
    let lines = s.split(|&c| c == b'\n');
    let (num_rows, num_cols) = lines
        .clone()
        .fold((0, 0), |(nr, nc), row| (nr + 1, nc.max(row.len())));
    if num_rows == 0 || num_cols == 0 {
        return Buf2::empty().into();
    }
    let (gw, gh) = font.glyph_size();

    let mut buf = Buf2::new_default(num_cols * gw, num_rows * gh);

    let (mut x, mut y) = (0, 0);
    for line in lines {
        for c in line {
            let mut dest = buf.slice_mut((x.., y..));
            font.glyph(*c).blit(&mut dest);
            x += gw;
        }
        (x, y) = (0, y + gh);
    }
    buf.into()
}
