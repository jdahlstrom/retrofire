#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

#[cfg(test)]
mod tests {
    #[test]
    fn test() {}
}
