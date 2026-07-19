// Hand-authored 16x16 foliage sprites, drawn as ASCII art and encoded to
// 2-bit texels for the raymarch shader (storage binding 18).
//
// This is the Allumeria/Minecraft foliage recipe: leaves and plants get their
// look from deliberately drawn cutout textures — clumped holes, silhouettes,
// two-tone shading — not from hash noise. Editing a sprite = editing the
// ASCII art below; the encoder packs it at startup and a unit test guards the
// dimensions and charset.
//
// Legend: '.' transparent  '#' primary  'o' secondary (dark/stem)  '*' accent
// Rows are written top-down as you read them; the encoder flips them so texel
// y = 0 is the sprite's BOTTOM row (the shader's v coordinate grows upward).

pub const SPRITE_DIM: usize = 16;
/// 16x16 texels x 2 bits = 512 bits = 16 u32 words per sprite.
pub const SPRITE_WORDS: usize = SPRITE_DIM * SPRITE_DIM * 2 / 32;

// Sprite indices — keep in sync with the SPR_* consts in shaders/raymarch.wgsl.
pub const SPR_LEAF_DENSE: usize = 0;
pub const SPR_LEAF_LIGHT: usize = 1;
pub const SPR_LEAF_PINE: usize = 2;
pub const SPR_TALL_GRASS: usize = 3;
pub const SPR_POPPY: usize = 4;
pub const SPR_DAISY: usize = 5;

#[rustfmt::skip]
const ART: [[&str; SPRITE_DIM]; 6] = [
    // SPR_LEAF_DENSE — oak/autumn: ~80% opaque, clumped holes, two-tone.
    [
        "##o##..###o##o##",
        "#o###..###o###o#",
        "o###o##o##..##o#",
        "##o#####o#..###o",
        "..##o##o###o##.#",
        "#.##.###o####o##",
        "###o#o##.##o###o",
        "o##.####o###o##.",
        "#o##o#..###o###o",
        "###o###o##o###.#",
        ".##o#..##o###o##",
        "#o###o##.####..#",
        "##.####o###o##.#",
        "o###o##.##o###o#",
        "#.##o###o####.##",
        "##o#..o###o##o##",
    ],
    // SPR_LEAF_LIGHT — birch: airier (~65% opaque), smaller clumps.
    [
        "#o.##..##o.##o#.",
        ".###o..###.##..#",
        "o#..#o#o##..#o##",
        "#.o####.o#..##.o",
        "..##o#.o###o#..#",
        "#..#.###o##.#o##",
        ".##o#o#..##o###.",
        "o#..###.o#.#o##.",
        "#o#.o#..##o#.##o",
        ".##o##.o##o###..",
        ".#.o#..##o##.o##",
        "#o.##o##..###..#",
        "##..###o#.#o##.#",
        "o#.#o##..#o#.#o.",
        "#..##o##o###..##",
        ".#o#..o##.o##o#.",
    ],
    // SPR_LEAF_PINE — needles: sparse (~50%), diagonal strokes.
    [
        "#o..#..#o..#..o#",
        ".#.o.#..#.#.#.#.",
        "..#.#.o#.o#.#..#",
        "#.#o#.#..#.o#.#.",
        ".o#.#.#.#.#.#o..",
        "#.#.o#.#o#.#..#.",
        ".#.#.#.#.#.o#.#o",
        "o..#.#o#.#.#.#..",
        ".#.#.#.#o#.#.#.#",
        "#.o#.#.#.#.#o..#",
        ".#.#o#.#.#o#.#..",
        "..#.#.#o#.#.#.#o",
        "#.#.#.#.#.#.#.#.",
        ".#o#.#.#.o#.#.o#",
        "#.#.#.o#.#.#.#..",
        "o#..#.#.#.#.#.#.",
    ],
    // SPR_TALL_GRASS — a tuft of tapering blades, dark toward the base.
    [
        "................",
        ".....#..........",
        ".....#....#.....",
        "..#..#....#.....",
        "..#..#...##...#.",
        "..#.##...#....#.",
        "...#.#...#...##.",
        "...#.#..##...#..",
        "...#.##.#...##..",
        "....#o#.#...#...",
        ".#..#o#.#..##..#",
        ".#.#oo#o#..#..#.",
        "..#.#o#o#.##.#..",
        "..#o#oo#o##..#..",
        "...#oo#o#o#.##..",
        "..o#o#oo#oo#o...",
    ],
    // SPR_POPPY — red petal head, dark centre, leafy stem.
    [
        "................",
        "................",
        ".....###........",
        "....#####.......",
        "....##*##.......",
        "....#***#.......",
        ".....###........",
        "......o.........",
        "......o.........",
        "......o..o......",
        "......o.o.......",
        "...o..oo........",
        "....o.o.........",
        ".....oo.........",
        "......o.........",
        ".....oo.........",
    ],
    // SPR_DAISY — white radiating petals, yellow centre, stem.
    [
        "................",
        ".......#........",
        "....#..#..#.....",
        ".....#.#.#......",
        "....#######.....",
        ".....##*##......",
        "....##***##.....",
        ".....##*##......",
        "....#######.....",
        ".....#.#.#......",
        "....#..#..#.....",
        ".......o........",
        ".......o........",
        "....o..o........",
        ".....o.o........",
        "......oo........",
    ],
];

/// Encode all sprites into the flat u32 word array the shader indexes.
/// Texel (x, y) of sprite s lives at bit `(y*16 + x) * 2` of word block
/// `s * SPRITE_WORDS`.
pub fn encoded() -> Vec<u32> {
    let mut out = vec![0u32; ART.len() * SPRITE_WORDS];
    for (si, art) in ART.iter().enumerate() {
        for (row, line) in art.iter().enumerate() {
            assert_eq!(
                line.len(),
                SPRITE_DIM,
                "sprite {si} row {row} must be {SPRITE_DIM} chars"
            );
            let y = SPRITE_DIM - 1 - row; // top-down art -> bottom-up texels
            for (x, ch) in line.bytes().enumerate() {
                let v = match ch {
                    b'.' => 0u32,
                    b'#' => 1,
                    b'o' => 2,
                    b'*' => 3,
                    _ => panic!("sprite {si} row {row}: bad char '{}'", ch as char),
                };
                let bit = (y * SPRITE_DIM + x) * 2;
                out[si * SPRITE_WORDS + bit / 32] |= v << (bit % 32);
            }
        }
    }
    out
}

/// Decode one texel back out (test + tooling mirror of the WGSL sprite_texel).
pub fn texel(words: &[u32], sprite: usize, x: usize, y: usize) -> u32 {
    let bit = (y * SPRITE_DIM + x) * 2;
    (words[sprite * SPRITE_WORDS + bit / 32] >> (bit % 32)) & 3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_all_sprites() {
        let w = encoded();
        assert_eq!(w.len(), 6 * SPRITE_WORDS);
    }

    #[test]
    fn round_trips_known_texels() {
        let w = encoded();
        // SPR_POPPY art row 5 (top-down) = "....#***#......." -> texel y = 10.
        assert_eq!(texel(&w, SPR_POPPY, 4, 10), 1); // '#'
        assert_eq!(texel(&w, SPR_POPPY, 5, 10), 3); // '*'
        assert_eq!(texel(&w, SPR_POPPY, 0, 10), 0); // '.'
        // SPR_POPPY art row 7 "......o........." -> y = 8, stem at x=6.
        assert_eq!(texel(&w, SPR_POPPY, 6, 8), 2); // 'o'
    }

    #[test]
    fn leaf_opacity_in_authored_range() {
        let w = encoded();
        let opacity = |s: usize| {
            let n: usize = (0..SPRITE_DIM)
                .flat_map(|y| (0..SPRITE_DIM).map(move |x| (x, y)))
                .filter(|&(x, y)| texel(&w, s, x, y) != 0)
                .count();
            n as f32 / 256.0
        };
        let dense = opacity(SPR_LEAF_DENSE);
        let light = opacity(SPR_LEAF_LIGHT);
        let pine = opacity(SPR_LEAF_PINE);
        assert!((0.70..=0.90).contains(&dense), "dense leaves {dense}");
        assert!((0.55..=0.75).contains(&light), "light leaves {light}");
        assert!((0.35..=0.60).contains(&pine), "pine needles {pine}");
        assert!(dense > light && light > pine, "opacity ordering");
    }

    #[test]
    fn grass_is_rooted_and_tapers() {
        let w = encoded();
        let row_count = |y: usize| {
            (0..SPRITE_DIM)
                .filter(|&x| texel(&w, SPR_TALL_GRASS, x, y) != 0)
                .count()
        };
        // Dense near the ground, sparse at the tips, empty at the very top.
        assert!(row_count(0) >= 8, "base row density");
        assert!(row_count(12) <= 4, "tip row density");
        assert_eq!(row_count(15), 0, "top row clear");
    }
}
