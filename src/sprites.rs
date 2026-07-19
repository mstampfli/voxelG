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
pub const SPR_LEAF_A: usize = 0; // upright X-quad leaf cluster, variant A
pub const SPR_LEAF_B: usize = 1; // upright X-quad leaf cluster, variant B
pub const SPR_LEAF_PINE: usize = 2; // drooping needle fan for pine X-quads
pub const SPR_TALL_GRASS: usize = 3;
pub const SPR_POPPY: usize = 4;
pub const SPR_DAISY: usize = 5;
pub const SPR_LEAF_TOP: usize = 6; // horizontal canopy cap, seen from above
pub const SPR_LEAF_FACE: usize = 7; // solid block-face texture (0 = shadow crevice)

/// IMPORTANT (flowers): the cross-quad renderer draws the SAME sprite on two
/// diagonal planes through the voxel centre. The stem must sit exactly on the
/// centre columns (7-8, which are also mirror-invariant: 15-7 = 8) or the two
/// quads render two separate stems instead of one X.

#[rustfmt::skip]
const ART: [[&str; SPRITE_DIM]; 8] = [
    // SPR_LEAF_A — bushy leaf-cluster quad for canopy fringes: large clear
    // leaves with highlight tips (*), dark understory (o), ragged silhouette
    // (transparent border texels so quad edges never read as straight lines).
    [
        "....#*....##....",
        "..######.####*..",
        ".o##*#########..",
        "#######o####o#*.",
        "o####o##*######.",
        ".#o####*####o##o",
        "..######o######.",
        ".#*##o####*###o.",
        "###*####o######.",
        "o######o####*##o",
        ".####*####o####.",
        "..o###o##*###o..",
        ".##*#####o##*#..",
        "..#o##*####o#...",
        "....###.##o.....",
        "......#*...#....",
    ],
    // SPR_LEAF_B — fringe cluster variant with a different silhouette.
    [
        ".....##....#*...",
        "...#####.#####..",
        ".#########*###o.",
        ".o###*####o####.",
        "#########o###*#o",
        "o###o#*########.",
        ".#####o###o####o",
        "#*####o#*######.",
        ".######o######o.",
        "o##*#####o##*##.",
        "#####o#*#######o",
        ".o####o###o###..",
        "..##*######*#o..",
        "..o######o##.#..",
        "....##o.###.....",
        ".....#...#o.....",
    ],
    // SPR_LEAF_PINE — drooping needle fan for pine X-quads.
    [
        ".......##.......",
        "....o.####.o....",
        "..#..######..#..",
        ".#o.###*###.o#..",
        "#..##o####o##..#",
        ".#.#####o####.#.",
        "#.###o##*##o##.#",
        ".###.##o##.###o.",
        "#o#.####o##.#o#.",
        ".##.#o####.##.#o",
        "#.#.####o#.#.#..",
        ".#..#o###.#o.#..",
        "#...####.#..#...",
        ".#..#o##.#...o..",
        "....###..o......",
        ".....#o...#.....",
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
    // SPR_POPPY — red petal head, dark centre, stem dead-centre on cols 7-8.
    [
        "................",
        "......####......",
        ".....######.....",
        ".....##**##.....",
        ".....##**##.....",
        "......####......",
        ".......##.......",
        ".......oo.......",
        ".......oo.......",
        ".......oo.......",
        "....o..oo.......",
        ".....o.oo..o....",
        "......ooo.o.....",
        ".......oo.......",
        ".......oo.......",
        ".......oo.......",
    ],
    // SPR_DAISY — white radiating petals, yellow centre, stem on cols 7-8.
    [
        "................",
        "......#..#......",
        "...#..####..#...",
        "....########....",
        "....##****##....",
        "...###****###...",
        "....##****##....",
        "....########....",
        "...#..####..#...",
        "......#..#......",
        ".......oo.......",
        ".......oo.......",
        "...o...oo...o...",
        "....o..oo..o....",
        ".....o.oo.o.....",
        ".......oo.......",
    ],
    // SPR_LEAF_TOP — horizontal canopy layer seen from above: a ragged
    // radial rosette of leaves, transparent at the corners so stacked layers
    // never read as square plates.
    [
        ".....o#..#o.....",
        "...####*###o....",
        "..o######*###...",
        ".#####o######o..",
        ".###*#####o###*.",
        "o###o##*######..",
        "######o###o####o",
        ".#*###o#*#####*.",
        "o####*#o######o.",
        "#####o####o####.",
        ".####o##*###o##.",
        "..##*######o##..",
        ".o######*#####..",
        "..####o####o#...",
        "....###*##o.....",
        "......#o.#......",
    ],
    // SPR_LEAF_FACE — the LEAF MOSAIC: distinct overlapping oval leaves,
    // each with a dark outline side (o), a lit body (#) and a bright tip
    // (*), separated by deep-shadow gaps (.). Triplanar-projected onto the
    // canopy surface — this is what makes individual leaves readable.
    [
        "..o#*..*#o...o#*",
        ".o##*..*##o..o##",
        "o###*..*###o.o##",
        "o#o#....o#o..o#o",
        ".*#o..o#*...*#o.",
        "*##o..o##*..*##o",
        "###o.o####..###o",
        ".o#..o#o#o...o#.",
        "..o#*..o#*...o#*",
        ".o##*.o###*..o##",
        "o###..o###o..o##",
        "o#o...o#o#......",
        ".*#o...*#o..*#o.",
        "*##o..*##o.*###o",
        "###o..###o..###o",
        ".o#....o#....o#.",
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
        assert_eq!(w.len(), 8 * SPRITE_WORDS);
    }

    /// The leaf mosaic needs enough leaf coverage that the gaps read as
    /// shadow crevices between leaves, not as the dominant surface.
    #[test]
    fn leaf_mosaic_coverage() {
        let w = encoded();
        let o = opacity(&w, SPR_LEAF_FACE);
        assert!((0.60..=0.85).contains(&o), "leaf mosaic coverage: {o}");
    }

    #[test]
    fn round_trips_known_texels() {
        let w = encoded();
        // SPR_POPPY art row 3 (top-down) = ".....##**##....." -> texel y = 12.
        assert_eq!(texel(&w, SPR_POPPY, 5, 12), 1); // '#'
        assert_eq!(texel(&w, SPR_POPPY, 7, 12), 3); // '*'
        assert_eq!(texel(&w, SPR_POPPY, 0, 12), 0); // '.'
        // SPR_POPPY art row 8 ".......oo......." -> y = 7, stem at x=7.
        assert_eq!(texel(&w, SPR_POPPY, 7, 7), 2); // 'o'
    }

    /// The cross-quad renderer draws the same sprite on two planes through
    /// the voxel centre: a flower's stem must sit exactly on the centre
    /// columns 7-8 (mirror-invariant) or the X renders as two split stems.
    #[test]
    fn flower_stems_centred() {
        let w = encoded();
        for s in [SPR_POPPY, SPR_DAISY] {
            for y in [0usize, 1, 4, 5] {
                assert_eq!(texel(&w, s, 7, y), 2, "sprite {s} stem col 7 y {y}");
                assert_eq!(texel(&w, s, 8, y), 2, "sprite {s} stem col 8 y {y}");
                assert_eq!(texel(&w, s, 6, y), 0, "sprite {s} col 6 clear y {y}");
                assert_eq!(texel(&w, s, 9, y), 0, "sprite {s} col 9 clear y {y}");
            }
        }
    }


    fn opacity(w: &[u32], s: usize) -> f32 {
        let n: usize = (0..SPRITE_DIM)
            .flat_map(|y| (0..SPRITE_DIM).map(move |x| (x, y)))
            .filter(|&(x, y)| texel(w, s, x, y) != 0)
            .count();
        n as f32 / 256.0
    }

    #[test]
    fn leaf_clusters_in_authored_range() {
        let w = encoded();
        let a = opacity(&w, SPR_LEAF_A);
        let b = opacity(&w, SPR_LEAF_B);
        let pine = opacity(&w, SPR_LEAF_PINE);
        let top = opacity(&w, SPR_LEAF_TOP);
        assert!((0.45..=0.78).contains(&a), "cluster A {a}");
        assert!((0.45..=0.78).contains(&b), "cluster B {b}");
        assert!((0.30..=0.60).contains(&pine), "pine fan {pine}");
        assert!((0.45..=0.80).contains(&top), "top rosette {top}");
    }

    /// Quad edges must never read as straight lines: every leaf-cluster
    /// sprite needs transparent corners (ragged silhouette).
    #[test]
    fn leaf_clusters_have_ragged_corners() {
        let w = encoded();
        for s in [SPR_LEAF_A, SPR_LEAF_B, SPR_LEAF_PINE, SPR_LEAF_TOP] {
            for (x, y) in [(0, 0), (15, 0), (0, 15), (15, 15)] {
                assert_eq!(texel(&w, s, x, y), 0, "sprite {s} corner ({x},{y})");
            }
        }
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
