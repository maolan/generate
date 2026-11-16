fn main() {
    let mut tok = tokie::Tokenizer::from_json("/home/meka/repos/ace/tokenizer.json").unwrap();
    let prompt = "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\nMetal guitar with a lot of distortion\n\n# Metas\n- bpm: 120\n- timesignature: 4/4\n- keyscale: A minor\n- duration: 4 seconds\n<|endoftext|>\n";
    let mut ids = Vec::new();
    tok.encode(prompt, &mut ids).unwrap();
    println!("count: {}", ids.len());
    for id in &ids {
        print!("{id} ");
    }
    println!();
    let lyric = "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>";
    let mut lids = Vec::new();
    tok.encode(lyric, &mut lids).unwrap();
    println!("lyric count: {}", lids.len());
    for id in &lids {
        print!("{id} ");
    }
    println!();
}
