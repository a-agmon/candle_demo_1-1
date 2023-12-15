#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::Tensor;
use models_hf::bert::BertInferenceModel;
use rayon::prelude::*;
use std::collections::HashMap;

fn main() {
    // extract the file name from the command line arg
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Usage: embedding_generator <file_name>");
        std::process::exit(1);
    }
    let file_name = &args[1];
    println!("Starting to generate embeddings from {}", file_name);
    //let file_name = "/Users/alonagmon/MyData/work/bbc_news.csv";
    let text_map: HashMap<String, String> = get_textcsv_as_map(file_name, 0, 0).unwrap();
    println!("text_map loaded - size: {}", text_map.len());

    // tale all the values to a vec string
    let sentences: Vec<String> = text_map.values().map(|s| s.to_string()).collect();
    // serialize the map to a binary file
    let mut file = std::fs::File::create("text_map.bin").unwrap();
    bincode::encode_into_std_write(&sentences, &mut file, bincode::config::standard())
        .expect("failed to encode sentences");
    println!("text_map serialized to text_map.bin");

    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        "",
        "",
    )
    .unwrap();
    println!("bert model loaded");

    // try to do this in parallel using rayon
    let results: Vec<Result<Tensor, _>> = sentences
        .par_chunks(350)
        .map(|chunk| bert_model.create_embeddings(chunk.to_vec()))
        .collect();
    println!("results generated");
    let embeddings = Tensor::cat(
        &results
            .iter()
            .map(|r| r.as_ref().unwrap())
            .collect::<Vec<_>>(),
        0,
    )
    .unwrap();

    embeddings
        .save_safetensors("my_embedding", "embeddings.bin")
        .unwrap();
    println!("embeddings.bin saved");
}

fn get_textcsv_as_map(
    filename: &str,
    name_col_index: usize,
    text_col_index: usize,
) -> anyhow::Result<HashMap<String, String>> {
    let mut map = HashMap::new();
    let mut rdr = csv::Reader::from_path(filename)?;
    for result in rdr.records() {
        let record = result?;
        let name = record.get(name_col_index).unwrap().to_string();
        let text = record.get(text_col_index).unwrap().to_string();
        map.insert(name, text);
    }
    Ok(map)
}
