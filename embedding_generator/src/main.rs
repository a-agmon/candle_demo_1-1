#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::Tensor;
use models_hf::bert::BertInferenceModel;
use rayon::prelude::*;

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
    let assets = get_textcsv_as_map(file_name, 0, 1);
    let (names, texts) = assets.unwrap();
    println!("text_map loaded - size: {}", names.len());

    ser_string_vec_file("texts.bin", &texts).unwrap();
    ser_string_vec_file("keys.bin", &names).unwrap();
    println!("mapping files serialized to disk");

    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        "",
        "",
    )
    .unwrap();
    println!("bert model loaded");

    // try to do this in parallel using rayon
    let results: Vec<Result<Tensor, _>> = texts
        .par_chunks(500)
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
) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let mut text_vec: Vec<String> = Vec::new();
    let mut name_vec: Vec<String> = Vec::new();
    let mut rdr = csv::Reader::from_path(filename)?;
    for result in rdr.records() {
        let record = result?;
        let name = record.get(name_col_index).unwrap().to_string();
        let text = record.get(text_col_index).unwrap().to_string();
        text_vec.push(text);
        name_vec.push(name);
    }
    Ok((name_vec, text_vec))
}

fn ser_string_vec_file(filename: &str, vec: &Vec<String>) -> anyhow::Result<()> {
    let mut file = std::fs::File::create(filename)?;
    bincode::encode_into_std_write(vec, &mut file, bincode::config::standard())?;
    Ok(())
}
