use std::fs::File;
use std::str::FromStr;

use models_hf::bert::BertInferenceModel;
use models_hf::quant_phi::QuantPhiTextGenerator;

fn main() -> anyhow::Result<()> {
    println!("Starting testing_app...");
    let key_words = "self driving cars navigation";
    let mut keys_map_file = File::open("keys.bin").unwrap();
    let mut text_map_file = File::open("texts.bin").unwrap();
    let keys_map: Vec<String> =
        bincode::decode_from_std_read(&mut keys_map_file, bincode::config::standard())?;
    let text_map: Vec<String> =
        bincode::decode_from_std_read(&mut text_map_file, bincode::config::standard())?;

    let mut gen_model = QuantPhiTextGenerator::load()?;
    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        "embeddings.bin",
        "my_embedding",
    )?;
    let query_vector = bert_model
        .infer_sentence_embedding(key_words)
        .expect("error infering sentence embedding");
    let results: Vec<(usize, f32)> = bert_model.score_vector_similarity(query_vector, 3).unwrap();

    let results: Vec<String> = results
        .into_iter()
        .map(|r| String::from_str(text_map.get(r.0).unwrap()).unwrap())
        .collect();

    let prompt_rag = format!(
        r#"
        USER: I have an abstract of a scientific article.
        abstract 1: {}
        Can you explain in a sentence or two what is this abstract about?
        ASSISTANT:"#,
        results[1]
    );
    gen_model.run_test(&prompt_rag, 250)?;

    Ok(())
}
