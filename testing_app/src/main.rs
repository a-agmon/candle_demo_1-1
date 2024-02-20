use std::fs::File;
use std::str::FromStr;
use std::vec;

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
        USER: Given the following  abstract of a scientific article:
        abstract 1: {}
        Can you suggest keywords that describe the topic of this article?
        ASSISTANT:"#,
        results[1]
    );
    let prompt_rag2 = format!(
        r#"
        USER:Given the following  abstract of a scientific article:
        abstract 1: {}
        Can you explain, in a sentence or two, what is this article about?
        ASSISTANT:"#,
        results[1]
    );
    let prompt_rag_sql = r#"
        USER: Given the following schema of the database table customer_usage: 
        dt (date), media_source (string), installs (int), 
        please write an sql query that will fetch from the customer_usage table the data required to answer the question: how many installs per media source did we have over the last 7 days.
        please just write the sql query without further explanations or comments
        SQL QUERY:"#.to_string();

    let prompt_rag_sql2 = r#"
        Consider the following reasoning given by one of our customers for why they unsubscribed from the service 
        and answer the question that follows: 

        customer response: 'The primary reason for my decision is the price. 
        While I value the service you provide, it no longer aligns with my current budget. 
        Additionally, my needs have evolved, and I find myself requiring a different type of service at this time.
        I appreciate the support and assistance you've offered during my subscription and hope to 
        possibly return in the future when my circumstances change.'

        What is the churn reason? 
        1. the cost of the service
        2. the quality of the service
        3. they found a better service
        
        Answer:"#
        .to_string();

    gen_model.run_test(&prompt_rag_sql2, 200)?;

    Ok(())
}
