use axum::extract::State;
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::sync::Arc;
use tokio::net::TcpListener;
use models_hf::bert::BertInferenceModel;
#[derive(Deserialize)]
struct ReqPayload {
    text: String,
    num_results: u32,
}

#[derive(Serialize)]
struct ResPayload {
    text: Vec<String>,
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let filename = "embeddings.bin";
    let embedding_key = "my_embedding";
    let bert_model = BertInferenceModel::load(
        "sentence-transformers/all-MiniLM-L6-v2",
        "refs/pr/21",
        filename,
        embedding_key,
    )?;

    let mut text_map_file = File::open("text_map.bin").unwrap();
    let text_map: Vec<String> = bincode::decode_from_std_read(
        &mut text_map_file,
        bincode::config::standard(),
    )?;
    let shared_state = Arc::new((bert_model, text_map));

    let app = Router::new()
        .route("/similar", post(find_similar))
        .with_state(shared_state);

    let listener = TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;

    Ok(())
}

//use models_hf::bert;
async fn find_similar(
    State(model_ctx): State<Arc<(BertInferenceModel, Vec<String>)>>,
    Json(payload): Json<ReqPayload>,
) -> Json<ResPayload> {
    let (model, text_map) = &*model_ctx;
    let query_vector = model
        .infer_sentence_embedding(&payload.text)
        .expect("error infering sentence embedding");
    let results: Vec<(usize, f32)> = model
        .score_vector_similarity(
            query_vector,
            payload.num_results as usize,
        )
        .unwrap();

    let results: Vec<String> = results
        .into_iter()
        .map(|r| {
            let top_item_text = text_map.get(r.0).unwrap();
            format!(
                "Item:{} (index: {} score:{:?})",
                top_item_text, r.0, r.1
            )
        })
        .collect();

    Json(ResPayload { text: results })
}


