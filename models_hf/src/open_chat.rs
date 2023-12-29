use candle::quantized::gguf_file;
use candle_transformers::models::{quantized_llama::ModelWeights, whisper::model};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

static MODEL_REPO: &str = "TheBloke/openchat_3.5-GGUF";
static MODEL_FILE: &str = "openchat_3.5.Q4_K_M.gguf";

pub struct OpenChatTextGenerator {
    temperature: f64,
    repetition_penalty: f32,
    repatition_last_n: usize,
    tokenizer: Tokenizer,
    model: ModelWeights,
}

impl OpenChatTextGenerator {
    pub fn load() -> anyhow::Result<Self> {
        let api = Api::new()?;
        let api = api.model(MODEL_REPO.to_string());
        let model_path = api.get(MODEL_FILE)?;
        let tokenizer_path = api.get("tokenizer.json")?;

        // read the actual files - model
        let mut model_file = std::fs::File::open(model_path)?;
        let model = gguf_file::Content::read(&mut model_file)?;
        let model = ModelWeights::from_gguf(model, &mut model_file)?;

        // read the tokenizer file
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        //let mut tos = TokenOutputStream::new(tokenizer);
        Ok(Self {
            temperature: 0.0,
            repetition_penalty: 1.1,
            repatition_last_n: 64,
            tokenizer,
            model,
        })
    }

    pub fn generate(&self, prompt_str: &str, max_len: usize) -> anyhow::Result<String> {
        let mut pre_prompt_tokens: Vec<u32> = vec![];
        let tokens = self.tokenizer
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
    let to_sample = max_len.saturating_sub(1);
        todo!()
    }
}
