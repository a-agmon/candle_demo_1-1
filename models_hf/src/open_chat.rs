#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as model_type;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::token_output_stream::TokenOutputStream;

static MODEL_REPO: &str = "TheBloke/openchat_3.5-GGUF";
static MODEL_FILE: &str = "openchat_3.5.Q4_K_M.gguf";
static TOKENIZER_REPO: &str = "openchat/openchat_3.5";

pub struct OpenChatTextGenerator {
    temperature: f64,
    seed: u64,
    //repetition_penalty: f32,
    //  repatition_last_n: usize,
    tos: TokenOutputStream,
    model: ModelWeights,
}

// implement clone for this
impl Clone for OpenChatTextGenerator {
    fn clone(&self) -> Self {
        Self {
            temperature: self.temperature,
            seed: self.seed,
            tos: self.tos.clone(),
            model: self.model.clone(),
        }
    }
    
}

impl OpenChatTextGenerator {
    pub fn load() -> anyhow::Result<Self> {
        let api = Api::new()?;
        let api = api.model(MODEL_REPO.to_string());
        let model_path = api.get(MODEL_FILE)?;
        let mut model_file = std::fs::File::open(model_path)?;
        let model = gguf_file::Content::read(&mut model_file)?;
        let model = ModelWeights::from_gguf(model, &mut model_file)?;

        // read the tokenizer file
        let api = Api::new()?;
        let api = api.model(TOKENIZER_REPO.to_string());
        let tokenizer_path = api.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        let tos = TokenOutputStream::new(tokenizer);
        //let mut tos = TokenOutputStream::new(tokenizer);
        Ok(Self {
            temperature: 0.0,
            seed: 299792458,
            // repetition_penalty: 1.1,
            // repatition_last_n: 64,
            tos,
            model,
        })
    }

    pub fn generate(&mut self, prompt_str: &str, max_len: usize) -> anyhow::Result<String> {
        let mut tos = self.tos.clone();
        let pre_prompt_tokens: Vec<u32> = vec![]; // in case we want to use this
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = max_len.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > model_type::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model_type::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        let mut all_tokens = vec![];
        let mut logits_processor = LogitsProcessor::new(self.seed, Some(self.temperature), None);
        let mut next_token = {
            let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        all_tokens.push(next_token);

        let mut generated_text = String::new();
        let eos_token = "<|end_of_turn|>";
        if let Some(t) = tos.next_token(next_token)? {
            generated_text.push_str(&t);
            //print!("{t}");
            //std::io::stdout().flush()?;
        }
        let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            // im ignoring here some code that actually applies the repetition penalty
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                generated_text.push_str(&t);
            }
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(candle::Error::msg)? {
            generated_text.push_str(&rest);
        }
        Ok(generated_text)
    }

    pub fn generate_once(&mut self, prompt_str: &str, max_len: usize) -> anyhow::Result<String> {
        let pre_prompt_tokens: Vec<u32> = vec![]; // in case we want to use this
        let tokens = self
            .tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = max_len.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > model_type::MAX_SEQ_LEN - 10 {
            println!("truncating the prompt - its longer than the max seq len");
            let to_remove = prompt_tokens.len() + to_sample + 10 - model_type::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        let mut all_tokens = vec![];
        let mut logits_processor = LogitsProcessor::new(self.seed, Some(self.temperature), None);
        let mut next_token = {
            let input = Tensor::new(prompt_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        all_tokens.push(next_token);

        let mut generated_text = String::new();
        let eos_token = "<|end_of_turn|>";
        let next_str = self.tos.tokenizer().decode(&[next_token], true).unwrap();
        generated_text.push_str(&next_str);

        let binding = self.tos.tokenizer().get_vocab(true);
        let eos_token = binding.get(eos_token).unwrap();
        for index in 0..to_sample {
            let input = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            // im ignoring here some code that actually applies the repetition penalty
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            let next_str = self.tos.tokenizer().decode(&[next_token], true).unwrap();
            let next_str = format!("{} ", next_str);
            generated_text.push_str(next_str.as_str());
            if next_token == *eos_token {
                break;
            };
        }
        let next_str = self.tos.tokenizer().decode(&[next_token], true).unwrap();
        generated_text.push_str(&next_str);
        Ok(generated_text)
    }
}
