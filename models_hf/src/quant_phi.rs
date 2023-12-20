#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
//use clap::{Parser, ValueEnum};

use candle_transformers::models::mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;

use candle::{DType, Device, Tensor};
//use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: QMixFormer,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: QMixFormer,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            print!("{token}");
            std::io::stdout().flush()?;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

pub struct QuantPhiTextGenerator {
    text_generator:TextGeneration,
}

impl QuantPhiTextGenerator {
    pub fn load() -> Result<Self> {
        let api = Api::new()?;
        let model_id = "lmz/candle-quantized-phi".to_string();
        let revision = "main".to_string();
        // =>
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let model_filenames = vec![repo.get("model-puffin-phi-v2-q4k.gguf")?];
        // =>
        let tokenizer_filename = repo.get("tokenizer-puffin-phi-v2.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        // =>
        let config = Config::puffin_phi_v2();
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&model_filenames[0])?;
        let model = QMixFormer::new(&config, vb)?;
        let pipeline = TextGeneration::new(
            model,
            tokenizer,
            299792458,
            Some(0.),
            None,
            1.1,
            64,
            false,
            &Device::Cpu,
        );
        //let prompt = "USER: given the following context in square brackets, please answer the question: what is special about today? \n[today is the first day of the year and my birthday]\nnASSISTANT: ";
        //pipeline.run(prompt, 100)?;
        Ok(Self {text_generator: pipeline})
    }

    pub fn run_test(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        //let prompt = "USER: given the following context in square brackets, please answer the question: what is special about today? \n[today is the first day of the year and my birthday]\nnASSISTANT: ";
        self.text_generator.run(prompt, sample_len)?;
        Ok(())
    }
}
