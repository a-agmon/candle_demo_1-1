#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
//use models_hf::bert::BertInferenceModel;
use models_hf::quant_phi::QuantPhiTextGenerator;
fn main() -> anyhow::Result<()> {
    println!("starting to check the RAG use case for this model");
    let mut gen_model = QuantPhiTextGenerator::load()?;
    let mut oc_model = models_hf::open_chat::OpenChatTextGenerator::load()?;
    let context = r#"
    Daisy Sarah Bacon (May 23, 1898 to March 1, 1986) was an American pulp fiction magazine editor and writer, best known as the editor of Love Story Magazine from 1928 to 1947. She moved to New York in about 1917, and worked at several jobs before she was hired in 1926 by Street & Smith, a major pulp magazine publisher, to assist with "Friends in Need", an advice column in Love Story Magazine. Two years later she was promoted to editor of the magazine, and stayed in that role for nearly twenty years. Love Story was one of the most successful pulp magazines, and Bacon was frequently interviewed about her role and her opinions of modern romance. Some interviews commented on the contrast between her personal life as a single woman, and the romance in the stories she edited; she did not reveal in these interviews that she had a long affair with a married man, Henry Miller, whose wife was the writer Alice Duer Miller.
        Street & Smith gave Bacon other magazines to edit: Ainslee's in the mid-1930s and Pocket Love in the late 1930s; neither lasted until 1940. In 1940 she took over as editor of Romantic Range, which featured love stories set in the American West, and the following year she was also given the editorship of Detective Story. Romantic Range and Love Story ceased publication in 1947, but in 1948 she became the editor of both The Shadow and Doc Savage, two of Street & Smith's hero pulps. However, Street & Smith shut down all their pulps the following April, and she was let go.
        In 1954 she published a book, Love Story Writer, about writing romance stories. She wrote a romance novel of her own in the 1930s but could not get it published, and in the 1950s also worked on a novel set in the publishing industry. She struggled with depression and alcoholism for much of her life and attempted suicide at least once. After she died, a scholarship fund was established in her name.
    "#;
    let question = "what was tragic about Sarah Bacon's life";
    run_prompt_on_phi(context, question, &mut gen_model)?;
    // print a long seperating line
    println!("\n{:-<40}", "");
    run_prompt_on_oc(context, question, &mut oc_model)?;
    Ok(())
}

// we create a function that send the preompt to the model
fn run_prompt_on_phi(
    context: &str,
    question: &str,
    model_ref: &mut models_hf::quant_phi::QuantPhiTextGenerator,
) -> anyhow::Result<()> {
    let prompt = format!(
        r#"
    USER: Please first carefuly read the following context:
    {} 
    
    USER: Using the given context, please answer the following question very briefly:{} ?
    ASSISTANT:"#,
        context, question
    );
    let answer = model_ref.generate_text(&prompt, 250)?;
    println!("{}", answer);
    Ok(())
}

fn run_prompt_on_oc(
    context: &str,
    question: &str,
    model_ref: &mut models_hf::open_chat::OpenChatTextGenerator,
) -> anyhow::Result<()> {
    let prompt = format!(
        r#"
    GPT4 User: Please first carefuly read the following context:
    {} 
    
    GPT4 User: Using the given context, please answer the following question:{} ?
    <|end_of_turn|>
    GPT4 Assistant::"#,
        context, question
    );
    let answer = model_ref.generate(&prompt, 250)?;
    println!("{}", answer);
    Ok(())
}
