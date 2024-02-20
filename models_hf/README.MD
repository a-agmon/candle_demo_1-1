# Streamlining Serverless ML Inference with Candle in Rust

## Introduction

This repository is a companion to the Medium post: ["Streamlining Serverless ML Inference: Unleashing Candle Framework's Power in Rust"](https://medium.com/towards-data-science/streamlining-serverless-ml-inference-unleashing-candle-frameworks-power-in-rust-c6775d558545). It provides a practical guide to implementing a vector embedding and search REST service using the Candle framework and Axum in Rust.

The post addresses challenges in scaling machine learning models for high-throughput, low-latency environments. It explores using the Candle framework, a Rust-based minimalist ML framework focused on performance and ease of use, ideal for cloud native serverless environments.

Feel free to examine other branches which also add a few LLMs such OpenChat and Phi to the toolset here

## Post (and Repository) Contents

- **Section 2**: Design and components of the vector embedding and search service.
- **Section 3**: Detailed implementation using the Candle framework with a Bert model.
- **Section 4**: Wrapping the model inference in a REST web service using Axum.
- **Section 5**: Creation of embedding artifacts and service setup.
- **Section 6**: Conclusion and further insights.


## Acknowledgements

Special thanks to Hugging Face for the Candle framework and examples

---

For a detailed walkthrough and insights into the challenges and solutions of serverless ML inference at scale, read the full post on Medium.
