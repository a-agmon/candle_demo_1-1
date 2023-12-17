//our main function will return a Result type
fn main() -> anyhow::Result<()> {
    println!("Starting testing_app...");
    let mut csv_data = csv::Reader::from_path("/Users/alonagmon/Downloads/archive/arxiv_data.csv")?;
    let mut record_iter = csv_data.records();
    if let Some(result) = record_iter.next() {
        let record = result?;
        let terms = record.get(0).unwrap().to_string();
        let title = record.get(1).unwrap().to_string();
        let abstrct = record.get(2).unwrap().to_string();
        println!("terms: {}, title: {}, abstract: {}", terms, title, abstrct)
    }

    Ok(())
}
