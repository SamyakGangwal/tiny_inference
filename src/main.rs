pub mod linear_layer;

use std::io;

use crate::linear_layer::linear_layer::LinearLayer;

fn relu(input: &f32) -> f32 {
    input.max(0f32)
}

fn relu_layer(inputs: &Vec<f32>) -> Vec<f32> {
    

    inputs.iter().map(relu).collect()
}

fn main() {
    println!("Enter a list of numbers separated by spaces:");
    let mut input = String::new();

    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read the line");

    let inputs: Vec<f32> = input
        .split_whitespace()
        .map(|s| s.parse().expect("Failed to parse the number"))
        .collect();

    let output = relu_layer(&inputs);

    let linear_layer = LinearLayer::new(vec![vec![0.5, 0.2], vec![0.3, 0.7]], vec![0.1, 0.2]);

    

    println!("ReLU output: {:?}", output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(relu(&5.0), 5.0);
        assert_eq!(relu(&-3.0), 0.0);
        assert_eq!(relu(&0.0), 0.0);
    }

    #[test]
    fn test_relu_layer() {
        let inputs = vec![1.0, -2.0, 3.0, -4.0];
        let expected_output = vec![1.0, 0.0, 3.0, 0.0];
        assert_eq!(relu_layer(&inputs), expected_output);
    }
}
