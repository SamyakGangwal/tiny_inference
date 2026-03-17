use crate::linear_layer::LinearLayer;

pub mod linear_layer;


fn relu(input: &f32) -> f32 {
    input.max(0f32)
}

fn relu_layer(inputs: &[f32]) -> Vec<f32> {
    

    inputs.iter().map(relu).collect()
}

fn main() {
    let input1 = vec![1.0, -2.0, 3.0, -4.0];

    let output = relu_layer(&input1);

    let input2 = vec![4.0, 6.0];

    let linear_layer = LinearLayer::new(vec![vec![0.5, 0.2], vec![0.3, 0.7]], vec![0.1, 0.2]);


    println!("Linear layer output: {:?}", linear_layer.forward(&input2));

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

    #[test]
    fn test_linear_layer_forward() {
        let linear_layer = LinearLayer::new(vec![vec![0.5, 0.2], vec![0.3, 0.7]], vec![0.1, 0.2]);
        let inputs = vec![4.0, 6.0];
        let expected_output = vec![(0.5 * 4.0) + (0.2 * 6.0) + 0.1, (0.3 * 4.0) + (0.7 * 6.0) + 0.2];
        assert_eq!(linear_layer.forward(&inputs), expected_output);
    }
}
